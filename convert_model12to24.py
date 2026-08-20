"""Convert a 24x24 HistoryTransformer checkpoint into a 12x12 checkpoint.

Despite the historical filename, this script performs the downsampling needed
for 24x24 BC -> 12x12 PPO initialization. With patch_size=2, it mean-pools the
12x12 spatial positional-embedding grid into a 6x6 grid. The value and two
temporal positional embeddings are copied unchanged, as are all other learned
weights.

Example:
    python convert_model12to24.py \
        checkpoints_run1/BC_S_24_ema_step_100000.eqx \
        checkpoints_run1/BC_S_24_ema_step_100000.yaml \
        checkpoints/BC_S_12_init.eqx

The converted config is written next to the output checkpoint by default.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from ruamel.yaml import YAML

from config import Config
from evals.agent import Agent
from networks import build_network
from networks.transformer import HistoryTransformer


SOURCE_PAD_TO = 24
TARGET_PAD_TO = 12
NUM_SPECIAL_TOKENS = 3


def resize_spatial_positional_embeddings(
    pos_encoding: jnp.ndarray,
    *,
    old_size: tuple[int, int] = (12, 12),
    new_size: tuple[int, int] = (6, 6),
) -> jnp.ndarray:
    """Downsample spatial positions with non-overlapping mean pooling.

    ``pos_encoding`` includes [VALUE, TEMPORAL_ARMY, TEMPORAL_LAND] before
    the row-major spatial patch positions. The special positions are preserved.
    This implementation requires an integer pooling factor in each dimension.
    """
    if pos_encoding.ndim != 2:
        raise ValueError(
            f"Expected positional encoding [tokens, d_model], got {pos_encoding.shape}"
        )

    old_h, old_w = old_size
    new_h, new_w = new_size
    expected_tokens = NUM_SPECIAL_TOKENS + old_h * old_w
    if pos_encoding.shape[0] != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} positional tokens for {old_size}, "
            f"got {pos_encoding.shape[0]}"
        )
    if old_h % new_h or old_w % new_w:
        raise ValueError(
            f"Mean pooling requires integer scale factors: {old_size} -> {new_size}"
        )

    special = pos_encoding[:NUM_SPECIAL_TOKENS]
    spatial = pos_encoding[NUM_SPECIAL_TOKENS:].reshape(
        old_h, old_w, pos_encoding.shape[-1]
    )
    pool_h, pool_w = old_h // new_h, old_w // new_w
    spatial = spatial.reshape(
        new_h, pool_h, new_w, pool_w, pos_encoding.shape[-1]
    ).mean(axis=(1, 3))
    spatial = spatial.reshape(new_h * new_w, pos_encoding.shape[-1])
    return jnp.concatenate([special, spatial], axis=0)


def convert_network(source: HistoryTransformer, target_cfg: Config) -> HistoryTransformer:
    """Create a native 12x12 model containing the source model's learned weights."""
    if not isinstance(source, HistoryTransformer):
        raise TypeError(
            "This converter currently supports only network=history_transformer"
        )
    if source.pad_to != SOURCE_PAD_TO:
        raise ValueError(
            f"Expected a pad_to={SOURCE_PAD_TO} source model, got {source.pad_to}"
        )
    if source.patch_size != 2:
        raise ValueError(
            f"Expected patch_size=2 for 12x12 -> 6x6 pooling, got {source.patch_size}"
        )

    target = build_network(target_cfg, jrandom.PRNGKey(0))
    resized_pos = resize_spatial_positional_embeddings(source.pos_encoding)
    if resized_pos.shape != target.pos_encoding.shape:
        raise ValueError(
            f"Converted positional shape {resized_pos.shape} does not match target "
            f"shape {target.pos_encoding.shape}"
        )

    # Target static fields (grid_size and pad_to) deliberately remain 12. Every
    # learned, shape-compatible component is transplanted from the source.
    target = eqx.tree_at(
        lambda model: (
            model.embedder,
            model.value_token,
            model.pos_encoding,
            model.transformer_layers,
            model.norm_out,
            model.policy_head,
            model.value_head,
            model.temporal_encoder,
            model.temporal_type_embed,
            model.bin_centers,
        ),
        target,
        replace=(
            source.embedder,
            source.value_token,
            resized_pos,
            source.transformer_layers,
            source.norm_out,
            source.policy_head,
            source.value_head,
            source.temporal_encoder,
            source.temporal_type_embed,
            source.bin_centers,
        ),
    )
    return target


def converted_config(source_cfg: Config) -> Config:
    """Return the source architecture configured for native 12x12 inputs."""
    return replace(
        source_cfg,
        run_name=f"{source_cfg.run_name}_12",
        pad_to=TARGET_PAD_TO,
        min_grid_size=TARGET_PAD_TO,
        max_grid_size=TARGET_PAD_TO,
        init_checkpoint="",
        ema_checkpoint="",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Source 24x24 .eqx checkpoint")
    parser.add_argument("config", type=Path, help="YAML config for the source checkpoint")
    parser.add_argument("output", type=Path, help="Destination 12x12 .eqx checkpoint")
    parser.add_argument(
        "--output-config",
        type=Path,
        default=None,
        help="Destination YAML (default: output checkpoint with .yaml suffix)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_config = args.output_config or args.output.with_suffix(".yaml")

    source_agent = Agent.load(args.model, args.config)
    source_cfg = source_agent.config
    if source_cfg.pad_to != SOURCE_PAD_TO:
        raise ValueError(
            f"Source config must have pad_to={SOURCE_PAD_TO}, got {source_cfg.pad_to}"
        )

    target_cfg = converted_config(source_cfg)
    target = convert_network(source_agent.network, target_cfg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(args.output, target)

    yaml = YAML()
    yaml.default_flow_style = False
    with output_config.open("w", encoding="utf-8") as handle:
        yaml.dump(target_cfg.to_dict(), handle)

    # Verify that the serialized weights load into the native target template.
    template = build_network(target_cfg, jrandom.PRNGKey(1))
    reloaded = eqx.tree_deserialise_leaves(args.output, template)
    if reloaded.pos_encoding.shape != (39, source_agent.network.pos_encoding.shape[-1]):
        raise RuntimeError(
            f"Unexpected reloaded positional shape: {reloaded.pos_encoding.shape}"
        )

    print(f"Loaded source: {args.model}")
    print(
        "Positional encoding: "
        f"{source_agent.network.pos_encoding.shape} -> {target.pos_encoding.shape} "
        "(12x12 spatial grid -> 6x6 by 2x2 mean pooling)"
    )
    print(f"Saved converted weights: {args.output}")
    print(f"Saved converted config:  {output_config}")
    print("Use a fresh optimizer state when starting PPO from this checkpoint.")


if __name__ == "__main__":
    main()
