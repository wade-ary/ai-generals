"""Behavioral-cloning training for PPO-compatible replay shards.

This module intentionally starts from fresh weights while retaining the S
transformer architecture and categorical value head. The collected replay
observations are 24x24, so only the spatial input/action dimensions differ
from the original 12x12 S checkpoints.

The remaining training pipeline is layered on top of the initialization in
this file: double-buffered shard streaming, one-pass shard updates, EMA,
epoch evaluation, and checkpointing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.random as jrandom
import optax
from ruamel.yaml import YAML

from config import Config
from networks import build_network, get_network_bundle


BC_PAD_TO = 24
BC_NUM_ACTIONS = 9 * BC_PAD_TO * BC_PAD_TO


@dataclass(frozen=True)
class BCTrainConfig:
    """Configuration specific to the offline BC training process."""

    model_config: Path = Path("configs/S.yaml")
    data_dir: Path = Path("/content/drive/MyDrive/generals_bc/shards")
    stage_dir: Path = Path("/content/bc_stage")
    checkpoint_dir: Path = Path(
        "/content/drive/MyDrive/generals_bc/checkpoints"
    )
    metrics_path: Path = Path(
        "/content/drive/MyDrive/generals_bc/metrics.jsonl"
    )

    s500_checkpoint: Path = Path("S/S_500/S_500.eqx")
    s500_config: Path = Path("S/S_500/config.yaml")
    s750_checkpoint: Path = Path("S/S_750/S_750.eqx")
    s750_config: Path = Path("S/S_750/config.yaml")

    seed: int = 44
    max_epochs: int = 50
    minibatch_size: int = 2_048
    value_beta: float = 0.25
    ema_decay: float = 0.999
    checkpoint_every: int = 5

    initial_lr: float = 1e-4
    minimum_lr: float = 5e-6
    lr_decay_factor: float = 0.5
    lr_plateau_patience: int = 2

    eval_games_per_opponent: int = 50
    eval_seed: int = 12_345
    skip_eval: bool = False

    @property
    def run_config_path(self) -> Path:
        return self.checkpoint_dir / "run_config.json"


def make_bc_model_config(path: str | Path) -> Config:
    """Load S architecture settings and adapt only its spatial size to BC data."""

    source = Config.from_yaml(str(path))
    if source.network != "history_transformer":
        raise ValueError(
            f"BC requires history_transformer, found {source.network!r}"
        )
    if source.value_loss != "ce":
        raise ValueError(
            "BC retains S's categorical value head; expected value_loss='ce'"
        )

    # Fresh model with S depth/width/patch/value design. Replay observations and
    # targets are 24x24, yielding a 5,184-class policy head.
    return replace(
        source,
        run_name="BC_S_24",
        pad_to=BC_PAD_TO,
        min_grid_size=17,
        max_grid_size=23,
        init_checkpoint="",
        ema_checkpoint="",
        minibatch_size=2_048,
        use_bf16=True,
        value_loss="ce",
        num_bins=128,
        v_min=-1.0,
        v_max=1.0,
        hl_sigma=0.04,
    )


def make_optimizer(max_grad_norm: float) -> optax.GradientTransformation:
    """Adam with PPO's global gradient clipping and externally supplied LR."""

    # Learning rate is applied to updates inside the eventual train step. This
    # keeps Adam state intact when the epoch-level plateau schedule changes LR.
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.scale_by_adam(),
    )


def initialize_training(
    cfg: BCTrainConfig,
) -> tuple[
    Config,
    Any,
    Any,
    optax.OptState,
    optax.GradientTransformation,
    Any,
    jax.Array,
]:
    """Create fresh network, optimizer state, EMA weights, and PRNG state."""

    if not 0.0 < cfg.ema_decay < 1.0:
        raise ValueError("ema_decay must be between zero and one")
    if cfg.minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    if cfg.value_beta < 0:
        raise ValueError("value_beta must be non-negative")
    if cfg.eval_games_per_opponent <= 0 or cfg.eval_games_per_opponent % 2:
        raise ValueError("eval_games_per_opponent must be positive and even")

    model_cfg = make_bc_model_config(cfg.model_config)
    model_cfg = replace(model_cfg, minibatch_size=cfg.minibatch_size)
    bundle = get_network_bundle(model_cfg.network)

    key = jrandom.PRNGKey(cfg.seed)
    key, network_key = jrandom.split(key)
    network = build_network(model_cfg, network_key)

    if network.pad_to != BC_PAD_TO:
        raise AssertionError(f"network pad_to={network.pad_to}, expected 24")
    if model_cfg.num_actions != BC_NUM_ACTIONS:
        raise AssertionError(
            f"action count={model_cfg.num_actions}, expected {BC_NUM_ACTIONS}"
        )
    if network.num_bins != 128:
        raise AssertionError(
            f"value bins={network.num_bins}, expected S's 128-bin head"
        )

    optimizer = make_optimizer(model_cfg.max_grad_norm)
    trainable = eqx.filter(network, eqx.is_array)
    opt_state = optimizer.init(trainable)
    ema_network = jax.tree.map(
        lambda value: value.copy() if eqx.is_array(value) else value,
        network,
    )
    return (
        model_cfg,
        bundle,
        network,
        opt_state,
        optimizer,
        ema_network,
        key,
    )


def _parameter_count(network: Any) -> int:
    arrays = jax.tree.leaves(eqx.filter(network, eqx.is_array))
    return sum(array.size for array in arrays)


def _write_startup_config(cfg: BCTrainConfig, model_cfg: Config) -> None:
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bc": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in cfg.__dict__.items()
        },
        "model": model_cfg.to_dict(),
        "policy_classes": BC_NUM_ACTIONS,
        "initialization": "fresh",
    }
    with cfg.run_config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    # Agent.load consumes this YAML next to BC checkpoints during evaluation.
    yaml = YAML()
    yaml.default_flow_style = False
    with (cfg.checkpoint_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.dump(model_cfg.to_dict(), handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/S.yaml"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/generals_bc/shards"),
    )
    parser.add_argument("--stage-dir", type=Path, default=Path("/content/bc_stage"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/generals_bc/checkpoints"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("/content/drive/MyDrive/generals_bc/metrics.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--minibatch-size", type=int, default=2_048)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="build and validate fresh model state, then exit",
    )
    return parser.parse_args()


def main() -> None:
    # Match PPO's high-throughput matmul policy on Ampere and newer GPUs.
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")
    args = parse_args()
    cfg = BCTrainConfig(
        model_config=args.config,
        data_dir=args.data_dir,
        stage_dir=args.stage_dir,
        checkpoint_dir=args.checkpoint_dir,
        metrics_path=args.metrics_path,
        seed=args.seed,
        max_epochs=args.max_epochs,
        minibatch_size=args.minibatch_size,
        skip_eval=args.skip_eval,
    )

    devices = jax.devices()
    print(f"JAX devices: {devices}")
    (
        model_cfg,
        _,
        network,
        _,
        _,
        ema_network,
        _,
    ) = initialize_training(cfg)
    _write_startup_config(cfg, model_cfg)

    print("Initialized fresh BC model")
    print(
        f"architecture=S history_transformer depth={model_cfg.depth} "
        f"embed={model_cfg.embed_dim} heads={model_cfg.n_head}"
    )
    print(
        f"input={BC_PAD_TO}x{BC_PAD_TO} actions={BC_NUM_ACTIONS:,} "
        f"value_head={model_cfg.num_bins}-bin HL-Gauss"
    )
    print(
        f"minibatch={cfg.minibatch_size:,} bf16={model_cfg.use_bf16} "
        f"grad_clip={model_cfg.max_grad_norm} ema={cfg.ema_decay}"
    )
    print(f"parameters={_parameter_count(network):,}")
    print(f"checkpoint config: {cfg.run_config_path}")

    # Ensure the EMA tree is structurally identical at initialization.
    if jax.tree.structure(network) != jax.tree.structure(ema_network):
        raise AssertionError("EMA tree does not match network tree")

    if not args.initialize_only:
        raise NotImplementedError(
            "BC initialization is complete; streaming and epoch training are "
            "the next implementation layer. Run with --initialize-only for now."
        )


if __name__ == "__main__":
    main()
