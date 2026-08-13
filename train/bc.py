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
import atexit
import json
import os
import shutil
import time
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
from ruamel.yaml import YAML

from config import Config
from networks import build_network, get_network_bundle
from networks.common import encode_action


BC_PAD_TO = 24
BC_NUM_ACTIONS = 9 * BC_PAD_TO * BC_PAD_TO
BC_TRAIN_FIELDS = ("obs", "action_mask", "temporal", "action", "reward")


@dataclass(frozen=True)
class BCTrainConfig:
    """Configuration specific to the offline BC training process."""

    model_config: Path = Path("configs/S.yaml")
    data_dir: Path = Path("data")
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


@dataclass(frozen=True)
class StagedShard:
    """One atomically extracted shard ready for memory-mapped consumption."""

    source: Path
    directory: Path
    sample_count: int
    compressed_bytes: int
    extracted_bytes: int
    extraction_seconds: float


@dataclass(frozen=True)
class PreparedBatch:
    """Host-resident batch plus the time spent reading it from mmap files."""

    arrays: tuple[np.ndarray, ...]
    preparation_seconds: float


def _format_bytes(value: int | float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units[:-1]:
        if abs(size) < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}{units[-1]}"


def _stream_extract_shard(source: Path, destination: Path) -> StagedShard:
    """Stream selected NPY members from NPZ without materializing them in RAM."""

    started = time.perf_counter()
    temporary = destination.with_name(f"{destination.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    try:
        with zipfile.ZipFile(source) as archive:
            members = {Path(name).stem: name for name in archive.namelist()}
            missing = set(BC_TRAIN_FIELDS) - set(members)
            if missing:
                raise ValueError(f"{source} is missing fields: {sorted(missing)}")

            for field in BC_TRAIN_FIELDS:
                output = temporary / f"{field}.npy"
                with archive.open(members[field]) as reader, output.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)

        arrays = {
            field: np.load(temporary / f"{field}.npy", mmap_mode="r")
            for field in BC_TRAIN_FIELDS
        }
        sample_count = int(arrays["action"].shape[0])
        for field, array in arrays.items():
            if array.shape[0] != sample_count:
                raise ValueError(
                    f"{source}: {field} has {array.shape[0]} samples, "
                    f"expected {sample_count}"
                )
        del arrays

        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    extracted_bytes = sum(path.stat().st_size for path in destination.glob("*.npy"))
    return StagedShard(
        source=source,
        directory=destination,
        sample_count=sample_count,
        compressed_bytes=source.stat().st_size,
        extracted_bytes=extracted_bytes,
        extraction_seconds=time.perf_counter() - started,
    )


def _open_staged_arrays(shard: StagedShard) -> dict[str, np.memmap]:
    return {
        field: np.load(shard.directory / f"{field}.npy", mmap_mode="r")
        for field in BC_TRAIN_FIELDS
    }


def _shuffled_batch_indices(
    sample_count: int,
    minibatch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Shuffle sample order while retaining large, mmap-friendly read blocks."""

    usable = sample_count - sample_count % minibatch_size
    blocks = np.arange(usable, dtype=np.int64).reshape(-1, minibatch_size)
    rng.shuffle(blocks, axis=0)
    # Randomize examples inside each contiguous read block. Preparing a batch
    # sorts these indices for I/O and then restores this randomized order.
    for block in blocks:
        rng.shuffle(block)
    return [block for block in blocks]


def _prepare_host_batch(
    arrays: Mapping[str, np.memmap], indices: np.ndarray
) -> PreparedBatch:
    """Read one shuffled block efficiently and return contiguous host arrays."""

    started = time.perf_counter()
    order = np.argsort(indices)
    sorted_indices = indices[order]
    restore = np.argsort(order)
    start = int(sorted_indices[0])
    stop = int(sorted_indices[-1]) + 1
    if stop - start != len(indices):
        raise AssertionError("batch indices must form one contiguous I/O block")
    batch = tuple(
        np.ascontiguousarray(arrays[field][start:stop][restore])
        for field in BC_TRAIN_FIELDS
    )
    return PreparedBatch(batch, time.perf_counter() - started)


def _tree_block_until_ready(tree: Any) -> Any:
    return jax.tree.map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready")
        else value,
        tree,
    )


def make_bc_train_step(optimizer: optax.GradientTransformation):
    """Build a compiled policy-only BC update; value training is added later."""

    @eqx.filter_jit
    def train_step(network, opt_state, batch, learning_rate):
        obs, masks, temporal, actions, _rewards = batch

        def loss_fn(net):
            def sample_loss(single_obs, single_mask, single_temporal, action):
                _, _, logprob, entropy, _, probabilities = net(
                    single_obs, single_mask, single_temporal, None, action
                )
                target_index = encode_action(action, BC_PAD_TO)
                correct = jnp.argmax(probabilities) == target_index
                return -logprob, (entropy, correct)

            losses, (entropies, correct) = jax.vmap(sample_loss)(
                obs, masks, temporal, actions
            )
            return losses.mean(), {
                "policy_loss": losses.mean(),
                "entropy": entropies.mean(),
                "accuracy": correct.mean(),
            }

        (loss, metrics), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(network)
        updates, opt_state = optimizer.update(grads, opt_state, network)
        updates = jax.tree.map(lambda update: -learning_rate * update, updates)
        network = eqx.apply_updates(network, updates)
        metrics = dict(metrics, total_loss=loss)
        return network, opt_state, metrics

    return train_step


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


def _append_metrics(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")


def _device_memory_summary(device: Any) -> dict[str, int]:
    try:
        stats = device.memory_stats() or {}
    except (AttributeError, RuntimeError):
        return {}
    result = {}
    for source, target in (
        ("bytes_in_use", "device_bytes_in_use"),
        ("peak_bytes_in_use", "device_peak_bytes_in_use"),
        ("bytes_limit", "device_bytes_limit"),
    ):
        if source in stats:
            result[target] = int(stats[source])
    return result


@eqx.filter_jit
def _update_ema(ema_network: Any, network: Any, decay: float) -> Any:
    return jax.tree.map(
        lambda ema, current: decay * ema + (1.0 - decay) * current
        if eqx.is_array(ema)
        else ema,
        ema_network,
        network,
    )


def _save_checkpoints(
    cfg: BCTrainConfig,
    epoch: int,
    network: Any,
    opt_state: optax.OptState,
    ema_network: Any,
) -> None:
    model_path = cfg.checkpoint_dir / f"BC_S_24_epoch_{epoch:04d}.eqx"
    ema_path = cfg.checkpoint_dir / f"BC_S_24_ema_epoch_{epoch:04d}.eqx"
    eqx.tree_serialise_leaves(model_path, (network, opt_state))
    eqx.tree_serialise_leaves(ema_path, ema_network)
    eqx.tree_serialise_leaves(cfg.checkpoint_dir / "BC_S_24_ema.eqx", ema_network)
    print(f"Saved checkpoint: {model_path}")


def _train_staged_shard(
    *,
    shard: StagedShard,
    epoch: int,
    shard_number: int,
    shard_total: int,
    rng: np.random.Generator,
    network: Any,
    opt_state: optax.OptState,
    ema_network: Any,
    train_step: Any,
    cfg: BCTrainConfig,
    device: Any,
) -> tuple[Any, optax.OptState, Any, dict[str, Any]]:
    """Train one mmap-backed shard with one asynchronously prepared batch."""

    arrays = _open_staged_arrays(shard)
    indices = _shuffled_batch_indices(
        shard.sample_count, cfg.minibatch_size, rng
    )
    dropped = shard.sample_count - len(indices) * cfg.minibatch_size
    if not indices:
        raise ValueError(
            f"{shard.source} has {shard.sample_count} samples, fewer than one "
            f"minibatch of {cfg.minibatch_size}"
        )

    prep_seconds = 0.0
    host_future_wait_seconds = 0.0
    device_enqueue_seconds = 0.0
    device_exposed_seconds = 0.0
    loss_sum = 0.0
    entropy_sum = 0.0
    accuracy_sum = 0.0
    shard_started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="bc-batch") as pool:
        current_host = _prepare_host_batch(arrays, indices[0])
        prep_seconds += current_host.preparation_seconds
        transfer_started = time.perf_counter()
        current_device = jax.device_put(current_host.arrays, device=device)
        device_enqueue_seconds += time.perf_counter() - transfer_started
        ready_started = time.perf_counter()
        _tree_block_until_ready(current_device)
        initial_transfer_seconds = time.perf_counter() - ready_started

        next_host: Future[PreparedBatch] | None = None
        for batch_number in range(len(indices)):
            if batch_number + 1 < len(indices):
                next_host = pool.submit(
                    _prepare_host_batch, arrays, indices[batch_number + 1]
                )
            else:
                next_host = None

            # JAX dispatch is asynchronous: computation for the current batch
            # starts before we wait for and enqueue the following host batch.
            network, opt_state, batch_metrics = train_step(
                network,
                opt_state,
                current_device,
                jnp.asarray(cfg.initial_lr, dtype=jnp.float32),
            )

            next_device = None
            if next_host is not None:
                host_wait_started = time.perf_counter()
                prepared = next_host.result()
                host_wait = time.perf_counter() - host_wait_started
                prep_seconds += prepared.preparation_seconds

                transfer_started = time.perf_counter()
                next_device = jax.device_put(prepared.arrays, device=device)
                device_enqueue_seconds += time.perf_counter() - transfer_started
            else:
                host_wait = 0.0

            _tree_block_until_ready((network, opt_state, batch_metrics))
            # This is the transfer delay actually exposed after current-batch
            # compute completes; zero means transfer was fully hidden.
            if next_device is not None:
                ready_started = time.perf_counter()
                _tree_block_until_ready(next_device)
                device_exposed_seconds += time.perf_counter() - ready_started

            host_future_wait_seconds += host_wait
            loss_sum += float(batch_metrics["policy_loss"])
            entropy_sum += float(batch_metrics["entropy"])
            accuracy_sum += float(batch_metrics["accuracy"])
            ema_network = _update_ema(ema_network, network, cfg.ema_decay)
            current_device = next_device

    _tree_block_until_ready(ema_network)
    elapsed = time.perf_counter() - shard_started
    trained_samples = len(indices) * cfg.minibatch_size
    metrics: dict[str, Any] = {
        "type": "shard",
        "epoch": epoch,
        "shard": shard_number,
        "shards_in_epoch": shard_total,
        "source": shard.source.name,
        "samples": shard.sample_count,
        "trained_samples": trained_samples,
        "dropped_samples": dropped,
        "batches": len(indices),
        "compressed_bytes": shard.compressed_bytes,
        "extracted_bytes": shard.extracted_bytes,
        "expansion_ratio": shard.extracted_bytes / max(shard.compressed_bytes, 1),
        "decompression_seconds": shard.extraction_seconds,
        "decompression_mib_s": (
            shard.extracted_bytes / (1024**2) / max(shard.extraction_seconds, 1e-9)
        ),
        "training_seconds": elapsed,
        "samples_per_second": trained_samples / max(elapsed, 1e-9),
        "host_preparation_seconds": prep_seconds,
        "host_future_wait_seconds": host_future_wait_seconds,
        "device_enqueue_seconds": device_enqueue_seconds,
        "initial_device_wait_seconds": initial_transfer_seconds,
        "device_transfer_exposed_seconds": device_exposed_seconds,
        "policy_loss": loss_sum / len(indices),
        "entropy": entropy_sum / len(indices),
        "accuracy": accuracy_sum / len(indices),
    }
    metrics.update(_device_memory_summary(device))
    del arrays, current_device
    return network, opt_state, ema_network, metrics


def _print_shard_metrics(metrics: Mapping[str, Any]) -> None:
    print(
        f"[epoch {metrics['epoch']:03d} shard "
        f"{metrics['shard']:03d}/{metrics['shards_in_epoch']:03d}] "
        f"{metrics['source']} | samples={metrics['trained_samples']:,} "
        f"({metrics['dropped_samples']:,} dropped) | "
        f"extract={metrics['decompression_seconds']:.1f}s "
        f"@ {metrics['decompression_mib_s']:.0f}MiB/s "
        f"({_format_bytes(metrics['compressed_bytes'])} -> "
        f"{_format_bytes(metrics['extracted_bytes'])}, "
        f"{metrics['expansion_ratio']:.1f}x) | "
        f"train={metrics['training_seconds']:.1f}s "
        f"@ {metrics['samples_per_second']:.0f} samples/s | "
        f"loss={metrics['policy_loss']:.4f} "
        f"acc={metrics['accuracy']:.3f} | "
        f"exposed: shard={metrics['next_shard_wait_seconds']:.2f}s "
        f"cleanup={metrics['cleanup_seconds']:.2f}s "
        f"host-wait={metrics['host_future_wait_seconds']:.2f}s "
        f"h2d={metrics['device_transfer_exposed_seconds']:.2f}s"
    )
    warnings = []
    if metrics["next_shard_wait_seconds"] > 0.05:
        warnings.append("next-shard decompression did not finish before training")
    if metrics["host_preparation_seconds"] >= metrics["training_seconds"]:
        warnings.append("memory-map batch preparation may be limiting throughput")
    if metrics["device_transfer_exposed_seconds"] > 0.05:
        warnings.append("host-to-device transfer was not fully hidden")
    if metrics["decompression_seconds"] >= metrics["training_seconds"]:
        warnings.append("decompression is at least as slow as shard training")
    if warnings:
        print("  PIPELINE WARNING: " + "; ".join(warnings))


def train_bc(
    cfg: BCTrainConfig,
    network: Any,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    ema_network: Any,
) -> tuple[Any, optax.OptState, Any]:
    shards = sorted(cfg.data_dir.glob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"No .npz shards found in {cfg.data_dir}")
    if jax.device_count() != 1:
        raise NotImplementedError(
            "The first BC streaming implementation currently supports one JAX device"
        )

    run_stage = cfg.stage_dir / f"bc-stage-{os.getpid()}"
    run_stage.mkdir(parents=True, exist_ok=False)
    atexit.register(shutil.rmtree, run_stage, True)
    device = jax.devices()[0]
    rng = np.random.default_rng(cfg.seed)
    train_step = make_bc_train_step(optimizer)

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="bc-shard") as pool:
            for epoch in range(1, cfg.max_epochs + 1):
                epoch_shards = list(shards)
                rng.shuffle(epoch_shards)
                print(
                    f"\nEpoch {epoch}/{cfg.max_epochs}: "
                    f"{len(epoch_shards)} shards, order shuffled"
                )
                if epoch == 1:
                    print(
                        "  Note: first-shard training time includes the one-time "
                        "JAX/XLA compilation cost."
                    )

                current = _stream_extract_shard(epoch_shards[0], run_stage / "slot-0")
                for shard_index, _source in enumerate(epoch_shards):
                    next_future: Future[StagedShard] | None = None
                    if shard_index + 1 < len(epoch_shards):
                        free_bytes = shutil.disk_usage(run_stage).free
                        estimated_required = int(current.extracted_bytes * 1.1)
                        if free_bytes < estimated_required:
                            raise OSError(
                                "Insufficient staging space for a second shard: "
                                f"free={_format_bytes(free_bytes)}, estimated "
                                f"required={_format_bytes(estimated_required)}"
                            )
                        next_slot = run_stage / f"slot-{(shard_index + 1) % 2}"
                        next_future = pool.submit(
                            _stream_extract_shard,
                            epoch_shards[shard_index + 1],
                            next_slot,
                        )

                    network, opt_state, ema_network, metrics = _train_staged_shard(
                        shard=current,
                        epoch=epoch,
                        shard_number=shard_index + 1,
                        shard_total=len(epoch_shards),
                        rng=rng,
                        network=network,
                        opt_state=opt_state,
                        ema_network=ema_network,
                        train_step=train_step,
                        cfg=cfg,
                        device=device,
                    )
                    cleanup_started = time.perf_counter()
                    shutil.rmtree(current.directory)
                    metrics["cleanup_seconds"] = (
                        time.perf_counter() - cleanup_started
                    )
                    if next_future is not None:
                        wait_started = time.perf_counter()
                        current = next_future.result()
                        metrics["next_shard_wait_seconds"] = (
                            time.perf_counter() - wait_started
                        )
                    else:
                        metrics["next_shard_wait_seconds"] = 0.0
                    _append_metrics(cfg.metrics_path, metrics)
                    _print_shard_metrics(metrics)

                epoch_metrics = {
                    "type": "epoch",
                    "epoch": epoch,
                    "shard_order": [path.name for path in epoch_shards],
                }
                _append_metrics(cfg.metrics_path, epoch_metrics)
                if epoch % cfg.checkpoint_every == 0 or epoch == cfg.max_epochs:
                    _save_checkpoints(
                        cfg, epoch, network, opt_state, ema_network
                    )
    finally:
        shutil.rmtree(run_stage, ignore_errors=True)
        atexit.unregister(shutil.rmtree)

    return network, opt_state, ema_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/S.yaml"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
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
        opt_state,
        optimizer,
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
        train_bc(
            cfg=cfg,
            network=network,
            opt_state=opt_state,
            optimizer=optimizer,
            ema_network=ema_network,
        )


if __name__ == "__main__":
    main()
