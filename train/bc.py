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
from threading import Event, Thread
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
BC_VALUE_MIN = -1.0
BC_VALUE_MAX = 1.0
BC_VALUE_BINS = 128
BC_HL_SIGMA = 0.04
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
    skip_checkpoints: bool = False

    initial_lr: float = 2e-4
    lr_cosine_steps: int = 100_000

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
        num_bins=BC_VALUE_BINS,
        v_min=BC_VALUE_MIN,
        v_max=BC_VALUE_MAX,
        hl_sigma=BC_HL_SIGMA,
    )


def make_optimizer(max_grad_norm: float) -> optax.GradientTransformation:
    """Adam with PPO's global gradient clipping and externally supplied LR."""

    # Learning rate is applied to updates inside the eventual train step. This
    # keeps Adam state intact when the epoch-level plateau schedule changes LR.
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.scale_by_adam(),
    )


def _cosine_learning_rate(step: int, maximum: float, schedule_steps: int) -> float:
    """Cosine decay over global optimizer steps, clamped after the schedule."""

    progress = min(step, schedule_steps) / schedule_steps
    return 0.5 * maximum * (1.0 + np.cos(np.pi * progress))


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


@dataclass(frozen=True)
class BatchSpec:
    """One contiguous mmap read block and its in-batch shuffle order."""

    start: int
    permutation: np.ndarray


class GPUUtilizationSampler:
    """Sample NVML in the background without synchronizing JAX execution."""

    def __init__(
        self,
        device_index: int = 0,
        interval_seconds: float = 0.2,
        idle_threshold_percent: int = 5,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.idle_threshold_percent = idle_threshold_percent
        self._stop = Event()
        self._thread: Thread | None = None
        self._gpu_samples: list[int] = []
        self.error: str | None = None
        self._pynvml: Any = None
        self._handle: Any = None

        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as exc:
            self.error = str(exc)

    @property
    def available(self) -> bool:
        return self._handle is not None

    def start(self) -> None:
        if not self.available:
            return
        if self._thread is not None:
            raise RuntimeError("GPU utilization sampler is already running")
        self._gpu_samples = []
        self._stop.clear()
        self._thread = Thread(
            target=self._sample_loop,
            name="bc-gpu-utilization",
            daemon=True,
        )
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                utilization = self._pynvml.nvmlDeviceGetUtilizationRates(
                    self._handle
                )
                self._gpu_samples.append(int(utilization.gpu))
            except Exception as exc:
                self.error = str(exc)
                break
            self._stop.wait(self.interval_seconds)

    def stop(self) -> dict[str, int | float]:
        if self._thread is None:
            return {}
        self._stop.set()
        self._thread.join(timeout=max(1.0, 2.0 * self.interval_seconds))
        self._thread = None
        if not self._gpu_samples:
            return {}

        count = len(self._gpu_samples)
        idle_count = sum(
            sample <= self.idle_threshold_percent
            for sample in self._gpu_samples
        )
        return {
            "gpu_utilization_mean_percent": sum(self._gpu_samples) / count,
            "gpu_idle_sample_fraction": idle_count / count,
        }


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


def _shuffled_batch_specs(
    sample_count: int,
    minibatch_size: int,
    rng: np.random.Generator,
) -> list[BatchSpec]:
    """Shuffle sample order while retaining large, mmap-friendly read blocks."""

    usable = sample_count - sample_count % minibatch_size
    starts = np.arange(0, usable, minibatch_size, dtype=np.int64)
    rng.shuffle(starts)
    return [
        BatchSpec(
            start=int(start),
            permutation=rng.permutation(minibatch_size),
        )
        for start in starts
    ]


def _prepare_host_batch(
    arrays: Mapping[str, np.memmap], spec: BatchSpec
) -> PreparedBatch:
    """Read one shuffled block efficiently and return contiguous host arrays."""

    started = time.perf_counter()
    stop = spec.start + len(spec.permutation)
    batch = tuple(
        np.ascontiguousarray(
            arrays[field][spec.start:stop][spec.permutation]
        )
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


def _ready_timestamp(tree: Any) -> float:
    """Return when a JAX tree becomes ready, for a background timing observer."""

    _tree_block_until_ready(tree)
    return time.perf_counter()


def _hl_gauss_value_loss(
    logits: jax.Array,
    target: jax.Array,
    bin_centers: jax.Array,
    sigma: float,
) -> jax.Array:
    """HL-Gauss cross-entropy used by PPO's categorical value head."""

    half_width = (
        (bin_centers[-1] - bin_centers[0])
        / (bin_centers.size - 1)
        / 2.0
    )
    upper = (bin_centers + half_width - target) / sigma
    lower = (bin_centers - half_width - target) / sigma
    target_probs = jax.scipy.stats.norm.cdf(upper) - jax.scipy.stats.norm.cdf(lower)
    target_probs /= jnp.maximum(target_probs.sum(), 1e-8)
    return -jnp.sum(target_probs * jax.nn.log_softmax(logits))


def make_bc_train_step(
    optimizer: optax.GradientTransformation,
    value_beta: float,
    ema_decay: float,
):
    """Build one compiled BC update including EMA and metric accumulation."""

    # Keep target bins outside the differentiated network tree, matching PPO.
    bin_centers = jnp.linspace(BC_VALUE_MIN, BC_VALUE_MAX, BC_VALUE_BINS)

    @eqx.filter_jit
    def train_step(
        network,
        opt_state,
        ema_network,
        metric_sums,
        batch,
        learning_rate,
    ):
        obs, masks, temporal, actions, rewards = batch

        def loss_fn(net):
            def sample_loss(single_obs, single_mask, single_temporal, action, reward):
                _, _, logprob, entropy, value_logits, probabilities = net(
                    single_obs, single_mask, single_temporal, None, action
                )
                target_index = encode_action(action, BC_PAD_TO)
                correct = jnp.argmax(probabilities) == target_index
                action_loss = -logprob
                value_loss = _hl_gauss_value_loss(
                    value_logits,
                    reward.astype(jnp.float32),
                    bin_centers,
                    BC_HL_SIGMA,
                )
                total_loss = action_loss + value_beta * value_loss
                return total_loss, (action_loss, value_loss, entropy, correct)

            losses, (action_losses, value_losses, entropies, correct) = jax.vmap(
                sample_loss
            )(
                obs, masks, temporal, actions, rewards
            )
            return losses.mean(), {
                "policy_loss": action_losses.mean(),
                "value_loss": value_losses.mean(),
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
        ema_network = jax.tree.map(
            lambda ema, current: ema_decay * ema + (1.0 - ema_decay) * current
            if eqx.is_array(ema)
            else ema,
            ema_network,
            network,
        )
        metric_sums = {
            name: metric_sums[name] + value for name, value in metrics.items()
        }
        return network, opt_state, ema_network, metric_sums

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
    if cfg.max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if cfg.value_beta < 0:
        raise ValueError("value_beta must be non-negative")
    if cfg.initial_lr <= 0:
        raise ValueError("initial_lr must be positive")
    if cfg.lr_cosine_steps <= 0:
        raise ValueError("lr_cosine_steps must be positive")
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
    if network.num_bins != BC_VALUE_BINS:
        raise AssertionError(
            f"value bins={network.num_bins}, expected S's "
            f"{BC_VALUE_BINS}-bin head"
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
    optimizer_step: int,
) -> tuple[Any, optax.OptState, Any, int, dict[str, Any]]:
    """Train one mmap-backed shard with one asynchronously prepared batch."""

    arrays = _open_staged_arrays(shard)
    batch_specs = _shuffled_batch_specs(
        shard.sample_count, cfg.minibatch_size, rng
    )
    dropped = shard.sample_count - len(batch_specs) * cfg.minibatch_size
    if not batch_specs:
        raise ValueError(
            f"{shard.source} has {shard.sample_count} samples, fewer than one "
            f"minibatch of {cfg.minibatch_size}"
        )

    prep_seconds = 0.0
    batch_training_seconds = 0.0
    device_transfer_seconds = 0.0
    batch_data_exposed_seconds = 0.0
    metric_sums = {
        "policy_loss": jnp.asarray(0.0, dtype=jnp.float32),
        "value_loss": jnp.asarray(0.0, dtype=jnp.float32),
        "entropy": jnp.asarray(0.0, dtype=jnp.float32),
        "accuracy": jnp.asarray(0.0, dtype=jnp.float32),
        "total_loss": jnp.asarray(0.0, dtype=jnp.float32),
    }
    shard_started = time.perf_counter()
    first_optimizer_step = optimizer_step
    first_learning_rate = _cosine_learning_rate(
        optimizer_step,
        cfg.initial_lr,
        cfg.lr_cosine_steps,
    )
    last_learning_rate = first_learning_rate

    with (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="bc-batch") as pool,
        ThreadPoolExecutor(max_workers=2, thread_name_prefix="bc-ready") as ready_pool,
    ):
        # Warm up both host buffers before training. Batch 1 is transferred
        # while batch 2 is prepared, so batch 2 is ready on the host when the
        # first training operation begins.
        current_host = _prepare_host_batch(arrays, batch_specs[0])
        prep_seconds += current_host.preparation_seconds
        next_host: PreparedBatch | Future[PreparedBatch] | None = None
        if len(batch_specs) > 1:
            next_host = pool.submit(
                _prepare_host_batch,
                arrays,
                batch_specs[1],
            )
        transfer_started = time.perf_counter()
        current_device = jax.device_put(current_host.arrays, device=device)
        _tree_block_until_ready(current_device)
        device_transfer_seconds += time.perf_counter() - transfer_started
        del current_host

        if isinstance(next_host, Future):
            next_host = next_host.result()
            prep_seconds += next_host.preparation_seconds

        for batch_number in range(len(batch_specs)):
            learning_rate = _cosine_learning_rate(
                optimizer_step,
                cfg.initial_lr,
                cfg.lr_cosine_steps,
            )
            training_started = time.perf_counter()
            network, opt_state, ema_network, metric_sums = train_step(
                network,
                opt_state,
                ema_network,
                metric_sums,
                current_device,
                jnp.asarray(learning_rate, dtype=jnp.float32),
            )
            training_ready = ready_pool.submit(
                _ready_timestamp,
                (network, opt_state, ema_network, metric_sums),
            )
            last_learning_rate = learning_rate
            optimizer_step += 1

            next_device = None
            transfer_ready: Future[float] | None = None
            next_transfer_started = 0.0
            following_host: Future[PreparedBatch] | None = None
            if batch_number + 1 < len(batch_specs):
                if isinstance(next_host, Future):
                    next_host = next_host.result()
                    prep_seconds += next_host.preparation_seconds
                if not isinstance(next_host, PreparedBatch):
                    raise AssertionError("next host buffer was not prepared")
                next_transfer_started = time.perf_counter()
                next_device = jax.device_put(next_host.arrays, device=device)
                transfer_ready = ready_pool.submit(
                    _ready_timestamp,
                    next_device,
                )

                if batch_number + 2 < len(batch_specs):
                    following_host = pool.submit(
                        _prepare_host_batch,
                        arrays,
                        batch_specs[batch_number + 2],
                    )

            # Wait only at buffer reuse: the current update must finish before
            # its device slot becomes available, and the next transfer must
            # finish before that batch can train.
            training_ready_at = training_ready.result()
            batch_training_seconds += training_ready_at - training_started
            if transfer_ready is not None:
                transfer_ready_at = transfer_ready.result()
                device_transfer_seconds += (
                    transfer_ready_at - next_transfer_started
                )
                batch_data_exposed_seconds += max(
                    0.0,
                    transfer_ready_at - training_ready_at,
                )

            current_device = next_device
            next_host = following_host

    elapsed = time.perf_counter() - shard_started
    trained_samples = len(batch_specs) * cfg.minibatch_size
    metrics: dict[str, Any] = {
        "type": "shard",
        "epoch": epoch,
        "shard": shard_number,
        "shards_in_epoch": shard_total,
        "source": shard.source.name,
        "samples": shard.sample_count,
        "trained_samples": trained_samples,
        "dropped_samples": dropped,
        "batches": len(batch_specs),
        "optimizer_step_start": first_optimizer_step,
        "optimizer_step_end": optimizer_step,
        "learning_rate_start": first_learning_rate,
        "learning_rate_end": last_learning_rate,
        "shard_decompression_seconds": shard.extraction_seconds,
        "training_seconds": elapsed,
        "samples_per_second": trained_samples / max(elapsed, 1e-9),
        "batch_training_ms": 1_000.0 * batch_training_seconds / len(batch_specs),
        "disk_to_cpu_ms": 1_000.0 * prep_seconds / len(batch_specs),
        "cpu_to_gpu_ms": 1_000.0 * device_transfer_seconds / len(batch_specs),
        "batch_data_exposed_ms": (
            1_000.0 * batch_data_exposed_seconds / len(batch_specs)
        ),
        "policy_loss": float(metric_sums["policy_loss"]) / len(batch_specs),
        "value_loss": float(metric_sums["value_loss"]) / len(batch_specs),
        "total_loss": float(metric_sums["total_loss"]) / len(batch_specs),
        "entropy": float(metric_sums["entropy"]) / len(batch_specs),
        "accuracy": float(metric_sums["accuracy"]) / len(batch_specs),
    }
    del arrays, current_device
    return network, opt_state, ema_network, optimizer_step, metrics


def _print_shard_metrics(metrics: Mapping[str, Any]) -> None:
    gpu_summary = "gpu=unavailable"
    if "gpu_utilization_mean_percent" in metrics:
        gpu_summary = (
            f"gpu={metrics['gpu_utilization_mean_percent']:.0f}% "
            f"idle={metrics['gpu_idle_sample_fraction']:.1%}"
        )
    print(
        f"[epoch {metrics['epoch']:03d} shard "
        f"{metrics['shard']:03d}/{metrics['shards_in_epoch']:03d}] "
        f"{metrics['source']} | samples={metrics['trained_samples']:,} "
        f"({metrics['dropped_samples']:,} dropped) | "
        f"train={metrics['training_seconds']:.1f}s "
        f"@ {metrics['samples_per_second']:.0f} samples/s | "
        f"loss={metrics['total_loss']:.4f} "
        f"(action={metrics['policy_loss']:.4f}, "
        f"value={metrics['value_loss']:.4f}) "
        f"acc={metrics['accuracy']:.3f} | "
        f"step={metrics['optimizer_step_start']:,}->"
        f"{metrics['optimizer_step_end']:,} "
        f"lr={metrics['learning_rate_start']:.2e}->"
        f"{metrics['learning_rate_end']:.2e}"
    )
    print(
        "  PIPELINE | "
        f"{gpu_summary} | batch avg: "
        f"train={metrics['batch_training_ms']:.1f}ms "
        f"disk->cpu={metrics['disk_to_cpu_ms']:.1f}ms "
        f"cpu->gpu={metrics['cpu_to_gpu_ms']:.1f}ms "
        f"data-stall={metrics['batch_data_exposed_ms']:.1f}ms | "
        f"shard: decompress={metrics['shard_decompression_seconds']:.1f}s "
        f"exposed={metrics['shard_exposed_wait_seconds']:.2f}s"
    )
    warnings = []
    if metrics["shard_exposed_wait_seconds"] > 0.05:
        warnings.append("next-shard decompression did not finish before training")
    if metrics["batch_data_exposed_ms"] > 1.0:
        warnings.append("next-batch data was not fully ready when training ended")
    if metrics.get("gpu_idle_sample_fraction", 0.0) > 0.05:
        warnings.append("GPU was near-idle in more than 5% of telemetry samples")
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
    gpu_sampler = GPUUtilizationSampler(device_index=getattr(device, "id", 0))
    if gpu_sampler.available:
        print(
            "GPU telemetry: NVML sampling every "
            f"{gpu_sampler.interval_seconds:.1f}s; idle means <= "
            f"{gpu_sampler.idle_threshold_percent}% utilization"
        )
    else:
        print(f"GPU telemetry unavailable: {gpu_sampler.error}")
    rng = np.random.default_rng(cfg.seed)
    train_step = make_bc_train_step(
        optimizer,
        cfg.value_beta,
        cfg.ema_decay,
    )
    optimizer_step = 0

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="bc-shard") as pool:
            epoch_shards = list(shards)
            rng.shuffle(epoch_shards)
            current_slot = 0
            run_started = time.perf_counter()
            current = _stream_extract_shard(
                epoch_shards[0],
                run_stage / f"slot-{current_slot}",
            )

            for epoch in range(1, cfg.max_epochs + 1):
                epoch_started = (
                    run_started if epoch == 1 else time.perf_counter()
                )
                epoch_batches = 0
                epoch_trained_samples = 0
                print(
                    f"\nEpoch {epoch}/{cfg.max_epochs}: "
                    f"{len(epoch_shards)} shards, order shuffled"
                )
                if epoch == 1:
                    print(
                        "  Note: first-shard training time includes the one-time "
                        "JAX/XLA compilation cost."
                    )

                next_epoch_shards: list[Path] | None = None
                for shard_index, _source in enumerate(epoch_shards):
                    next_future: Future[StagedShard] | None = None
                    next_source: Path | None = None
                    if shard_index + 1 < len(epoch_shards):
                        next_source = epoch_shards[shard_index + 1]
                    elif epoch < cfg.max_epochs:
                        # Decide the next epoch's order early enough to overlap
                        # its first extraction with this epoch's final training.
                        next_epoch_shards = list(shards)
                        rng.shuffle(next_epoch_shards)
                        next_source = next_epoch_shards[0]

                    next_slot_index: int | None = None
                    if next_source is not None:
                        free_bytes = shutil.disk_usage(run_stage).free
                        estimated_required = int(current.extracted_bytes * 1.1)
                        if free_bytes < estimated_required:
                            raise OSError(
                                "Insufficient staging space for a second shard: "
                                f"free={_format_bytes(free_bytes)}, estimated "
                                f"required={_format_bytes(estimated_required)}"
                            )
                        next_slot_index = 1 - current_slot
                        next_slot = run_stage / f"slot-{next_slot_index}"
                        next_future = pool.submit(
                            _stream_extract_shard,
                            next_source,
                            next_slot,
                        )

                    gpu_sampler.start()
                    try:
                        network, opt_state, ema_network, optimizer_step, metrics = (
                            _train_staged_shard(
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
                                optimizer_step=optimizer_step,
                            )
                        )
                    finally:
                        gpu_metrics = gpu_sampler.stop()
                    metrics.update(gpu_metrics)
                    epoch_batches += int(metrics["batches"])
                    epoch_trained_samples += int(metrics["trained_samples"])
                    shutil.rmtree(current.directory)
                    if next_future is not None:
                        wait_started = time.perf_counter()
                        current = next_future.result()
                        if next_slot_index is None:
                            raise AssertionError("next staging slot was not set")
                        current_slot = next_slot_index
                        metrics["shard_exposed_wait_seconds"] = (
                            time.perf_counter() - wait_started
                        )
                    else:
                        metrics["shard_exposed_wait_seconds"] = 0.0
                    _append_metrics(cfg.metrics_path, metrics)
                    _print_shard_metrics(metrics)

                epoch_seconds = time.perf_counter() - epoch_started
                epoch_metrics = {
                    "type": "epoch",
                    "epoch": epoch,
                    "shard_order": [path.name for path in epoch_shards],
                    "minibatches": epoch_batches,
                    "trained_samples": epoch_trained_samples,
                    "epoch_seconds": epoch_seconds,
                    "samples_per_second": (
                        epoch_trained_samples / max(epoch_seconds, 1e-9)
                    ),
                }
                _append_metrics(cfg.metrics_path, epoch_metrics)
                print(
                    f"Epoch {epoch} complete | minibatches={epoch_batches:,} "
                    f"samples={epoch_trained_samples:,} | "
                    f"time={epoch_seconds / 60.0:.1f}m "
                    f"@ {epoch_metrics['samples_per_second']:.0f} samples/s"
                )
                if (
                    not cfg.skip_checkpoints
                    and (
                        epoch % cfg.checkpoint_every == 0
                        or epoch == cfg.max_epochs
                    )
                ):
                    _save_checkpoints(
                        cfg, epoch, network, opt_state, ema_network
                    )
                if next_epoch_shards is not None:
                    epoch_shards = next_epoch_shards
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
    parser.add_argument("--initial-lr", type=float, default=2e-4)
    parser.add_argument("--lr-cosine-steps", type=int, default=100_000)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-checkpoints", action="store_true")
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
        initial_lr=args.initial_lr,
        lr_cosine_steps=args.lr_cosine_steps,
        skip_eval=args.skip_eval,
        skip_checkpoints=args.skip_checkpoints,
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
        f"grad_clip={model_cfg.max_grad_norm} ema={cfg.ema_decay} "
        "host_buffers=2 device_buffers=2"
    )
    print(
        f"learning_rate=cosine max={cfg.initial_lr:.1e} "
        f"steps={cfg.lr_cosine_steps:,}"
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
