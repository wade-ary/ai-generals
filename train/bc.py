"""Behavioral-cloning training for PPO-compatible replay shards.

This module intentionally starts from fresh weights while retaining the S
transformer architecture and categorical value head. The collected replay
observations are 24x24, so only the spatial input/action dimensions differ
from the original 12x12 S checkpoints.

The remaining training pipeline is layered on top of the initialization in
this file: four-shard in-memory streaming, one-pass shard updates, EMA, and
checkpointing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
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
BC_SHARD_WORKERS = 2
BC_RAM_SLOTS = 4
BC_CHECKPOINT_STEPS = (15_000, 30_000, 45_000, 60_000, 75_000, 90_000, 100_000)


@dataclass(frozen=True)
class BCTrainConfig:
    """Configuration specific to the offline BC training process."""

    model_config: Path = Path("configs/S.yaml")
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path(
        "/content/drive/MyDrive/generals_bc/checkpoints"
    )
    metrics_path: Path = Path(
        "/content/drive/MyDrive/generals_bc/metrics.jsonl"
    )

    seed: int = 44
    max_epochs: int = 50
    minibatch_size: int = 1_024
    value_beta: float = 0.25
    ema_decay: float = 0.999
    max_grad_norm: float = 2.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.9999
    weight_decay: float = 0.01
    checkpoint_steps: tuple[int, ...] = BC_CHECKPOINT_STEPS
    skip_checkpoints: bool = False

    initial_lr: float = 2e-4
    lr_cosine_steps: int = 100_000

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
        minibatch_size=1_024,
        use_bf16=True,
        value_loss="ce",
        num_bins=BC_VALUE_BINS,
        v_min=BC_VALUE_MIN,
        v_max=BC_VALUE_MAX,
        hl_sigma=BC_HL_SIGMA,
    )


def _weight_decay_mask(params: Any) -> Any:
    """Decay learned matrices/embeddings, excluding all one-dimensional state."""

    return jax.tree.map(
        lambda value: eqx.is_array(value) and value.ndim > 1,
        params,
    )


def make_optimizer(cfg: BCTrainConfig) -> optax.GradientTransformation:
    """AdamW with global gradient clipping and masked weight decay."""

    # Learning rate is applied to updates inside the train step so the cosine
    # schedule can be driven by the exact global optimizer step.
    # Decay matrix/embedding weights, but not biases, normalization parameters,
    # or fixed one-dimensional arrays such as the HL-Gauss value-bin centers.
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.scale_by_adam(b1=cfg.adam_beta1, b2=cfg.adam_beta2),
        optax.add_decayed_weights(cfg.weight_decay, mask=_weight_decay_mask),
    )


def _cosine_learning_rate(step: int, maximum: float, schedule_steps: int) -> float:
    """Cosine decay over global optimizer steps, clamped after the schedule."""

    progress = min(step, schedule_steps) / schedule_steps
    return 0.5 * maximum * (1.0 + np.cos(np.pi * progress))


@dataclass(frozen=True)
class LoadedShard:
    """One replay shard fully decompressed into CPU-resident NumPy arrays."""

    source: Path
    arrays: Mapping[str, np.ndarray]
    sample_count: int
    compressed_bytes: int
    resident_bytes: int
    load_seconds: float


@dataclass(frozen=True)
class ShardPlan:
    """The ordered epoch position associated with one source shard."""

    epoch: int
    shard_number: int
    shards_in_epoch: int
    source: Path


@dataclass(frozen=True)
class PreparedBatch:
    """Contiguous host batch plus the time spent gathering it from RAM."""

    arrays: tuple[np.ndarray, ...]
    preparation_seconds: float


@dataclass(frozen=True)
class BatchSpec:
    """One contiguous source block and its in-batch shuffle order."""

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


def _load_shard_to_ram(source: Path) -> LoadedShard:
    """Decompress the five training members of an NPZ directly into CPU RAM."""

    started = time.perf_counter()
    with np.load(source, allow_pickle=False) as archive:
        missing = set(BC_TRAIN_FIELDS) - set(archive.files)
        if missing:
            raise ValueError(f"{source} is missing fields: {sorted(missing)}")
        arrays = {field: archive[field] for field in BC_TRAIN_FIELDS}

    sample_count = int(arrays["action"].shape[0])
    for field, array in arrays.items():
        if array.shape[0] != sample_count:
            raise ValueError(
                f"{source}: {field} has {array.shape[0]} samples, "
                f"expected {sample_count}"
            )
        if not array.flags.c_contiguous:
            arrays[field] = np.ascontiguousarray(array)

    return LoadedShard(
        source=source,
        arrays=arrays,
        sample_count=sample_count,
        compressed_bytes=source.stat().st_size,
        resident_bytes=sum(array.nbytes for array in arrays.values()),
        load_seconds=time.perf_counter() - started,
    )


def _shuffled_batch_specs(
    sample_count: int,
    minibatch_size: int,
    rng: np.random.Generator,
) -> list[BatchSpec]:
    """Shuffle block order and sample order within each contiguous RAM block."""

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
    arrays: Mapping[str, np.ndarray], spec: BatchSpec
) -> PreparedBatch:
    """Gather one shuffled block from RAM into contiguous host arrays."""

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
    if cfg.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if not 0.0 <= cfg.adam_beta1 < 1.0:
        raise ValueError("adam_beta1 must be in [0, 1)")
    if not 0.0 <= cfg.adam_beta2 < 1.0:
        raise ValueError("adam_beta2 must be in [0, 1)")
    if cfg.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if cfg.initial_lr <= 0:
        raise ValueError("initial_lr must be positive")
    if cfg.lr_cosine_steps <= 0:
        raise ValueError("lr_cosine_steps must be positive")
    model_cfg = make_bc_model_config(cfg.model_config)
    model_cfg = replace(
        model_cfg,
        minibatch_size=cfg.minibatch_size,
        max_grad_norm=cfg.max_grad_norm,
    )
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

    optimizer = make_optimizer(cfg)
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
    optimizer_step: int,
    network: Any,
    opt_state: optax.OptState,
    ema_network: Any,
) -> None:
    model_path = cfg.checkpoint_dir / f"BC_S_24_step_{optimizer_step:06d}.eqx"
    ema_path = cfg.checkpoint_dir / f"BC_S_24_ema_step_{optimizer_step:06d}.eqx"
    model_config_path = model_path.with_suffix(".yaml")
    ema_config_path = ema_path.with_suffix(".yaml")
    run_config_path = cfg.checkpoint_dir / (
        f"BC_S_24_step_{optimizer_step:06d}_run_config.json"
    )
    eqx.tree_serialise_leaves(model_path, (network, opt_state))
    eqx.tree_serialise_leaves(ema_path, ema_network)
    eqx.tree_serialise_leaves(cfg.checkpoint_dir / "BC_S_24_ema.eqx", ema_network)
    shutil.copyfile(cfg.checkpoint_dir / "config.yaml", model_config_path)
    shutil.copyfile(cfg.checkpoint_dir / "config.yaml", ema_config_path)
    shutil.copyfile(cfg.run_config_path, run_config_path)
    print(f"Saved checkpoint: {model_path}")
    print(f"Saved EMA checkpoint: {ema_path}")


def _train_loaded_shard(
    *,
    shard: LoadedShard,
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
    """Train one RAM-resident shard with double-buffered host/device batches."""

    arrays = shard.arrays
    batch_specs = _shuffled_batch_specs(
        shard.sample_count, cfg.minibatch_size, rng
    )
    max_steps = max(cfg.checkpoint_steps)
    batch_specs = batch_specs[: max(0, max_steps - optimizer_step)]
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
            if (
                not cfg.skip_checkpoints
                and optimizer_step in cfg.checkpoint_steps
            ):
                _save_checkpoints(
                    cfg, optimizer_step, network, opt_state, ema_network
                )
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
        "shard_load_to_ram_seconds": shard.load_seconds,
        "shard_resident_bytes": shard.resident_bytes,
        "training_seconds": elapsed,
        "samples_per_second": trained_samples / max(elapsed, 1e-9),
        "batch_training_ms": 1_000.0 * batch_training_seconds / len(batch_specs),
        "ram_batch_prepare_ms": 1_000.0 * prep_seconds / len(batch_specs),
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
        f"ram->batch={metrics['ram_batch_prepare_ms']:.1f}ms "
        f"cpu->gpu={metrics['cpu_to_gpu_ms']:.1f}ms "
        f"data-stall={metrics['batch_data_exposed_ms']:.1f}ms | "
        f"shard: load->ram={metrics['shard_load_to_ram_seconds']:.1f}s "
        f"resident={_format_bytes(metrics['shard_resident_bytes'])} "
        f"exposed={metrics['shard_exposed_wait_seconds']:.2f}s"
    )
    warnings = []
    if metrics["shard_exposed_wait_seconds"] > 0.05:
        warnings.append("next in-memory shard pair was not ready after training")
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
    if BC_RAM_SLOTS != 2 * BC_SHARD_WORKERS:
        raise AssertionError(
            "RAM pipeline requires one resident pair and one loading pair"
        )

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
    # Keep epoch ordering independent from the number of random draws used for
    # in-shard batch shuffling. This lets us plan across epoch boundaries early
    # enough to keep both RAM-loading workers occupied.
    order_rng = np.random.default_rng(cfg.seed)
    batch_rng = np.random.default_rng(cfg.seed + 1)
    train_step = make_bc_train_step(
        optimizer,
        cfg.value_beta,
        cfg.ema_decay,
    )
    optimizer_step = 0

    epoch_orders: list[list[Path]] = []
    plans: list[ShardPlan] = []
    for epoch in range(1, cfg.max_epochs + 1):
        epoch_order = list(shards)
        order_rng.shuffle(epoch_order)
        epoch_orders.append(epoch_order)
        plans.extend(
            ShardPlan(epoch, index, len(epoch_order), source)
            for index, source in enumerate(epoch_order, start=1)
        )

    plan_pairs = [
        plans[index : index + BC_SHARD_WORKERS]
        for index in range(0, len(plans), BC_SHARD_WORKERS)
    ]
    epoch_started = 0.0
    epoch_batches = 0
    epoch_trained_samples = 0
    run_started = time.perf_counter()
    max_steps = max(cfg.checkpoint_steps)
    training_complete = False

    def submit_pair(
        pool: ThreadPoolExecutor, pair: list[ShardPlan]
    ) -> list[Future[LoadedShard]]:
        return [pool.submit(_load_shard_to_ram, plan.source) for plan in pair]

    with ThreadPoolExecutor(
        max_workers=BC_SHARD_WORKERS,
        thread_name_prefix="bc-shard-ram",
    ) as pool:
        # Only the first pair is loaded synchronously. Every later pair loads
        # while the GPU consumes the preceding two RAM-resident shards.
        current_pair: list[LoadedShard | None] = [
            future.result() for future in submit_pair(pool, plan_pairs[0])
        ]

        for pair_index, pair_plans in enumerate(plan_pairs):
            next_futures = (
                submit_pair(pool, plan_pairs[pair_index + 1])
                if pair_index + 1 < len(plan_pairs)
                else []
            )
            next_pair: list[LoadedShard] = []

            for pair_offset, plan in enumerate(pair_plans):
                current = current_pair[pair_offset]
                if current is None:
                    raise AssertionError("RAM shard slot was released before training")
                if plan.shard_number == 1:
                    epoch_started = (
                        run_started if plan.epoch == 1 else time.perf_counter()
                    )
                    epoch_batches = 0
                    epoch_trained_samples = 0
                    print(
                        f"\nEpoch {plan.epoch}/{cfg.max_epochs}: "
                        f"{plan.shards_in_epoch} shards, order shuffled"
                    )
                    if plan.epoch == 1:
                        print(
                            "  Note: startup includes the first two RAM loads; "
                            "first-shard training includes one-time JAX/XLA "
                            "compilation."
                        )

                gpu_sampler.start()
                try:
                    network, opt_state, ema_network, optimizer_step, metrics = (
                        _train_loaded_shard(
                            shard=current,
                            epoch=plan.epoch,
                            shard_number=plan.shard_number,
                            shard_total=plan.shards_in_epoch,
                            rng=batch_rng,
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

                # Drop the trained shard immediately. At the end of each pair,
                # wait until both following shards are resident before rotating.
                current_pair[pair_offset] = None
                del current
                if optimizer_step >= max_steps:
                    training_complete = True
                    metrics["shard_exposed_wait_seconds"] = 0.0
                elif pair_offset + 1 == len(pair_plans) and next_futures:
                    wait_started = time.perf_counter()
                    next_pair = [future.result() for future in next_futures]
                    metrics["shard_exposed_wait_seconds"] = (
                        time.perf_counter() - wait_started
                    )
                else:
                    metrics["shard_exposed_wait_seconds"] = 0.0

                _append_metrics(cfg.metrics_path, metrics)
                _print_shard_metrics(metrics)

                if plan.shard_number == plan.shards_in_epoch:
                    epoch_seconds = time.perf_counter() - epoch_started
                    epoch_metrics = {
                        "type": "epoch",
                        "epoch": plan.epoch,
                        "shard_order": [
                            path.name for path in epoch_orders[plan.epoch - 1]
                        ],
                        "minibatches": epoch_batches,
                        "trained_samples": epoch_trained_samples,
                        "epoch_seconds": epoch_seconds,
                        "samples_per_second": (
                            epoch_trained_samples / max(epoch_seconds, 1e-9)
                        ),
                    }
                    _append_metrics(cfg.metrics_path, epoch_metrics)
                    print(
                        f"Epoch {plan.epoch} complete | "
                        f"minibatches={epoch_batches:,} "
                        f"samples={epoch_trained_samples:,} | "
                        f"time={epoch_seconds / 60.0:.1f}m "
                        f"@ {epoch_metrics['samples_per_second']:.0f} samples/s"
                    )
                if training_complete:
                    break

            if training_complete:
                break
            current_pair = list(next_pair)

    if optimizer_step < max_steps:
        raise RuntimeError(
            f"Training data ended at step {optimizer_step:,} before the "
            f"configured final step {max_steps:,}"
        )
    print(f"Training complete at optimizer step {optimizer_step:,}")

    return network, opt_state, ema_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/S.yaml"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )
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
    parser.add_argument("--minibatch-size", type=int, default=1_024)
    parser.add_argument("--initial-lr", type=float, default=2e-4)
    parser.add_argument("--lr-cosine-steps", type=int, default=100_000)
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
        checkpoint_dir=args.checkpoint_dir,
        metrics_path=args.metrics_path,
        seed=args.seed,
        max_epochs=args.max_epochs,
        minibatch_size=args.minibatch_size,
        initial_lr=args.initial_lr,
        lr_cosine_steps=args.lr_cosine_steps,
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
        f"grad_clip={cfg.max_grad_norm} ema={cfg.ema_decay} "
        f"host_buffers=2 device_buffers=2 shard_workers={BC_SHARD_WORKERS} "
        f"ram_shard_slots={BC_RAM_SLOTS}"
    )
    print(
        f"optimizer=AdamW beta1={cfg.adam_beta1} beta2={cfg.adam_beta2} "
        f"weight_decay={cfg.weight_decay}"
    )
    print(
        f"learning_rate=cosine max={cfg.initial_lr:.1e} "
        f"steps={cfg.lr_cosine_steps:,}"
    )
    print(
        "checkpoint steps="
        + ", ".join(f"{step:,}" for step in cfg.checkpoint_steps)
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
