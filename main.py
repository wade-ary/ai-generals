"""Entry point: load config, initialize everything, run training."""

import argparse
import os
from ruamel.yaml import YAML
from dataclasses import fields

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax

from generals.core.env import GeneralsEnv

from config import Config
from networks import get_network_bundle, build_network
from logger import Logger
from train.ppo import train


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training for generals.io")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    # Add every Config field as an optional CLI override
    for f in fields(Config):
        flag = f"--{f.name}"
        if f.type is int or f.type == "int":
            parser.add_argument(flag, type=int, default=None)
        elif f.type is float or f.type == "float":
            parser.add_argument(flag, type=float, default=None)
        elif f.type is bool or f.type == "bool":
            parser.add_argument(flag, action="store_true", default=None)
        else:
            parser.add_argument(flag, default=None)

    return parser.parse_args()


def load_secret(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def main():
    # TF32 matmul: free ~2x speedup on Ampere+ GPUs (H100/A100), no accuracy loss for training
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")

    args = parse_args()

    # Load base config from YAML, then override with any CLI flags
    cfg = Config.from_yaml(args.config)
    for f in fields(Config):
        cli_val = getattr(args, f.name)
        if cli_val is not None:
            object.__setattr__(cfg, f.name, cli_val)

    bundle = get_network_bundle(cfg.network)
    NetworkClass = bundle["cls"]
    print(f"JAX PPO with {cfg.network} ({NetworkClass.__name__})")
    if cfg.min_grid_size != cfg.max_grid_size:
        print(f"Grid: {cfg.min_grid_size}-{cfg.max_grid_size} variable (padded to {cfg.pad_to}), Envs: {cfg.num_envs}")
    else:
        print(f"Grid: {cfg.min_grid_size}x{cfg.min_grid_size} (padded to {cfg.pad_to}), Envs: {cfg.num_envs}")
    print(f"LR: {cfg.lr}")
    print(f"Devices ({jax.device_count()}): {jax.devices()}")
    print()

    key = jrandom.PRNGKey(cfg.seed)
    key, net_key = jrandom.split(key)

    network = build_network(cfg, net_key)

    # LR schedule
    data_mult = 2  # self-play: data from both player seats
    steps_per_iter = cfg.num_epochs * (data_mult * cfg.num_envs * cfg.num_steps // cfg.minibatch_size)

    if cfg.lr_schedule == "power_law":
        # Ataraxos-style: clip(numerator / iter^exponent, min, max)
        # iter is the training iteration (1-indexed), applied per optimizer step
        num = cfg.lr_power_law_numerator
        exp = cfg.lr_power_law_exponent
        lr_min = cfg.lr_power_law_min
        lr_max = cfg.lr_power_law_max
        def lr(step):
            # Convert optimizer steps to 1-indexed training iterations
            iteration = step / steps_per_iter + 1.0
            raw = num / (iteration ** exp)
            return jnp.clip(raw, lr_min, lr_max)
        print(f"LR schedule: power_law clip({num} / iter^{exp}, {lr_min}, {lr_max})")
    else:
        decay_steps = cfg.lr_decay_iters * steps_per_iter
        if decay_steps > 0:
            lr = optax.linear_schedule(
                init_value=cfg.lr,
                end_value=cfg.final_lr,
                transition_steps=decay_steps,
            )
            print(f"LR schedule: linear decay {cfg.lr} -> {cfg.final_lr} over {cfg.lr_decay_iters} iters ({decay_steps} steps)")
        else:
            lr = cfg.lr

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(lr),
    )
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

    # Load checkpoint: try (network, opt_state) tuple first, fall back to network-only
    if cfg.init_checkpoint:
        try:
            network, opt_state = eqx.tree_deserialise_leaves(
                cfg.init_checkpoint, (network, opt_state)
            )
            print(f"Loaded weights + optimizer state from {cfg.init_checkpoint}")
        except Exception:
            network = eqx.tree_deserialise_leaves(cfg.init_checkpoint, network)
            opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
            print(f"Loaded weights from {cfg.init_checkpoint} (fresh optimizer)")

    params, _ = eqx.partition(network, eqx.is_array)
    print(f"Parameters: {sum(x.size for x in jax.tree.leaves(params)):,}")

    run_name = cfg.run_name
    ckpt_dir = os.path.join("checkpoints", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save merged config (base YAML + CLI overrides) to checkpoint dir for reproducibility
    yaml = YAML()
    yaml.default_flow_style = False
    with open(os.path.join(ckpt_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg.to_dict(), f)

    logger = Logger(
        project="averagejoe",
        wandb_token=load_secret(".secrets/wandb_token.txt"),
        hparams=cfg.to_dict(),
        run_name=run_name,
    )

    env_kwargs = dict(
        min_generals_distance=cfg.min_generals_distance,
        max_generals_distance=cfg.max_generals_distance,
        truncation=cfg.truncation,
        pool_size=cfg.pool_size,
        castle_val_range=(cfg.castle_val_min, cfg.castle_val_max),
        num_cities_range=(cfg.num_cities_min, cfg.num_cities_max),
        mountain_density_range=(cfg.mountain_density_min, cfg.mountain_density_max),
    )
    env = GeneralsEnv(
        min_grid_size=cfg.min_grid_size,
        max_grid_size=cfg.max_grid_size,
        pad_to=cfg.pad_to,
        **env_kwargs,
    )
    stages = cfg.curriculum_stages
    if stages:
        stage0 = stages[0]
        env.min_generals_distance = stage0.min_generals_distance
        env.max_generals_distance = stage0.max_generals_distance
        if stage0.castle_val_min is not None:
            env.castle_val_range = (stage0.castle_val_min, stage0.castle_val_max)
        if stage0.num_cities_min is not None:
            env.num_cities_range = (stage0.num_cities_min, stage0.num_cities_max)
        if stage0.gamma is not None:
            object.__setattr__(cfg, 'gamma', stage0.gamma)
        print(f"Curriculum: {len(stages)} stages")
        for i, s in enumerate(stages):
            extra = ""
            if s.gamma is not None:
                extra += f", gamma={s.gamma}"
            wr_str = f"wr>={s.win_rate_threshold:.0%}" if i > 0 else "start"
            print(f"  stage {i}: {wr_str} → dist={s.min_generals_distance}-{s.max_generals_distance}{extra}")

    key, pool_key = jrandom.split(key)
    pool, _ = env.reset(pool_key)

    network, opt_state = train(env, pool, network, optimizer, opt_state, logger, key, cfg, bundle, ckpt_dir, run_name)

    logger.finish()
    final_path = os.path.join(ckpt_dir, f"{run_name}_final.eqx")
    eqx.tree_serialise_leaves(final_path, (network, opt_state))
    print(f"\nDone! Model saved to {final_path}")


if __name__ == "__main__":
    main()
