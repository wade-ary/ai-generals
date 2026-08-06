"""Behavioral cloning sample: supervised loop on expert move sequences.

Same observation / augment / mask pipeline as self-play PPO, but actions come
from a fixed list instead of the policy. Train with action NLL.

Expert actions shape convention (matches env.step):
    (T, N, 2, 5)  — time, envs, both seats, [pass, row, col, dir, split]

Run as a sketch:

    python -m train.bc
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import equinox as eqx
import optax
from datasets import load_dataset

from generals.core.env import GeneralsEnv, TimeStep
from generals.core.game import get_observation, create_initial_state, step as game_step
from generals.core.action import compute_valid_move_mask, create_action

from networks import get_network_bundle, build_network, obs_to_array, reset_done_envs
from config import Config
from train.rewards import win_lose_reward


# ---- Expert data: HuggingFace replays → grids + move sequences ----

HF_DATASET = "strakammm/generals_io_replays"
PAD_TO = 24  # pad with mountains / hills (-2); dataset maps are 17–23

PASS = np.asarray(create_action(to_pass=True), dtype=np.int32)
DELTA_TO_DIR = {
    (-1, 0): 0,  # UP
    (1, 0): 1,   # DOWN
    (0, -1): 2,  # LEFT
    (0, 1): 3,   # RIGHT
}


def tile_to_rc(tile: int, width: int) -> tuple[int, int]:
    return divmod(int(tile), int(width))


def tiles_to_direction(start_tile: int, end_tile: int, width: int) -> int:
    sr, sc = tile_to_rc(start_tile, width)
    er, ec = tile_to_rc(end_tile, width)
    key = (er - sr, ec - sc)
    if key not in DELTA_TO_DIR:
        raise ValueError(f"non-adjacent move {start_tile}->{end_tile} (delta={key})")
    return DELTA_TO_DIR[key]


def dataset_move_to_action(start_tile: int, end_tile: int, is_50: int, width: int) -> np.ndarray:
    """Convert one HF move into env action [pass, row, col, direction, split]."""
    row, col = tile_to_rc(start_tile, width)
    direction = tiles_to_direction(start_tile, end_tile, width)
    return np.array([0, row, col, direction, int(is_50)], dtype=np.int32)


def replay_to_grid(replay: dict[str, Any], pad_to: int = PAD_TO) -> np.ndarray:
    """Build a padded numeric grid from one HF replay (pad with mountains / hills)."""
    h, w = int(replay["mapHeight"]), int(replay["mapWidth"])
    if h > pad_to or w > pad_to:
        raise ValueError(f"map {(h, w)} exceeds pad_to={pad_to}")

    grid = np.zeros((h, w), dtype=np.int32)
    for tile in replay["mountains"]:
        r, c = tile_to_rc(tile, w)
        grid[r, c] = -2
    for tile, army in zip(replay["cities"], replay["cityArmies"]):
        r, c = tile_to_rc(tile, w)
        grid[r, c] = int(army)
    for player, tile in enumerate(replay["generals"]):
        r, c = tile_to_rc(tile, w)
        grid[r, c] = player + 1  # 1 or 2

    padded = np.full((pad_to, pad_to), -2, dtype=np.int32)
    padded[:h, :w] = grid
    return padded


def replay_to_actions(replay: dict[str, Any], truncation: int) -> np.ndarray:
    """Build a (T, 2, 5) action sequence from HF moves; missing turns stay pass."""
    w = int(replay["mapWidth"])
    seq = np.broadcast_to(np.stack([PASS, PASS]), (truncation, 2, 5)).copy()
    for move in replay["moves"]:
        player, start, end, is_50, turn = (int(x) for x in move)
        if turn >= truncation:
            continue
        seq[turn, player] = dataset_move_to_action(start, end, is_50, w)
    return seq


def load_expert_batch(
    num_envs: int,
    truncation: int,
    pad_to: int = PAD_TO,
    seed: int = 0,
    split: str = "train",
) -> tuple[Any, jnp.ndarray]:
    """Load HF replays and build batched initial states + expert actions.

    Returns:
        states: batched GameState (N, ...) from padded custom maps
        expert_actions: (T, N, 2, 5) int32
    """
    print(f"Loading HuggingFace dataset {HF_DATASET!r}...")
    dataset = load_dataset(HF_DATASET)
    train_dataset = dataset[split]
    print(f"Replays: {len(train_dataset)}")

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(train_dataset), size=num_envs, replace=False)
    replays = [train_dataset[int(i)] for i in idxs]

    grids = jnp.stack([jnp.asarray(replay_to_grid(r, pad_to)) for r in replays])
    actions_nt = jnp.stack(
        [jnp.asarray(replay_to_actions(r, truncation)) for r in replays]
    )  # (N, T, 2, 5)
    expert_actions = jnp.transpose(actions_nt, (1, 0, 2, 3))  # (T, N, 2, 5)
    states = jax.vmap(create_initial_state)(grids)
    return states, expert_actions


# ---- Collect: fixed moves → (obs, mask, temporal, action) -------------------


def _step_no_reset(state, actions, truncation: int):
    """Game step without pool auto-reset (needed for fixed demo maps)."""
    new_state, info = game_step(state, actions)
    terminated = info.is_done
    truncated = (new_state.time >= truncation) & ~terminated
    reward_p0 = jnp.where(info.winner == 0, 1.0, jnp.where(info.winner == 1, -1.0, 0.0))
    rewards = jnp.array([reward_p0, -reward_p0])
    obs_p0 = get_observation(new_state, 0)
    obs_p1 = get_observation(new_state, 1)
    observation = jax.tree.map(lambda a, b: jnp.stack([a, b], axis=0), obs_p0, obs_p1)
    timestep = TimeStep(
        observation=observation,
        reward=rewards,
        terminated=terminated,
        truncated=truncated,
        info=info,
        last_state=new_state,
    )
    return timestep, new_state


@jax.jit(static_argnames=["env", "num_steps", "augment_fn", "truncation"])
def collect_bc_rollout(
    states,
    env,
    expert_actions,
    key,
    num_steps,
    obs_state_p0,
    obs_state_p1,
    augment_fn,
    pool=None,
    truncation: int = 1001,
):
    """Drive N games with expert actions; record both seats as supervised pairs.

    Args:
        states: batched GameState (N, ...)
        expert_actions: (T, N, 2, 5)
        obs_state_*: AugmentedObsState batched (N, ...)

    Returns:
        final_states, data, key, osp0, osp1
        data fields are (T, 2N, ...): obs, masks, temporal, actions, rewards,
        terminated, truncated  (rewards useful if you also fit a value head)
    """
    n = states.armies.shape[0]
    if pool is not None:
        step_fn = lambda s, a: env.step(s, a, pool)
    else:
        step_fn = lambda s, a: _step_no_reset(s, a, truncation)
    cat = lambda a, b: jax.tree.map(lambda x, y: jnp.concatenate([x, y]), a, b)
    osp_init = cat(obs_state_p0, obs_state_p1)

    def scan_body(carry, expert_t):
        # expert_t: (N, 2, 5)
        states, key, osp = carry

        obs_p0 = jax.vmap(lambda s: get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: get_observation(s, 1))(states)
        obs_both = cat(obs_p0, obs_p1)
        obs_arr = jax.vmap(obs_to_array)(obs_both)
        masks = jax.vmap(
            lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains)
        )(obs_both)
        obs_aug, new_osp = jax.vmap(augment_fn)(obs_arr, osp)
        temporal = jnp.stack(
            [new_osp.opponent_army_history, new_osp.opponent_land_history], axis=1
        )

        # Expert actions for both seats → (2N, 5)
        a0, a1 = expert_t[:, 0], expert_t[:, 1]
        actions = jnp.concatenate([a0, a1], axis=0)

        timesteps, new_states = jax.vmap(step_fn)(states, expert_t)

        terminated = timesteps.terminated
        truncated = timesteps.truncated
        winners = timesteps.info.winner

        next_obs_p0 = jax.vmap(lambda s: get_observation(s, 0))(timesteps.last_state)
        next_obs_p1 = jax.vmap(lambda s: get_observation(s, 1))(timesteps.last_state)
        rewards_p0 = win_lose_reward(obs_p0, a0, next_obs_p0, winners)
        winners_p1 = jnp.where(winners >= 0, 1 - winners, winners)
        rewards_p1 = win_lose_reward(obs_p1, a1, next_obs_p1, winners_p1)

        dones_both = jnp.concatenate([terminated | truncated, terminated | truncated])
        osp = reset_done_envs(new_osp, dones_both)

        data = (
            obs_aug.astype(jnp.bfloat16),
            masks,
            temporal,
            actions,
            jnp.concatenate([rewards_p0, rewards_p1]),
            jnp.concatenate([terminated, terminated]),
            jnp.concatenate([truncated, truncated]),
        )
        return (new_states, key, osp), data

    (final_states, final_key, final_osp), rollout = jax.lax.scan(
        scan_body,
        (states, key, osp_init),
        expert_actions,  # scanned over T
        length=num_steps,
    )
    final_osp0 = jax.tree.map(lambda x: x[:n], final_osp)
    final_osp1 = jax.tree.map(lambda x: x[n:], final_osp)
    return final_states, rollout, final_key, final_osp0, final_osp1


# ---- Supervised update ------------------------------------------------------


def bc_update(network, opt_state, batch, optimizer, key, minibatch_size):
    """One BC epoch: minimize -log π(a_expert | s) on shuffled minibatches."""
    obs, masks, temporal, actions, train_mask = batch
    total = obs.shape[0] * obs.shape[1]

    obs_f = obs.reshape(total, *obs.shape[2:])
    masks_f = masks.reshape(total, *masks.shape[2:])
    temporal_f = temporal.reshape(total, *temporal.shape[2:])
    actions_f = actions.reshape(total, -1)
    mask_f = train_mask.reshape(-1)

    # Keep non-truncated steps (same idea as PPO train_mask)
    valid_idx = jnp.where(mask_f > 0.5, size=total, fill_value=0)[0]
    n_valid = int(jnp.maximum(mask_f.sum(), 1.0))
    n_keep = (n_valid // minibatch_size) * minibatch_size
    n_keep = max(n_keep, minibatch_size)
    sample_idx = valid_idx[:n_keep]

    n_samples = sample_idx.shape[0]
    num_batches = max(n_samples // minibatch_size, 1)
    perm = jrandom.permutation(key, n_samples)
    shuffled = sample_idx[perm][: num_batches * minibatch_size]
    idx_mb = shuffled.reshape(num_batches, minibatch_size)

    def scan_body(carry, mb_idx):
        network, opt_state = carry
        mb_obs = obs_f[mb_idx]
        mb_masks = masks_f[mb_idx]
        mb_temporal = temporal_f[mb_idx]
        mb_actions = actions_f[mb_idx]

        def loss_fn(net):
            def single(o, m, td, a):
                # Pass expert action → network returns log π(a|s)
                _, _, lp, ent, _, _ = net(o, m, td, None, a)
                return -lp, ent

            nlls, ents = jax.vmap(single)(mb_obs, mb_masks, mb_temporal, mb_actions)
            return nlls.mean(), {"nll": nlls.mean(), "entropy": ents.mean()}

        (loss, stats), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(network)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(network, eqx.is_array))
        network = eqx.apply_updates(network, updates)
        return (network, opt_state), stats

    (network, opt_state), stats = jax.lax.scan(
        scan_body, (network, opt_state), idx_mb
    )
    # Average stats over minibatches
    stats = jax.tree.map(lambda x: x.mean(), stats)
    return network, opt_state, stats


# ---- Tiny training loop (sample) --------------------------------------------


def train_bc(cfg: Config | None = None, num_iters: int = 3):
    """Minimal BC loop: HF demos → collect on custom maps → supervised update."""
    cfg = cfg or Config(
        pad_to=PAD_TO,
        min_grid_size=PAD_TO,
        max_grid_size=PAD_TO,
        num_envs=4,
        num_steps=32,
    )
    object.__setattr__(cfg, "pad_to", PAD_TO)
    # Smaller net defaults for a smoke sample
    object.__setattr__(cfg, "depth", min(cfg.depth, 2))
    object.__setattr__(cfg, "embed_dim", min(cfg.embed_dim, 64))
    object.__setattr__(cfg, "conv_dim", min(cfg.conv_dim, 64))
    object.__setattr__(cfg, "minibatch_size", min(cfg.minibatch_size, 64))

    bundle = get_network_bundle(cfg.network)
    augment_fn = bundle["augment_obs"]
    init_obs_state_fn = bundle["init_obs_state"]

    key = jrandom.PRNGKey(cfg.seed)
    key, net_key = jrandom.split(key)
    network = build_network(cfg, net_key)
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

    # Custom maps + expert moves from HuggingFace (pad with mountains / hills)
    init_states, expert = load_expert_batch(
        num_envs=cfg.num_envs,
        truncation=cfg.num_steps,
        pad_to=PAD_TO,
        seed=cfg.seed,
    )
    num_steps = int(expert.shape[0])

    env = GeneralsEnv(
        min_grid_size=PAD_TO,
        max_grid_size=PAD_TO,
        pad_to=PAD_TO,
        truncation=cfg.truncation,
        pool_size=min(cfg.pool_size, 256),
    )

    osp_template = jax.tree.map(
        lambda x: jnp.tile(x, (cfg.num_envs, *([1] * x.ndim))),
        init_obs_state_fn(PAD_TO, PAD_TO),
    )

    print(f"BC sample | envs={cfg.num_envs} steps={num_steps} pad={PAD_TO}")

    for it in range(num_iters):
        key, roll_key, upd_key = jrandom.split(key, 3)
        # Replay the same demo maps from t=0 each iter (no pool auto-reset)
        states = jax.tree.map(lambda x: x.copy(), init_states)
        osp0 = jax.tree.map(lambda x: x.copy(), osp_template)
        osp1 = jax.tree.map(lambda x: x.copy(), osp_template)
        states, data, roll_key, osp0, osp1 = collect_bc_rollout(
            states,
            env,
            expert,
            roll_key,
            num_steps,
            osp0,
            osp1,
            augment_fn,
            pool=None,
            truncation=cfg.truncation,
        )
        obs, masks, temporal, actions, _rews, _term, trunc = data
        train_mask = 1.0 - trunc.astype(jnp.float32)
        batch = (obs, masks, temporal, actions, train_mask)

        network, opt_state, stats = bc_update(
            network, opt_state, batch, optimizer, upd_key, cfg.minibatch_size
        )
        print(f"  iter {it + 1}/{num_iters}  nll={float(stats['nll']):.4f}  ent={float(stats['entropy']):.3f}")

    return network


def main():
    train_bc()


if __name__ == "__main__":
    main()
