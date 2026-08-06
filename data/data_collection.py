"""Barebones BC data collection: HuggingFace replays → fast JAX rollouts.

Pipeline:
  1. Load HF replays
  2. Convert maps (pad 24, hills/mountains) + move lists → env grids / actions
  3. Simulate with jit + lax.scan over time (both seats batched like PPO)
  4. Keep only acting-player pairs (drop passes): obs → action

Run a smoke batch:

    python -m data.data_collection
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset

from generals.core.game import (
    GameState,
    create_initial_state,
    get_observation,
    step as game_step,
)
from generals.core.action import compute_valid_move_mask, create_action
from networks.common import (
    augment_obs,
    init_obs_state,
    obs_to_array,
    reset_done_envs,
)


# ---- Config defaults ----

HF_DATASET = "strakammm/generals_io_replays"
PAD_TO = 24  # dataset maps are 17–23; pad with mountains (-2)


PASS = np.asarray(create_action(to_pass=True), dtype=np.int32)
DELTA_TO_DIR = {
    (-1, 0): 0,  # UP
    (1, 0): 1,   # DOWN
    (0, -1): 2,  # LEFT
    (0, 1): 3,   # RIGHT
}


# ---- HF replay → grid / actions ----


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
    """HF move → env action [pass, row, col, direction, split]."""
    row, col = tile_to_rc(start_tile, width)
    direction = tiles_to_direction(start_tile, end_tile, width)
    return np.array([0, row, col, direction, int(is_50)], dtype=np.int32)


def replay_to_grid(replay: dict[str, Any], pad_to: int = PAD_TO) -> np.ndarray:
    """Numeric grid padded to (pad_to, pad_to) with mountains / hills (-2)."""
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
        grid[r, c] = player + 1

    padded = np.full((pad_to, pad_to), -2, dtype=np.int32)
    padded[:h, :w] = grid
    return padded


def replay_to_actions(replay: dict[str, Any], truncation: int) -> np.ndarray:
    """(T, 2, 5) action sequence; missing turns stay pass."""
    w = int(replay["mapWidth"])
    seq = np.broadcast_to(np.stack([PASS, PASS]), (truncation, 2, 5)).copy()
    for move in replay["moves"]:
        player, start, end, is_50, turn = (int(x) for x in move)
        if turn >= truncation:
            continue
        seq[turn, player] = dataset_move_to_action(start, end, is_50, w)
    return seq


def load_hf_replays(split: str = "train"):
    """Load the HuggingFace generals.io replay dataset split."""
    print(f"Loading {HF_DATASET!r} [{split}]...")
    ds = load_dataset(HF_DATASET)[split]
    print(f"Replays: {len(ds)}")
    return ds


def sample_replays(dataset, n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=n, replace=False)
    return [dataset[int(i)] for i in idxs]


def build_batch(
    replays: list[dict[str, Any]],
    truncation: int,
    pad_to: int = PAD_TO,
) -> tuple[GameState, jnp.ndarray]:
    """Stack replays into vmappable initial states + actions.

    Returns:
        states: batched GameState (N, ...)
        actions: (T, N, 2, 5)  — time-major for lax.scan (PPO style)
    """
    grids = jnp.stack([jnp.asarray(replay_to_grid(r, pad_to)) for r in replays])
    actions_nt = jnp.stack(
        [jnp.asarray(replay_to_actions(r, truncation)) for r in replays]
    )  # (N, T, 2, 5)
    actions = jnp.transpose(actions_nt, (1, 0, 2, 3))  # (T, N, 2, 5)
    states = jax.vmap(create_initial_state)(grids)
    return states, actions


# ---- Fast JAX rollout (scan over T, both seats as 2N) ----


@jax.jit
def collect_trajectories(
    states: GameState,
    expert_actions: jnp.ndarray,
    obs_state_p0,
    obs_state_p1,
):
    """Replay expert actions; record augmented obs for both seats.

    Same structure as PPO / train.bc collect: one lax.scan over time, both
    players concatenated into a 2N batch for obs → array → augment → mask.

    Args:
        states: (N, ...) GameState
        expert_actions: (T, N, 2, 5)
        obs_state_p0 / p1: AugmentedObsState batched (N, ...)

    Returns:
        final_states,
        data = (obs_aug, masks, temporal, actions, terminated, truncated)
            each time-leading with env dim 2N:
            obs_aug   (T, 2N, C, H, W) bfloat16
            masks     (T, 2N, H, W, 4)
            temporal  (T, 2N, 2, temporal_window)
            actions   (T, 2N, 5)
            terminated / truncated (T, 2N)
    """
    n = states.armies.shape[0]
    cat = lambda a, b: jax.tree.map(lambda x, y: jnp.concatenate([x, y]), a, b)
    osp = cat(obs_state_p0, obs_state_p1)

    def scan_body(carry, expert_t):
        # expert_t: (N, 2, 5)
        states, osp = carry

        obs_p0 = jax.vmap(lambda s: get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: get_observation(s, 1))(states)
        obs_both = cat(obs_p0, obs_p1)

        obs_arr = jax.vmap(obs_to_array)(obs_both)
        masks = jax.vmap(
            lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains)
        )(obs_both)
        obs_aug, new_osp = jax.vmap(augment_obs)(obs_arr, osp)
        temporal = jnp.stack(
            [new_osp.opponent_army_history, new_osp.opponent_land_history], axis=1
        )

        a0, a1 = expert_t[:, 0], expert_t[:, 1]
        actions = jnp.concatenate([a0, a1], axis=0)  # (2N, 5)

        new_states, infos = jax.vmap(game_step)(states, expert_t)
        terminated = infos.is_done
        # no env truncation here — demos define their own horizon
        truncated = jnp.zeros_like(terminated)

        dones_both = jnp.concatenate([terminated | truncated, terminated | truncated])
        osp = reset_done_envs(new_osp, dones_both)

        data = (
            obs_aug.astype(jnp.bfloat16),
            masks,
            temporal,
            actions,
            jnp.concatenate([terminated, terminated]),
            jnp.concatenate([truncated, truncated]),
        )
        return (new_states, osp), data

    (final_states, final_osp), rollout = jax.lax.scan(
        scan_body, (states, osp), expert_actions
    )
    final_osp0 = jax.tree.map(lambda x: x[:n], final_osp)
    final_osp1 = jax.tree.map(lambda x: x[n:], final_osp)
    return final_states, rollout, final_osp0, final_osp1


def init_batched_obs_state(num_envs: int, pad_to: int = PAD_TO):
    """Tile a fresh AugmentedObsState across N envs (for both seats)."""
    single = init_obs_state(pad_to, pad_to)
    batched = jax.tree.map(
        lambda x: jnp.tile(x, (num_envs, *([1] * x.ndim))), single
    )
    return batched, jax.tree.map(lambda x: x.copy(), batched)


def acting_pairs_from_rollout(data) -> dict[str, np.ndarray]:
    """Keep only (obs → action) for seats that actually moved (non-pass).

    Simulation still steps both players (passes for idle seats), but we drop
    pass frames so the dataset is only actor decisions.
    """
    obs, masks, temporal, acts, _terminated, _truncated = data
    acts_np = np.asarray(acts)  # (T, 2N, 5)
    keep = acts_np[..., 0] == 0  # pass flag == 0 → real move

    def _take(x):
        arr = np.asarray(x)
        return arr[keep]

    return {
        "obs": _take(obs),
        "masks": _take(masks),
        "temporal": _take(temporal),
        "actions": _take(acts_np),
    }


# ---- High-level barebones entry ----


def collect_batch(
    num_games: int = 512,
    num_steps: int = 64,
    pad_to: int = PAD_TO,
    seed: int = 0,
    split: str = "train",
):
    """Load a small HF batch, simulate, return acting-player pairs only.

    Returns flat arrays: one row per real (non-pass) expert move.
    """
    dataset = load_hf_replays(split=split)
    replays = sample_replays(dataset, num_games, seed=seed)
    states, actions = build_batch(replays, truncation=num_steps, pad_to=pad_to)
    osp0, osp1 = init_batched_obs_state(num_games, pad_to=pad_to)

    print(f"Collecting | games={num_games} steps={num_steps} pad={pad_to}")
    _, data, _, _ = collect_trajectories(states, actions, osp0, osp1)
    jax.block_until_ready(data)

    pairs = acting_pairs_from_rollout(data)
    pairs["game_ids"] = [str(r.get("id", i)) for i, r in enumerate(replays)]
    print(f"Acting pairs: {pairs['actions'].shape[0]}  (dropped passes)")
    return pairs


def main():
    out = collect_batch(num_games=4, num_steps=32)
    print("shapes:")
    for k, v in out.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: len={len(v)}" if hasattr(v, "__len__") else f"  {k}: {v}")


if __name__ == "__main__":
    main()
