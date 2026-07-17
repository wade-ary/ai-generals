"""
Pipeline: custom maps → scan one full game → vmap across a batch.

Architecture
------------
1. ``scan_one_game`` — ``jax.lax.scan`` over timesteps for a *single* game
2. ``run_batch``      — ``jax.vmap(scan_one_game)`` over N games, then ``jit``

Fill in ``MAP_INPUT`` / ``MOVES_INPUT`` later. Empty inputs use defaults so this
runs today (same default map × N, pass-actions for every turn).

    python -m examples.custom_game_pipeline
    python -m examples.custom_game_pipeline --batch-size 64
    python -m examples.custom_game_pipeline --replay   # scrub game 0
"""
from __future__ import annotations

import argparse
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals import create_action, create_initial_state, step
from generals.core.game import GameInfo, GameState, get_info

# =============================================================================
# Inputs — leave empty for now; plug specs in later
# =============================================================================
# Map: None / {} → default board. Later e.g. {"size": (10, 10), "generals": [...], ...}
MAP_INPUT: dict[str, Any] | None = None

# Moves: None / {} → all pass. Later: list of length T with (2, 5) actions,
# or array shaped (T, 2, 5) / (N, T, 2, 5).
MOVES_INPUT: list | dict[str, Any] | jnp.ndarray | None = None

BATCH_SIZE = 64
TRUNCATION = 200
SEED = 0
FPS = 8


# Grid encoding (same as create_initial_state):
#   -2 mountain | 0 empty | 1 P0 general | 2 P1 general | >2 city army value
def build_map(spec: dict[str, Any] | None) -> jnp.ndarray:
    """Build a numeric grid from an optional map spec."""
    if not spec:
        h, w = 8, 8
        grid = jnp.zeros((h, w), dtype=jnp.int32)
        grid = grid.at[1, 1].set(1)  # P0 general
        grid = grid.at[6, 6].set(2)  # P1 general
        grid = grid.at[3, 3].set(40)  # city
        grid = grid.at[2, 5].set(-2)  # mountain
        grid = grid.at[5, 2].set(-2)
        return grid

    # --- fill in when you have a real map schema ---
    raise NotImplementedError("MAP_INPUT schema not implemented yet — pass None for default")


def build_action_sequence(
    moves_spec: list | dict[str, Any] | jnp.ndarray | None,
    truncation: int,
    batch_size: int,
) -> jnp.ndarray:
    """
    Build actions for the scan.

    Returns:
        (N, T, 2, 5) int32 — per-game, per-timestep, both players.
    """
    pass_pair = jnp.stack([create_action(to_pass=True), create_action(to_pass=True)])

    if not moves_spec:
        # Empty input → everyone passes every turn (placeholder for BC scripts)
        return jnp.broadcast_to(pass_pair, (batch_size, truncation, 2, 5))

    if isinstance(moves_spec, list):
        seq = jnp.stack([jnp.asarray(a, dtype=jnp.int32) for a in moves_spec])  # (T', 2, 5)
        t = seq.shape[0]
        if t < truncation:
            pad = jnp.broadcast_to(pass_pair, (truncation - t, 2, 5))
            seq = jnp.concatenate([seq, pad], axis=0)
        else:
            seq = seq[:truncation]
        return jnp.broadcast_to(seq, (batch_size, truncation, 2, 5))

    arr = jnp.asarray(moves_spec, dtype=jnp.int32)
    if arr.ndim == 3:  # (T, 2, 5)
        return jnp.broadcast_to(arr[:truncation], (batch_size, truncation, 2, 5))
    if arr.ndim == 4:  # (N, T, 2, 5)
        return arr[:batch_size, :truncation]

    raise NotImplementedError("MOVES_INPUT dict / unsupported shape — pass None, list, or array")


# =============================================================================
# Part 1 — scan: one full game
# =============================================================================
def scan_one_game(
    initial_state: GameState,
    actions: jnp.ndarray,
) -> tuple[GameState, GameInfo]:
    """
    Run one game for T steps with ``lax.scan``.

    Args:
        initial_state: Starting ``GameState`` (single env).
        actions: (T, 2, 5) action sequence.

    Returns:
        states: GameState pytree with leading time axis (T, ...)
        infos:  GameInfo  pytree with leading time axis (T, ...)
    """

    def body(state: GameState, action: jnp.ndarray):
        new_state, info = step(state, action)
        return new_state, (new_state, info)

    _final_state, (states, infos) = jax.lax.scan(body, initial_state, actions)
    return states, infos


# =============================================================================
# Part 2 — vmap: scan across a batch of games
# =============================================================================
@jax.jit
def run_batch(
    initial_states: GameState,
    actions: jnp.ndarray,
) -> tuple[GameState, GameInfo]:
    """
    Parallel full-game rollouts.

    Args:
        initial_states: Batched GameState (N, ...)
        actions: (N, T, 2, 5)

    Returns:
        states: (N, T, ...)
        infos:  (N, T, ...)
    """
    return jax.vmap(scan_one_game)(initial_states, actions)


def build_batch_states(
    map_spec: dict[str, Any] | None,
    batch_size: int,
) -> tuple[jnp.ndarray, GameState]:
    """Create N initial states. Empty map spec → same default grid for every env."""
    grid = build_map(map_spec)
    # Stack identical grids for now; later: different maps per batch index.
    grids = jnp.stack([grid for _ in range(batch_size)])
    initial_states = jax.vmap(create_initial_state)(grids)
    return grid, initial_states


def _tree_index(tree, idx: int):
    """Take element ``idx`` from the leading axis of a pytree."""
    return jax.tree.map(lambda x: x[idx], tree)


def trajectory_to_lists(
    initial_state: GameState,
    states: GameState,
    infos: GameInfo,
) -> tuple[list[GameState], list[GameInfo]]:
    """Convert a single-game (T, ...) trajectory into Python lists for ReplayGUI."""
    t = int(states.time.shape[0])
    states_log = [initial_state] + [_tree_index(states, i) for i in range(t)]
    infos_log = [get_info(initial_state)] + [_tree_index(infos, i) for i in range(t)]
    return states_log, infos_log


def run_pipeline(
    map_spec: dict[str, Any] | None = MAP_INPUT,
    moves_spec: list | dict[str, Any] | jnp.ndarray | None = MOVES_INPUT,
    batch_size: int = BATCH_SIZE,
    truncation: int = TRUNCATION,
    seed: int = SEED,
    replay: bool = False,
    replay_env: int = 0,
) -> tuple[GameState, GameInfo]:
    """
    Build a batch of games, scan each with ``lax.scan``, vmap across the batch.

    Returns batched ``(states, infos)`` with shapes (N, T, ...).
    """
    del seed  # reserved for when maps/moves become stochastic

    _, initial_states = build_batch_states(map_spec, batch_size)
    actions = build_action_sequence(moves_spec, truncation, batch_size)

    # Warmup / run: scan per game, vmap over batch
    states, infos = run_batch(initial_states, actions)

    winners = infos.winner[:, -1]
    n_p0 = int(jnp.sum(winners == 0))
    n_p1 = int(jnp.sum(winners == 1))
    n_ongoing = int(jnp.sum(winners < 0))
    print(
        f"Batch {batch_size} × {truncation} steps  |  "
        f"P0 wins={n_p0}  P1 wins={n_p1}  unfinished={n_ongoing}"
    )

    if replay:
        from generals.gui import ReplayGUI
        from generals.gui.properties import GuiMode

        init_i = _tree_index(initial_states, replay_env)
        states_i = _tree_index(states, replay_env)
        infos_i = _tree_index(infos, replay_env)
        states_log, infos_log = trajectory_to_lists(init_i, states_i, infos_i)

        gui = ReplayGUI(
            states_log[0],
            agent_ids=["P0", "P1"],
            mode=GuiMode.REPLAY,
            start_paused=True,
            fps=FPS,
        )
        print(f"Replay env {replay_env}: SPACE play/pause | ←/→ step | R restart | Q quit")
        gui.play(states_log, infos_log)

    return states, infos


def main():
    parser = argparse.ArgumentParser(description="Scan-one-game + vmap-batch pipeline")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--truncation", type=int, default=TRUNCATION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--replay", action="store_true", help="Open scrubable replay for one env")
    parser.add_argument("--replay-env", type=int, default=0, help="Which batch index to replay")
    args = parser.parse_args()

    run_pipeline(
        map_spec=MAP_INPUT,
        moves_spec=MOVES_INPUT,
        batch_size=args.batch_size,
        truncation=args.truncation,
        seed=args.seed,
        replay=args.replay,
        replay_env=args.replay_env,
    )


if __name__ == "__main__":
    main()
