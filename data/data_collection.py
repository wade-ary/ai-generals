"""Build a fresh PPO-compatible behavioral-cloning dataset from HF replays.

The raw replay source is ``strakammm/generals_io_replays``.  Collection is
deliberately independent of a policy network: observations, history features,
legal-action masks, and temporal features are produced by the same helpers used
by PPO, while actions come from the replay.

Each outer batch contains 1,024 games.  Samples are flushed every 256 replay
turns, but games and their observation histories continue across flushes until
they finish.  Only recorded, non-pass moves become supervised samples.

Run from the repository root with, for example::

    python -m data.data_collection --output-dir data/bc_data

For a small end-to-end check::

    python -m data.data_collection --output-dir /tmp/bc_smoke --max-games 8 \
        --game-batch-size 8 --turns-per-shard 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset

from generals.core.action import compute_valid_move_mask, create_action
from generals.core.game import GameState, create_initial_state, get_observation, step
from networks.common import augment_obs, init_obs_state, obs_to_array


HF_DATASET = "strakammm/generals_io_replays"
HF_SPLIT = "train"
PAD_TO = 24
GAME_BATCH_SIZE = 1_024
TURNS_PER_SHARD = 256
HISTORY_SIZE = 7
TEMPORAL_WINDOW = 512

PASS_ACTION = np.asarray(create_action(to_pass=True), dtype=np.int32)
DELTAS_TO_DIRECTIONS = {
    (-1, 0): 0,  # up
    (1, 0): 1,   # down
    (0, -1): 2,  # left
    (0, 1): 3,   # right
}


@dataclass(frozen=True)
class CleanMoves:
    """Environment-ready actions for one replay.

    ``actions[t, p]`` is the PPO/environment action for player ``p`` at turn
    ``t``. ``has_move`` distinguishes a real replay move from the pass padding
    used to step both seats together.
    """

    actions: np.ndarray  # (T, 2, 5), int32
    has_move: np.ndarray  # (T, 2), bool


def initialise_map(replay: Mapping[str, Any], pad_to: int = PAD_TO) -> np.ndarray:
    """Recreate a replay's initial map and mountain-pad it to ``pad_to`` square.

    Grid encoding is the encoding expected by ``create_initial_state``:
    mountains=-2, empty=0, player generals=1/2, and cities=their army count.
    """

    height = int(replay["mapHeight"])
    width = int(replay["mapWidth"])
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid map dimensions {height}x{width}")
    if height > pad_to or width > pad_to:
        raise ValueError(
            f"map {height}x{width} does not fit in {pad_to}x{pad_to} padding"
        )

    tile_count = height * width

    def row_col(tile: Any) -> tuple[int, int]:
        tile = int(tile)
        if tile < 0 or tile >= tile_count:
            raise ValueError(
                f"tile {tile} is outside a {height}x{width} replay map"
            )
        return divmod(tile, width)

    grid = np.zeros((height, width), dtype=np.int32)

    for tile in replay["mountains"]:
        grid[row_col(tile)] = -2

    cities = replay["cities"]
    city_armies = replay["cityArmies"]
    if len(cities) != len(city_armies):
        raise ValueError(
            f"cities/cityArmies length mismatch: {len(cities)} != {len(city_armies)}"
        )
    for tile, army in zip(cities, city_armies):
        army = int(army)
        if army <= 2:
            raise ValueError(f"city at tile {tile} has invalid initial army {army}")
        grid[row_col(tile)] = army

    generals = replay["generals"]
    if len(generals) != 2:
        raise ValueError(f"expected exactly two generals, found {len(generals)}")
    for player, tile in enumerate(generals):
        location = row_col(tile)
        if grid[location] != 0:
            raise ValueError(f"general {player} overlaps another structure at tile {tile}")
        grid[location] = player + 1

    padded = np.full((pad_to, pad_to), -2, dtype=np.int32)
    padded[:height, :width] = grid
    return padded


def moves_to_env_actions(replay: Mapping[str, Any]) -> CleanMoves:
    """Convert raw replay moves to PPO/environment actions grouped by turn.

    Raw moves are ``[player, start_tile, end_tile, is_50%, turn]``. Missing
    player-turns are represented by pass actions for simulation only and are
    excluded from the supervised dataset through ``has_move``.
    """

    width = int(replay["mapWidth"])
    height = int(replay["mapHeight"])
    tile_count = width * height
    raw_moves: Sequence[Sequence[Any]] = replay["moves"]

    parsed: list[tuple[int, int, int, int, int, int]] = []
    max_turn = -1
    for index, raw in enumerate(raw_moves):
        if len(raw) < 5:
            raise ValueError(f"move {index} has {len(raw)} fields; expected at least 5")
        player, start, end, is_half, turn = (int(value) for value in raw[:5])
        if player not in (0, 1):
            raise ValueError(f"move {index} has invalid player {player}")
        if turn < 0:
            raise ValueError(f"move {index} has negative turn {turn}")
        if start < 0 or start >= tile_count or end < 0 or end >= tile_count:
            raise ValueError(f"move {index} has out-of-bounds tiles {start}->{end}")
        if is_half not in (0, 1):
            raise ValueError(f"move {index} has invalid split flag {is_half}")

        start_row, start_col = divmod(start, width)
        end_row, end_col = divmod(end, width)
        delta = (end_row - start_row, end_col - start_col)
        if delta not in DELTAS_TO_DIRECTIONS:
            raise ValueError(
                f"move {index} is not adjacent: {start}->{end}, delta={delta}"
            )

        parsed.append((turn, player, start_row, start_col,
                       DELTAS_TO_DIRECTIONS[delta], is_half))
        max_turn = max(max_turn, turn)

    num_turns = max_turn + 1
    actions = np.broadcast_to(PASS_ACTION, (num_turns, 2, 5)).copy()
    has_move = np.zeros((num_turns, 2), dtype=np.bool_)
    for turn, player, row, col, direction, is_half in parsed:
        if has_move[turn, player]:
            raise ValueError(f"multiple moves for player {player} on turn {turn}")
        actions[turn, player] = np.array(
            [0, row, col, direction, is_half], dtype=np.int32
        )
        has_move[turn, player] = True

    return CleanMoves(actions=actions, has_move=has_move)


def _stack_trees(trees: Sequence[Any]) -> Any:
    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


@jax.jit
def _ppo_inputs_and_next_history(states: GameState, obs_states: Any):
    """Build exactly the three policy inputs PPO constructs for both seats."""

    batch_size = states.armies.shape[0]
    obs_p0 = jax.vmap(lambda state: get_observation(state, 0))(states)
    obs_p1 = jax.vmap(lambda state: get_observation(state, 1))(states)
    both_obs = jax.tree.map(
        lambda p0, p1: jnp.concatenate((p0, p1), axis=0), obs_p0, obs_p1
    )
    obs_arrays = jax.vmap(obs_to_array)(both_obs)
    augmented, next_obs_states = jax.vmap(augment_obs)(obs_arrays, obs_states)
    masks = jax.vmap(
        lambda obs: compute_valid_move_mask(
            obs.armies, obs.owned_cells, obs.mountains
        )
    )(both_obs)
    temporal = jnp.stack(
        (
            next_obs_states.opponent_army_history,
            next_obs_states.opponent_land_history,
        ),
        axis=1,
    )

    # PPO concatenates all p0 samples followed by all p1 samples. Return a seat
    # axis to make replay-player selection unambiguous: (N, 2, ...).
    def add_seat_axis(array: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack((array[:batch_size], array[batch_size:]), axis=1)

    return (
        add_seat_axis(augmented).astype(jnp.bfloat16),
        add_seat_axis(masks),
        add_seat_axis(temporal),
        next_obs_states,
    )


@jax.jit
def _step_batch(states: GameState, actions: jnp.ndarray):
    return jax.vmap(step)(states, actions)


def _empty_sample_buffer() -> dict[str, list[np.ndarray]]:
    return {
        "obs": [],
        "action_mask": [],
        "temporal": [],
        "action": [],
        "reward": [],
        "game_id": [],
        "player": [],
        "turn": [],
    }


def _append_selected_samples(
    buffer: dict[str, list[np.ndarray]],
    augmented: jax.Array,
    masks: jax.Array,
    temporal: jax.Array,
    actions: np.ndarray,
    selected: np.ndarray,
    game_ids: np.ndarray,
    turn: int,
) -> None:
    """Transfer only selected mover samples from device to the host buffer."""

    game_indices, players = np.nonzero(selected)
    if len(game_indices) == 0:
        return

    # Persist floating-point policy inputs in portable float16 to keep shards
    # compact. The augmented values have already passed through PPO's bfloat16
    # cast; training casts the loaded arrays to its compute dtype.
    buffer["obs"].append(
        np.asarray(augmented[game_indices, players], dtype=np.float16)
    )
    buffer["action_mask"].append(
        np.asarray(masks[game_indices, players], dtype=np.bool_)
    )
    buffer["temporal"].append(
        np.asarray(temporal[game_indices, players], dtype=np.float16)
    )
    buffer["action"].append(actions[game_indices, players].astype(np.int16))
    buffer["reward"].append(
        np.full(len(game_indices), np.nan, dtype=np.float16)
    )
    buffer["game_id"].append(game_ids[game_indices].astype(str))
    buffer["player"].append(players.astype(np.int16))
    buffer["turn"].append(np.full(len(game_indices), turn, dtype=np.int16))


def _write_shard(
    output_dir: Path,
    batch_index: int,
    chunk_index: int,
    turn_start: int,
    turn_stop: int,
    buffer: dict[str, list[np.ndarray]],
) -> Path | None:
    """Atomically write one compressed 256-turn sample shard."""

    if not buffer["action"]:
        return None

    arrays = {key: np.concatenate(parts, axis=0) for key, parts in buffer.items()}
    path = output_dir / (
        f"bc_batch_{batch_index:04d}_chunk_{chunk_index:04d}_"
        f"turns_{turn_start:06d}_{turn_stop:06d}.npz"
    )
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    return path


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _actions_are_legal(
    masks: jax.Array,
    actions: np.ndarray,
    has_move: np.ndarray,
) -> np.ndarray:
    """Return one boolean per game; both recorded moves must be pre-step legal."""

    mask_host = np.asarray(masks)
    legal = np.ones(actions.shape[0], dtype=np.bool_)
    for player in (0, 1):
        movers = np.flatnonzero(has_move[:, player])
        if len(movers) == 0:
            continue
        mover_actions = actions[movers, player]
        legal[movers] &= mask_host[
            movers,
            player,
            mover_actions[:, 1],
            mover_actions[:, 2],
            mover_actions[:, 3],
        ]
    return legal


def _collect_game_batch(
    replays: Sequence[Mapping[str, Any]],
    output_dir: Path,
    batch_index: int,
    turns_per_shard: int,
    winner_lookup: dict[str, int],
) -> dict[str, int]:
    """Simulate one game batch through completion, flushing turn chunks."""

    grids: list[np.ndarray] = []
    cleaned_moves: list[CleanMoves] = []
    game_ids: list[str] = []
    preprocessing_failed = 0
    for replay in replays:
        game_id = str(replay["id"])
        try:
            grids.append(initialise_map(replay))
            cleaned_moves.append(moves_to_env_actions(replay))
            game_ids.append(game_id)
        except (KeyError, TypeError, ValueError) as error:
            preprocessing_failed += 1
            print(f"  cancel {game_id}: cleaning failed: {error}")

    if not grids:
        return {"completed": 0, "draws": 0, "failed": preprocessing_failed, "samples": 0}

    ids = np.asarray(game_ids, dtype=str)
    states = jax.vmap(create_initial_state)(jnp.asarray(np.stack(grids)))
    single_obs_state = init_obs_state(
        PAD_TO,
        PAD_TO,
        history_size=HISTORY_SIZE,
        temporal_window=TEMPORAL_WINDOW,
    )
    obs_states = _stack_trees([single_obs_state] * (2 * len(grids)))

    max_turn = max(clean.actions.shape[0] for clean in cleaned_moves)
    active = np.ones(len(grids), dtype=np.bool_)
    failed = np.zeros(len(grids), dtype=np.bool_)
    completed = np.zeros(len(grids), dtype=np.bool_)
    samples_written = 0

    for chunk_index, turn_start in enumerate(range(0, max_turn, turns_per_shard)):
        turn_stop = min(turn_start + turns_per_shard, max_turn)
        buffer = _empty_sample_buffer()

        for turn in range(turn_start, turn_stop):
            if not np.any(active):
                break

            actions = np.broadcast_to(PASS_ACTION, (len(grids), 2, 5)).copy()
            has_move = np.zeros((len(grids), 2), dtype=np.bool_)
            for game_index, clean in enumerate(cleaned_moves):
                if active[game_index] and turn < clean.actions.shape[0]:
                    actions[game_index] = clean.actions[turn]
                    has_move[game_index] = clean.has_move[turn]

            augmented, masks, temporal, next_obs_states = (
                _ppo_inputs_and_next_history(states, obs_states)
            )

            legal = _actions_are_legal(masks, actions, has_move)
            newly_failed = active & ~legal
            if np.any(newly_failed):
                for game_index in np.flatnonzero(newly_failed):
                    print(
                        f"  cancel {ids[game_index]}: illegal/divergent move at turn {turn}"
                    )
                failed |= newly_failed
                active &= ~newly_failed

            selected = has_move & active[:, None]
            _append_selected_samples(
                buffer, augmented, masks, temporal, actions, selected, ids, turn
            )

            # Failed and already-finished games receive passes. Their states are
            # retained only to keep static batch shapes for JIT compilation.
            step_actions = actions.copy()
            step_actions[~active] = PASS_ACTION
            states, info = _step_batch(states, jnp.asarray(step_actions))
            obs_states = next_obs_states

            winners = np.asarray(info.winner)
            newly_completed = active & (winners >= 0)
            for game_index in np.flatnonzero(newly_completed):
                winner_lookup[ids[game_index]] = int(winners[game_index])
            completed |= newly_completed
            active &= ~newly_completed

            # Exhausting a replay without an environment winner is a draw. This
            # is checked after its final listed turn has been stepped.
            exhausted = np.fromiter(
                (
                    active[i] and turn + 1 >= cleaned_moves[i].actions.shape[0]
                    for i in range(len(cleaned_moves))
                ),
                dtype=np.bool_,
                count=len(cleaned_moves),
            )
            for game_index in np.flatnonzero(exhausted):
                winner_lookup[ids[game_index]] = -1
            completed |= exhausted
            active &= ~exhausted

        shard = _write_shard(
            output_dir,
            batch_index,
            chunk_index,
            turn_start,
            turn_stop,
            buffer,
        )
        chunk_samples = sum(len(part) for part in buffer["action"])
        samples_written += chunk_samples
        if shard is not None:
            print(
                f"  shard {shard.name}: samples={chunk_samples:,}, "
                f"active={int(active.sum()):,}"
            )

    # A non-failed replay with no moves is a draw and never enters the turn loop.
    if max_turn == 0:
        for game_index in np.flatnonzero(~failed):
            winner_lookup[ids[game_index]] = -1
            completed[game_index] = True

    return {
        "completed": int(completed.sum()),
        "draws": sum(winner_lookup.get(game_id) == -1 for game_id in ids),
        "failed": int(failed.sum()) + preprocessing_failed,
        "samples": samples_written,
    }


def collect_dataset(
    output_dir: str | Path,
    *,
    dataset_name: str = HF_DATASET,
    split: str = HF_SPLIT,
    game_batch_size: int = GAME_BATCH_SIZE,
    turns_per_shard: int = TURNS_PER_SHARD,
    max_games: int | None = None,
) -> None:
    """Collect one sequential epoch over the requested Hugging Face split."""

    if game_batch_size <= 0 or turns_per_shard <= 0:
        raise ValueError("game_batch_size and turns_per_shard must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    winner_path = output_dir / "game_winners.json"

    print(f"Loading Hugging Face dataset {dataset_name!r}, split={split!r}...")
    dataset = load_dataset(dataset_name, split=split)
    total_games = len(dataset)
    if max_games is not None:
        total_games = min(total_games, max(0, int(max_games)))
    total_batches = math.ceil(total_games / game_batch_size) if total_games else 0
    print(
        f"Collecting {total_games:,} games once: batch={game_batch_size:,}, "
        f"flush={turns_per_shard} turns, batches={total_batches:,}"
    )

    winner_lookup: dict[str, int] = {}
    totals = {"completed": 0, "draws": 0, "failed": 0, "samples": 0}
    started = time.monotonic()

    for batch_index, start in enumerate(range(0, total_games, game_batch_size)):
        stop = min(start + game_batch_size, total_games)
        print(f"batch {batch_index + 1}/{total_batches}: games[{start}:{stop}]")
        replays = [dataset[index] for index in range(start, stop)]
        stats = _collect_game_batch(
            replays,
            output_dir,
            batch_index,
            turns_per_shard,
            winner_lookup,
        )
        for key in totals:
            totals[key] += stats[key]

        # Persist outcomes after every outer batch so completed work survives an
        # interruption. Failed/cancelled games are intentionally absent.
        _write_json_atomic(winner_path, winner_lookup)
        elapsed = time.monotonic() - started
        print(
            f"  progress games={stop:,}/{total_games:,}, "
            f"samples={totals['samples']:,}, failed={totals['failed']:,}, "
            f"elapsed={elapsed / 60:.1f}m"
        )

    manifest = {
        "dataset": dataset_name,
        "split": split,
        "games_requested": total_games,
        "game_batch_size": game_batch_size,
        "turns_per_shard": turns_per_shard,
        "pad_to": PAD_TO,
        "history_size": HISTORY_SIZE,
        "temporal_window": TEMPORAL_WINDOW,
        "storage_dtypes": {
            "obs": "float16",
            "action_mask": "bool",
            "temporal": "float16",
            "action": "int16",
            "reward": "float16",
            "player": "int16",
            "turn": "int16",
            "game_id": "unicode",
        },
        "reward_placeholder": "NaN",
        "winner_encoding": {"player_0": 0, "player_1": 1, "draw": -1},
        **totals,
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    print(f"Done. Manifest: {output_dir / 'manifest.json'}")
    print(f"Winners: {winner_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/bc_data"))
    parser.add_argument("--dataset", default=HF_DATASET)
    parser.add_argument("--split", default=HF_SPLIT)
    parser.add_argument("--game-batch-size", type=int, default=GAME_BATCH_SIZE)
    parser.add_argument("--turns-per-shard", type=int, default=TURNS_PER_SHARD)
    parser.add_argument("--max-games", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    collect_dataset(
        args.output_dir,
        dataset_name=args.dataset,
        split=args.split,
        game_batch_size=args.game_batch_size,
        turns_per_shard=args.turns_per_shard,
        max_games=args.max_games,
    )


if __name__ == "__main__":
    main()
