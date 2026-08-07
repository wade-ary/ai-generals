"""Build a fresh PPO-compatible behavioral-cloning dataset from HF replays.

The raw replay source is ``strakammm/generals_io_replays``.  Collection is
deliberately independent of a policy network: observations, history features,
legal-action masks, and temporal features are produced by the same helpers used
by PPO, while actions come from the replay.

A fixed pool contains 1,024 games. Completed slots are immediately replaced by
the next replay, just like PPO auto-reset. Samples are flushed every 256 global
rollout steps, while unfinished games and their histories continue across
flushes. Only recorded, non-pass, legal moves become supervised samples.

Run from the repository root with, for example::

    python -m data.data_collection --output-dir data/bc_data

For a small end-to-end check::

    python -m data.data_collection --output-dir /tmp/bc_smoke --max-games 8 \
        --game-batch-size 8 --turns-per-shard 16
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset

from generals.core.action import compute_valid_move_mask, create_action
from generals.core.game import (
    GameState,
    create_initial_state,
    get_observation,
    step,
)
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


class AsyncArchiver:
    """Copy completed files to durable storage without blocking collection."""

    def __init__(self, archive_dir: Path, delete_source: bool = False) -> None:
        self.archive_dir = archive_dir
        self.delete_source = delete_source
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bc-archive"
        )
        self._pending: list[concurrent.futures.Future[Path]] = []

    def submit(self, source: Path) -> None:
        """Queue one atomic copy and report any earlier background failure."""

        self._raise_completed_errors()
        self._pending.append(self._executor.submit(self._copy, source))

    def _copy(self, source: Path) -> Path:
        destination = self.archive_dir / source.name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            # Never leave a partial file looking like a completed shard.
            if temporary.exists():
                temporary.unlink()
        if self.delete_source:
            source.unlink()
        return destination

    def _raise_completed_errors(self) -> None:
        remaining: list[concurrent.futures.Future[Path]] = []
        for future in self._pending:
            if future.done():
                future.result()
            else:
                remaining.append(future)
        self._pending = remaining

    def close(self) -> None:
        """Wait for queued copies and propagate background exceptions."""

        try:
            for future in self._pending:
                future.result()
        finally:
            self._executor.shutdown(wait=True)
        self._pending.clear()


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
    """Step replay games with the exact same turn resolution used by PPO."""

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
    turns: np.ndarray,
) -> None:
    """Transfer only selected mover samples from device to the host buffer."""

    game_indices, players = np.nonzero(selected)
    if len(game_indices) == 0:
        return

    # Selection count varies every step. Indexing a JAX device array with these
    # variable-length host indices can trigger a fresh XLA gather compilation
    # for every distinct count. Transfer fixed-shape rollout outputs once, then
    # perform the dynamic mover selection with NumPy on the host.
    augmented_host = np.asarray(augmented)
    masks_host = np.asarray(masks)
    temporal_host = np.asarray(temporal)

    # Persist floating-point policy inputs in portable float16 to keep shards
    # compact. The augmented values have already passed through PPO's bfloat16
    # cast; training casts the loaded arrays to its compute dtype.
    buffer["obs"].append(
        np.asarray(augmented_host[game_indices, players], dtype=np.float16)
    )
    buffer["action_mask"].append(
        np.asarray(masks_host[game_indices, players], dtype=np.bool_)
    )
    buffer["temporal"].append(
        np.asarray(temporal_host[game_indices, players], dtype=np.float16)
    )
    buffer["action"].append(actions[game_indices, players].astype(np.int16))
    buffer["reward"].append(
        np.full(len(game_indices), np.nan, dtype=np.float16)
    )
    buffer["game_id"].append(game_ids[game_indices].astype(str))
    buffer["player"].append(players.astype(np.int16))
    buffer["turn"].append(turns[game_indices].astype(np.int16))


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
    """Return pre-step legality for each player action; passes/padding are true."""

    mask_host = np.asarray(masks)
    legal = np.ones(actions.shape[:2], dtype=np.bool_)
    for player in (0, 1):
        movers = np.flatnonzero(has_move[:, player])
        if len(movers) == 0:
            continue
        mover_actions = actions[movers, player]
        legal[movers, player] = mask_host[
            movers,
            player,
            mover_actions[:, 1],
            mover_actions[:, 2],
            mover_actions[:, 3],
        ]
    return legal


def _illegal_move_event(
    states: GameState,
    actions: np.ndarray,
    game_id: str,
    game_index: int,
    player: int,
    turn: int,
) -> dict[str, Any]:
    """Capture enough state to distinguish isolated no-ops from divergence."""

    action = actions[game_index, player].astype(np.int32)
    _, row, col, direction, _ = action
    deltas = np.asarray(((-1, 0), (1, 0), (0, -1), (0, 1)))
    dest_row, dest_col = (np.array((row, col)) + deltas[direction]).tolist()
    armies = np.asarray(states.armies[game_index])
    ownership = np.asarray(states.ownership[game_index])
    mountains = np.asarray(states.mountains[game_index])
    source_owned = bool(ownership[player, row, col])
    source_army = int(armies[row, col])
    destination_in_bounds = (
        0 <= dest_row < armies.shape[0] and 0 <= dest_col < armies.shape[1]
    )
    destination_mountain = (
        bool(mountains[dest_row, dest_col]) if destination_in_bounds else None
    )
    if ownership[0, row, col]:
        source_owner = 0
    elif ownership[1, row, col]:
        source_owner = 1
    else:
        source_owner = -1

    reasons: list[str] = []
    if not source_owned:
        reasons.append("source_not_owned")
    if source_army <= 1:
        reasons.append("source_army_le_1")
    if not destination_in_bounds:
        reasons.append("destination_out_of_bounds")
    elif destination_mountain:
        reasons.append("destination_mountain")

    return {
        "game_id": game_id,
        "turn": int(turn),
        "state_time": int(np.asarray(states.time[game_index])),
        "player": int(player),
        "priority_system": "ppo_current",
        "action": action.tolist(),
        "both_actions": actions[game_index].astype(np.int32).tolist(),
        "source": [int(row), int(col)],
        "destination": [int(dest_row), int(dest_col)],
        "source_owner": source_owner,
        "source_army": source_army,
        "destination_in_bounds": destination_in_bounds,
        "destination_mountain": destination_mountain,
        "reasons": reasons or ["unknown_mask_failure"],
    }


def _set_tree_rows(tree: Any, indices: np.ndarray, rows: Any) -> Any:
    """Functionally replace selected rows in a JAX pytree."""

    device_indices = jnp.asarray(indices, dtype=jnp.int32)
    return jax.tree.map(
        lambda current, replacement: current.at[device_indices].set(replacement),
        tree,
        rows,
    )


def _collect_rolling_pool(
    dataset: Sequence[Mapping[str, Any]],
    total_games: int,
    output_dir: Path,
    pool_size: int,
    turns_per_shard: int,
    winner_lookup: dict[str, int],
    illegal_move_events: list[dict[str, Any]],
    archiver: AsyncArchiver | None = None,
) -> dict[str, int]:
    """Collect all replays with immediate PPO-style slot replacement.

    JAX array shapes remain fixed at ``pool_size``. When a game completes, only
    that state's row is replaced and both of its PPO observation-history rows
    are zeroed. Other slots continue without interruption or recompilation.
    """

    single_obs_state = init_obs_state(
        PAD_TO,
        PAD_TO,
        history_size=HISTORY_SIZE,
        temporal_window=TEMPORAL_WINDOW,
    )
    slot_moves: list[CleanMoves | None] = [None] * pool_size
    slot_ids = np.full(pool_size, "", dtype=object)
    local_turns = np.zeros(pool_size, dtype=np.int32)
    active = np.zeros(pool_size, dtype=np.bool_)
    next_source_index = 0
    completed = 0
    failed = 0
    draws = 0
    samples_written = 0
    global_step = 0
    chunk_index = 0

    def load_slots(slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fill slots sequentially, skipping structurally malformed replays."""

        nonlocal next_source_index, failed
        loaded_slots: list[int] = []
        loaded_grids: list[np.ndarray] = []
        for slot in slots.tolist():
            while next_source_index < total_games:
                replay = dataset[next_source_index]
                next_source_index += 1
                game_id = str(replay.get("id", f"source_index_{next_source_index - 1}"))
                try:
                    grid = initialise_map(replay)
                    clean = moves_to_env_actions(replay)
                except (KeyError, TypeError, ValueError) as error:
                    failed += 1
                    print(f"  cancel {game_id}: cleaning failed: {error}")
                    continue
                slot_moves[slot] = clean
                slot_ids[slot] = game_id
                local_turns[slot] = 0
                active[slot] = True
                loaded_slots.append(slot)
                loaded_grids.append(grid)
                break
            else:
                slot_moves[slot] = None
                slot_ids[slot] = ""
                local_turns[slot] = 0
                active[slot] = False
        return np.asarray(loaded_slots, dtype=np.int32), np.asarray(loaded_grids)

    initial_slots, initial_grids = load_slots(np.arange(pool_size, dtype=np.int32))
    if len(initial_slots) == 0:
        return {
            "completed": 0,
            "draws": 0,
            "failed": failed,
            "samples": 0,
            "illegal_moves": 0,
            "slot_refills": 0,
        }

    # Initial fill is dense because pool_size <= total_games and malformed
    # entries are replaced from the remaining source queue.
    if len(initial_slots) != pool_size:
        kept_slots = initial_slots.tolist()
        slot_moves = [slot_moves[slot] for slot in kept_slots]
        slot_ids = np.asarray([slot_ids[slot] for slot in kept_slots], dtype=object)
        local_turns = np.zeros(len(kept_slots), dtype=np.int32)
        active = np.ones(len(kept_slots), dtype=np.bool_)
        pool_size = len(kept_slots)
    states = jax.vmap(create_initial_state)(jnp.asarray(initial_grids))
    obs_states = _stack_trees([single_obs_state] * (2 * pool_size))
    slot_refills = 0

    while np.any(active):
        buffer = _empty_sample_buffer()
        chunk_start = global_step
        chunk_illegal_start = len(illegal_move_events)

        for _ in range(turns_per_shard):
            if not np.any(active):
                break

            actions = np.broadcast_to(PASS_ACTION, (pool_size, 2, 5)).copy()
            has_move = np.zeros((pool_size, 2), dtype=np.bool_)
            for slot in np.flatnonzero(active):
                clean = slot_moves[slot]
                assert clean is not None
                turn = int(local_turns[slot])
                if turn < clean.actions.shape[0]:
                    actions[slot] = clean.actions[turn]
                    has_move[slot] = clean.has_move[turn]

            augmented, masks, temporal, next_obs_states = (
                _ppo_inputs_and_next_history(states, obs_states)
            )
            legal_actions = _actions_are_legal(masks, actions, has_move)
            illegal_actions = has_move & active[:, None] & ~legal_actions
            for slot, player in zip(*np.nonzero(illegal_actions)):
                event = _illegal_move_event(
                    states,
                    actions,
                    str(slot_ids[slot]),
                    slot,
                    player,
                    int(local_turns[slot]),
                )
                illegal_move_events.append(event)

            selected = has_move & legal_actions & active[:, None]
            _append_selected_samples(
                buffer,
                augmented,
                masks,
                temporal,
                actions,
                selected,
                slot_ids,
                local_turns,
            )

            step_actions = actions.copy()
            step_actions[~active] = PASS_ACTION
            states, info = _step_batch(states, jnp.asarray(step_actions))
            obs_states = next_obs_states
            local_turns[active] += 1
            winners = np.asarray(info.winner)

            finished_slots: list[int] = []
            for slot in np.flatnonzero(active):
                clean = slot_moves[slot]
                assert clean is not None
                winner = int(winners[slot])
                exhausted = local_turns[slot] >= clean.actions.shape[0]
                if winner >= 0 or exhausted:
                    outcome = winner if winner >= 0 else -1
                    winner_lookup[str(slot_ids[slot])] = outcome
                    draws += int(outcome == -1)
                    completed += 1
                    active[slot] = False
                    finished_slots.append(slot)

            # Refill immediately for the next rollout step, exactly as PPO does.
            if finished_slots:
                refill_slots, refill_grids = load_slots(
                    np.asarray(finished_slots, dtype=np.int32)
                )
                if len(refill_slots):
                    replacement_states = jax.vmap(create_initial_state)(
                        jnp.asarray(refill_grids)
                    )
                    states = _set_tree_rows(states, refill_slots, replacement_states)
                    history_indices = np.concatenate(
                        (refill_slots, refill_slots + pool_size)
                    )
                    zero_histories = _stack_trees(
                        [single_obs_state] * len(history_indices)
                    )
                    obs_states = _set_tree_rows(
                        obs_states, history_indices, zero_histories
                    )
                    slot_refills += len(refill_slots)

            global_step += 1

        shard = _write_shard(
            output_dir,
            0,
            chunk_index,
            chunk_start,
            global_step,
            buffer,
        )
        chunk_samples = sum(len(part) for part in buffer["action"])
        chunk_illegal = len(illegal_move_events) - chunk_illegal_start
        samples_written += chunk_samples
        if shard is not None:
            if archiver is not None:
                archiver.submit(shard)
            print(
                f"  shard {shard.name}: samples={chunk_samples:,}, "
                f"invalid_skipped={chunk_illegal:,}, "
                f"active={int(active.sum()):,}, loaded={next_source_index:,}/"
                f"{total_games:,}, completed={completed:,}"
            )
        # Checkpoint small metadata after every flush so an interrupted Colab
        # run retains outcomes and diagnostics for all completed shards.
        _write_json_atomic(output_dir / "game_winners.json", winner_lookup)
        _write_json_atomic(
            output_dir / "illegal_moves.json", {"events": illegal_move_events}
        )
        chunk_index += 1

    return {
        "completed": completed,
        "draws": draws,
        "failed": failed,
        "samples": samples_written,
        "illegal_moves": len(illegal_move_events),
        "slot_refills": slot_refills,
    }


def collect_dataset(
    output_dir: str | Path,
    *,
    dataset_name: str = HF_DATASET,
    split: str = HF_SPLIT,
    game_batch_size: int = GAME_BATCH_SIZE,
    turns_per_shard: int = TURNS_PER_SHARD,
    max_games: int | None = None,
    archive_dir: str | Path | None = None,
    delete_after_archive: bool = False,
) -> None:
    """Collect one sequential epoch over the requested Hugging Face split."""

    if game_batch_size <= 0 or turns_per_shard <= 0:
        raise ValueError("game_batch_size and turns_per_shard must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = Path(archive_dir) if archive_dir is not None else None
    if archive_path is not None and archive_path.resolve() == output_dir.resolve():
        raise ValueError("archive_dir must differ from output_dir")
    if delete_after_archive and archive_path is None:
        raise ValueError("delete_after_archive requires archive_dir")
    archiver = (
        AsyncArchiver(archive_path, delete_after_archive)
        if archive_path is not None
        else None
    )
    winner_path = output_dir / "game_winners.json"
    illegal_path = output_dir / "illegal_moves.json"

    print(f"Loading Hugging Face dataset {dataset_name!r}, split={split!r}...")
    dataset = load_dataset(dataset_name, split=split)
    total_games = len(dataset)
    if max_games is not None:
        total_games = min(total_games, max(0, int(max_games)))
    pool_size = min(game_batch_size, total_games) if total_games else 0
    print(
        f"Collecting {total_games:,} games once: rolling_pool={pool_size:,}, "
        f"flush={turns_per_shard} global steps"
    )

    winner_lookup: dict[str, int] = {}
    totals = {
        "completed": 0,
        "draws": 0,
        "failed": 0,
        "samples": 0,
        "illegal_moves": 0,
        "slot_refills": 0,
    }
    illegal_move_events: list[dict[str, Any]] = []
    started = time.monotonic()

    try:
        if total_games:
            stats = _collect_rolling_pool(
                dataset,
                total_games,
                output_dir,
                pool_size,
                turns_per_shard,
                winner_lookup,
                illegal_move_events,
                archiver,
            )
            for key in totals:
                totals[key] += stats[key]
    finally:
        if archiver is not None:
            archiver.close()
    _write_json_atomic(winner_path, winner_lookup)
    _write_json_atomic(illegal_path, {"events": illegal_move_events})
    elapsed = time.monotonic() - started
    print(
        f"  progress games={totals['completed'] + totals['failed']:,}/"
        f"{total_games:,}, samples={totals['samples']:,}, "
        f"failed={totals['failed']:,}, elapsed={elapsed / 60:.1f}m"
    )

    manifest = {
        "dataset": dataset_name,
        "split": split,
        "games_requested": total_games,
        "game_batch_size": game_batch_size,
        "rolling_pool_size": pool_size,
        "archive_dir": str(archive_path) if archive_path is not None else None,
        "delete_after_archive": delete_after_archive,
        "turns_per_shard": turns_per_shard,
        "pad_to": PAD_TO,
        "history_size": HISTORY_SIZE,
        "temporal_window": TEMPORAL_WINDOW,
        "replay_priority": "ppo_current",
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
    print(f"Illegal-move diagnostics: {illegal_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/bc_data"))
    parser.add_argument("--dataset", default=HF_DATASET)
    parser.add_argument("--split", default=HF_SPLIT)
    parser.add_argument("--game-batch-size", type=int, default=GAME_BATCH_SIZE)
    parser.add_argument("--turns-per-shard", type=int, default=TURNS_PER_SHARD)
    parser.add_argument("--max-games", type=int)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="copy each completed shard here in a background thread",
    )
    parser.add_argument(
        "--delete-after-archive",
        action="store_true",
        help="delete each local shard after its background archive succeeds",
    )
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
        archive_dir=args.archive_dir,
        delete_after_archive=args.delete_after_archive,
    )


if __name__ == "__main__":
    main()
