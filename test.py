"""Ten-game end-to-end smoke test for the fresh BC data collector.

This downloads/loads ten real Hugging Face replays, collects them with the
production preprocessing path, and audits every output row.

Colab usage (from the repository root)::

    python test.py --output-dir /content/bc_test_10

Reusing a previous test directory requires an explicit ``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import jax
import numpy as np
from datasets import load_dataset

from data.data_collection import (
    HF_DATASET,
    HF_SPLIT,
    PAD_TO,
    TURNS_PER_SHARD,
    collect_dataset,
)


NUM_GAMES = 10
TEST_POOL_SIZE = 4
OBS_CHANNELS = 38
TEMPORAL_WINDOW = 512

EXPECTED_DTYPES = {
    "obs": np.dtype(np.float16),
    "action_mask": np.dtype(np.bool_),
    "temporal": np.dtype(np.float16),
    "action": np.dtype(np.int16),
    "reward": np.dtype(np.float16),
    "player": np.dtype(np.int16),
    "turn": np.dtype(np.int16),
}

EXPECTED_TAIL_SHAPES = {
    "obs": (OBS_CHANNELS, PAD_TO, PAD_TO),
    "action_mask": (PAD_TO, PAD_TO, 4),
    "temporal": (2, TEMPORAL_WINDOW),
    "action": (5,),
    "reward": (),
    "game_id": (),
    "player": (),
    "turn": (),
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{path} is not empty; use a new directory or pass --overwrite"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _assert_shard(
    path: Path,
    source_ids: set[str],
    seen_samples: set[tuple[str, int, int]],
) -> tuple[int, set[str], int]:
    """Validate one shard and return row count, IDs, and maximum turn."""

    required = set(EXPECTED_TAIL_SHAPES)
    with np.load(path, allow_pickle=False) as shard:
        missing = required - set(shard.files)
        assert not missing, f"{path.name}: missing arrays {sorted(missing)}"

        row_count = len(shard["action"])
        assert row_count > 0, f"{path.name}: empty shards should not be written"

        for key, tail_shape in EXPECTED_TAIL_SHAPES.items():
            array = shard[key]
            assert len(array) == row_count, (
                f"{path.name}: {key} has {len(array)} rows, expected {row_count}"
            )
            assert array.shape[1:] == tail_shape, (
                f"{path.name}: {key} shape {array.shape}, "
                f"expected ({row_count}, {tail_shape})"
            )

        for key, expected_dtype in EXPECTED_DTYPES.items():
            assert shard[key].dtype == expected_dtype, (
                f"{path.name}: {key} dtype {shard[key].dtype}, "
                f"expected {expected_dtype}"
            )

        obs = shard["obs"]
        masks = shard["action_mask"]
        temporal = shard["temporal"]
        actions = shard["action"].astype(np.int32)
        rewards = shard["reward"]
        ids = shard["game_id"].astype(str)
        players = shard["player"].astype(np.int32)
        turns = shard["turn"].astype(np.int32)

        assert np.isfinite(obs).all(), f"{path.name}: observation contains NaN/Inf"
        assert np.isfinite(temporal).all(), f"{path.name}: temporal contains NaN/Inf"
        assert np.isnan(rewards).all(), f"{path.name}: rewards are not all NaN"
        assert set(ids).issubset(source_ids), f"{path.name}: unknown game ID stored"
        assert np.isin(players, (0, 1)).all(), f"{path.name}: invalid player index"
        assert (turns >= 0).all(), f"{path.name}: negative turn stored"

        # Only demonstrated moves should be stored; pass is always zero.
        assert (actions[:, 0] == 0).all(), f"{path.name}: pass action was stored"
        assert ((0 <= actions[:, 1]) & (actions[:, 1] < PAD_TO)).all()
        assert ((0 <= actions[:, 2]) & (actions[:, 2] < PAD_TO)).all()
        assert ((0 <= actions[:, 3]) & (actions[:, 3] < 4)).all()
        assert np.isin(actions[:, 4], (0, 1)).all()

        rows = np.arange(row_count)
        demonstrated_legal = masks[
            rows, actions[:, 1], actions[:, 2], actions[:, 3]
        ]
        assert demonstrated_legal.all(), (
            f"{path.name}: stored action is illegal under its stored PPO mask"
        )

        for sample in zip(ids.tolist(), players.tolist(), turns.tolist()):
            assert sample not in seen_samples, f"duplicate sample {sample}"
            seen_samples.add(sample)

        return row_count, set(ids), int(turns.max())


def run_test(output_dir: Path, require_gpu: bool, overwrite: bool) -> None:
    devices = jax.devices()
    print("JAX devices:", devices)
    has_gpu = any(device.platform == "gpu" for device in devices)
    if require_gpu and not has_gpu:
        raise RuntimeError(
            "No JAX GPU detected. In Colab select Runtime -> Change runtime type -> GPU."
        )
    if not has_gpu:
        print("WARNING: GPU not detected; ten-game test will run on CPU.")

    _prepare_output_dir(output_dir, overwrite)

    print(f"Loading the first {NUM_GAMES} source IDs for independent validation...")
    source = load_dataset(HF_DATASET, split=HF_SPLIT)
    assert len(source) >= NUM_GAMES
    source_ids = {str(source[index]["id"]) for index in range(NUM_GAMES)}
    assert len(source_ids) == NUM_GAMES, "source replay IDs are not unique"

    print("Running the production collector on 10 real games...")
    collect_dataset(
        output_dir,
        dataset_name=HF_DATASET,
        split=HF_SPLIT,
        game_batch_size=TEST_POOL_SIZE,
        turns_per_shard=TURNS_PER_SHARD,
        max_games=NUM_GAMES,
    )

    shard_paths = sorted(output_dir.glob("bc_batch_*.npz"))
    assert shard_paths, "collector produced no shards"
    assert (output_dir / "manifest.json").is_file(), "manifest.json missing"
    assert (output_dir / "game_winners.json").is_file(), "game_winners.json missing"
    assert (output_dir / "illegal_moves.json").is_file(), "illegal_moves.json missing"
    assert not list(output_dir.glob("*.tmp")), "temporary output file was left behind"

    seen_samples: set[tuple[str, int, int]] = set()
    sampled_ids: set[str] = set()
    total_rows = 0
    max_turn = -1
    for shard_path in shard_paths:
        rows, ids, shard_max_turn = _assert_shard(
            shard_path, source_ids, seen_samples
        )
        total_rows += rows
        sampled_ids |= ids
        max_turn = max(max_turn, shard_max_turn)
        print(f"  PASS {shard_path.name}: {rows:,} rows")

    manifest = _load_json(output_dir / "manifest.json")
    winners = _load_json(output_dir / "game_winners.json")
    illegal_events = _load_json(output_dir / "illegal_moves.json")["events"]

    assert manifest["games_requested"] == NUM_GAMES
    assert manifest["game_batch_size"] == TEST_POOL_SIZE
    assert manifest["rolling_pool_size"] == TEST_POOL_SIZE
    assert manifest["slot_refills"] == NUM_GAMES - TEST_POOL_SIZE, (
        "every game after the initial pool must enter through immediate refill"
    )
    assert manifest["turns_per_shard"] == TURNS_PER_SHARD
    assert manifest["pad_to"] == PAD_TO
    assert manifest["history_size"] == 7
    assert manifest["temporal_window"] == TEMPORAL_WINDOW
    assert manifest["replay_priority"] == "ppo_current"
    assert manifest["illegal_moves"] == len(illegal_events)
    assert manifest["samples"] == total_rows, (
        f"manifest samples={manifest['samples']} but shards contain {total_rows}"
    )
    assert manifest["completed"] + manifest["failed"] == NUM_GAMES, (
        "every requested game must either complete or be explicitly cancelled"
    )
    assert set(winners).issubset(source_ids), "winner file contains unknown game IDs"
    assert set(winners.values()).issubset({-1, 0, 1}), "invalid winner encoding"
    assert len(winners) == manifest["completed"], (
        "winner lookup must contain exactly one entry per completed game"
    )
    assert sampled_ids.issubset(source_ids)

    # Invalid commands continue as no-ops, so only structurally malformed
    # replays may be cancelled. The event report below verifies that no invalid
    # command was stored as a BC target.
    assert manifest["failed"] == 0, (
        f"collector cancelled {manifest['failed']} structurally invalid games"
    )
    assert len(winners) == NUM_GAMES, "not all ten games received an outcome"
    for event in illegal_events:
        sample = (event["game_id"], event["player"], event["turn"])
        assert sample not in seen_samples, f"invalid action was stored: {sample}"

    print("\nALL CHECKS PASSED")
    print(f"games:       {NUM_GAMES}")
    print(f"samples:     {total_rows:,}")
    print(f"shards:      {len(shard_paths)}")
    print(f"maximum turn represented: {max_turn}")
    if max_turn >= TURNS_PER_SHARD:
        print("history/collection crossed a 256-turn shard boundary")
    else:
        print("none of these ten games crossed turn 256; persistence was not exercised")
    print(f"winner counts: p0={list(winners.values()).count(0)}, "
          f"p1={list(winners.values()).count(1)}, "
          f"draw={list(winners.values()).count(-1)}")
    print(f"illegal commands excluded: {len(illegal_events)}")
    events_by_game: dict[str, list[dict]] = {}
    for event in illegal_events:
        events_by_game.setdefault(event["game_id"], []).append(event)
    for game_id, events in sorted(events_by_game.items()):
        turns = [event["turn"] for event in events]
        classification = "isolated" if len(events) == 1 else "cascading/repeated"
        reasons = sorted({reason for event in events for reason in event["reasons"]})
        print(
            f"  {game_id}: {classification}, count={len(events)}, "
            f"turns={turns}, reasons={reasons}"
        )
    print(f"diagnostics: {(output_dir / 'illegal_moves.json').resolve()}")
    print(f"output:      {output_dir.resolve()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/bc_test_10")
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="fail instead of warning when JAX cannot see a GPU",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete and recreate an existing non-empty test output directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_test(args.output_dir, args.require_gpu, args.overwrite)
