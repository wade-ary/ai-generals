"""Attach terminal game outcomes to behavioral-cloning NPZ shards.

Each shard is extracted, updated, recompressed, and removed from the working
directory before the next shard is processed.  This keeps peak disk usage to
one decompressed shard and avoids loading large observation arrays into RAM.

Example:
    python -m data.add_winners \
        --input-dir data/bc_data \
        --winners game_winners.json \
        --output-dir data/bc_data_with_winners
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


REQUIRED_ARRAYS = {"game_id.npy", "player.npy", "reward.npy"}


@dataclass(frozen=True)
class ShardResult:
    source: Path
    destination: Path
    samples: int
    missing_games: int
    decompression_seconds: float
    reward_seconds: float
    compression_seconds: float


def inspect_npz(path: Path) -> tuple[float, float]:
    """Return compressed and decompressed archive sizes in GiB."""
    with zipfile.ZipFile(path) as archive:
        compressed = sum(info.compress_size for info in archive.infolist())
        uncompressed = sum(info.file_size for info in archive.infolist())
    return compressed / 1024**3, uncompressed / 1024**3


def load_winners(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by game ID")

    winners = {str(game_id): int(winner) for game_id, winner in raw.items()}
    invalid = {value for value in winners.values() if value not in (-1, 0, 1)}
    if invalid:
        raise ValueError(f"invalid winner values in {path}: {sorted(invalid)}")
    return winners


def _validate_archive(archive: zipfile.ZipFile) -> None:
    names = {info.filename for info in archive.infolist() if not info.is_dir()}
    missing = REQUIRED_ARRAYS - names
    if missing:
        raise ValueError(f"shard is missing required arrays: {sorted(missing)}")
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise ValueError("shard contains an unsafe archive path")


def _write_rewards(extracted_dir: Path, winners: Mapping[str, int]) -> tuple[int, int]:
    game_ids = np.load(extracted_dir / "game_id.npy", mmap_mode="r")
    players = np.load(extracted_dir / "player.npy", mmap_mode="r")
    old_rewards = np.load(extracted_dir / "reward.npy", mmap_mode="r")

    if game_ids.ndim != 1 or players.ndim != 1 or old_rewards.ndim != 1:
        raise ValueError("game_id, player, and reward arrays must all be one-dimensional")
    if not (len(game_ids) == len(players) == len(old_rewards)):
        raise ValueError("game_id, player, and reward arrays have different lengths")
    invalid_players = np.setdiff1d(np.unique(players), np.array([0, 1]))
    if invalid_players.size:
        raise ValueError(f"invalid player IDs: {invalid_players.tolist()}")

    row_winners = np.fromiter(
        (winners.get(str(game_id), -2) for game_id in game_ids),
        dtype=np.int8,
        count=len(game_ids),
    )
    missing_mask = row_winners == -2
    missing_game_count = 0
    if missing_mask.any():
        missing_ids = np.unique(np.asarray(game_ids[missing_mask]))
        missing_game_count = len(missing_ids)
        preview = ", ".join(map(str, missing_ids[:10]))
        print(
            f"  warning: {missing_game_count} game IDs absent from winners JSON; "
            f"assigning reward 0 to both players. First: {preview}",
            flush=True,
        )
        # The winner encoding for a draw is -1, which maps both players to 0.
        row_winners[missing_mask] = -1

    # Draw -> 0; otherwise winner -> +1 and loser -> -1.
    rewards = np.where(
        row_winners == -1,
        0,
        np.where(players == row_winners, 1, -1),
    ).astype(old_rewards.dtype, copy=False)

    temporary = extracted_dir / "reward.updated.npy"
    np.save(temporary, rewards, allow_pickle=False)
    os.replace(temporary, extracted_dir / "reward.npy")
    return len(rewards), missing_game_count


def _compress_npz(source_dir: Path, destination: Path, compresslevel: int) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=compresslevel,
            allowZip64=True,
        ) as archive:
            for path in sorted(source_dir.iterdir()):
                if path.is_file():
                    archive.write(path, arcname=path.name)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def process_shard(
    shard: Path,
    winners: Mapping[str, int],
    output_dir: Path,
    *,
    compresslevel: int = 6,
    overwrite: bool = False,
) -> ShardResult:
    """Process one shard, cleaning its decompressed working copy afterward."""
    destination = output_dir / shard.name
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{shard.stem}-", dir=output_dir) as work:
        extracted_dir = Path(work)

        started = time.perf_counter()
        with zipfile.ZipFile(shard) as archive:
            _validate_archive(archive)
            archive.extractall(extracted_dir)
        decompression_seconds = time.perf_counter() - started

        started = time.perf_counter()
        samples, missing_games = _write_rewards(extracted_dir, winners)
        reward_seconds = time.perf_counter() - started

        started = time.perf_counter()
        _compress_npz(extracted_dir, destination, compresslevel)
        compression_seconds = time.perf_counter() - started

    return ShardResult(
        source=shard,
        destination=destination,
        samples=samples,
        missing_games=missing_games,
        decompression_seconds=decompression_seconds,
        reward_seconds=reward_seconds,
        compression_seconds=compression_seconds,
    )


def discover_shards(inputs: Iterable[Path], input_dir: Path | None) -> list[Path]:
    shards = list(inputs)
    if input_dir is not None:
        shards.extend(sorted(input_dir.glob("*.npz")))
    shards = sorted({path.resolve() for path in shards})
    if not shards:
        raise ValueError("no NPZ shards were provided or found")
    missing = [path for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"shards do not exist: {missing}")
    return shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="*", type=Path, help="individual NPZ shards")
    parser.add_argument("--input-dir", type=Path, help="process every *.npz in this folder")
    parser.add_argument("--winners", type=Path, default=Path("game_winners.json"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/bc_data_with_winners")
    )
    parser.add_argument("--compresslevel", type=int, choices=range(1, 10), default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="resume a run by leaving already-written output shards untouched",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards = discover_shards(args.shards, args.input_dir)
    winners = load_winners(args.winners)

    if args.overwrite and args.skip_existing:
        raise ValueError("--overwrite and --skip-existing cannot be used together")

    for shard in shards:
        destination = args.output_dir / shard.name
        if args.skip_existing and destination.exists():
            print(f"{shard.name}: skipped (output already exists)", flush=True)
            continue
        compressed, uncompressed = inspect_npz(shard)
        print(
            f"{shard.name}: {compressed:.2f} GiB compressed -> "
            f"{uncompressed:.2f} GiB decompressed",
            flush=True,
        )
        result = process_shard(
            shard,
            winners,
            args.output_dir,
            compresslevel=args.compresslevel,
            overwrite=args.overwrite,
        )
        print(f"  decompress: {result.decompression_seconds:.2f} s", flush=True)
        print(
            f"  attach rewards: {result.reward_seconds:.2f} s "
            f"({result.samples:,} samples)",
            flush=True,
        )
        print(f"  compress/store: {result.compression_seconds:.2f} s", flush=True)
        print(f"  wrote: {result.destination}", flush=True)


if __name__ == "__main__":
    main()
