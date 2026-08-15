"""Combine compatible NPZ shards along their leading sample dimension."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np


def _extract_member(shard: Path, member: str, destination: Path) -> Path:
    with zipfile.ZipFile(shard) as archive:
        if member not in archive.namelist():
            raise ValueError(f"{shard} does not contain {member}")
        archive.extract(member, destination)
    return destination / member


def combine_shards(
    shards: list[Path], destination: Path, *, compresslevel: int = 6
) -> None:
    if len(shards) < 2:
        raise ValueError("provide at least two shards")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".combine-", dir=destination.parent) as work:
        work_dir = Path(work)
        lengths: list[int] = []
        for index, shard in enumerate(shards):
            metadata_dir = work_dir / f"metadata-{index}"
            game_ids_path = _extract_member(shard, "game_id.npy", metadata_dir)
            game_ids = np.load(game_ids_path, mmap_mode="r", allow_pickle=False)
            if game_ids.ndim != 1:
                raise ValueError(f"{shard}: game_id.npy must be one-dimensional")
            lengths.append(len(game_ids))

        total_samples = sum(lengths)
        combined_dir = work_dir / "combined"
        combined_dir.mkdir()
        outputs: dict[str, np.memmap] = {}
        expected_members: list[str] | None = None
        offset = 0

        for index, (shard, length) in enumerate(zip(shards, lengths)):
            extracted_dir = work_dir / f"shard-{index}"
            with zipfile.ZipFile(shard) as archive:
                members = sorted(
                    info.filename for info in archive.infolist() if not info.is_dir()
                )
                if any(
                    Path(name).is_absolute() or ".." in Path(name).parts
                    for name in members
                ):
                    raise ValueError(f"{shard} contains an unsafe archive path")
                archive.extractall(extracted_dir)

            if expected_members is None:
                expected_members = members
            elif members != expected_members:
                raise ValueError(f"{shard} has a different set of arrays")

            for member in members:
                source = np.load(extracted_dir / member, mmap_mode="r", allow_pickle=False)
                if source.ndim == 0 or source.shape[0] != length:
                    raise ValueError(
                        f"{shard}:{member} does not share the sample dimension"
                    )
                if member not in outputs:
                    outputs[member] = np.lib.format.open_memmap(
                        combined_dir / member,
                        mode="w+",
                        dtype=source.dtype,
                        shape=(total_samples, *source.shape[1:]),
                    )
                output = outputs[member]
                if source.dtype != output.dtype or source.shape[1:] != output.shape[1:]:
                    raise ValueError(f"{shard}:{member} has an incompatible schema")
                output[offset : offset + length] = source

            offset += length
            for output in outputs.values():
                output.flush()
            # Remove this large extracted shard before extracting the next one.
            for path in extracted_dir.iterdir():
                path.unlink()
            extracted_dir.rmdir()
            print(f"  copied {shard.name}: {length:,} samples", flush=True)

        outputs.clear()
        copy_seconds = time.perf_counter() - started

        compression_started = time.perf_counter()
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with zipfile.ZipFile(
                temporary,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=compresslevel,
                allowZip64=True,
            ) as archive:
                for member in expected_members or []:
                    archive.write(combined_dir / member, arcname=member)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        compression_seconds = time.perf_counter() - compression_started

    print(f"combined samples: {total_samples:,}", flush=True)
    print(f"extract/copy: {copy_seconds:.2f} s", flush=True)
    print(f"compress/store: {compression_seconds:.2f} s", flush=True)
    print(f"wrote: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compresslevel", type=int, choices=range(1, 10), default=6)
    args = parser.parse_args()
    combine_shards(args.shards, args.output, compresslevel=args.compresslevel)


if __name__ == "__main__":
    main()
