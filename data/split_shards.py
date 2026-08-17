"""Split large NPZ shards into smaller, row-aligned NPZ shards.

The input NPZ is extracted once and its NPY members are memory-mapped. Output
arrays are copied in bounded chunks, so peak RAM does not scale with shard size.

Example:
    python3 -m data.split_shards \
        --input-dir data/bc_data_with_winners \
        --output-dir data/bc_small_shards
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


GIB = 1024**3
MIB = 1024**2


@dataclass(frozen=True)
class ShardPlan:
    source: Path
    samples: int
    expanded_bytes: int
    parts: int


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


def _archive_members(archive: zipfile.ZipFile) -> list[str]:
    # Preserve archive order so np.load(...).files matches the source shard.
    members = [info.filename for info in archive.infolist() if not info.is_dir()]
    if not members:
        raise ValueError("archive contains no files")
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in members):
        raise ValueError("archive contains an unsafe path")
    if any(Path(name).parent != Path(".") or Path(name).suffix != ".npy" for name in members):
        raise ValueError("every NPZ member must be a top-level .npy file")
    return members


def _expanded_size(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        _archive_members(archive)
        return sum(info.file_size for info in archive.infolist() if not info.is_dir())


def _load_arrays(extracted_dir: Path, members: list[str]) -> dict[str, np.ndarray]:
    arrays = {
        member: np.load(extracted_dir / member, mmap_mode="r", allow_pickle=False)
        for member in members
    }
    first = next(iter(arrays.values()))
    if first.ndim == 0:
        raise ValueError("arrays must have a leading sample dimension")
    samples = first.shape[0]
    for member, array in arrays.items():
        if array.ndim == 0 or array.shape[0] != samples:
            raise ValueError(f"{member} does not share the leading sample dimension")
    return arrays


def _row_ranges(samples: int, parts: int) -> list[tuple[int, int]]:
    base, extra = divmod(samples, parts)
    ranges = []
    start = 0
    for index in range(parts):
        stop = start + base + (index < extra)
        ranges.append((start, stop))
        start = stop
    return ranges


def _write_array_slice(
    archive: zipfile.ZipFile,
    member: str,
    source: np.ndarray,
    start: int,
    stop: int,
    chunk_bytes: int,
) -> None:
    """Stream one row slice as an NPY member without an intermediate copy."""
    bytes_per_row = max(1, source[0:1].nbytes)
    rows_per_chunk = max(1, chunk_bytes // bytes_per_row)
    header = {
        "descr": np.lib.format.dtype_to_descr(source.dtype),
        "fortran_order": False,
        "shape": (stop - start, *source.shape[1:]),
    }
    with archive.open(member, mode="w", force_zip64=True) as output:
        np.lib.format.write_array_header_2_0(output, header)
        for chunk_start in range(start, stop, rows_per_chunk):
            chunk_stop = min(stop, chunk_start + rows_per_chunk)
            chunk = np.ascontiguousarray(source[chunk_start:chunk_stop])
            output.write(memoryview(chunk).cast("B"))


def _compress_slices(
    arrays: dict[str, np.ndarray],
    destination: Path,
    members: list[str],
    start: int,
    stop: int,
    chunk_bytes: int,
    compresslevel: int,
) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=compresslevel,
            allowZip64=True,
        ) as archive:
            for member in members:
                _write_array_slice(
                    archive, member, arrays[member], start, stop, chunk_bytes
                )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def split_shard(
    source: Path,
    output_dir: Path,
    *,
    target_bytes: int,
    chunk_bytes: int,
    compresslevel: int,
    overwrite: bool,
) -> ShardPlan:
    expanded_bytes = _expanded_size(source)
    parts = max(1, math.ceil(expanded_bytes / target_bytes))
    print(
        f"{source.name}: {expanded_bytes / GIB:.2f} GiB expanded -> {parts} parts "
        f"(~{expanded_bytes / parts / GIB:.2f} GiB each)",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [
        output_dir / f"{source.stem}_part_{index + 1:02d}_of_{parts:02d}.npz"
        for index in range(parts)
    ]
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output already exists: {existing[0]}")

    with tempfile.TemporaryDirectory(prefix=f".{source.stem}-", dir=output_dir) as work:
        work_dir = Path(work)
        extracted_dir = work_dir / "extracted"
        extracted_dir.mkdir()
        started = time.perf_counter()
        with zipfile.ZipFile(source) as archive:
            members = _archive_members(archive)
            archive.extractall(extracted_dir)
        print(f"  extracted in {time.perf_counter() - started:.1f}s", flush=True)

        arrays = _load_arrays(extracted_dir, members)
        samples = next(iter(arrays.values())).shape[0]
        if parts > samples:
            raise ValueError(f"cannot split {samples} samples into {parts} nonempty parts")

        for part_index, ((start, stop), destination) in enumerate(
            zip(_row_ranges(samples, parts), destinations), start=1
        ):
            part_started = time.perf_counter()
            _compress_slices(
                arrays,
                destination,
                members,
                start,
                stop,
                chunk_bytes,
                compresslevel,
            )
            print(
                f"  [{part_index}/{parts}] rows {start:,}:{stop:,} -> "
                f"{destination.name} ({time.perf_counter() - part_started:.1f}s)",
                flush=True,
            )

        del arrays

    return ShardPlan(source, samples, expanded_bytes, parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="*", type=Path, help="individual NPZ shards")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="directory of NPZ shards (default: data/bc_data_with_winners)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/bc_small_shards"))
    parser.add_argument(
        "--target-expanded-gib",
        type=float,
        default=3.5,
        help="maximum estimated expanded size per output (default: 3.5)",
    )
    parser.add_argument(
        "--copy-chunk-mib",
        type=int,
        default=256,
        help="maximum array-copy chunk held in memory (default: 256)",
    )
    parser.add_argument(
        "--compresslevel",
        type=int,
        choices=range(1, 10),
        default=1,
        help="ZIP compression level; 1 is fastest (default: 1)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="show split counts without extracting"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_expanded_gib <= 0:
        raise ValueError("--target-expanded-gib must be positive")
    if args.copy_chunk_mib <= 0:
        raise ValueError("--copy-chunk-mib must be positive")

    input_dir = args.input_dir
    if input_dir is None and not args.shards:
        input_dir = Path("data/bc_data_with_winners")
    shards = discover_shards(args.shards, input_dir)
    target_bytes = int(args.target_expanded_gib * GIB)
    if args.dry_run:
        for source in shards:
            expanded_bytes = _expanded_size(source)
            parts = max(1, math.ceil(expanded_bytes / target_bytes))
            print(
                f"{source.name}: {expanded_bytes / GIB:.2f} GiB -> {parts} parts "
                f"(~{expanded_bytes / parts / GIB:.2f} GiB each)"
            )
        return

    for source in shards:
        split_shard(
            source,
            args.output_dir,
            target_bytes=target_bytes,
            chunk_bytes=args.copy_chunk_mib * MIB,
            compresslevel=args.compresslevel,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
