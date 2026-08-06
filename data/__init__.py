"""Fresh Hugging Face replay cleaning and BC dataset collection.

Exports are loaded lazily so ``python -m data.data_collection`` does not import
the target module once through this package before executing it with ``runpy``.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "GAME_BATCH_SIZE",
    "HF_DATASET",
    "HF_SPLIT",
    "PAD_TO",
    "TURNS_PER_SHARD",
    "CleanMoves",
    "collect_dataset",
    "initialise_map",
    "moves_to_env_actions",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module("data.data_collection"), name)
