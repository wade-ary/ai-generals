"""
Pick a random HuggingFace generals replay, step it in the env, open pygame GUI.

Same flow as ``bc/data_test.ipynb`` (silent env replay → scrubable ReplayGUI).

    python -m bc.sample_game_runner
    python -m bc.sample_game_runner --seed 0
    python -m bc.sample_game_runner --index 42
    python -m bc.sample_game_runner --fps 12

Controls: SPACE play/pause | ←/→ step | R restart | Q quit
"""
from __future__ import annotations

import argparse
import random
from typing import Any

import jax.numpy as jnp
import numpy as np
from datasets import load_dataset

from generals import create_action, create_initial_state, step, get_observation
from generals.core.action import compute_valid_move_mask
from generals.core.game import GameInfo, GameState, get_info
from generals.gui import ReplayGUI
from generals.gui.properties import GuiMode

PAD_TO = 23
SEED: int | None = None  # None = fresh random each run
FPS = 8

PASS = np.asarray(create_action(to_pass=True), dtype=np.int32)
DELTA_TO_DIR = {
    (-1, 0): 0,  # UP
    (1, 0): 1,  # DOWN
    (0, -1): 2,  # LEFT
    (0, 1): 3,  # RIGHT
}


def tile_to_rc(tile: int, width: int) -> tuple[int, int]:
    return divmod(int(tile), int(width))


def dataset_move_to_action(start_tile: int, end_tile: int, is_50: int, width: int) -> np.ndarray:
    sr, sc = tile_to_rc(start_tile, width)
    er, ec = tile_to_rc(end_tile, width)
    direction = DELTA_TO_DIR[(er - sr, ec - sc)]
    return np.array([0, sr, sc, direction, int(is_50)], dtype=np.int32)


def replay_to_grid(replay: dict[str, Any], pad_to: int = PAD_TO) -> np.ndarray:
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


def replay_num_turns(replay: dict[str, Any]) -> int:
    moves = replay["moves"]
    if not moves:
        return 1
    return int(max(int(m[4]) for m in moves) + 1)


def replay_to_actions(replay: dict[str, Any], truncation: int) -> np.ndarray:
    """(T, 2, 5) — missing turns stay as pass."""
    w = int(replay["mapWidth"])
    seq = np.broadcast_to(np.stack([PASS, PASS]), (truncation, 2, 5)).copy()
    for move in replay["moves"]:
        player, start, end, is_50, turn = (int(x) for x in move)
        if turn >= truncation:
            continue
        seq[turn, player] = dataset_move_to_action(start, end, is_50, w)
    return seq


def pick_replay(dataset, index: int | None, seed: int | None) -> tuple[int, dict[str, Any]]:
    rng = random.Random(seed)
    idx = index if index is not None else rng.randrange(len(dataset))
    return idx, dataset[idx]


def step_replay(
    sample: dict[str, Any],
) -> tuple[list[GameState], list[GameInfo], dict[str, int]]:
    """Silent env rollout — collect trajectory for GUI (no textual board)."""
    T = replay_num_turns(sample)
    grid = replay_to_grid(sample)
    actions = replay_to_actions(sample, truncation=T)

    state = create_initial_state(jnp.asarray(grid))
    traj_states: list[GameState] = [state]
    traj_infos: list[GameInfo] = [get_info(state)]

    applied = 0
    illegal = 0
    info: GameInfo | None = None
    t = 0

    for t in range(T):
        joint = actions[t]
        for p in range(2):
            a = np.asarray(joint[p])
            if int(a[0]) == 1:
                continue
            applied += 1
            o = get_observation(state, int(p))
            mask = compute_valid_move_mask(o.armies, o.owned_cells, o.mountains)
            if not bool(mask[int(a[1]), int(a[2]), int(a[3])]):
                illegal += 1

        state, info = step(state, jnp.asarray(joint))
        traj_states.append(state)
        traj_infos.append(info)
        if bool(info.is_done):
            break

    assert info is not None
    stats = {
        "T": T,
        "end_turn": int(info.time),
        "winner": int(info.winner),
        "applied": applied,
        "illegal": illegal,
        "frames": len(traj_states),
    }
    return traj_states, traj_infos, stats


def open_gui(
    sample: dict[str, Any],
    traj_states: list[GameState],
    traj_infos: list[GameInfo],
    fps: int = FPS,
) -> None:
    agent_ids = [str(sample["usernames"][0]), str(sample["usernames"][1])]
    print(
        f"opening GUI  frames={len(traj_states)}  fps={fps}  "
        f"{agent_ids[0]} vs {agent_ids[1]}"
    )
    print("controls: SPACE play/pause | ←/→ step | R restart | Q quit")
    gui = ReplayGUI(
        traj_states[0],
        agent_ids=agent_ids,
        mode=GuiMode.REPLAY,
        start_paused=True,
        fps=fps,
    )
    gui.play(traj_states, traj_infos)
    print("GUI closed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick an HF generals replay and open scrubable pygame GUI"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="RNG seed for random pick (default: fresh)")
    parser.add_argument("--index", type=int, default=None, help="Fixed dataset index (overrides --seed)")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--no-gui", action="store_true", help="Step only; skip pygame window")
    args = parser.parse_args()

    print("Loading dataset...")
    train_dataset = load_dataset("strakammm/generals_io_replays")["train"]
    print(f"Replays: {len(train_dataset)}")

    idx, sample = pick_replay(train_dataset, args.index, args.seed)
    print(f"picked index={idx}  id={sample['id']}")
    print(f"players: {sample['usernames'][0]} vs {sample['usernames'][1]}")
    print(f"map: {sample['mapWidth']}x{sample['mapHeight']}  moves={len(sample['moves'])}")

    traj_states, traj_infos, stats = step_replay(sample)
    winner = stats["winner"]
    wlabel = "P0" if winner == 0 else "P1" if winner == 1 else "none/unfinished"
    print("--- replay summary ---")
    print(
        f"ended at turn={stats['end_turn']}  winner={winner} ({wlabel})  "
        f"non-pass={stats['applied']}  illegal={stats['illegal']}  "
        f"GUI frames={stats['frames']}"
    )

    if args.no_gui:
        print("--no-gui set; skipping window")
        return

    open_gui(sample, traj_states, traj_infos, fps=args.fps)


if __name__ == "__main__":
    main()
