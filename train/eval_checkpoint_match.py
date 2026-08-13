"""Evaluate two checkpoints on paired maps and record one viewable replay.

Typical S_750 vs S_500 comparison::

    python -m train.eval_checkpoint_match \
        checkpoints/S/S_750.eqx S/S_500.eqx \
        --config-a configs/S_500_to_750.yaml \
        --config-b S/config.yaml \
        --num-games 250 \
        --replay-output S/S_750_vs_S_500_replay.pkl

Add ``--show-replay`` when running on a machine with a graphical display.
Colab should save the replay and download it for viewing locally.

View a downloaded replay without loading either model::

    python -m train.eval_checkpoint_match --view-replay S_750_vs_S_500_replay.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from evals.agent import Agent
from evals.matchup import play_match
from generals.core.action import compute_valid_move_mask
from generals.core.env import GeneralsEnv
from generals.core.game import GameInfo, GameState, get_info, get_observation
from networks import obs_to_array


def make_eval_env(cfg, num_maps: int) -> GeneralsEnv:
    """Construct the final-stage environment described by a model config."""
    stages = cfg.curriculum_stages
    if stages:
        stage = stages[-1]
        min_distance = stage.min_generals_distance
        max_distance = stage.max_generals_distance
        cities = (
            (stage.num_cities_min, stage.num_cities_max)
            if stage.num_cities_min is not None
            else (cfg.num_cities_min, cfg.num_cities_max)
        )
        castles = (
            (stage.castle_val_min, stage.castle_val_max)
            if stage.castle_val_min is not None
            else (cfg.castle_val_min, cfg.castle_val_max)
        )
    else:
        min_distance = cfg.min_generals_distance
        max_distance = cfg.max_generals_distance
        cities = (cfg.num_cities_min, cfg.num_cities_max)
        castles = (cfg.castle_val_min, cfg.castle_val_max)

    return GeneralsEnv(
        grid_dims=(cfg.max_grid_size, cfg.max_grid_size),
        pad_to=cfg.pad_to,
        min_generals_distance=min_distance,
        max_generals_distance=max_distance,
        truncation=cfg.truncation,
        num_cities_range=cities,
        castle_val_range=castles,
        mountain_density_range=(cfg.mountain_density_min, cfg.mountain_density_max),
        pool_size=max(num_maps, 1_000),
    )


def evaluate_paired(agent_a, agent_b, env, pool, num_games: int, seed: int):
    """Evaluate on identical maps with player positions reversed.

    ``num_games`` is the total across both seat assignments and must be even.
    """
    if num_games <= 0 or num_games % 2:
        raise ValueError("--num-games must be a positive even number")

    num_maps = num_games // 2
    match_key = jrandom.PRNGKey(seed)

    # Reusing the same key makes both calls generate the same map batch.
    a_as_p0, b_as_p1, draws_ab = play_match(
        agent_a, agent_b, env, pool, num_maps, env.truncation, match_key
    )
    b_as_p0, a_as_p1, draws_ba = play_match(
        agent_b, agent_a, env, pool, num_maps, env.truncation, match_key
    )

    a_wins = a_as_p0 + a_as_p1
    b_wins = b_as_p1 + b_as_p0
    draws = draws_ab + draws_ba
    decisive = a_wins + b_wins
    score_a = (a_wins + 0.5 * draws) / num_games

    print("\n=== Paired checkpoint evaluation ===")
    print(f"Maps: {num_maps} | Games: {num_games} (each map played both ways)")
    print(f"{agent_a.name}: {a_wins} wins")
    print(f"{agent_b.name}: {b_wins} wins")
    print(f"Draws: {draws}")
    print(f"{agent_a.name} score: {100.0 * score_a:.1f}% (draws count as 0.5)")
    if decisive:
        print(f"{agent_a.name} decisive win rate: {100.0 * a_wins / decisive:.1f}%")
    print(
        f"Seat split: {agent_a.name} won {a_as_p0} as P0 and {a_as_p1} as P1; "
        f"{agent_b.name} won {b_as_p0} as P0 and {b_as_p1} as P1"
    )

    return {
        "num_maps": num_maps,
        "num_games": num_games,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_score": score_a,
        "a_wins_as_p0": a_as_p0,
        "a_wins_as_p1": a_as_p1,
        "b_wins_as_p0": b_as_p0,
        "b_wins_as_p1": b_as_p1,
    }


def record_game(agent_p0, agent_p1, env, pool, seed: int):
    """Play one greedy game and retain every state for replay."""
    key = jrandom.PRNGKey(seed)
    state = env.init_state(key)
    obs_state_p0 = agent_p0.init_obs_state_fn(env.pad_to, env.pad_to)
    obs_state_p1 = agent_p1.init_obs_state_fn(env.pad_to, env.pad_to)

    states: list[GameState] = [state]
    infos: list[GameInfo] = [get_info(state)]
    winner = -1

    for _ in range(env.truncation):
        obs_p0 = get_observation(state, 0)
        obs_p1 = get_observation(state, 1)

        aug_p0, obs_state_p0 = agent_p0.augment_fn(obs_to_array(obs_p0), obs_state_p0)
        mask_p0 = compute_valid_move_mask(obs_p0.armies, obs_p0.owned_cells, obs_p0.mountains)
        temporal_p0 = jnp.stack(
            [obs_state_p0.opponent_army_history, obs_state_p0.opponent_land_history]
        )

        aug_p1, obs_state_p1 = agent_p1.augment_fn(obs_to_array(obs_p1), obs_state_p1)
        mask_p1 = compute_valid_move_mask(obs_p1.armies, obs_p1.owned_cells, obs_p1.mountains)
        temporal_p1 = jnp.stack(
            [obs_state_p1.opponent_army_history, obs_state_p1.opponent_land_history]
        )

        action_p0 = agent_p0.greedy_fn(
            agent_p0.network, aug_p0, mask_p0, temporal_p0
        )
        action_p1 = agent_p1.greedy_fn(
            agent_p1.network, aug_p1, mask_p1, temporal_p1
        )
        timestep, new_state = env.step(
            state, jnp.stack([action_p0, action_p1]), pool
        )

        # last_state is the terminal board before env auto-reset.
        displayed_state = timestep.last_state
        states.append(displayed_state)
        infos.append(timestep.info)

        if bool(timestep.terminated) or bool(timestep.truncated):
            winner = int(timestep.info.winner)
            break
        state = new_state

    return states, infos, winner


def save_replay(path, states, infos, metadata):
    """Save a portable host-array replay."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    host_states = [jax.tree.map(np.asarray, state) for state in states]
    host_infos = [jax.tree.map(np.asarray, info) for info in infos]
    with open(path, "wb") as file:
        pickle.dump(
            {"states": host_states, "infos": host_infos, "metadata": metadata},
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def show_replay(states, infos, names, fps):
    """Open the interactive pygame replay viewer."""
    from generals.gui import ReplayGUI
    from generals.gui.properties import GuiMode

    gui = ReplayGUI(
        states[0],
        agent_ids=names,
        fps=fps,
        mode=GuiMode.REPLAY,
        start_paused=False,
    )
    gui.play(states, infos)


def save_replay_gif(states, infos, names, output_path, fps, frame_stride, speed):
    """Render a saved replay as a looping, GitHub-friendly GIF."""
    from pathlib import Path

    import pygame
    from PIL import Image

    from generals.gui import ReplayGUI

    gui = ReplayGUI(states[0], agent_ids=names)
    frames = []
    selected = list(range(0, len(states), frame_stride))
    if selected[-1] != len(states) - 1:
        selected.append(len(states) - 1)

    for frame_index in selected:
        gui.update(states[frame_index], infos[frame_index])
        gui.render()
        surface = pygame.display.get_surface()
        frames.append(Image.frombytes(
            "RGB", surface.get_size(), pygame.image.tobytes(surface, "RGB")
        ))
    gui.close()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, round(1000 * frame_stride / (max(1, fps) * speed)))
    durations = [duration_ms] * len(frames)
    durations[-1] = 2000
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"Saved GIF: {destination} ({len(frames)} frames, {speed:g}x speed)")


def load_replay(path):
    with open(path, "rb") as file:
        replay = pickle.load(file)
    return replay["states"], replay["infos"], replay.get("metadata", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_a", nargs="?", help="First checkpoint, e.g. S_750.eqx")
    parser.add_argument("model_b", nargs="?", help="Second checkpoint, e.g. S_500.eqx")
    parser.add_argument("--config-a", help="YAML used to build model A")
    parser.add_argument(
        "--config-b", default=None, help="YAML for model B (default: --config-a)"
    )
    parser.add_argument(
        "--num-games", type=int, default=250,
        help="Total paired evaluation games; must be even (default: 250)",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--replay-seed", type=int, default=124,
        help="Seed for the separately recorded display game",
    )
    parser.add_argument(
        "--replay-output", default="checkpoint_match_replay.pkl",
        help="Where to save the recorded game",
    )
    parser.add_argument(
        "--show-replay", action="store_true",
        help="Open the recorded game in pygame (requires a graphical display)",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--gif-output", default=None,
        help="Render the replay to a GIF instead of opening the interactive viewer",
    )
    parser.add_argument(
        "--gif-stride", type=int, default=1,
        help="Include every Nth replay state in the GIF (default: 1)",
    )
    parser.add_argument(
        "--gif-speed", type=float, default=1.0,
        help="GIF playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--view-replay", default=None,
        help="View a previously saved replay without running evaluation",
    )
    args = parser.parse_args()

    if args.view_replay:
        states, infos, metadata = load_replay(args.view_replay)
        names = [
            os.path.basename(metadata.get("model_a", "Model A")).replace(".eqx", ""),
            os.path.basename(metadata.get("model_b", "Model B")).replace(".eqx", ""),
        ]
        print(
            f"Replay: {metadata.get('winner_name', 'unknown winner')} | "
            f"Steps: {metadata.get('steps', len(states) - 1)}"
        )
        if args.gif_output:
            save_replay_gif(
                states, infos, names, args.gif_output, args.fps,
                max(1, args.gif_stride), max(0.1, args.gif_speed),
            )
            return
        show_replay(states, infos, names, args.fps)
        return

    if not args.model_a or not args.model_b or not args.config_a:
        parser.error(
            "model_a, model_b, and --config-a are required unless --view-replay is used"
        )

    config_b = args.config_b or args.config_a
    agent_a = Agent.load(args.model_a, args.config_a)
    agent_b = Agent.load(args.model_b, config_b)
    print(f"A: {agent_a.name} ({agent_a.param_count():,} parameters)")
    print(f"B: {agent_b.name} ({agent_b.param_count():,} parameters)")

    if agent_a.pad_to != agent_b.pad_to:
        raise ValueError(
            f"Models use incompatible pad sizes: {agent_a.pad_to} vs {agent_b.pad_to}"
        )

    num_maps = args.num_games // 2
    env = make_eval_env(agent_a.config, num_maps)
    pool_key = jrandom.PRNGKey(args.seed + 10_000)
    pool, _ = env.reset(pool_key)

    results = evaluate_paired(
        agent_a, agent_b, env, pool, args.num_games, args.seed
    )

    states, infos, replay_winner = record_game(
        agent_a, agent_b, env, pool, args.replay_seed
    )
    winner_name = (
        agent_a.name if replay_winner == 0
        else agent_b.name if replay_winner == 1
        else "draw"
    )
    metadata = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "config_a": args.config_a,
        "config_b": config_b,
        "seed": args.replay_seed,
        "winner": replay_winner,
        "winner_name": winner_name,
        "steps": len(states) - 1,
        "evaluation": results,
    }
    save_replay(args.replay_output, states, infos, metadata)
    print("\n=== Recorded replay ===")
    print(f"Winner: {winner_name} | Steps: {len(states) - 1}")
    print(f"Saved: {os.path.abspath(args.replay_output)}")

    if args.show_replay:
        show_replay(states, infos, [agent_a.name, agent_b.name], args.fps)
    else:
        print("Run again with --show-replay on a machine with a display to watch it.")


if __name__ == "__main__":
    main()
