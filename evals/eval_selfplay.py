"""Watch trained models play against each other (or one model against itself).

Usage:
    python evals/eval_selfplay.py model.eqx --config configs/S.yaml
    python evals/eval_selfplay.py model_a.eqx model_b.eqx --config configs/S.yaml
    python evals/eval_selfplay.py model.eqx --config configs/S.yaml --headless --num-games 100
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import time
import jax.numpy as jnp
import jax.random as jrandom

from generals.core.game import get_observation
from generals.core.action import compute_valid_move_mask
from generals.core.env import GeneralsEnv

from networks import obs_to_array
from evals.agent import Agent


def _make_eval_env(cfg):
    stages = cfg.curriculum_stages
    if stages:
        last = stages[-1]
        min_dist = last.min_generals_distance
        max_dist = last.max_generals_distance
        cities = (last.num_cities_min, last.num_cities_max) if last.num_cities_min is not None else (cfg.num_cities_min, cfg.num_cities_max)
        castle = (last.castle_val_min, last.castle_val_max) if last.castle_val_min is not None else (cfg.castle_val_min, cfg.castle_val_max)
    else:
        min_dist = cfg.min_generals_distance
        max_dist = cfg.max_generals_distance
        cities = (cfg.num_cities_min, cfg.num_cities_max)
        castle = (cfg.castle_val_min, cfg.castle_val_max)
    gs = cfg.eval_grid_size
    return GeneralsEnv(grid_dims=(gs, gs), pad_to=cfg.pad_to,
                       min_generals_distance=min_dist, max_generals_distance=max_dist,
                       truncation=cfg.truncation,
                       num_cities_range=cities, castle_val_range=castle)


def _play_one_game(state, env, pool, agent_p0, agent_p1, cfg, gui=None, fps=10):
    """Play a single game, return (winner, steps, p0_passes, p1_passes, state, timestep)."""
    obs_state_p0 = agent_p0.init_obs_state_fn(cfg.pad_to, cfg.pad_to)
    obs_state_p1 = agent_p1.init_obs_state_fn(cfg.pad_to, cfg.pad_to)

    p0_passes, p1_passes = 0, 0
    for step in range(cfg.truncation):
        obs_p0 = get_observation(state, 0)
        obs_p1 = get_observation(state, 1)

        obs_aug_p0, obs_state_p0 = agent_p0.augment_fn(obs_to_array(obs_p0), obs_state_p0)
        mask_p0 = compute_valid_move_mask(obs_p0.armies, obs_p0.owned_cells, obs_p0.mountains)
        temporal_p0 = jnp.stack([obs_state_p0.opponent_army_history, obs_state_p0.opponent_land_history])

        obs_aug_p1, obs_state_p1 = agent_p1.augment_fn(obs_to_array(obs_p1), obs_state_p1)
        mask_p1 = compute_valid_move_mask(obs_p1.armies, obs_p1.owned_cells, obs_p1.mountains)
        temporal_p1 = jnp.stack([obs_state_p1.opponent_army_history, obs_state_p1.opponent_land_history])

        action_p0 = agent_p0.greedy_fn(agent_p0.network, obs_aug_p0, mask_p0, temporal_p0)
        action_p1 = agent_p1.greedy_fn(agent_p1.network, obs_aug_p1, mask_p1, temporal_p1)

        p0_passes += int(action_p0[0])
        p1_passes += int(action_p1[0])

        actions = jnp.stack([action_p0, action_p1])
        timestep, state = env.step(state, actions, pool)

        if gui is not None:
            gui.update(state)
            gui.tick(fps=fps)

        if timestep.terminated or timestep.truncated:
            return int(timestep.info.winner), step + 1, p0_passes, p1_passes, state, timestep

    return -1, cfg.truncation, p0_passes, p1_passes, state, timestep


def run_selfplay_headless(agent_p0, agent_p1, cfg, num_games=100, seed=123):
    env = _make_eval_env(cfg)
    key = jrandom.PRNGKey(seed)
    key, pool_key = jrandom.split(key)
    pool, _ = env.reset(pool_key)

    names = [agent_p0.name or "P0", agent_p1.name or "P1"]
    p0_wins, p1_wins, draws = 0, 0, 0

    for g in range(num_games):
        key, init_key = jrandom.split(key)
        state = env.init_state(init_key)
        winner, steps, _, _, _, _ = _play_one_game(state, env, pool, agent_p0, agent_p1, cfg)

        if winner == 0: p0_wins += 1
        elif winner == 1: p1_wins += 1
        else: draws += 1

        total = g + 1
        print(f"Game {total:3d}/{num_games} | Winner: {names[winner] if winner >= 0 else 'Draw':20s} | "
              f"Steps: {steps:4d} | "
              f"{names[0]}={p0_wins} {names[1]}={p1_wins} D={draws} "
              f"({p0_wins/total*100:.0f}%-{p1_wins/total*100:.0f}%)")

    print(f"\nFinal: {names[0]}={p0_wins} {names[1]}={p1_wins} D={draws} / {num_games} games "
          f"({p0_wins/num_games*100:.1f}% - {p1_wins/num_games*100:.1f}%)")


def run_selfplay(agent_p0, agent_p1, cfg, fps=10, seed=123):
    from generals.gui import ReplayGUI

    env = _make_eval_env(cfg)
    key = jrandom.PRNGKey(seed)
    key, pool_key = jrandom.split(key)
    pool, _ = env.reset(pool_key)

    agents_display = [agent_p0.name or "Red (P0)", agent_p1.name or "Blue (P1)"]
    p0_wins, p1_wins, draws, games = 0, 0, 0, 0
    gui = None

    while True:
        key, init_key = jrandom.split(key)
        state = env.init_state(init_key)

        if gui is None:
            gui = ReplayGUI(state, agent_ids=agents_display)
        else:
            gui.update(state)
            gui.tick(fps=1000)
            time.sleep(1.5)

        print(f"\n--- Game {games + 1} ---")
        winner, steps, p0_passes, p1_passes, state, timestep = _play_one_game(
            state, env, pool, agent_p0, agent_p1, cfg, gui=gui, fps=fps)

        display_state = timestep.last_state if (timestep.terminated or timestep.truncated) else state
        gui.update(display_state, timestep.info)
        gui.tick(fps=1)

        if winner == 0: p0_wins += 1
        elif winner == 1: p1_wins += 1
        else: draws += 1
        games += 1

        label = "P0" if winner == 0 else "P1" if winner == 1 else "Draw"
        print(f"Winner: {label} | Steps: {steps} | "
              f"Pass rate: P0={p0_passes}/{steps} ({100*p0_passes/steps:.0f}%) "
              f"P1={p1_passes}/{steps} ({100*p1_passes/steps:.0f}%) | "
              f"Record: P0={p0_wins} P1={p1_wins} D={draws}")

    gui.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_p0", help="Checkpoint for player 0 (red)")
    parser.add_argument("model_p1", nargs="?", default=None,
                        help="Checkpoint for player 1 (blue). Defaults to model_p0.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--config-p1", type=str, default=None,
                        help="Separate config for P1 model (default: same as --config)")
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--truncation", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-games", type=int, default=100)
    args = parser.parse_args()

    agent_p0 = Agent.load(args.model_p0, args.config)
    print(f"P0: {args.model_p0} ({agent_p0.param_count():,} params)")

    path_p1 = args.model_p1 or args.model_p0
    cfg_p1 = args.config_p1 or args.config
    if path_p1 == args.model_p0 and cfg_p1 == args.config:
        agent_p1 = Agent.load(path_p1, cfg_p1)
        agent_p0.name = agent_p0.name + " (red)"
        agent_p1.name = agent_p1.name + " (blue)"
        print("Self-play: same checkpoint for both players")
    else:
        agent_p1 = Agent.load(path_p1, cfg_p1)
        print(f"P1: {path_p1} ({agent_p1.param_count():,} params)")

    # Use p0's config for env params, with optional overrides
    cfg = agent_p0.config
    if args.grid_size is not None:
        cfg.eval_grid_size = args.grid_size
    else:
        cfg.eval_grid_size = cfg.max_grid_size
    if args.truncation is not None:
        cfg.truncation = args.truncation

    if args.headless:
        run_selfplay_headless(agent_p0, agent_p1, cfg,
                              num_games=args.num_games, seed=args.seed)
    else:
        run_selfplay(agent_p0, agent_p1, cfg, fps=args.fps, seed=args.seed)


if __name__ == "__main__":
    main()
