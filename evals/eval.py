"""Visualize a trained PPO agent playing against a random opponent.

Usage:
    python evals/eval.py checkpoints/model.eqx --config configs/S.yaml
    python evals/eval.py checkpoints/model.eqx --config configs/S.yaml --fps 5
    python evals/eval.py checkpoints/model.eqx --config configs/S.yaml --headless
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import time
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core.game import get_observation
from generals.core.action import compute_valid_move_mask
from generals.core.env import GeneralsEnv

from networks import obs_to_array, random_action
from evals.agent import Agent


def _make_eval_env(cfg):
    return GeneralsEnv(
        min_grid_size=cfg.min_grid_size, max_grid_size=cfg.max_grid_size,
        pad_to=cfg.pad_to,
        min_generals_distance=cfg.min_generals_distance,
        max_generals_distance=cfg.max_generals_distance,
        truncation=cfg.truncation,
    )


def run_headless(agent, num_games=50):
    cfg = agent.config
    env = _make_eval_env(cfg)
    key = jrandom.PRNGKey(123)
    key, pool_key = jrandom.split(key)
    pool, _ = env.reset(pool_key)

    wins, losses, draws = 0, 0, 0

    for g in range(num_games):
        key, init_key = jrandom.split(key)
        state = env.init_state(init_key)
        obs_state = agent.init_obs_state_fn(cfg.pad_to, cfg.pad_to)

        for step in range(cfg.truncation):
            obs_p0 = get_observation(state, 0)
            obs_p1 = get_observation(state, 1)

            obs_arr = obs_to_array(obs_p0)
            mask = compute_valid_move_mask(obs_p0.armies, obs_p0.owned_cells, obs_p0.mountains)
            obs_augmented, obs_state = agent.augment_fn(obs_arr, obs_state)
            temporal = jnp.stack([obs_state.opponent_army_history, obs_state.opponent_land_history])

            key, k1 = jrandom.split(key)
            action_p0 = agent.greedy_fn(agent.network, obs_augmented, mask, temporal)
            action_p1 = random_action(k1, obs_p1)

            actions = jnp.stack([action_p0, action_p1])
            timestep, state = env.step(state, actions, pool)

            if timestep.terminated or timestep.truncated:
                winner = int(timestep.info.winner)
                if winner == 0:
                    wins += 1
                elif winner == 1:
                    losses += 1
                else:
                    draws += 1
                break

        print(
            f"Game {g + 1:3d}/{num_games} | "
            f"Winner: {'PPO' if winner == 0 else 'Random' if winner == 1 else 'Draw':6s} | "
            f"Steps: {step + 1:3d} | "
            f"Record: {wins}W-{losses}L-{draws}D ({wins / (g + 1) * 100:.0f}%)"
        )

    print(f"\nFinal: {wins}W-{losses}L-{draws}D out of {num_games} games "
          f"({wins / num_games * 100:.1f}% win rate)")


def run_gui(agent, fps=10):
    from generals.gui import ReplayGUI

    cfg = agent.config
    env = _make_eval_env(cfg)
    key = jrandom.PRNGKey(123)
    key, pool_key = jrandom.split(key)
    pool, _ = env.reset(pool_key)

    agents = ["PPO Agent", "Random"]
    wins, losses, games = 0, 0, 0

    while True:
        key, init_key = jrandom.split(key)
        state = env.init_state(init_key)
        obs_state = agent.init_obs_state_fn(cfg.pad_to, cfg.pad_to)

        if games == 0:
            gui = ReplayGUI(state, agent_ids=agents)
        else:
            gui.update(state)

        print(f"\n--- Game {games + 1} ---")

        for step in range(cfg.truncation):
            obs_p0 = get_observation(state, 0)
            obs_p1 = get_observation(state, 1)

            obs_arr = obs_to_array(obs_p0)
            mask = compute_valid_move_mask(obs_p0.armies, obs_p0.owned_cells, obs_p0.mountains)
            obs_augmented, obs_state = agent.augment_fn(obs_arr, obs_state)
            temporal = jnp.stack([obs_state.opponent_army_history, obs_state.opponent_land_history])

            key, k1 = jrandom.split(key)
            action_p0 = agent.greedy_fn(agent.network, obs_augmented, mask, temporal)
            action_p1 = random_action(k1, obs_p1)

            actions = jnp.stack([action_p0, action_p1])
            timestep, state = env.step(state, actions, pool)

            gui.update(state, timestep.info)
            gui.tick(fps=fps)

            if timestep.terminated or timestep.truncated:
                winner = int(timestep.info.winner)
                if winner == 0:
                    wins += 1
                elif winner == 1:
                    losses += 1
                games += 1
                label = "PPO" if winner == 0 else "Random" if winner == 1 else "Draw"
                print(f"Winner: {label} | Steps: {step + 1} | "
                      f"Record: {wins}W-{losses}L ({wins / games * 100:.0f}%)")
                time.sleep(1.5)
                break

    gui.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Checkpoint path")
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-games", type=int, default=50)
    args = parser.parse_args()

    agent = Agent.load(args.model, args.config)
    print(f"Loaded {args.model} ({agent.param_count():,} params)")

    if args.headless:
        run_headless(agent, num_games=args.num_games)
    else:
        run_gui(agent, fps=args.fps)


if __name__ == "__main__":
    main()
