"""Run the complete BC run-1 checkpoint tournament in Colab.

The script discovers every milestone model and EMA model in checkpoints_run1,
runs a combined round robin on both 12x12 and 24x24 maps, then plays the 12x12
tournament winner against S_2250. Results are printed and saved as JSON/CSV,
and one winner-vs-S_2250 game is rendered as a GIF.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

# Colab renders pygame into an off-screen surface.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from ruamel.yaml import YAML

from config import Config
from evals.agent import Agent
from generals.core.action import compute_valid_move_mask
from generals.core.env import GeneralsEnv
from generals.core.game import get_info, get_observation
from networks import build_network, get_network_bundle, obs_to_array, reset_done_envs
from train.bc import BCTrainConfig, make_optimizer
from train.eval_checkpoint_match import save_replay_gif


@dataclass(frozen=True)
class PairResult:
    agent_a: str
    agent_b: str
    games: int
    a_wins: int
    b_wins: int
    draws: int
    a_score: float


def _load_config(path: Path) -> Config:
    yaml = YAML(typ="safe")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.load(handle)
    valid = Config.__dataclass_fields__
    return Config(**{key: value for key, value in data.items() if key in valid})


def _load_bc_agent(model_path: Path) -> Agent:
    """Load either a standalone EMA tree or a regular (network, AdamW) tree."""

    config_path = model_path.with_suffix(".yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config for {model_path}: {config_path}")

    cfg = _load_config(config_path)
    network = build_network(cfg, jrandom.PRNGKey(0))
    if "_ema_step_" in model_path.name:
        network = eqx.tree_deserialise_leaves(model_path, network)
    else:
        # Regular BC checkpoints include the optimizer state. Its tree must
        # exactly match training for Equinox deserialization to succeed.
        train_cfg = BCTrainConfig(
            minibatch_size=cfg.minibatch_size,
            max_grad_norm=cfg.max_grad_norm,
        )
        optimizer = make_optimizer(train_cfg)
        opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
        network, _ = eqx.tree_deserialise_leaves(
            model_path, (network, opt_state)
        )

    return Agent(
        network,
        cfg,
        get_network_bundle(cfg.network),
        name=model_path.stem,
    )


def discover_bc_agents(checkpoint_dir: Path) -> list[Agent]:
    regular = sorted(checkpoint_dir.glob("BC_S_24_step_*.eqx"))
    ema = sorted(checkpoint_dir.glob("BC_S_24_ema_step_*.eqx"))
    paths = regular + ema
    if not paths:
        raise FileNotFoundError(
            f"No milestone checkpoints found under {checkpoint_dir}"
        )
    print(f"Discovered {len(regular)} regular and {len(ema)} EMA checkpoints")
    return [_load_bc_agent(path) for path in paths]


def make_eval_env(cfg: Config, grid_size: int, num_maps: int) -> GeneralsEnv:
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
        grid_dims=(grid_size, grid_size),
        pad_to=grid_size,
        min_generals_distance=min_distance,
        max_generals_distance=max_distance,
        truncation=cfg.truncation,
        num_cities_range=cities,
        castle_val_range=castles,
        mountain_density_range=(cfg.mountain_density_min, cfg.mountain_density_max),
        pool_size=max(1_000, num_maps),
    )


def make_bc_matcher(
    template: Agent,
    env: GeneralsEnv,
    pool: Any,
    num_games: int,
):
    """Compile one homogeneous BC match kernel and reuse it for every pair."""

    single_obs_state = template.init_obs_state_fn(template.pad_to, template.pad_to)
    initial_obs_state = jax.tree.map(
        lambda value: jnp.repeat(value[None], num_games, axis=0),
        single_obs_state,
    )

    def action_fn(network, obs, mask, temporal, _key):
        return template.greedy_fn(network, obs, mask, temporal)

    @jax.jit
    def play(net_a, net_b, key):
        key, *init_keys = jrandom.split(key, num_games + 1)
        states = jax.vmap(env.init_state)(jnp.stack(init_keys))
        finished = jnp.zeros(num_games, dtype=jnp.bool_)
        wins_a = jnp.int32(0)
        wins_b = jnp.int32(0)
        draws = jnp.int32(0)

        def body(carry, _):
            states, key, finished, wins_a, wins_b, draws, obs_a, obs_b = carry
            key, key_a, key_b = jrandom.split(key, 3)
            keys_a = jrandom.split(key_a, num_games)
            keys_b = jrandom.split(key_b, num_games)
            raw_a = jax.vmap(lambda state: get_observation(state, 0))(states)
            raw_b = jax.vmap(lambda state: get_observation(state, 1))(states)
            aug_a, obs_a = jax.vmap(template.augment_fn)(
                jax.vmap(obs_to_array)(raw_a), obs_a
            )
            aug_b, obs_b = jax.vmap(template.augment_fn)(
                jax.vmap(obs_to_array)(raw_b), obs_b
            )
            mask_a = jax.vmap(
                lambda obs: compute_valid_move_mask(
                    obs.armies, obs.owned_cells, obs.mountains
                )
            )(raw_a)
            mask_b = jax.vmap(
                lambda obs: compute_valid_move_mask(
                    obs.armies, obs.owned_cells, obs.mountains
                )
            )(raw_b)
            temporal_a = jnp.stack(
                [obs_a.opponent_army_history, obs_a.opponent_land_history], axis=1
            )
            temporal_b = jnp.stack(
                [obs_b.opponent_army_history, obs_b.opponent_land_history], axis=1
            )
            action_a = jax.vmap(action_fn, in_axes=(None, 0, 0, 0, 0))(
                net_a, aug_a, mask_a, temporal_a, keys_a
            )
            action_b = jax.vmap(action_fn, in_axes=(None, 0, 0, 0, 0))(
                net_b, aug_b, mask_b, temporal_b, keys_b
            )
            timesteps, new_states = jax.vmap(
                lambda state, actions: env.step(state, actions, pool)
            )(states, jnp.stack([action_a, action_b], axis=1))
            dones = timesteps.terminated | timesteps.truncated
            newly_done = dones & ~finished
            wins_a += jnp.sum(newly_done & (timesteps.info.winner == 0))
            wins_b += jnp.sum(newly_done & (timesteps.info.winner == 1))
            draws += jnp.sum(newly_done & timesteps.truncated & ~timesteps.terminated)
            return (
                new_states,
                key,
                finished | dones,
                wins_a,
                wins_b,
                draws,
                reset_done_envs(obs_a, dones),
                reset_done_envs(obs_b, dones),
            ), None

        (_, _, _, wins_a, wins_b, draws, _, _), _ = jax.lax.scan(
            body,
            (
                states,
                key,
                finished,
                wins_a,
                wins_b,
                draws,
                initial_obs_state,
                initial_obs_state,
            ),
            None,
            length=env.truncation,
        )
        return wins_a, wins_b, draws

    def match(agent_a: Agent, agent_b: Agent, key: jax.Array):
        wins_a, wins_b, draws = play(agent_a.network, agent_b.network, key)
        return int(wins_a), int(wins_b), int(draws)

    return match


def _paired_result(
    agent_a: Agent,
    agent_b: Agent,
    matcher: Any,
    games: int,
    seed: int,
) -> PairResult:
    if games <= 0 or games % 2:
        raise ValueError("games per pair must be a positive even number")
    key = jrandom.PRNGKey(seed)
    a_p0, b_p1, draws_ab = matcher(agent_a, agent_b, key)
    b_p0, a_p1, draws_ba = matcher(agent_b, agent_a, key)
    a_wins = a_p0 + a_p1
    b_wins = b_p1 + b_p0
    draws = draws_ab + draws_ba
    return PairResult(
        agent_a.name,
        agent_b.name,
        games,
        a_wins,
        b_wins,
        draws,
        (a_wins + 0.5 * draws) / games,
    )


def run_round_robin(
    agents: list[Agent],
    env: GeneralsEnv,
    pool: Any,
    games_per_pair: int,
    seed: int,
) -> tuple[list[PairResult], list[dict[str, Any]]]:
    totals = {
        agent.name: {"wins": 0, "losses": 0, "draws": 0, "games": 0}
        for agent in agents
    }
    pairs: list[PairResult] = []
    all_pairs = list(combinations(agents, 2))
    matcher = make_bc_matcher(agents[0], env, pool, games_per_pair // 2)
    for index, (agent_a, agent_b) in enumerate(all_pairs, start=1):
        result = _paired_result(
            agent_a,
            agent_b,
            matcher,
            games_per_pair,
            seed + index,
        )
        pairs.append(result)
        totals[result.agent_a]["wins"] += result.a_wins
        totals[result.agent_a]["losses"] += result.b_wins
        totals[result.agent_a]["draws"] += result.draws
        totals[result.agent_a]["games"] += result.games
        totals[result.agent_b]["wins"] += result.b_wins
        totals[result.agent_b]["losses"] += result.a_wins
        totals[result.agent_b]["draws"] += result.draws
        totals[result.agent_b]["games"] += result.games
        print(
            f"[{index:3d}/{len(all_pairs):3d}] {result.agent_a} vs "
            f"{result.agent_b}: {result.a_wins}-{result.b_wins}-"
            f"{result.draws} | A score={100 * result.a_score:.1f}%"
        )

    standings = []
    for name, row in totals.items():
        score = (row["wins"] + 0.5 * row["draws"]) / max(1, row["games"])
        standings.append({"name": name, **row, "score": score})
    standings.sort(key=lambda row: (row["score"], row["wins"]), reverse=True)
    return pairs, standings


def _print_standings(title: str, standings: list[dict[str, Any]]) -> None:
    print(f"\n=== {title} standings ===")
    print(f"{'rank':>4}  {'agent':44s} {'W':>5} {'L':>5} {'D':>5} {'score':>8}")
    for rank, row in enumerate(standings, start=1):
        print(
            f"{rank:4d}  {row['name'][:44]:44s} {row['wins']:5d} "
            f"{row['losses']:5d} {row['draws']:5d} {100 * row['score']:7.2f}%"
        )


def _save_tournament(
    output_dir: Path,
    label: str,
    grid_size: int,
    games_per_pair: int,
    pairs: list[PairResult],
    standings: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "grid_size": grid_size,
        "games_per_pair": games_per_pair,
        "pairs": [asdict(result) for result in pairs],
        "standings": standings,
    }
    with (output_dir / f"{label}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    with (output_dir / f"{label}_standings.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=standings[0].keys())
        writer.writeheader()
        writer.writerows(standings)


def play_match_mixed_padding(
    agent_a: Agent,
    agent_b: Agent,
    env: GeneralsEnv,
    pool: Any,
    num_games: int,
    key: jax.Array,
) -> tuple[int, int, int]:
    """Vectorized match for agents whose networks use different pad sizes."""

    state_a = agent_a.init_obs_state_fn(agent_a.pad_to, agent_a.pad_to)
    state_b = agent_b.init_obs_state_fn(agent_b.pad_to, agent_b.pad_to)
    batched_a = jax.tree.map(lambda x: jnp.repeat(x[None], num_games, 0), state_a)
    batched_b = jax.tree.map(lambda x: jnp.repeat(x[None], num_games, 0), state_b)

    @jax.jit
    def _play(net_a, net_b, match_key):
        match_key, *init_keys = jrandom.split(match_key, num_games + 1)
        states = jax.vmap(env.init_state)(jnp.stack(init_keys))
        finished = jnp.zeros(num_games, dtype=jnp.bool_)
        wins_a = jnp.int32(0)
        wins_b = jnp.int32(0)
        draws = jnp.int32(0)

        def body(carry, _):
            states, finished, wins_a, wins_b, draws, obs_a, obs_b = carry
            raw_a = jax.vmap(lambda s: get_observation(s, 0))(states)
            raw_b = jax.vmap(lambda s: get_observation(s, 1))(states)
            aug_a, obs_a = jax.vmap(agent_a.augment_fn)(
                jax.vmap(obs_to_array)(raw_a), obs_a
            )
            aug_b, obs_b = jax.vmap(agent_b.augment_fn)(
                jax.vmap(obs_to_array)(raw_b), obs_b
            )
            mask_a = jax.vmap(
                lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains)
            )(raw_a)
            mask_b = jax.vmap(
                lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains)
            )(raw_b)
            temporal_a = jnp.stack(
                [obs_a.opponent_army_history, obs_a.opponent_land_history], axis=1
            )
            temporal_b = jnp.stack(
                [obs_b.opponent_army_history, obs_b.opponent_land_history], axis=1
            )
            actions_a = jax.vmap(agent_a.greedy_fn, in_axes=(None, 0, 0, 0))(
                net_a, aug_a, mask_a, temporal_a
            )
            actions_b = jax.vmap(agent_b.greedy_fn, in_axes=(None, 0, 0, 0))(
                net_b, aug_b, mask_b, temporal_b
            )
            timesteps, new_states = jax.vmap(lambda s, a: env.step(s, a, pool))(
                states, jnp.stack([actions_a, actions_b], axis=1)
            )
            dones = timesteps.terminated | timesteps.truncated
            newly_done = dones & ~finished
            wins_a += jnp.sum(newly_done & (timesteps.info.winner == 0))
            wins_b += jnp.sum(newly_done & (timesteps.info.winner == 1))
            draws += jnp.sum(newly_done & timesteps.truncated & ~timesteps.terminated)
            return (
                new_states,
                finished | dones,
                wins_a,
                wins_b,
                draws,
                reset_done_envs(obs_a, dones),
                reset_done_envs(obs_b, dones),
            ), None

        (_, _, wins_a, wins_b, draws, _, _), _ = jax.lax.scan(
            body,
            (states, finished, wins_a, wins_b, draws, batched_a, batched_b),
            None,
            length=env.truncation,
        )
        return wins_a, wins_b, draws

    wins_a, wins_b, draws = _play(agent_a.network, agent_b.network, key)
    return int(wins_a), int(wins_b), int(draws)


def evaluate_mixed_paired(
    agent_a: Agent,
    agent_b: Agent,
    env: GeneralsEnv,
    pool: Any,
    games: int,
    seed: int,
) -> PairResult:
    if games <= 0 or games % 2:
        raise ValueError("S_2250 games must be a positive even number")
    maps = games // 2
    key = jrandom.PRNGKey(seed)
    a_p0, b_p1, draws_ab = play_match_mixed_padding(
        agent_a, agent_b, env, pool, maps, key
    )
    b_p0, a_p1, draws_ba = play_match_mixed_padding(
        agent_b, agent_a, env, pool, maps, key
    )
    a_wins = a_p0 + a_p1
    b_wins = b_p1 + b_p0
    draws = draws_ab + draws_ba
    return PairResult(
        agent_a.name,
        agent_b.name,
        games,
        a_wins,
        b_wins,
        draws,
        (a_wins + 0.5 * draws) / games,
    )


def record_mixed_game(
    agent_p0: Agent,
    agent_p1: Agent,
    env: GeneralsEnv,
    pool: Any,
    seed: int,
):
    state = env.init_state(jrandom.PRNGKey(seed))
    obs_p0_state = agent_p0.init_obs_state_fn(agent_p0.pad_to, agent_p0.pad_to)
    obs_p1_state = agent_p1.init_obs_state_fn(agent_p1.pad_to, agent_p1.pad_to)
    states = [state]
    infos = [get_info(state)]
    winner = -1
    for _ in range(env.truncation):
        raw_p0 = get_observation(state, 0)
        raw_p1 = get_observation(state, 1)
        aug_p0, obs_p0_state = agent_p0.augment_fn(
            obs_to_array(raw_p0), obs_p0_state
        )
        aug_p1, obs_p1_state = agent_p1.augment_fn(
            obs_to_array(raw_p1), obs_p1_state
        )
        mask_p0 = compute_valid_move_mask(
            raw_p0.armies, raw_p0.owned_cells, raw_p0.mountains
        )
        mask_p1 = compute_valid_move_mask(
            raw_p1.armies, raw_p1.owned_cells, raw_p1.mountains
        )
        temporal_p0 = jnp.stack(
            [obs_p0_state.opponent_army_history, obs_p0_state.opponent_land_history]
        )
        temporal_p1 = jnp.stack(
            [obs_p1_state.opponent_army_history, obs_p1_state.opponent_land_history]
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
        states.append(timestep.last_state)
        infos.append(timestep.info)
        if bool(timestep.terminated) or bool(timestep.truncated):
            winner = int(timestep.info.winner)
            break
        state = new_state
    return states, infos, winner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/generals_bc/checkpoints_run1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/generals_bc/eval_run1"),
    )
    parser.add_argument("--games-per-pair", type=int, default=20)
    parser.add_argument("--s-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument(
        "--s2250-checkpoint", type=Path, default=Path("S/sss/S_2250.eqx")
    )
    parser.add_argument(
        "--s2250-config", type=Path, default=Path("S/sss/config (3).yaml")
    )
    parser.add_argument("--gif-stride", type=int, default=4)
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--gif-speed", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.games_per_pair <= 0 or args.games_per_pair % 2:
        raise ValueError("--games-per-pair must be a positive even number")
    if args.s_games <= 0 or args.s_games % 2:
        raise ValueError("--s-games must be a positive even number")

    agents = discover_bc_agents(args.checkpoint_dir)
    agent_by_name = {agent.name: agent for agent in agents}
    s2250 = Agent.load(args.s2250_checkpoint, args.s2250_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    winners: dict[int, Agent] = {}
    for grid_size in (12, 24):
        # Use S_2250's map distribution at 12x12 and BC's at 24x24.
        env_cfg = s2250.config if grid_size == 12 else agents[0].config
        maps = args.games_per_pair // 2
        env = make_eval_env(env_cfg, grid_size, maps)
        pool, _ = env.reset(jrandom.PRNGKey(args.seed + grid_size * 1_000))
        print(
            f"\n##### {grid_size}x{grid_size} combined checkpoint/EMA tournament "
            f"({args.games_per_pair} paired games per matchup) #####"
        )
        pairs, standings = run_round_robin(
            agents,
            env,
            pool,
            args.games_per_pair,
            args.seed + grid_size * 10_000,
        )
        label = f"bc_run1_{grid_size}x{grid_size}"
        _print_standings(label, standings)
        _save_tournament(
            args.output_dir,
            label,
            grid_size,
            args.games_per_pair,
            pairs,
            standings,
        )
        winners[grid_size] = agent_by_name[standings[0]["name"]]

    winner = winners[12]
    final_env = make_eval_env(
        s2250.config, 12, max(args.s_games // 2, 1_000)
    )
    final_pool, _ = final_env.reset(jrandom.PRNGKey(args.seed + 900_000))
    final = evaluate_mixed_paired(
        winner,
        s2250,
        final_env,
        final_pool,
        args.s_games,
        args.seed + 910_000,
    )
    print("\n=== 12x12 tournament winner vs S_2250 ===")
    print(f"Winner checkpoint: {winner.name}")
    print(
        f"{winner.name}: {final.a_wins} wins | S_2250: {final.b_wins} wins | "
        f"draws: {final.draws} | winner score: {100 * final.a_score:.2f}%"
    )
    with (args.output_dir / "winner_12x12_vs_S_2250.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(asdict(final), handle, indent=2)
        handle.write("\n")

    states, infos, replay_winner = record_mixed_game(
        winner,
        s2250,
        final_env,
        final_pool,
        args.seed + 920_000,
    )
    gif_path = args.output_dir / "winner_12x12_vs_S_2250.gif"
    save_replay_gif(
        states,
        infos,
        [winner.name, s2250.name],
        gif_path,
        args.gif_fps,
        max(1, args.gif_stride),
        max(0.1, args.gif_speed),
    )
    replay_name = (
        winner.name if replay_winner == 0
        else s2250.name if replay_winner == 1
        else "draw"
    )
    print(f"Recorded GIF winner: {replay_name}")
    print(f"All results saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
