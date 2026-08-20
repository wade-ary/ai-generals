"""Run the BC run-1 checkpoint tournament on 24x24 maps in Colab.

The script discovers every milestone model and EMA model in checkpoints_run1,
runs one combined 24x24 round robin, and prints/saves JSON and CSV results.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from ruamel.yaml import YAML

from config import Config
from evals.agent import Agent
from generals.core.action import compute_valid_move_mask
from generals.core.env import GeneralsEnv
from generals.core.game import get_observation
from networks import build_network, get_network_bundle, obs_to_array, reset_done_envs
from train.bc import BCTrainConfig, make_optimizer


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
    parser.add_argument("--seed", type=int, default=12_345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.games_per_pair <= 0 or args.games_per_pair % 2:
        raise ValueError("--games-per-pair must be a positive even number")

    agents = discover_bc_agents(args.checkpoint_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grid_size = 24
    maps = args.games_per_pair // 2
    env = make_eval_env(agents[0].config, grid_size, maps)
    pool, _ = env.reset(jrandom.PRNGKey(args.seed + grid_size * 1_000))
    print(
        "\n##### 24x24 combined checkpoint/EMA tournament "
        f"({args.games_per_pair} paired games per matchup) #####"
    )
    pairs, standings = run_round_robin(
        agents,
        env,
        pool,
        args.games_per_pair,
        args.seed + grid_size * 10_000,
    )
    label = "bc_run1_24x24"
    _print_standings(label, standings)
    _save_tournament(
        args.output_dir,
        label,
        grid_size,
        args.games_per_pair,
        pairs,
        standings,
    )
    print(f"All results saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
