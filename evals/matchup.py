"""Play games between two agents."""

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core.game import get_observation
from generals.core.action import compute_valid_move_mask

from networks import obs_to_array, reset_done_envs
from evals.agent import Agent


def _keyless(fn):
    def wrapped(net, obs, mask, temporal, key):
        return fn(net, obs, mask, temporal)
    return wrapped


def play_match(agent_a, agent_b, env, pool, num_games, truncation, key):
    """Play num_games between two agents. Returns (wins_a, wins_b, draws) as ints.

    agent_a plays as P0, agent_b plays as P1.
    """
    assert agent_a.pad_to == agent_b.pad_to
    pad_to = agent_a.pad_to

    action_fn_a = _keyless(agent_a.greedy_fn)
    action_fn_b = _keyless(agent_b.greedy_fn)
    augment_fn_a = agent_a.augment_fn
    augment_fn_b = agent_b.augment_fn

    single_state = agent_a.init_obs_state_fn(pad_to, pad_to)
    batched_state = jax.tree.map(
        lambda x: jnp.repeat(x[None], num_games, axis=0), single_state
    )

    step_fn = (lambda s, a: env.step(s, a, pool)) if pool is not None else env.step

    @jax.jit
    def _play(net_a, net_b, key):
        key, *init_keys = jrandom.split(key, num_games + 1)
        states = jax.vmap(env.init_state)(jnp.stack(init_keys))

        finished = jnp.zeros(num_games, dtype=jnp.bool_)
        wins_a = jnp.int32(0)
        wins_b = jnp.int32(0)
        draws = jnp.int32(0)
        obs_state_a = batched_state
        obs_state_b = batched_state

        def scan_body(carry, _):
            states, key, finished, wins_a, wins_b, draws, obs_state_a, obs_state_b = carry

            key, key_a, key_b = jrandom.split(key, 3)
            keys_a = jrandom.split(key_a, num_games)
            keys_b = jrandom.split(key_b, num_games)

            obs_p0 = jax.vmap(lambda s: get_observation(s, 0))(states)
            obs_p1 = jax.vmap(lambda s: get_observation(s, 1))(states)

            obs_arr_a = jax.vmap(obs_to_array)(obs_p0)
            masks_a = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_p0)
            obs_aug_a, obs_state_a = jax.vmap(augment_fn_a)(obs_arr_a, obs_state_a)
            temporal_a = jnp.stack([obs_state_a.opponent_army_history, obs_state_a.opponent_land_history], axis=1)
            action_a = jax.vmap(action_fn_a, in_axes=(None, 0, 0, 0, 0))(net_a, obs_aug_a, masks_a, temporal_a, keys_a)

            obs_arr_b = jax.vmap(obs_to_array)(obs_p1)
            masks_b = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_p1)
            obs_aug_b, obs_state_b = jax.vmap(augment_fn_b)(obs_arr_b, obs_state_b)
            temporal_b = jnp.stack([obs_state_b.opponent_army_history, obs_state_b.opponent_land_history], axis=1)
            action_b = jax.vmap(action_fn_b, in_axes=(None, 0, 0, 0, 0))(net_b, obs_aug_b, masks_b, temporal_b, keys_b)

            actions = jnp.stack([action_a, action_b], axis=1)
            timesteps, new_states = jax.vmap(step_fn)(states, actions)

            dones = timesteps.terminated | timesteps.truncated
            new_done = dones & ~finished
            wins_a = wins_a + jnp.sum(new_done & (timesteps.info.winner == 0))
            wins_b = wins_b + jnp.sum(new_done & (timesteps.info.winner == 1))
            draws = draws + jnp.sum(new_done & timesteps.truncated & ~timesteps.terminated)
            finished = finished | dones

            obs_state_a = reset_done_envs(obs_state_a, dones)
            obs_state_b = reset_done_envs(obs_state_b, dones)

            return (new_states, key, finished, wins_a, wins_b, draws, obs_state_a, obs_state_b), None

        (_, _, _, wins_a, wins_b, draws, _, _), _ = jax.lax.scan(
            scan_body,
            (states, key, finished, wins_a, wins_b, draws, obs_state_a, obs_state_b),
            None,
            length=truncation
        )
        return wins_a, wins_b, draws

    w_a, w_b, d = _play(agent_a.network, agent_b.network, key)
    return int(w_a), int(w_b), int(d)


def round_robin(agents, env, pool, num_games, truncation, key):
    """Play all pairs both ways. Returns h2h dict.

    h2h[a][b] = {"wins": int, "losses": int, "draws": int}  (from a's perspective)
    """
    from itertools import combinations

    names = [a.name for a in agents]
    agent_map = {a.name: a for a in agents}
    h2h = {n: {o: {"wins": 0, "losses": 0, "draws": 0} for o in names if o != n} for n in names}

    for a, b in combinations(names, 2):
        for p0, p1 in [(a, b), (b, a)]:
            key, mk = jrandom.split(key)
            w0, w1, d = play_match(agent_map[p0], agent_map[p1], env, pool, num_games, truncation, mk)
            h2h[p0][p1]["wins"] += w0
            h2h[p0][p1]["losses"] += w1
            h2h[p0][p1]["draws"] += d
            h2h[p1][p0]["wins"] += w1
            h2h[p1][p0]["losses"] += w0
            h2h[p1][p0]["draws"] += d

    return h2h


def compute_elo(names, h2h, seed=42, passes=100, k=0.01):
    """Compute ELO from head-to-head dict.

    h2h[a][b] = {"wins": int, "losses": int, "draws": int}
    """
    import random
    from itertools import combinations

    outcomes = []
    for a, b in combinations(names, 2):
        if b not in h2h.get(a, {}):
            continue
        r = h2h[a][b]
        outcomes.extend([(a, b, 1.0)] * r["wins"])
        outcomes.extend([(a, b, 0.0)] * r["losses"])
        outcomes.extend([(a, b, 0.5)] * r["draws"])

    ratings = {n: 1500.0 for n in names}
    rng = random.Random(seed)
    for _ in range(passes):
        rng.shuffle(outcomes)
        for a, b, score in outcomes:
            ea = 1.0 / (1.0 + 10.0 ** ((ratings[b] - ratings[a]) / 400.0))
            ratings[a] += k * (score - ea)
            ratings[b] += k * ((1.0 - score) - (1.0 - ea))
    return ratings


def merge_h2h(base, new):
    """Merge two h2h dicts, summing W/L/D."""
    merged = {}
    all_names = set(base.keys()) | set(new.keys())
    for a in all_names:
        merged[a] = {}
        opponents = set(base.get(a, {}).keys()) | set(new.get(a, {}).keys())
        for b in opponents:
            rb = base.get(a, {}).get(b, {"wins": 0, "losses": 0, "draws": 0})
            rn = new.get(a, {}).get(b, {"wins": 0, "losses": 0, "draws": 0})
            merged[a][b] = {
                "wins": rb["wins"] + rn["wins"],
                "losses": rb["losses"] + rn["losses"],
                "draws": rb["draws"] + rn["draws"],
            }
    return merged
