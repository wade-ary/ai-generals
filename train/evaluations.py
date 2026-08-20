"""Evaluation functions: play network vs random or vs opponent."""

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from typing import NamedTuple

from generals.core.game import get_observation
from generals.core.action import compute_valid_move_mask

from networks import (
    obs_to_array,
    random_action,
    reset_done_envs,
)
from evals.agent import Agent
from evals.ref_eval import ref_eval
from evals.matchup import play_match


@partial(jax.jit, static_argnames=["env", "truncation", "n_maps", "grid_size", "augment_fn", "reset_fn", "greedy_fn"])
def evaluate(env, network, key, truncation, n_maps, grid_size,
             obs_state, augment_fn, reset_fn, greedy_fn, pool=None):
    """Play n_maps maps as both player positions vs random.

    Each map is played twice: once with network as p0, once as p1.
    Returns (wins, losses, draws, finished, won_both, lost_both, split, key).
    """
    step_fn = (lambda s, a: env.step(s, a, pool)) if pool is not None else env.step
    key, key_fwd, key_rev = jrandom.split(key, 3)
    key, *init_keys = jrandom.split(key, n_maps + 1)
    init_keys_arr = jnp.stack(init_keys)

    # ---- Forward: network=p0, random=p1 ----
    states_fwd = jax.vmap(env.init_state)(init_keys_arr)

    def scan_fwd(carry, _):
        states, rng, finished, net_won, net_lost, drew, obs_st = carry
        obs_p0 = jax.vmap(lambda s: get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: get_observation(s, 1))(states)

        obs_arr = jax.vmap(obs_to_array)(obs_p0)
        masks = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_p0)
        obs_aug, obs_st = jax.vmap(augment_fn)(obs_arr, obs_st)
        temporal = jnp.stack([obs_st.opponent_army_history, obs_st.opponent_land_history], axis=1)
        a0 = jax.vmap(greedy_fn, in_axes=(None, 0, 0, 0))(network, obs_aug, masks, temporal)

        rng, *ks = jrandom.split(rng, n_maps + 1)
        a1 = jax.vmap(random_action)(jnp.stack(ks), obs_p1)

        actions = jnp.stack([a0, a1], axis=1)
        timesteps, new_states = jax.vmap(step_fn)(states, actions)

        dones = timesteps.terminated | timesteps.truncated
        new_done = dones & ~finished
        net_won = net_won | (new_done & (timesteps.info.winner == 0))
        net_lost = net_lost | (new_done & (timesteps.info.winner == 1))
        drew = drew | (new_done & timesteps.truncated & ~timesteps.terminated)
        finished = finished | dones
        obs_st = reset_done_envs(obs_st, dones)

        return (new_states, rng, finished, net_won, net_lost, drew, obs_st), None

    z = jnp.zeros(n_maps, jnp.bool_)
    (_, _, fwd_fin, fwd_won, fwd_lost, fwd_drew, _), _ = jax.lax.scan(
        scan_fwd, (states_fwd, key_fwd, z, z, z, z, obs_state), None, length=truncation)

    # ---- Reverse: random=p0, network=p1 ----
    states_rev = jax.vmap(env.init_state)(init_keys_arr)

    def scan_rev(carry, _):
        states, rng, finished, net_won, net_lost, drew, obs_st = carry
        obs_p0 = jax.vmap(lambda s: get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: get_observation(s, 1))(states)

        obs_arr = jax.vmap(obs_to_array)(obs_p1)
        masks = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_p1)
        obs_aug, obs_st = jax.vmap(augment_fn)(obs_arr, obs_st)
        temporal = jnp.stack([obs_st.opponent_army_history, obs_st.opponent_land_history], axis=1)
        a1 = jax.vmap(greedy_fn, in_axes=(None, 0, 0, 0))(network, obs_aug, masks, temporal)

        rng, *ks = jrandom.split(rng, n_maps + 1)
        a0 = jax.vmap(random_action)(jnp.stack(ks), obs_p0)

        actions = jnp.stack([a0, a1], axis=1)
        timesteps, new_states = jax.vmap(step_fn)(states, actions)

        dones = timesteps.terminated | timesteps.truncated
        new_done = dones & ~finished
        net_won = net_won | (new_done & (timesteps.info.winner == 1))
        net_lost = net_lost | (new_done & (timesteps.info.winner == 0))
        drew = drew | (new_done & timesteps.truncated & ~timesteps.terminated)
        finished = finished | dones
        obs_st = reset_done_envs(obs_st, dones)

        return (new_states, rng, finished, net_won, net_lost, drew, obs_st), None

    z = jnp.zeros(n_maps, jnp.bool_)
    (_, _, rev_fin, rev_won, rev_lost, rev_drew, _), _ = jax.lax.scan(
        scan_rev, (states_rev, key_rev, z, z, z, z, obs_state), None, length=truncation)

    # ---- Aggregate ----
    total_wins = jnp.sum(fwd_won) + jnp.sum(rev_won)
    total_losses = jnp.sum(fwd_lost) + jnp.sum(rev_lost)
    total_draws = jnp.sum(fwd_drew) + jnp.sum(rev_drew)
    total_finished = jnp.sum(fwd_fin) + jnp.sum(rev_fin)

    # Paired map stats (maps where both games had decisive outcomes)
    both_decisive = (fwd_won | fwd_lost) & (rev_won | rev_lost)
    won_both = jnp.sum(both_decisive & fwd_won & rev_won)
    lost_both = jnp.sum(both_decisive & fwd_lost & rev_lost)
    split = jnp.sum(both_decisive & ((fwd_won & rev_lost) | (fwd_lost & rev_won)))

    return total_wins, total_losses, total_draws, total_finished, won_both, lost_both, split, key


class EvalCtx(NamedTuple):
    """Loop-invariant context for periodic_eval (network plumbing + reference setup)."""
    bundle: dict
    single_state: object
    augment_fn: object
    reset_fn: object
    greedy_fn: object
    ref_agents: object
    ref_h2h: object
    ref_eval_env: object
    ref_eval_pool: object
    eval_opponent_agent: object


def periodic_eval(it, cfg, eval_freq, network, ema_params, static,
                  eval_env, eval_pool, ev, logger, key, last_eval_wr):
    """Eval vs random (+ reference ELO) on eval iters; logs results.

    Returns (eval_ran, last_eval_wr, key). last_eval_wr updates only when the
    vs-random eval runs (it drives curriculum advancement).
    """
    eval_ran = False
    if eval_freq > 0 and it % eval_freq == 0:
        key, eval_key = jrandom.split(key)
        n_maps = cfg.eval_games // 2
        if ev.eval_opponent_agent is not None:
            current = Agent(network, cfg, ev.bundle, name="current")
            opponent = ev.eval_opponent_agent
            # Same key => identical map batch, with player seats reversed.
            current_p0, opponent_p1, draws_fwd = play_match(
                current, opponent, eval_env, eval_pool, n_maps, cfg.truncation, eval_key)
            opponent_p0, current_p1, draws_rev = play_match(
                opponent, current, eval_env, eval_pool, n_maps, cfg.truncation, eval_key)
            ew = current_p0 + current_p1
            el = opponent_p1 + opponent_p0
            ed = draws_fwd + draws_rev
            edone = ew + el + ed
            wb = lb = sp = 0  # paired aggregate categories are unavailable here
            last_eval_wr = ew / max(edone, 1)
            eval_ran = True
            print(
                f"  EVAL: {ew}W/{el}L/{ed}D ({last_eval_wr * 100:.0f}%) "
                f"greedy vs {opponent.name} | Maps({n_maps}) | "
                f"current wins P0/P1={current_p0}/{current_p1}")
            logger.log_eval(it, ew, el, ed, edone)
            logger.log(it, {
                "eval/current_wins_as_p0": current_p0 / max(n_maps, 1),
                "eval/current_wins_as_p1": current_p1 / max(n_maps, 1),
            })
        else:
            eval_obs_state = jax.tree.map(
                lambda x: jnp.tile(x, (n_maps, *([1] * x.ndim))), ev.single_state)
            ew, el, ed, edone, wb, lb, sp, _ = evaluate(
                eval_env, network, eval_key, cfg.truncation, n_maps, cfg.pad_to,
                eval_obs_state, ev.augment_fn, ev.reset_fn, ev.greedy_fn, pool=eval_pool)
            ew, el, ed, edone = int(ew), int(el), int(ed), int(edone)
            wb, lb, sp = int(wb), int(lb), int(sp)
            last_eval_wr = ew / max(edone, 1)
            eval_ran = True
            print(f"  EVAL: {ew}W/{el}L/{ed}D ({last_eval_wr * 100:.0f}%) greedy vs random | Maps({n_maps}): {wb}WB/{lb}LB/{sp}S")
            logger.log_eval(it, ew, el, ed, edone)
            logger.log(it, {
                "eval/won_both": wb / max(n_maps, 1),
                "eval/lost_both": lb / max(n_maps, 1),
                "eval/split": sp / max(n_maps, 1),
            })

    if ev.ref_agents and (it == 0 or (it + 1) % cfg.ref_eval_every == 0):
        ema_network = eqx.combine(ema_params, static)
        ema_agent = Agent(ema_network, cfg, ev.bundle, name="_ema")
        if cfg.eval_ema_only:
            candidates = [ema_agent]
        else:
            candidates = [Agent(network, cfg, ev.bundle, name="_current"), ema_agent]
        key, ref_key = jrandom.split(key)
        ratings, full_h2h = ref_eval(
            candidates=candidates,
            ref_agents=ev.ref_agents, ref_h2h=ev.ref_h2h,
            env=ev.ref_eval_env, pool=ev.ref_eval_pool,
            num_games=cfg.ref_eval_games,
            truncation=ev.ref_eval_env.truncation,
            key=ref_key,
        )
        ref_names = [a.name for a in ev.ref_agents]
        max_ref_elo = max(ratings[n] for n in ref_names)
        if cfg.eval_ema_only:
            print(f"  REF_EVAL ELO (iter {it+1}): ema={ratings['_ema']:.0f}")
        else:
            print(f"  REF_EVAL ELO (iter {it+1}): current={ratings['_current']:.0f} ema={ratings['_ema']:.0f}")
        ref_metrics = {
            "ref_elo/ema": ratings["_ema"],
            "ref_elo/ema_vs_max": ratings["_ema"] - max_ref_elo,
        }
        if not cfg.eval_ema_only:
            ref_metrics["ref_elo/current"] = ratings["_current"]
            ref_metrics["ref_elo/current_vs_max"] = ratings["_current"] - max_ref_elo
        for rn in ref_names:
            for cand_tag in [c.name for c in candidates]:
                r = full_h2h.get(cand_tag, {}).get(rn, {})
                w, l, d = r.get("wins", 0), r.get("losses", 0), r.get("draws", 0)
                tot = w + l + d
                wr = w / tot if tot > 0 else 0.0
                tag = "current" if cand_tag == "_current" else "ema"
                ref_metrics[f"ref_wr/{tag}_vs_{rn}"] = wr
        logger.log(it, ref_metrics)

    return eval_ran, last_eval_wr, key
