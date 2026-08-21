"""Rollout collection: self-play environment interaction logic."""

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core.game import get_observation
from generals.core.action import compute_valid_move_mask

from networks import (
    obs_to_array,
    reset_done_envs,
)


# ---- Rollout (self-play) ----


@partial(jax.jit, static_argnames=["env", "num_steps", "grid_size", "reward_fn", "augment_fn", "reset_fn"])
def collect_rollout(states, env, network, key, num_steps, obs_state_p0, obs_state_p1,
                    grid_size, reward_fn, augment_fn, reset_fn, gamma, pool=None):
    """Collect rollout with self-play: same network plays both sides.

    Concatenates both players into a single 2N batch for obs processing,
    augmentation, and network forward pass. Only splits for env.step and rewards.
    Returns training data with env dimension = 2*N.
    """
    n = states.armies.shape[0]
    step_fn = (lambda s, a: env.step(s, a, pool)) if pool is not None else env.step
    cat = lambda a, b: jax.tree.map(lambda x, y: jnp.concatenate([x, y]), a, b)

    # Combine obs states into single (2N, ...) batch
    osp_init = cat(obs_state_p0, obs_state_p1)

    def scan_body(carry, _):
        states, key, osp, _, _ = carry

        obs_p0 = jax.vmap(lambda s: get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: get_observation(s, 1))(states)

        # Concat both players: obs_to_array, masks, augment, network — single 2N batch
        obs_both = cat(obs_p0, obs_p1)
        obs_arr = jax.vmap(obs_to_array)(obs_both)
        masks = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_both)
        obs_aug, new_osp = jax.vmap(augment_fn)(obs_arr, osp)
        obs_aug = obs_aug.astype(jnp.bfloat16)

        # Extract temporal data from obs state for network
        temporal = jnp.stack([new_osp.opponent_army_history, new_osp.opponent_land_history], axis=1)

        keys = jrandom.split(key, 2 * n + 1)
        key = keys[0]
        actions, vals, lps, _, _, _ = jax.vmap(network, in_axes=(0, 0, 0, 0, None))(
            obs_aug, masks, temporal, keys[1:], None)

        # Split actions for env.step: needs (N, 2, 5)
        a0, a1 = actions[:n], actions[n:]
        env_actions = jnp.stack([a0, a1], axis=1)
        timesteps, new_states = jax.vmap(step_fn)(states, env_actions)

        terminated = timesteps.terminated
        truncated = timesteps.truncated
        dones = terminated | truncated
        winners = timesteps.info.winner

        # Rewards from each player's perspective
        next_obs_p0 = jax.vmap(lambda s: get_observation(s, 0))(timesteps.last_state)
        next_obs_p1 = jax.vmap(lambda s: get_observation(s, 1))(timesteps.last_state)
        rewards_p0 = reward_fn(obs_p0, a0, next_obs_p0, winners, truncated=truncated, gamma=gamma)
        winners_p1 = jnp.where(winners >= 0, 1 - winners, winners)
        rewards_p1 = reward_fn(obs_p1, a1, next_obs_p1, winners_p1, truncated=truncated, gamma=gamma)

        # Reset obs state for done envs (both players see same dones)
        dones_both = jnp.concatenate([dones, dones])
        osp = reset_done_envs(new_osp, dones_both)

        # Mean owned cities (p0 only, scalar)
        owned_cities_p0 = (obs_p0.cities * obs_p0.owned_cells).sum()

        data = (
            obs_aug,                                                  # (2N, C, H, W)
            masks,                                                    # (2N, H, W, 4)
            temporal,                                                 # (2N, 2, temporal_window)
            actions,                                                  # (2N, 5)
            lps,                                                      # (2N,)
            vals,                                                     # (2N,)
            jnp.concatenate([rewards_p0, rewards_p1]),                # (2N,)
            jnp.concatenate([terminated, terminated]),                # (2N,)
            jnp.concatenate([truncated, truncated]),                  # (2N,)
            jnp.concatenate([winners, winners_p1]),                   # (2N,)
            owned_cities_p0,                                          # scalar
        )

        return (new_states, key, osp, timesteps.last_state, new_osp), data

    (final_states, final_key, final_osp,
     last_pre_reset, last_osp), rollout_data = jax.lax.scan(
        scan_body,
        (states, key, osp_init, states, osp_init),
        None, length=num_steps
    )

    obs, masks, temporal, actions, lps, vals, rews, terminated, truncated, winners, owned_cities = rollout_data

    # Bootstrap values — single 2N batch
    boot_obs_p0 = jax.vmap(lambda s: get_observation(s, 0))(last_pre_reset)
    boot_obs_p1 = jax.vmap(lambda s: get_observation(s, 1))(last_pre_reset)
    boot_obs = cat(boot_obs_p0, boot_obs_p1)
    boot_arr = jax.vmap(obs_to_array)(boot_obs)
    boot_masks = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(boot_obs)
    boot_aug, boot_osp = jax.vmap(augment_fn)(boot_arr, last_osp)
    boot_temporal = jnp.stack([boot_osp.opponent_army_history, boot_osp.opponent_land_history], axis=1)

    boot_keys = jrandom.split(final_key, 2 * n)
    _, final_val, _, _, _, _ = jax.vmap(network, in_axes=(0, 0, 0, 0, None))(
        boot_aug, boot_masks, boot_temporal, boot_keys, None)

    next_vals = jnp.concatenate([vals[1:], final_val[None]], axis=0)

    rollout_data = (obs, masks, temporal, actions, lps, vals, next_vals, rews, terminated, truncated, winners, owned_cities)
    # Split obs state back for caller
    final_osp0 = jax.tree.map(lambda x: x[:n], final_osp)
    final_osp1 = jax.tree.map(lambda x: x[n:], final_osp)
    return final_states, rollout_data, final_key, final_osp0, final_osp1


@partial(jax.jit, static_argnames=["env", "num_steps", "grid_size", "reward_fn", "augment_fn", "reset_fn", "greedy_fn"])
def collect_rollout_frozen_opponent(
        states, env, learner, opponent, key, num_steps, obs_state_p0,
        obs_state_p1, grid_size, reward_fn, augment_fn, reset_fn, greedy_fn,
        gamma, pool=None):
    """Collect PPO data for p0 against a frozen p1 opponent.

    Only learner (p0) transitions are returned. The frozen opponent acts
    greedily, matching the historical-opponent setup used in Straka (2025).
    """
    n = states.armies.shape[0]
    step_fn = (lambda s, a: env.step(s, a, pool)) if pool is not None else env.step

    def scan_body(carry, _):
        states, key, osp0, osp1, _, _ = carry
        obs0 = jax.vmap(lambda s: get_observation(s, 0))(states)
        obs1 = jax.vmap(lambda s: get_observation(s, 1))(states)

        arr0, arr1 = jax.vmap(obs_to_array)(obs0), jax.vmap(obs_to_array)(obs1)
        mask_fn = lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains)
        masks0, masks1 = jax.vmap(mask_fn)(obs0), jax.vmap(mask_fn)(obs1)
        aug0, new_osp0 = jax.vmap(augment_fn)(arr0, osp0)
        aug1, new_osp1 = jax.vmap(augment_fn)(arr1, osp1)
        aug0, aug1 = aug0.astype(jnp.bfloat16), aug1.astype(jnp.bfloat16)
        temporal0 = jnp.stack(
            [new_osp0.opponent_army_history, new_osp0.opponent_land_history], axis=1)
        temporal1 = jnp.stack(
            [new_osp1.opponent_army_history, new_osp1.opponent_land_history], axis=1)

        keys = jrandom.split(key, 2 * n + 1)
        key = keys[0]
        a0, vals, lps, _, _, _ = jax.vmap(
            learner, in_axes=(0, 0, 0, 0, None))(
                aug0, masks0, temporal0, keys[1:n + 1], None)
        a1 = jax.vmap(greedy_fn, in_axes=(None, 0, 0, 0))(
            opponent, aug1, masks1, temporal1)

        timesteps, new_states = jax.vmap(step_fn)(
            states, jnp.stack([a0, a1], axis=1))
        terminated, truncated = timesteps.terminated, timesteps.truncated
        dones, winners = terminated | truncated, timesteps.info.winner
        next_obs0 = jax.vmap(lambda s: get_observation(s, 0))(timesteps.last_state)
        rewards = reward_fn(
            obs0, a0, next_obs0, winners, truncated=truncated, gamma=gamma)
        osp0 = reset_done_envs(new_osp0, dones)
        osp1 = reset_done_envs(new_osp1, dones)
        owned_cities = (obs0.cities * obs0.owned_cells).sum()
        data = (aug0, masks0, temporal0, a0, lps, vals, rewards,
                terminated, truncated, winners, owned_cities)
        return (new_states, key, osp0, osp1, timesteps.last_state, new_osp0), data

    (final_states, final_key, final_osp0, final_osp1,
     last_pre_reset, last_osp0), data = jax.lax.scan(
        scan_body,
        (states, key, obs_state_p0, obs_state_p1, states, obs_state_p0),
        None, length=num_steps)

    (obs, masks, temporal, actions, lps, vals, rews, terminated,
     truncated, winners, owned_cities) = data
    boot_obs = jax.vmap(lambda s: get_observation(s, 0))(last_pre_reset)
    boot_arr = jax.vmap(obs_to_array)(boot_obs)
    boot_masks = jax.vmap(
        lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(boot_obs)
    boot_aug, boot_osp = jax.vmap(augment_fn)(boot_arr, last_osp0)
    boot_temporal = jnp.stack(
        [boot_osp.opponent_army_history, boot_osp.opponent_land_history], axis=1)
    boot_keys = jrandom.split(final_key, n)
    _, final_val, _, _, _, _ = jax.vmap(
        learner, in_axes=(0, 0, 0, 0, None))(
            boot_aug, boot_masks, boot_temporal, boot_keys, None)
    next_vals = jnp.concatenate([vals[1:], final_val[None]], axis=0)
    rollout_data = (obs, masks, temporal, actions, lps, vals, next_vals,
                    rews, terminated, truncated, winners, owned_cities)
    return (final_states, rollout_data, final_key, final_osp0, final_osp1)
