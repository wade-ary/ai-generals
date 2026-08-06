"""Shared utilities for all network architectures.

Includes: observation state, augmentation, action encoding/decoding,
mask preparation, normalization, and environment helpers.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array

from generals.core.action import compute_valid_move_mask


# ---- Observation state (for rich augmented networks) ----


class AugmentedObsState(NamedTuple):
    army_stack: Array                     # (history_size, pad_to, pad_to)
    enemy_stack: Array                    # (history_size, pad_to, pad_to)
    last_army: Array                      # (pad_to, pad_to)
    last_enemy_army: Array                # (pad_to, pad_to)
    cities: Array                         # (pad_to, pad_to) bool
    generals: Array                       # (pad_to, pad_to) bool
    mountains: Array                      # (pad_to, pad_to) bool
    seen: Array                           # (pad_to, pad_to) bool
    enemy_seen: Array                     # (pad_to, pad_to) bool
    last_enemy_army_seen_value: Array     # (pad_to, pad_to)
    last_enemy_army_seen_timestep: Array  # (pad_to, pad_to)
    opponent_army_history: Array          # (temporal_window,) — time series
    opponent_land_history: Array          # (temporal_window,) — time series
    temporal_step: Array                  # () int — current write index


def init_obs_state(grid_size, pad_to=None, history_size=7, temporal_window=512):
    """Create zero-initialized obs state for a single environment."""
    p = pad_to if pad_to is not None else grid_size
    return AugmentedObsState(
        army_stack=jnp.zeros((history_size, p, p)),
        enemy_stack=jnp.zeros((history_size, p, p)),
        last_army=jnp.zeros((p, p)),
        last_enemy_army=jnp.zeros((p, p)),
        cities=jnp.zeros((p, p), dtype=jnp.bool_),
        generals=jnp.zeros((p, p), dtype=jnp.bool_),
        mountains=jnp.zeros((p, p), dtype=jnp.bool_),
        seen=jnp.zeros((p, p), dtype=jnp.bool_),
        enemy_seen=jnp.zeros((p, p), dtype=jnp.bool_),
        last_enemy_army_seen_value=jnp.zeros((p, p)),
        last_enemy_army_seen_timestep=jnp.zeros((p, p)),
        opponent_army_history=jnp.zeros((temporal_window,)),
        opponent_land_history=jnp.zeros((temporal_window,)),
        temporal_step=jnp.int32(0),
    )


def reset_obs_state(obs_state):
    """Reset all state to zeros."""
    return jax.tree.map(jnp.zeros_like, obs_state)


def _max_pool_2d(x, window_size=3):
    """Max pool a single (H, W) array with SAME padding."""
    x = x[None, :, :]  # (1, H, W)
    pooled = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max,
        (1, window_size, window_size), (1, 1, 1), 'SAME'
    )
    return pooled[0, :, :]  # (H, W)


def augment_obs(obs_arr, obs_state):
    """Augment raw observation with history and memory.

    Args:
        obs_arr: (14, H, W) from obs_to_array — single sample, no batch dim
        obs_state: AugmentedObsState — single sample

    Returns:
        augmented_obs: (24 + 2*history_size, pad_to, pad_to) float array
        new_obs_state: AugmentedObsState
    """
    # Channel indices in obs_arr (from obs_to_array)
    armies, generals_ch, cities_ch, mountains_ch = 0, 1, 2, 3
    neutral_cells, owned_cells, opponent_cells = 4, 5, 6
    fog_cells, structures_in_fog = 7, 8
    owned_land_count, owned_army_count = 9, 10
    opponent_land_count, opponent_army_count = 11, 12
    timestep_ch = 13

    # Infer pad_to from obs_state spatial dimensions
    p = obs_state.last_army.shape[0]

    # Pad obs from grid_size to pad_to x pad_to (border = mountains)
    h, w = obs_arr.shape[1], obs_arr.shape[2]
    pad_h, pad_w = p - h, p - w
    obs = jnp.pad(obs_arr, ((0, 0), (0, pad_h), (0, pad_w)))
    pad_mask = (jnp.arange(p)[:, None] >= h) | (jnp.arange(p)[None, :] >= w)
    # Accumulate visible padding cells as confirmed mountains in obs_state
    visible = _max_pool_2d(obs[owned_cells]) > 0
    seen_pad_mountains = obs_state.mountains | (pad_mask & visible)
    # Visible (or previously seen) padding → mountains; rest → structures_in_fog
    obs = obs.at[mountains_ch].set(jnp.where(pad_mask & seen_pad_mountains, 1.0, obs[mountains_ch]))
    obs = obs.at[structures_in_fog].set(jnp.where(pad_mask & ~seen_pad_mountains, 1.0, obs[structures_in_fog]))
    # Broadcast scalar channels into padding (training has them everywhere)
    for ch in (owned_land_count, owned_army_count, opponent_land_count,
               opponent_army_count, timestep_ch):
        obs = obs.at[ch].set(jnp.where(pad_mask, obs[ch, 0, 0], obs[ch]))

    # Calculate current army states
    current_army = obs[armies] * obs[owned_cells]
    current_enemy_army = obs[armies] * obs[opponent_cells]

    # Update history stacks
    new_army_stack = jnp.concatenate([
        (current_army - obs_state.last_army)[None, :, :],
        obs_state.army_stack[:-1, :, :]
    ], axis=0)
    new_enemy_stack = jnp.concatenate([
        (current_enemy_army - obs_state.last_enemy_army)[None, :, :],
        obs_state.enemy_stack[:-1, :, :]
    ], axis=0)

    # Max pooling for visibility
    new_seen = obs_state.seen | (_max_pool_2d(obs[owned_cells]) > 0)
    new_enemy_seen = obs_state.enemy_seen | (_max_pool_2d(obs[opponent_cells]) > 0)

    # Accumulate static structures
    new_cities = obs_state.cities | (obs[cities_ch] > 0)
    new_generals = obs_state.generals | (obs[generals_ch] > 0)
    new_mountains = obs_state.mountains | (obs[mountains_ch] > 0) | seen_pad_mountains

    # Update last seen enemy army
    new_last_enemy_army_seen_value = jnp.where(
        current_enemy_army > 0, current_enemy_army,
        obs_state.last_enemy_army_seen_value
    )
    # Store raw step count; log decay applied only in observation channel
    new_last_enemy_army_seen_timestep = jnp.where(
        current_enemy_army > 0, 0.0, obs_state.last_enemy_army_seen_timestep + 1.0
    )

    # Update temporal opponent stat histories (sliding window: roll left, write newest at end)
    opp_army_val = obs[opponent_army_count, 0, 0]  # scalar (broadcast channel)
    opp_land_val = obs[opponent_land_count, 0, 0]
    new_opponent_army_history = jnp.roll(obs_state.opponent_army_history, -1).at[-1].set(opp_army_val)
    new_opponent_land_history = jnp.roll(obs_state.opponent_land_history, -1).at[-1].set(opp_land_val)
    new_temporal_step = obs_state.temporal_step + 1

    # Coordinate channels (normalized to [0, 1])
    coords_x = jnp.broadcast_to(jnp.arange(p, dtype=jnp.float32)[None, :] / (p - 1), (p, p))
    coords_y = jnp.broadcast_to(jnp.arange(p, dtype=jnp.float32)[:, None] / (p - 1), (p, p))

    # Build augmented observation (24 base channels)
    ones = jnp.ones((p, p))
    channels = jnp.stack([
        obs[armies],                                  # 0
        current_army,                                 # 1
        current_enemy_army,                           # 2
        obs[armies] * obs[neutral_cells],             # 3
        new_seen.astype(jnp.float32),                 # 4
        new_enemy_seen.astype(jnp.float32),           # 5
        new_generals.astype(jnp.float32),             # 6
        new_cities.astype(jnp.float32),               # 7
        new_mountains.astype(jnp.float32),            # 8
        obs[neutral_cells],                           # 9
        obs[owned_cells],                             # 10
        obs[opponent_cells],                          # 11
        obs[fog_cells],                               # 12
        obs[structures_in_fog],                       # 13
        obs[timestep_ch] * ones,                      # 14
        (obs[timestep_ch] % 50) * ones / 50,          # 15
        obs[owned_land_count] * ones,                 # 16
        obs[owned_army_count] * ones,                 # 17
        obs[opponent_land_count] * ones,              # 18
        obs[opponent_army_count] * ones,              # 19
        new_last_enemy_army_seen_value,               # 20
        jnp.log1p(new_last_enemy_army_seen_timestep) / 5.0,  # 21: log decay
        coords_x,                                     # 22
        coords_y,                                     # 23
    ], axis=0)

    # Concatenate history stacks
    augmented_obs = jnp.concatenate([channels, new_army_stack, new_enemy_stack], axis=0)

    new_state = AugmentedObsState(
        army_stack=new_army_stack,
        enemy_stack=new_enemy_stack,
        last_army=current_army,
        last_enemy_army=current_enemy_army,
        cities=new_cities,
        generals=new_generals,
        mountains=new_mountains,
        seen=new_seen,
        enemy_seen=new_enemy_seen,
        last_enemy_army_seen_value=new_last_enemy_army_seen_value,
        last_enemy_army_seen_timestep=new_last_enemy_army_seen_timestep,
        opponent_army_history=new_opponent_army_history,
        opponent_land_history=new_opponent_land_history,
        temporal_step=new_temporal_step,
    )
    return augmented_obs, new_state


# ---- Observation conversion ----


def obs_to_array(obs):
    """Convert Observation to (14, H, W) float array.

    Channels 0-8: spatial grids, 9-13: scalar stats broadcast to (H, W).
    """
    shape = obs.armies.shape
    return jnp.stack(
        [
            obs.armies,
            obs.generals,
            obs.cities,
            obs.mountains,
            obs.neutral_cells,
            obs.owned_cells,
            obs.opponent_cells,
            obs.fog_cells,
            obs.structures_in_fog,
            jnp.broadcast_to(obs.owned_land_count, shape),
            jnp.broadcast_to(obs.owned_army_count, shape),
            jnp.broadcast_to(obs.opponent_land_count, shape),
            jnp.broadcast_to(obs.opponent_army_count, shape),
            jnp.broadcast_to(obs.timestep, shape),
        ],
        axis=0,
    ).astype(jnp.float32)


# ---- Action encoding/decoding ----


def decode_action(idx, pad_to):
    """Convert flat logit index to action array.

    Args:
        idx: scalar index into (9 * pad_to * pad_to,) logit vector
        pad_to: grid dimension

    Returns:
        (5,) int32 array: [is_pass, row, col, direction, is_half]
    """
    gc = pad_to * pad_to
    d, pos = idx // gc, idx % gc
    r, c = pos // pad_to, pos % pad_to
    is_pass = d == 8
    is_half = (d >= 4) & (d < 8)
    ad = jnp.where(is_pass, 0, jnp.where(is_half, d - 4, d))
    return jnp.array([is_pass, r, c, ad, is_half], dtype=jnp.int32)


def encode_action(action, pad_to):
    """Convert action array back to flat logit index.

    Args:
        action: (5,) array [is_pass, row, col, direction, is_half]
        pad_to: grid dimension

    Returns:
        scalar int32 index into (9 * pad_to * pad_to,) logit vector
    """
    gc = pad_to * pad_to
    ip = action[0].astype(jnp.int32)
    r = action[1].astype(jnp.int32)
    c = action[2].astype(jnp.int32)
    d = action[3].astype(jnp.int32)
    ih = action[4].astype(jnp.int32)
    ed = jnp.where(ip > 0, 8, jnp.where(ih > 0, d + 4, d))
    return ed * gc + r * pad_to + c


# ---- Mask and normalization ----


def prepare_action_mask(direction_mask, pad_to, allow_pass=True):
    """Pad direction mask and build 9-direction action penalty.

    Args:
        direction_mask: (H, W, 4) validity mask (1=valid, 0=invalid)
        pad_to: target spatial dimension
        allow_pass: if False, mask out the pass action

    Returns:
        (9, pad_to, pad_to) penalty array (-1e9 for invalid, 0 for valid)
    """
    mask_t = 1 - jnp.transpose(direction_mask, (2, 0, 1))
    pad_h = pad_to - mask_t.shape[1]
    pad_w = pad_to - mask_t.shape[2]
    mask = jnp.pad(mask_t, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=1)
    full_mask = mask[:4]
    half_mask = mask[:4]
    pass_val = 0.0 if allow_pass else 1.0
    pass_mask = jnp.full((1, pad_to, pad_to), pass_val)
    return jnp.concatenate([full_mask, half_mask, pass_mask], axis=0) * -1e9


def normalize_observations(obs):
    """Normalize augmented observation channels (24 + history).

    For networks using AugmentedObsState (unet, simple_augmented_cnn, transformer).
    All divisors standardized to 50.

    Args:
        obs: (C, pad_to, pad_to) augmented observation

    Returns:
        normalized observation, same shape
    """
    army_channels = [0, 1, 2, 3, 17, 19, 20] + list(range(24, obs.shape[0]))
    obs = obs.at[jnp.array(army_channels)].divide(50.0)
    obs = obs.at[14].divide(50.0)
    obs = obs.at[jnp.array([16, 18])].divide(50.0)
    return obs


# ---- Environment helpers ----


def reset_done_envs(obs_state, dones):
    """Reset obs_state for envs where done=True. Works on batched (N, ...) state."""
    def reset_leaf(leaf):
        expand = dones.reshape(dones.shape[0], *([1] * (leaf.ndim - 1)))
        return jnp.where(expand, jnp.zeros_like(leaf), leaf)
    return jax.tree.map(reset_leaf, obs_state)


def random_action(key, obs):
    """Sample a random valid action for the opponent."""
    mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    valid = jnp.argwhere(mask, size=100, fill_value=-1)
    nv = jnp.sum(jnp.all(valid >= 0, axis=-1))
    k1, k2 = jrandom.split(key)
    should_pass = nv == 0
    idx = jnp.minimum(jrandom.randint(k1, (), 0, jnp.maximum(nv, 1)), nv - 1)
    m = valid[idx]
    ih = jrandom.randint(k2, (), 0, 2)
    return jnp.array([should_pass, m[0], m[1], m[2], ih], dtype=jnp.int32)
