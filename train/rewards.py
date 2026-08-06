"""Reward functions for PPO training.

Signature matches the rollout call site (extra args unused, kept for that interface):
    win_lose_reward(prior_obs, action, next_obs, winners, truncated=False, gamma=1.0) -> (N,)
"""

from functools import partial

import jax.numpy as jnp


def win_lose_reward(prior_obs, action, next_obs, winners, truncated=False, gamma=1.0):
    """+1 if we won (winner == 0), -1 if we lost (winner == 1), else 0."""
    return jnp.where(winners == 0, 1.0, jnp.where(winners == 1, -1.0, 0.0))


def _normalized_log_ratio(mine, opponent, max_ratio):
    """Map mine/opponent to [-1, 1] on a symmetric logarithmic scale.

    Counts are floored at one so zero-castle states remain finite.
    """
    mine = jnp.maximum(mine.astype(jnp.float32), 1.0)
    opponent = jnp.maximum(opponent.astype(jnp.float32), 1.0)
    ratio = mine / opponent
    return jnp.clip(jnp.log(ratio) / jnp.log(max_ratio), -1.0, 1.0)


def relative_progress_reward(
    prior_obs,
    action,
    next_obs,
    winners,
    truncated=False,
    gamma=1.0,
    *,
    army_weight=0.30,
    land_weight=0.30,
    castle_weight=0.40,
    max_army_ratio=2.0,
    max_land_ratio=2.0,
    max_castle_ratio=2.0,
):
    """Reward improvements in relative army, land, and castle ownership.

    Each metric's potential is log(mine / opponent), normalized
    by log(max_ratio) and clipped to [-1, 1]. The shaping reward is the change
    in that potential from ``prior_obs`` to ``next_obs``. On a terminal step,
    the original +/-1 win/loss reward is returned instead of shaping.
    """
    del action, gamma  # Kept for the common rollout reward interface.

    prior_my_castles = jnp.sum(prior_obs.cities & prior_obs.owned_cells, axis=(-2, -1))
    prior_opp_castles = jnp.sum(prior_obs.cities & prior_obs.opponent_cells, axis=(-2, -1))
    next_my_castles = jnp.sum(next_obs.cities & next_obs.owned_cells, axis=(-2, -1))
    next_opp_castles = jnp.sum(next_obs.cities & next_obs.opponent_cells, axis=(-2, -1))

    def improvement(prior_mine, prior_opponent, next_mine, next_opponent, max_ratio):
        prior_value = _normalized_log_ratio(prior_mine, prior_opponent, max_ratio)
        next_value = _normalized_log_ratio(next_mine, next_opponent, max_ratio)
        return next_value - prior_value

    army_reward = improvement(
        prior_obs.owned_army_count,
        prior_obs.opponent_army_count,
        next_obs.owned_army_count,
        next_obs.opponent_army_count,
        max_army_ratio,
    )
    land_reward = improvement(
        prior_obs.owned_land_count,
        prior_obs.opponent_land_count,
        next_obs.owned_land_count,
        next_obs.opponent_land_count,
        max_land_ratio,
    )
    castle_reward = improvement(
        prior_my_castles,
        prior_opp_castles,
        next_my_castles,
        next_opp_castles,
        max_castle_ratio,
    )

    shaping = (
        army_weight * army_reward
        + land_weight * land_reward
        + castle_weight * castle_reward
    )
    terminal = win_lose_reward(prior_obs, None, next_obs, winners, truncated=truncated)
    return jnp.where(winners >= 0, terminal, shaping)


def get_reward_fn(cfg):
    """Build the configured reward callable used by PPO rollouts."""
    if cfg.reward_fn == "win_lose_reward":
        return win_lose_reward
    if cfg.reward_fn == "relative_progress_reward":
        return partial(
            relative_progress_reward,
            max_army_ratio=cfg.max_army_ratio,
            max_land_ratio=cfg.max_land_ratio,
            max_castle_ratio=cfg.max_castle_ratio,
        )
    raise ValueError(
        f"Unknown reward_fn '{cfg.reward_fn}'. Available: "
        "win_lose_reward, relative_progress_reward"
    )
