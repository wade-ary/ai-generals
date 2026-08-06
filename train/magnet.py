"""Magnet policy: lightweight heuristic action priors for KL anchoring."""

import jax
import jax.numpy as jnp

# Augmented obs channel indices
_OWN_ARMY = 1
_ENEMY_ARMY = 2
_CITIES = 7
_NEUTRAL = 9
_OWNED = 10
_OPPONENT = 11


def expander_magnet(
    obs, mask,
    score_pass=0.2,
    score_default=1.0,
    score_neutral=2.0,
    score_enemy=3.0,
    score_city=5.0,
    score_topk=2.0,
):
    """Heuristic distribution favoring expansion and capture.

    Args:
        obs: (C, H, W) augmented observation array
        mask: (H, W, 4) direction validity mask

    Returns:
        (9*H*W,) probability distribution
    """
    H, W = mask.shape[:2]

    own_army = obs[_OWN_ARMY]
    enemy_army = obs[_ENEMY_ARMY]
    cities = obs[_CITIES]
    neutral = obs[_NEUTRAL]
    owned = obs[_OWNED]
    opponent = obs[_OPPONENT]

    # Destination values for each direction: roll(+1) looks at (r-1,c) = UP dest, etc.
    # UP: dest=(r-1,c), DOWN: dest=(r+1,c), LEFT: dest=(r,c-1), RIGHT: dest=(r,c+1)
    def _dest(arr):
        return jnp.stack([
            jnp.roll(arr, 1, axis=0),   # UP
            jnp.roll(arr, -1, axis=0),  # DOWN
            jnp.roll(arr, 1, axis=1),   # LEFT
            jnp.roll(arr, -1, axis=1),  # RIGHT
        ])  # (4, H, W)

    dest_neutral = _dest(neutral)
    dest_opponent = _dest(opponent)
    dest_cities = _dest(cities)
    dest_enemy_army = _dest(enemy_army)
    dest_owned = _dest(owned)
    dest_army = _dest(obs[0])  # total army at destination (for neutral cities)

    can_capture_enemy = (own_army[None] > dest_enemy_army + 1) & (dest_opponent > 0)
    can_capture_city = (dest_cities > 0) & (dest_owned == 0) & (own_army[None] > dest_army + 1)

    scores = jnp.full((4, H, W), score_default)
    scores = jnp.where(dest_neutral > 0, score_neutral, scores)
    scores = jnp.where(can_capture_enemy, score_enemy, scores)
    scores = jnp.where(can_capture_city, score_city, scores)

    # Boost top-5 owned cells by army count
    top5_thresh = jax.lax.sort(own_army.reshape(-1), is_stable=False)[-5]
    top5_mask = (own_army >= top5_thresh) & (own_army > 0)  # (H, W)
    scores = scores + top5_mask[None] * score_topk  # +score_topk to all 4 directions from top cells

    pass_score = jnp.full((1, H, W), score_pass)
    logits = jnp.concatenate([scores, scores, pass_score], axis=0)
    return jax.nn.softmax(logits.reshape(-1))
