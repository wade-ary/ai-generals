import math

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array

from networks.common import decode_action, encode_action, prepare_action_mask, normalize_observations


# ---- Helpers ----

def _to_bf16(tree):
    """Cast all float arrays in a pytree to bfloat16 for mixed-precision compute."""
    return jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating) else x,
        tree,
    )


# ---- Transformer building blocks ----

class MultiHeadSelfAttention(eqx.Module):
    """Multi-head self-attention with Q, K, V projections."""
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    n_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_head: int, *, key):
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.q_proj = eqx.nn.Linear(d_model, d_model, key=k1)
        self.k_proj = eqx.nn.Linear(d_model, d_model, key=k2)
        self.v_proj = eqx.nn.Linear(d_model, d_model, key=k3)
        self.out_proj = eqx.nn.Linear(d_model, d_model, key=k4)

    def __call__(self, x):
        """x: (seq_len, d_model) -> (seq_len, d_model)"""
        seq_len = x.shape[0]
        q = jax.vmap(self.q_proj)(x).reshape(seq_len, self.n_head, self.head_dim)
        k = jax.vmap(self.k_proj)(x).reshape(seq_len, self.n_head, self.head_dim)
        v = jax.vmap(self.v_proj)(x).reshape(seq_len, self.n_head, self.head_dim)

        q = jnp.transpose(q, (1, 0, 2))
        k = jnp.transpose(k, (1, 0, 2))
        v = jnp.transpose(v, (1, 0, 2))

        scale = math.sqrt(self.head_dim)
        attn = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) / scale
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(q.dtype)

        out = jnp.matmul(attn, v)
        out = jnp.transpose(out, (1, 0, 2)).reshape(seq_len, -1)
        return jax.vmap(self.out_proj)(out)


class SelfAttentionLayer(eqx.Module):
    """Pre-norm transformer block: LN -> MHSA -> residual -> LN -> FFN -> residual."""
    norm1: eqx.nn.LayerNorm
    attn: MultiHeadSelfAttention
    norm2: eqx.nn.LayerNorm
    ff_linear1: eqx.nn.Linear
    ff_linear2: eqx.nn.Linear

    def __init__(self, d_model: int, n_head: int, ff_factor: int = 4, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, key=k1)
        self.norm2 = eqx.nn.LayerNorm(d_model)
        self.ff_linear1 = eqx.nn.Linear(d_model, ff_factor * d_model, key=k2)
        self.ff_linear2 = eqx.nn.Linear(ff_factor * d_model, d_model, key=k3)

    def __call__(self, x):
        """x: (seq_len, d_model) -> (seq_len, d_model)"""
        x = x + self.attn(jax.vmap(self.norm1)(x))
        h = jax.vmap(self.norm2)(x)
        h = jax.nn.silu(jax.vmap(self.ff_linear1)(h))
        h = jax.vmap(self.ff_linear2)(h)
        x = x + h
        return x


# ---- Temporal encoder ----

class TemporalEncoder(eqx.Module):
    """Encode opponent stat time-series into 2 summary tokens via separate MLPs.

    Input: (2, temporal_window) — opponent army and land count histories.
    Output: (2, embed_dim) — one summary token per channel.

    Each channel (army, land) has its own independent MLP.
    """
    army_l1: eqx.nn.Linear    # temporal_window -> hidden
    army_l2: eqx.nn.Linear    # hidden -> embed_dim
    land_l1: eqx.nn.Linear    # temporal_window -> hidden
    land_l2: eqx.nn.Linear    # hidden -> embed_dim

    def __init__(self, embed_dim: int, temporal_window: int = 512, hidden: int = 1024, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.army_l1 = eqx.nn.Linear(temporal_window, hidden, key=k1)
        self.army_l2 = eqx.nn.Linear(hidden, embed_dim, key=k2)
        self.land_l1 = eqx.nn.Linear(temporal_window, hidden, key=k3)
        self.land_l2 = eqx.nn.Linear(hidden, embed_dim, key=k4)

    def __call__(self, temporal_data):
        """Encode temporal opponent stats into 2 summary tokens.

        Args:
            temporal_data: (2, temporal_window) — [army_history, land_history]

        Returns:
            (2, embed_dim) — army summary token and land summary token
        """
        army_hist = temporal_data[0] / 50.0   # (temporal_window,)
        land_hist = temporal_data[1] / 50.0   # (temporal_window,)
        army_token = self.army_l2(jax.nn.silu(self.army_l1(army_hist)))  # (embed_dim,)
        land_token = self.land_l2(jax.nn.silu(self.land_l1(land_hist)))  # (embed_dim,)
        return jnp.stack([army_token, land_token])  # (2, embed_dim)


# ---- History transformer (with temporal encoder) ----

_TEMPORAL_WINDOW = 512
_TEMPORAL_HIDDEN = 512

class HistoryTransformer(eqx.Module):
    """Transformer policy-value network with temporal opponent stat encoding.

    Encodes opponent army/land count
    time series into 2 summary tokens via TemporalEncoder MLPs. These tokens
    are prepended to the sequence alongside the value token.

    Single-sample, vmappable.
    """
    embedder: eqx.nn.Linear
    value_token: Array
    pos_encoding: Array
    transformer_layers: list
    norm_out: eqx.nn.LayerNorm
    policy_head: eqx.nn.Linear
    value_head: eqx.nn.Linear
    temporal_encoder: TemporalEncoder
    temporal_type_embed: Array
    bin_centers: Array
    grid_size: int = eqx.field(static=True)
    pad_to: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    use_bf16: bool = eqx.field(static=True)
    num_bins: int = eqx.field(static=True)
    temporal_window: int = eqx.field(static=True)

    def __init__(
        self,
        grid_size: int = 24,
        pad_to: int = None,
        history_size: int = 7,
        patch_size: int = 1,
        depth: int = 6,
        embed_dim: int = 256,
        n_head: int = 8,
        ff_factor: int = 4,
        use_bf16: bool = False,
        value_loss: str = "mse",
        num_bins: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        *,
        key,
    ):
        temporal_window = _TEMPORAL_WINDOW
        temporal_hidden = _TEMPORAL_HIDDEN

        self.grid_size = grid_size
        self.pad_to = pad_to if pad_to is not None else grid_size
        self.patch_size = patch_size
        self.n_channels = 24 + 2 * history_size
        self.use_bf16 = use_bf16
        self.temporal_window = temporal_window

        assert self.pad_to % patch_size == 0, \
            f"pad_to ({self.pad_to}) must be divisible by patch_size ({patch_size})"

        keys = jax.random.split(key, depth + 7)

        patch_dim = self.n_channels * patch_size * patch_size
        self.embedder = eqx.nn.Linear(patch_dim, embed_dim, key=keys[0])

        n_patches = (self.pad_to // patch_size) ** 2
        n_temporal_tokens = 2  # army + land
        n_tokens = n_patches + 1 + n_temporal_tokens  # patches + value + temporal
        self.value_token = jax.random.normal(keys[1], (1, embed_dim)) * 0.02
        self.pos_encoding = jax.random.truncated_normal(
            keys[2], -2.0, 2.0, (n_tokens, embed_dim)
        ) * 0.1

        self.transformer_layers = [
            SelfAttentionLayer(embed_dim, n_head, ff_factor, key=keys[3 + i])
            for i in range(depth)
        ]

        self.norm_out = eqx.nn.LayerNorm(embed_dim)
        self.policy_head = eqx.nn.Linear(embed_dim, 9 * patch_size * patch_size, key=keys[3 + depth])

        # Value head: scalar (MSE) or categorical bins (CE)
        self.num_bins = num_bins if value_loss == "ce" else 0
        if self.num_bins > 0:
            self.value_head = eqx.nn.Linear(embed_dim, num_bins, key=keys[3 + depth + 1])
            self.bin_centers = jnp.linspace(v_min, v_max, num_bins)
        else:
            self.value_head = eqx.nn.Linear(embed_dim, 1, key=keys[3 + depth + 1])
            self.bin_centers = jnp.zeros(0)

        # Temporal encoder + type embedding
        self.temporal_encoder = TemporalEncoder(embed_dim, temporal_window=temporal_window, hidden=temporal_hidden, key=keys[3 + depth + 2])
        self.temporal_type_embed = jax.random.normal(keys[3 + depth + 3], (2, embed_dim)) * 0.02


    def _forward(self, obs, mask, temporal_data, allow_pass=True):
        """Shared forward trunk -> (flat masked action logits, value, value_aux).

        value_aux is the (num_bins,) logits for CE loss, or the scalar value for MSE.
        """
        p = self.pad_to
        M = self.patch_size
        gp = p // M  # grid of patches
        obs_norm = normalize_observations(obs)
        mask_prep = prepare_action_mask(mask, p, allow_pass=allow_pass)  # float32 (contains -1e9)

        # Mixed precision: cast params and activations to bfloat16
        net = _to_bf16(self) if self.use_bf16 else self
        if self.use_bf16:
            obs_norm = obs_norm.astype(jnp.bfloat16)
            temporal_data = temporal_data.astype(jnp.bfloat16)

        # Patchify: (C, p, p) -> (n_patches, C*M*M)
        x = obs_norm.reshape(self.n_channels, gp, M, gp, M)
        x = x.transpose(1, 3, 0, 2, 4).reshape(gp * gp, -1)

        # Embed each patch token
        x = jax.vmap(net.embedder)(x)  # (n_patches, embed_dim)

        # Temporal tokens from opponent stat history (+ type embedding)
        temporal_tokens = net.temporal_encoder(temporal_data) + net.temporal_type_embed

        # Token sequence: [VALUE, TEMPORAL_ARMY, TEMPORAL_LAND, PATCH_0, ..., PATCH_N]
        x = jnp.concatenate([net.value_token, temporal_tokens, x], axis=0)
        x = x + net.pos_encoding

        for layer in net.transformer_layers:
            x = layer(x)
        x = jax.vmap(net.norm_out)(x)

        value_embedding = x[0]
        patch_embeddings = x[3:]  # skip value + 2 temporal tokens

        # Value head
        value_raw = net.value_head(value_embedding)
        if self.use_bf16:
            value_raw = value_raw.astype(jnp.float32)
        if self.num_bins > 0:
            value = jnp.sum(jax.nn.softmax(value_raw) * self.bin_centers)
            value_aux = value_raw  # (num_bins,) logits for CE loss
        else:
            value = value_raw[0]
            value_aux = value  # scalar for MSE loss

        # Policy head: per-patch logits, then unpatchify to (9, p, p)
        patch_logits = jax.vmap(net.policy_head)(patch_embeddings)  # (n_patches, 9*M*M)
        if self.use_bf16:
            patch_logits = patch_logits.astype(jnp.float32)
        action_logits = patch_logits.reshape(gp, gp, 9, M, M)
        action_logits = action_logits.transpose(2, 0, 3, 1, 4).reshape(9, p, p)
        action_logits = action_logits + mask_prep
        return action_logits.reshape(-1), value, value_aux

    def __call__(self, obs, mask, temporal_data, key, action=None, allow_pass=True):
        """Forward pass on a single sample (vmappable).

        Returns (action, value, logprob, entropy, value_aux, p_dist).
        """
        logits, value, value_aux = self._forward(obs, mask, temporal_data, allow_pass=allow_pass)

        if action is None:
            idx = jax.random.categorical(key, logits)
            action = decode_action(idx, self.pad_to)
        else:
            idx = encode_action(action, self.pad_to)

        lp = jax.nn.log_softmax(logits)
        logprob = lp[idx.astype(jnp.int32)]
        p_dist = jax.nn.softmax(logits)
        entropy = -jnp.sum(p_dist * lp)

        return action, value, logprob, entropy, value_aux, p_dist


# ---- Shared greedy/sample functions ----

def greedy_action_transformer(network, obs, mask, temporal_data):
    """Select action greedily (argmax) for the transformer policy-value network."""
    logits, _, _ = network._forward(obs, mask, temporal_data)
    return decode_action(jnp.argmax(logits), network.pad_to)
