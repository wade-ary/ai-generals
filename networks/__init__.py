"""Network architectures for generals.io PPO agent."""

import inspect

from networks import transformer, common
from networks.transformer import HistoryTransformer, greedy_action_transformer
from networks.common import (
    obs_to_array,
    random_action,
    reset_done_envs,
)

NETWORK_REGISTRY = {
    "history_transformer": {
        "cls": HistoryTransformer,
        "init_obs_state": common.init_obs_state,
        "augment_obs": common.augment_obs,
        "reset_obs_state": common.reset_obs_state,
        "greedy_action": transformer.greedy_action_transformer,
    },
}


def get_network_bundle(name: str) -> dict:
    """Look up a network bundle (cls + obs state functions) by name."""
    if name not in NETWORK_REGISTRY:
        available = ", ".join(NETWORK_REGISTRY.keys())
        raise ValueError(f"Unknown network '{name}'. Available: {available}")
    return NETWORK_REGISTRY[name]


# Config fields forwarded to a network constructor when present in its signature.
_NET_CFG_FIELDS = [
    "depth", "embed_dim", "n_head", "ff_factor", "patch_size", "conv_dim",
    "use_bf16", "value_loss", "num_bins", "v_min", "v_max",
]


def build_network(cfg, key):
    """Instantiate the configured network, forwarding only the config fields
    that match its constructor signature."""
    cls = get_network_bundle(cfg.network)["cls"]
    params = inspect.signature(cls).parameters
    kwargs = {"grid_size": cfg.pad_to, "pad_to": cfg.pad_to, "key": key}
    kwargs.update({f: getattr(cfg, f) for f in _NET_CFG_FIELDS if f in params})
    return cls(**kwargs)
