"""Agent loading from checkpoint + config."""

import os

import jax
import jax.random as jrandom
import equinox as eqx

from config import Config
from networks import get_network_bundle, build_network


def _safe_load_config(path):
    """Load config YAML, ignoring unknown keys."""
    from dataclasses import fields as dc_fields
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    with open(path) as f:
        data = yaml.load(f)
    valid = {f.name for f in dc_fields(Config)}
    return Config(**{k: v for k, v in data.items() if k in valid})


def _try_load(network, model_path):
    try:
        import optax
        optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(1e-4))
        opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
        network, _ = eqx.tree_deserialise_leaves(model_path, (network, opt_state))
    except Exception:
        network = eqx.tree_deserialise_leaves(model_path, network)
    return network


class Agent:
    """A loaded model ready to play games."""

    def __init__(self, network, config, bundle, name=None):
        self.network = network
        self.config = config
        self.greedy_fn = bundle["greedy_action"]
        self.augment_fn = bundle["augment_obs"]
        self.init_obs_state_fn = bundle["init_obs_state"]
        self.bundle = bundle
        self.name = name

    @staticmethod
    def load(eqx_path, yaml_path):
        cfg = _safe_load_config(yaml_path)
        return Agent.from_config(eqx_path, cfg)

    @staticmethod
    def from_config(eqx_path, cfg):
        bundle = get_network_bundle(cfg.network)
        key = jrandom.PRNGKey(0)

        network = build_network(cfg, key)
        try:
            network = _try_load(network, eqx_path)
        except Exception:
            # Older checkpoints used a scalar (MSE) value head
            from dataclasses import replace
            network = build_network(replace(cfg, value_loss="mse"), key)
            network = _try_load(network, eqx_path)

        display_name = os.path.basename(eqx_path).replace(".eqx", "")
        return Agent(network, cfg, bundle, name=display_name)

    @property
    def pad_to(self):
        return self.config.pad_to

    def param_count(self):
        params, _ = eqx.partition(self.network, eqx.is_array)
        return sum(x.size for x in jax.tree.leaves(params))
