"""Evaluate candidates against reference checkpoints and compute ELO."""

import os
import json

import jax.random as jrandom

from generals.core.env import GeneralsEnv

from evals.agent import Agent, _safe_load_config
from evals.matchup import play_match, compute_elo, merge_h2h


def discover_refs(folder):
    """Find reference checkpoints: each .eqx paired with same-name .yaml."""
    refs = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".eqx"):
            continue
        name = fname.replace(".eqx", "")
        yaml_path = os.path.join(folder, name + ".yaml")
        if not os.path.exists(yaml_path):
            yaml_path = os.path.join(folder, "config.yaml")
        if not os.path.exists(yaml_path):
            print(f"WARNING: No config for {fname}, skipping")
            continue
        refs.append({"path": os.path.join(folder, fname), "yaml": yaml_path, "name": name})
    return refs


def load_refs(folder):
    """Load all reference agents from a folder."""
    agents = []
    for ref in discover_refs(folder):
        agent = Agent.load(ref["path"], ref["yaml"])
        agent.name = ref["name"]
        agents.append(agent)
    return agents


def ref_eval(candidates, ref_agents, ref_h2h, env, pool, num_games, truncation, key):
    """Play candidates vs refs, merge with precomputed ref-ref h2h, compute ELO.

    Candidate is always P0. Returns (elo_ratings, h2h).
    """
    cand_names = [a.name for a in candidates]
    ref_names = [a.name for a in ref_agents]
    all_names = cand_names + ref_names

    h2h = {n: {} for n in all_names}
    for cand in candidates:
        for ref in ref_agents:
            key, mk = jrandom.split(key)
            w0, w1, d = play_match(cand, ref, env, pool, num_games, truncation, mk)
            h2h[cand.name][ref.name] = {"wins": w0, "losses": w1, "draws": d}
            h2h[ref.name][cand.name] = {"wins": w1, "losses": w0, "draws": d}
            total = w0 + w1 + d
            decisive = w0 + w1
            dwr = w0 / decisive * 100 if decisive > 0 else 0.0
            print(f"  {cand.name} vs {ref.name}: {w0}W/{w1}L/{d}D ({dwr:.0f}% of decisive, {w0/total*100:.0f}% overall)")

    full_h2h = merge_h2h(ref_h2h, h2h)
    ratings = compute_elo(all_names, full_h2h)
    return ratings, full_h2h
