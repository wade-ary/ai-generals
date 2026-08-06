"""Play against the AI — you control Red (player 0).

Controls:
  Click       Select a tile
  W/A/S/D     Queue move (up/left/down/right) from selected tile
  Shift+WASD  Queue half-army move
  E           Undo last queued move
  Q           Clear entire queue + deselect
  Space       Deselect selected tile
  P           Pause / unpause
  [ / ]       Decrease / increase game speed
  V           Toggle full-map visibility (cheat)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import math
import time
import threading

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import pygame

from generals.core.game import get_observation, get_info, create_initial_state
from generals.core.action import compute_valid_move_mask
from generals.core.env import GeneralsEnv
from generals.core.config import Dimension
from generals.core.rendering import JaxGameAdapter, JaxChannelsAdapter
from generals.gui.properties import Properties, GuiMode
from generals.gui.rendering import Renderer

from networks import obs_to_array
from evals.agent import Agent, _safe_load_config

# Direction: 0=UP 1=DOWN 2=LEFT 3=RIGHT
DIR_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
KEY_TO_DIR = {pygame.K_w: 0, pygame.K_s: 1, pygame.K_a: 2, pygame.K_d: 3}

SEL_COLOR = (255, 255, 255)
ARROW_FULL = (255, 255, 255)
ARROW_HALF = (180, 220, 255)
PASS_ACTION = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)


def _make_env(cfg):
    stages = cfg.curriculum_stages
    if stages:
        last = stages[-1]
        min_d = last.min_generals_distance
        max_d = last.max_generals_distance
        cities = (last.num_cities_min, last.num_cities_max) if last.num_cities_min is not None else (cfg.num_cities_min, cfg.num_cities_max)
        castle = (last.castle_val_min, last.castle_val_max) if last.castle_val_min is not None else (cfg.castle_val_min, cfg.castle_val_max)
    else:
        min_d = cfg.min_generals_distance
        max_d = cfg.max_generals_distance
        cities = (cfg.num_cities_min, cfg.num_cities_max)
        castle = (cfg.castle_val_min, cfg.castle_val_max)
    gs = cfg.eval_grid_size
    return GeneralsEnv(grid_dims=(gs, gs), pad_to=cfg.pad_to,
                       min_generals_distance=min_d, max_generals_distance=max_d,
                       truncation=cfg.truncation, num_cities_range=cities,
                       castle_val_range=castle)


def _cache_visibility(adapter):
    from generals.core import game as _game
    cache = {}
    for i, ag in enumerate(adapter.agents):
        cache[ag] = np.array(_game.get_visibility(adapter.channels._state.ownership[i]))
    adapter.channels.get_visibility = lambda agent_id: cache[agent_id]


def extract_grid_from_log(path, pad_to):
    """Extract map grid from an online game debug log for replay."""
    import pickle
    with open(path, "rb") as f:
        log = pickle.load(f)
    h, w = log[0]["map_size"]
    mountains = np.zeros((h, w), dtype=bool)
    cities = np.zeros((h, w), dtype=bool)
    fog_structures = np.zeros((h, w), dtype=bool)
    for entry in log:
        mountains |= np.asarray(entry["raw_mountains"], dtype=bool)
        cities |= np.asarray(entry["raw_cities"], dtype=bool)
        fog_structures |= np.asarray(entry["raw_structures_in_fog"], dtype=bool)
    mountains |= fog_structures & ~cities
    t1 = log[0]
    our_mask = np.asarray(t1["raw_owned_cells"], dtype=bool) & np.asarray(t1["raw_generals"], dtype=bool)
    our_pos = np.argwhere(our_mask)
    assert len(our_pos) == 1
    our_gen = tuple(our_pos[0])
    empty = [(r, c) for r in range(h) for c in range(w)
             if not mountains[r, c] and not cities[r, c] and (r, c) != our_gen]
    opp_gen = empty[np.random.randint(len(empty))]
    city_armies = np.full((h, w), 40, dtype=np.int32)
    for entry in log:
        vis_cities = np.asarray(entry["raw_cities"], dtype=bool)
        arms = np.asarray(entry["raw_armies"])
        mask = vis_cities & (arms > 0)
        city_armies = np.where(mask, arms, city_armies)
    playable = np.zeros((h, w), dtype=np.int32)
    playable[mountains] = -2
    playable[cities] = city_armies[cities]
    grid = np.full((pad_to, pad_to), -2, dtype=np.int32)
    grid[:h, :w] = playable
    grid[our_gen] = 1
    grid[opp_gen] = 2
    n_mt = int(np.sum(grid[:h, :w] == -2))
    n_ct = int(np.sum(grid[:h, :w] >= 40))
    print(f"Replay map: {h}x{w}, {len(log)} turns, {n_mt} mountains, {n_ct} cities, "
          f"P0={our_gen} P1={opp_gen} (random)")
    return jnp.array(grid, dtype=jnp.int32)


def draw_arrow(surface, sx, sy, dx, dy, color, width=2):
    mx = sx + (dx - sx) * 0.5
    my = sy + (dy - sy) * 0.5
    pygame.draw.line(surface, color, (sx, sy), (mx, my), width)
    angle = math.atan2(my - sy, mx - sx)
    hl, ha = 7, math.pi / 5
    p1 = (mx - hl * math.cos(angle - ha), my - hl * math.sin(angle - ha))
    p2 = (mx - hl * math.cos(angle + ha), my - hl * math.sin(angle + ha))
    pygame.draw.polygon(surface, color, [(mx, my), p1, p2])


def run(agent, cfg, game_speed=2.0, seed=42, replay_grid=None, human_player=0):
    env = _make_env(cfg)
    key = jrandom.PRNGKey(seed)
    key, pk = jrandom.split(key)
    pool, _ = env.reset(pk)

    greedy_fn = eqx.filter_jit(agent.greedy_fn)

    ai_idx = 1 - human_player
    if human_player == 0:
        agents = ["Human (Red)", "AI (Blue)"]
    else:
        agents = ["AI (Red)", "Human (Blue)"]
    sq = Dimension.SQUARE_SIZE.value

    if replay_grid is not None:
        state = create_initial_state(replay_grid)
    else:
        key, ik = jrandom.split(key)
        state = env.init_state(ik)
    mountains = np.array(state.mountains)
    gh, gw = mountains.shape

    # Warmup JIT
    print("Warming up JIT...", end=" ", flush=True)
    _obs = get_observation(state, 1)
    _arr = obs_to_array(_obs)
    _mask = compute_valid_move_mask(_obs.armies, _obs.owned_cells, _obs.mountains)
    _obs_st = agent.init_obs_state_fn(cfg.pad_to, cfg.pad_to)
    _aug, _obs_st = agent.augment_fn(_arr, _obs_st)
    _temporal = jnp.stack([_obs_st.opponent_army_history, _obs_st.opponent_land_history])
    _ = greedy_fn(agent.network, _aug, _mask, _temporal)
    print("done.")

    adapter = JaxGameAdapter(state, agents, get_info(state))
    agent_data = {agents[0]: {"color": (220, 50, 50)}, agents[1]: {"color": (50, 100, 220)}}
    props = Properties(adapter, agent_data, GuiMode.TRAIN)
    props._Properties__agent_fov[agents[ai_idx]] = False
    renderer = Renderer(props)

    font_hud = pygame.font.Font(None, 22)
    font_big = pygame.font.Font(None, 48)

    h_wins, a_wins, games = 0, 0, 0
    clock = pygame.time.Clock()

    while True:  # per-game
        if games > 0:
            key, ik = jrandom.split(key)
            state = env.init_state(ik)
            mountains = np.array(state.mountains)
            adapter.update_from_state(state, get_info(state))
            _cache_visibility(adapter)

        obs_state_ai = agent.init_obs_state_fn(cfg.pad_to, cfg.pad_to)
        selected = None
        queues = []
        last_tick = time.time()
        paused = False
        game_log = []
        turn_counter = [0]

        def _save_game_log():
            if not game_log:
                return
            import pickle
            from datetime import datetime
            log_dir = "evals/eval_human_logs"
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"{log_dir}/game_{ts}_{games}.pkl"
            with open(log_path, "wb") as f:
                pickle.dump(game_log, f)
            print(f"Saved {len(game_log)} turns to {log_path}")

        ai_result = [None]
        ai_lock = threading.Lock()

        def _precompute_ai(state, obs_state_ai_snap, rng_key):
            obs_ai = get_observation(state, ai_idx)
            arr = obs_to_array(obs_ai)
            mask = compute_valid_move_mask(obs_ai.armies, obs_ai.owned_cells, obs_ai.mountains)
            aug, new_obs_state = agent.augment_fn(arr, obs_state_ai_snap)
            temporal = jnp.stack([new_obs_state.opponent_army_history,
                                  new_obs_state.opponent_land_history])
            act = greedy_fn(agent.network, aug, mask, temporal)
            h, w = obs_ai.armies.shape
            turn_counter[0] += 1
            game_log.append({
                "turn": turn_counter[0],
                "map_size": (int(h), int(w)),
                "dt_ms": 0.0,
                "player_index": ai_idx,
                "raw_scores": None,
                "raw_armies": np.array(obs_ai.armies),
                "raw_generals": np.array(obs_ai.generals),
                "raw_cities": np.array(obs_ai.cities),
                "raw_mountains": np.array(obs_ai.mountains),
                "raw_neutral_cells": np.array(obs_ai.neutral_cells),
                "raw_owned_cells": np.array(obs_ai.owned_cells),
                "raw_opponent_cells": np.array(obs_ai.opponent_cells),
                "raw_fog_cells": np.array(obs_ai.fog_cells),
                "raw_structures_in_fog": np.array(obs_ai.structures_in_fog),
                "raw_owned_land_count": int(obs_ai.owned_land_count),
                "raw_owned_army_count": int(obs_ai.owned_army_count),
                "raw_opponent_land_count": int(obs_ai.opponent_land_count),
                "raw_opponent_army_count": int(obs_ai.opponent_army_count),
                "raw_timestep": int(obs_ai.timestep),
                "obs_arr": np.array(arr),
                "obs_augmented": np.array(aug),
                "mask": np.array(mask),
                "temporal": np.array(temporal),
                "action": np.array(act),
            })
            with ai_lock:
                ai_result[0] = (act, new_obs_state)

        key, k1 = jrandom.split(key)
        t = threading.Thread(target=_precompute_ai, args=(state, obs_state_ai, k1), daemon=True)
        t.start()

        while True:  # per-frame
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    _save_game_log()
                    pygame.quit()
                    return
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    mx, my = ev.pos
                    c, r = mx // sq, my // sq
                    if 0 <= r < gh and 0 <= c < gw and mx < renderer.display_grid_width:
                        selected = (r, c)
                        queues.append([])
                    else:
                        selected = None
                elif ev.type == pygame.KEYDOWN:
                    if ev.key in KEY_TO_DIR and selected is not None:
                        d = KEY_TO_DIR[ev.key]
                        dr, dc = DIR_DELTA[d]
                        nr, nc = selected[0] + dr, selected[1] + dc
                        if 0 <= nr < gh and 0 <= nc < gw and not mountains[nr, nc]:
                            half = 1 if (ev.mod & pygame.KMOD_SHIFT) else 0
                            if not queues:
                                queues.append([])
                            queues[-1].append((selected[0], selected[1], d, half))
                            selected = (nr, nc)
                    elif ev.key == pygame.K_SPACE:
                        selected = None
                    elif ev.key == pygame.K_p:
                        paused = not paused
                    elif ev.key == pygame.K_e:
                        while queues and not queues[-1]:
                            queues.pop()
                        if queues:
                            last = queues[-1].pop()
                            selected = (last[0], last[1])
                            if not queues[-1]:
                                queues.pop()
                    elif ev.key == pygame.K_q:
                        queues.clear()
                        selected = None
                    elif ev.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
                        game_speed = max(0.5, game_speed / 2)
                    elif ev.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS):
                        game_speed = min(16, game_speed * 2)
                    elif ev.key == pygame.K_v:
                        renderer.agent_fov[agents[ai_idx]] = not renderer.agent_fov[agents[ai_idx]]

            now = time.time()
            game_over = False
            if not paused and now - last_tick >= 1.0 / game_speed:
                last_tick = now

                t.join()
                with ai_lock:
                    ai_act, obs_state_ai = ai_result[0]
                    ai_result[0] = None

                obs_human = get_observation(state, human_player)
                human_act = PASS_ACTION
                while queues:
                    if not queues[0]:
                        queues.pop(0)
                        continue
                    r, c, d, h = queues[0][0]
                    if not obs_human.owned_cells[r, c] or obs_human.armies[r, c] <= 1:
                        queues.pop(0)
                        continue
                    queues[0].pop(0)
                    human_act = jnp.array([0, r, c, d, h], dtype=jnp.int32)
                    if not queues[0]:
                        queues.pop(0)
                    break

                if human_player == 0:
                    actions = jnp.stack([human_act, ai_act])
                else:
                    actions = jnp.stack([ai_act, human_act])
                timestep, state = env.step(state, actions, pool)
                if timestep.terminated or timestep.truncated:
                    adapter.update_from_state(timestep.last_state, timestep.info)
                    _cache_visibility(adapter)
                else:
                    adapter.update_from_state(state, timestep.info)

                key, k1 = jrandom.split(key)
                ai_result[0] = None
                t = threading.Thread(target=_precompute_ai, args=(state, obs_state_ai, k1), daemon=True)
                t.start()

                if timestep.terminated or timestep.truncated:
                    w = int(timestep.info.winner)
                    if w == human_player:
                        h_wins += 1
                    elif w == ai_idx:
                        a_wins += 1
                    games += 1
                    result = "YOU WIN!" if w == human_player else "AI WINS" if w == ai_idx else "DRAW"
                    print(f"{result} | Turn {int(timestep.last_state.time)} | "
                          f"Score: Human {h_wins} - AI {a_wins}")
                    _save_game_log()
                    game_over = True

            renderer.render_grid()
            renderer.render_stats()

            if selected:
                sr, sc = selected
                dark = pygame.Surface((sq, sq))
                dark.fill((0, 0, 0))
                dark.set_alpha(60)
                for dd in DIR_DELTA.values():
                    ar, ac = sr + dd[0], sc + dd[1]
                    if 0 <= ar < gh and 0 <= ac < gw and not mountains[ar, ac]:
                        renderer.screen.blit(dark, (ac * sq, ar * sq))
                rect = pygame.Rect(sc * sq, sr * sq, sq, sq)
                pygame.draw.rect(renderer.screen, SEL_COLOR, rect, 3)

            for chain in queues:
                for qr, qc, qd, qh in chain:
                    dr, dc = DIR_DELTA[qd]
                    nr, nc = qr + dr, qc + dc
                    color = ARROW_HALF if qh else ARROW_FULL
                    draw_arrow(renderer.screen,
                               qc * sq + sq // 2, qr * sq + sq // 2,
                               nc * sq + sq // 2, nr * sq + sq // 2, color)

            spd = f"{game_speed:g}"
            total_moves = sum(len(c) for c in queues)
            status = f"{'PAUSED' if paused else 'Playing'} | Speed: {spd}x | Queue: {total_moves}"
            hud = font_hud.render(status, True, (255, 255, 255), (0, 0, 0))
            renderer.screen.blit(hud, (4, renderer.display_grid_height - 22))

            if game_over:
                cx = renderer.display_grid_width // 2
                cy = renderer.display_grid_height // 2
                text = font_big.render(result, True, (255, 255, 255))
                sub = font_hud.render("Press any key for next game", True, (200, 200, 200))
                bw = max(text.get_width(), sub.get_width()) + 30
                bh = text.get_height() + sub.get_height() + 30
                bg = pygame.Surface((bw, bh))
                bg.fill((0, 0, 0))
                bg.set_alpha(200)
                renderer.screen.blit(bg, (cx - bw // 2, cy - bh // 2))
                renderer.screen.blit(text, text.get_rect(center=(cx, cy - 12)))
                renderer.screen.blit(sub, sub.get_rect(center=(cx, cy + 20)))

            pygame.display.flip()
            clock.tick(30)

            if game_over:
                waiting = True
                while waiting:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            return
                        elif ev.type == pygame.KEYDOWN:
                            waiting = False
                    clock.tick(30)
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Checkpoint path")
    parser.add_argument("--speed", type=float, default=2.0, help="Game ticks/sec (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--truncation", type=int, default=None)
    parser.add_argument("--replay", type=str, default=None,
                        help="Online game log pickle to replay the same map")
    parser.add_argument("--player", type=int, default=0, choices=[0, 1],
                        help="Which player to control: 0=Red, 1=Blue (default: 0)")
    args = parser.parse_args()

    agent = Agent.load(args.model, args.config)
    cfg = agent.config
    cfg.eval_grid_size = args.grid_size or cfg.max_grid_size
    if args.truncation:
        cfg.truncation = args.truncation

    print(f"Loaded {args.model} ({agent.param_count():,} params)")
    print(__doc__)

    replay_grid = None
    if args.replay:
        replay_grid = extract_grid_from_log(args.replay, cfg.pad_to)

    run(agent, cfg, game_speed=args.speed, seed=args.seed,
        replay_grid=replay_grid, human_player=args.player)


if __name__ == "__main__":
    main()
