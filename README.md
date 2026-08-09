# Generals AI

A transformer-based reinforcement-learning agent for [Generals.io](https://generals.io/), trained with PPO and self-play.

## S_750 self-play

The animation below shows the trained `S_750` checkpoint playing against itself. Both players use greedy action selection from the same policy.

![S_750 trained agent self-play](generals/assets/gifs/s750-selfplay.gif)

Run the matchup locally:

```bash
.venv/bin/python evals/eval_selfplay.py S/S_750/S_750.eqx \
  --config S/S_750/config.yaml \
  --fps 10
```

Record another deterministic game as a GIF:

```bash
.venv/bin/python evals/eval_selfplay.py S/S_750/S_750.eqx \
  --config S/S_750/config.yaml \
  --seed 123 \
  --fps 10 \
  --gif-game-index 1 \
  --gif-speed 2 \
  --gif generals/assets/gifs/s750-selfplay.gif
```

GitHub renders an animated GIF referenced with ordinary Markdown, so the replay plays directly on the repository page.
