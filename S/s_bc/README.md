# S BC 12x12 self-play initialization

This directory contains the non-EMA `BC_S_24_step_090000` model converted
from `pad_to: 24` to `pad_to: 12` for self-play training.

## Files

- `BC_S_12_step_090000_init.eqx`: weights-only 12x12 initialization.
- `config.yaml`: self-play configuration that loads the converted checkpoint
  with a fresh optimizer. EMA is initialized from the loaded model because
  `ema_checkpoint` is intentionally empty.

## Conversion

The source model has a 12x12 grid of spatial patch positional embeddings
because its 24x24 observations use `patch_size: 2`. The converted model has a
6x6 patch grid for 12x12 observations. Each new spatial positional embedding
is the mean of the corresponding non-overlapping 2x2 region. The value token,
two temporal-token positions, transformer, policy head, and value head are
copied unchanged.

Source files:

- `checkpoints_run1/BC_S_24_step_090000.eqx`
- `checkpoints_run1/BC_S_24_step_090000.yaml`

Conversion command:

```bash
.venv/bin/python convert_model12to24.py \
  checkpoints_run1/BC_S_24_step_090000.eqx \
  checkpoints_run1/BC_S_24_step_090000.yaml \
  S/s_bc/BC_S_12_step_090000_init.eqx \
  --output-config S/s_bc/config.yaml
```

The run is configured for exactly 500 PPO iterations with 2,048 environments
per GPU. At completion, the trainer saves
`checkpoints/S_BC_12/S_BC_12_final.eqx`.

Curriculum is disabled. Every iteration uses the top-level 12x12 environment
settings: general distance 4-8, 9-11 cities, and castle values 13-18.

Run training and the end-of-run evaluation from the repository root:

```bash
S/s_bc/run_500_selfplay_and_eval.sh
```

After training, the script compares the final model against the 12x12 base
self-play agent `S/S_500/S_500.eqx` over 200 paired games (100 maps with both
seat assignments). It saves both the recorded matchup and a GIF:

- `S/s_bc/S_BC_12_final_vs_S_500_replay.pkl`
- `S/s_bc/S_BC_12_final_vs_S_500.gif`

Set `PYTHON_BIN` if the Python executable is elsewhere, for example:

```bash
PYTHON_BIN=python S/s_bc/run_500_selfplay_and_eval.sh
```
