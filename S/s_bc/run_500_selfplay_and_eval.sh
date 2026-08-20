#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-.venv/bin/python}"
run_dir="checkpoints/S_BC_12"
trained_model="${run_dir}/S_BC_12_final.eqx"
base_model="S/S_500/S_500.eqx"

"${python_bin}" main.py --config S/s_bc/config.yaml

# Allow GIF rendering on headless training machines such as Colab.
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"

"${python_bin}" -m train.eval_checkpoint_match \
  "${trained_model}" \
  "${base_model}" \
  --config-a "${run_dir}/config.yaml" \
  --config-b S/S_500/config.yaml \
  --num-games 200 \
  --replay-output S/s_bc/S_BC_12_final_vs_S_500_replay.pkl \
  --gif-output S/s_bc/S_BC_12_final_vs_S_500.gif \
  --gif-stride 4 \
  --gif-speed 2.0
