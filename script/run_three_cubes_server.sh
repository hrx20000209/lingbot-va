#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:-/home/rxhuang/Projects/models/lingbot-va-three-cubes/action_last2/checkpoints/last/transformer}
GPU=${GPU:-3}

cd /home/rxhuang/Projects/lingbot-va
CUDA_VISIBLE_DEVICES=${GPU} /home/rxhuang/anaconda3/envs/lingbot/bin/torchrun \
  --standalone --nproc-per-node=1 \
  -m wan_va.wan_va_server \
  --config-name three_cubes \
  --transformer-path "${CHECKPOINT}" \
  --save_root /home/rxhuang/Projects/models/lingbot-va-three-cubes/inference_debug
