#!/usr/bin/env bash
# Partial fine-tune (last-N transformer blocks + action branches, see
# wan_va/configs/va_so101_train_cfg.py's train_mode="action_last_n") of
# LingBot-VA on the 3-camera (front/right/wrist) three_cubes_1_lingbot
# conversion. Full-parameter fine-tuning was measured to OOM 24GB GPUs even at
# 6 free GPUs (see va_so101_train_cfg.py's comment) -- not attempted by
# default here. Wraps script/run_va_posttrain.sh's launch pattern (torchrun
# -m wan_va.train) with:
#   - free-GPU auto-detection (falls back to explicit GPUS=<comma list>)
#   - gradient_accumulation_steps scaled to keep the demo's effective batch
#     size (batch_size=1 x grad_accum=8 x 8 GPUs = 64) on however many GPUs
#     are actually free
#   - an explicit assertion that wan_va/train.py still hardcodes attn_mode
#     "flex" for training (inference uses "torch" -- see
#     script/launch_server_so101.sh), since that flag isn't exposed as a CLI
#     override and this script must not modify train.py to set it.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: script/run_va_posttrain_so101.sh [-- <extra wan_va.train overrides>]

Environment variables:
  GPUS                 Comma-separated CUDA device indices to use, e.g. "1,2,3,5,7".
                        If unset, free GPUs (near-zero memory.used) are auto-detected.
  EFFECTIVE_BATCH       Target effective batch size to preserve (default: 64,
                        matching va_demo_train_cfg's batch_size=1 x grad_accum=8 x 8 GPUs).
  CONFIG_NAME           wan_va config to train (default: so101_train).
  SAVE_ROOT             Checkpoint/metrics output directory
                        (default: ~/Projects/lingbot-va/train_out/so101_three_cubes).
  NUM_STEPS             Total training steps (default: 1500).
  MASTER_PORT           torchrun master port (default: 29501).

Examples:
  script/run_va_posttrain_so101.sh
  GPUS=1,2,3 script/run_va_posttrain_so101.sh
  script/run_va_posttrain_so101.sh -- --learning-rate 5e-5
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_NAME=${CONFIG_NAME:-"so101_train"}
SAVE_ROOT=${SAVE_ROOT:-"${REPO_ROOT}/train_out/so101_three_cubes"}
NUM_STEPS=${NUM_STEPS:-1500}
MASTER_PORT=${MASTER_PORT:-29501}
EFFECTIVE_BATCH=${EFFECTIVE_BATCH:-64}
BATCH_SIZE=1

# --- attn_mode assertion (train.py:94 hardcodes "flex" for training) ---
if ! grep -q 'attn_mode="flex"' wan_va/train.py; then
  echo "ERROR: wan_va/train.py no longer hardcodes attn_mode=\"flex\" for training." >&2
  echo "This script assumes training uses flex-attention (inference server uses torch" >&2
  echo "attention instead, see script/launch_server_so101.sh). Re-check before training." >&2
  exit 1
fi
echo "[run_va_posttrain_so101] confirmed train.py uses attn_mode=flex for training"

# --- GPU selection ---
if [[ -n "${GPUS:-}" ]]; then
  gpu_list="${GPUS}"
else
  gpu_list=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 500) print $1}' \
    | paste -sd, -)
fi
if [[ -z "${gpu_list}" ]]; then
  echo "ERROR: no free GPUs detected and GPUS not set." >&2
  exit 1
fi
n_gpus=$(awk -F',' '{print NF}' <<<"${gpu_list}")
echo "[run_va_posttrain_so101] using GPUs: ${gpu_list} (n_gpus=${n_gpus})"

# --- gradient accumulation scaled to preserve the demo's effective batch ---
grad_accum=$(( (EFFECTIVE_BATCH + BATCH_SIZE * n_gpus - 1) / (BATCH_SIZE * n_gpus) ))
if (( grad_accum < 1 )); then
  grad_accum=1
fi
echo "[run_va_posttrain_so101] effective_batch=${EFFECTIVE_BATCH} batch_size=${BATCH_SIZE} n_gpus=${n_gpus} -> gradient_accumulation_steps=${grad_accum}"

overrides=""
if [[ $# -ne 0 ]]; then
  if [[ "$1" == "--" ]]; then shift; fi
  overrides="$*"
fi

export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="${gpu_list}" PYTORCH_ALLOC_CONF="expandable_segments:True" \
  /home/rxhuang/anaconda3/envs/lingbot/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${n_gpus}" \
  --master_port "${MASTER_PORT}" \
  --tee 3 \
  -m wan_va.train \
  --config-name "${CONFIG_NAME}" \
  --save-root "${SAVE_ROOT}" \
  --num-steps "${NUM_STEPS}" \
  --gradient-accumulation-steps "${grad_accum}" \
  ${overrides}
