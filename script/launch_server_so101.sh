#!/usr/bin/env bash
# Launch the LingBot-VA inference server for the 3-camera SO101 checkpoint.
# Forked from script/run_three_cubes_server.sh's single-GPU torchrun pattern.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: script/launch_server_so101.sh [checkpoint_dir]

Positional:
  checkpoint_dir   Directory containing transformer/ (default: train_out/so101_three_cubes/checkpoints/last)

Environment variables:
  GPU          CUDA device index to use (default: auto-detect first free GPU).
  CONFIG_NAME  wan_va config to serve (default: so101).
  SAVE_ROOT    Server debug output directory (default: train_out/so101_three_cubes/server_debug).
  PORT         Server port override (default: config's own port, 29536).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT=${1:-"${REPO_ROOT}/train_out/so101_three_cubes/checkpoints/last"}
CONFIG_NAME=${CONFIG_NAME:-"so101"}
SAVE_ROOT=${SAVE_ROOT:-"${REPO_ROOT}/train_out/so101_three_cubes/server_debug"}

# --- attn_mode assertion (wan_va_server.py:95 hardcodes "torch" for inference) ---
if ! grep -q 'attn_mode="torch"' wan_va/wan_va_server.py; then
  echo "ERROR: wan_va/wan_va_server.py no longer hardcodes attn_mode=\"torch\" for inference." >&2
  echo "This script assumes inference uses torch attention (training uses flex, see" >&2
  echo "script/run_va_posttrain_so101.sh). Re-check before serving." >&2
  exit 1
fi
echo "[launch_server_so101] confirmed wan_va_server.py uses attn_mode=torch for inference"

NORM_STAT="/data/rxhuang/three_cubes_1_lingbot/norm_stat.json"
if [[ -f "${NORM_STAT}" ]]; then
  echo "[launch_server_so101] norm_stat.json md5: $(md5sum "${NORM_STAT}" | awk '{print $1}')"
else
  echo "WARNING: ${NORM_STAT} not found; server config will fail to load norm_stat." >&2
fi

if [[ -n "${GPU:-}" ]]; then
  gpu="${GPU}"
else
  gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 500) {print $1; exit}}')
fi
if [[ -z "${gpu}" ]]; then
  echo "ERROR: no free GPU detected and GPU not set." >&2
  exit 1
fi
echo "[launch_server_so101] using GPU ${gpu}, checkpoint=${CHECKPOINT}"

CUDA_VISIBLE_DEVICES="${gpu}" /home/rxhuang/anaconda3/envs/lingbot/bin/torchrun \
  --standalone --nproc-per-node=1 \
  -m wan_va.wan_va_server \
  --config-name "${CONFIG_NAME}" \
  --transformer-path "${CHECKPOINT}/transformer" \
  --save_root "${SAVE_ROOT}"
