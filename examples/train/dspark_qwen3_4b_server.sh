#!/bin/bash
# Launch vLLM (Ascend) for online DSpark training against local Qwen3-4B.
# Run inside container: dspark-qwen34b-train
#
# Usage:
#   bash examples/train/dspark_qwen3_4b_server.sh

set -euo pipefail
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1

MODEL="/home/z50062551/weights/Qwen3-4B"
VLLM_PORT="${VLLM_PORT:-8000}"
# Match deepseek-ai/dspark_qwen3_4b_block7 target taps
TARGET_LAYER_IDS="1 9 17 25 33"
# NPU0 may already be occupied on this host; default to 1-2 for vLLM.
VLLM_NPUS="${VLLM_NPUS:-1,2}"
DP_SIZE="${DP_SIZE:-2}"

cd /home/z50062551/speculators

echo "=== Launching vLLM on Ascend NPUs: $VLLM_NPUS (port=$VLLM_PORT) ==="
env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py "$MODEL" \
  --target-layer-ids $TARGET_LAYER_IDS \
  -- --port "$VLLM_PORT" --data-parallel-size "$DP_SIZE" --enforce-eager
