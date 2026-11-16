#!/bin/bash
# TRPO smoke trial on perfectblend 5k (Qwen3-4B DSpark backbone).
# Prerequisites: vLLM server already running (dspark_qwen3_4b_server.sh).
# Run inside container: dspark-qwen34b-train
#
# Usage:
#   bash examples/train/trpo_qwen3_4b_trial.sh

set -euo pipefail
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1

MODEL="/home/z50062551/weights/Qwen3-4B"
FROM_PRETRAINED="${FROM_PRETRAINED:-/home/z50062551/weights/dspark_qwen3_4b_block7_speculators}"
DATA_PATH="/home/z50062551/datasets/perfectblend_qwen3_4b_5k_processed"
OUTPUT_DIR="/home/z50062551/speculators/output/trpo_qwen3_4b_perfectblend_5k_trial"
VLLM_PORT="${VLLM_PORT:-8000}"

SEQ_LENGTH=2048
EPOCHS=1
LR=6e-4
LOGGER="tensorboard"

BLOCK_SIZE=7
MAX_ANCHORS=32
MAX_TRPO_ANCHORS=4
TREE_TOP_K=4
TREE_MAX_NODES=16
TREE_PACK_MODE=leaf-up
LAMBDA_CE=0.1
MAX_STEPS="${MAX_STEPS:-20}"
TARGET_LAYER_IDS="1 9 17 25 33"
DRAFT_ATTN_IMPL="sdpa"

MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"

TRAIN_NPUS="${TRAIN_NPUS:-3}"
NUM_TRAIN_NPUS="${NUM_TRAIN_NPUS:-1}"

cd /home/z50062551/speculators
mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/checkpoints"
LOG_FILE="$OUTPUT_DIR/logs/trpo_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$OUTPUT_DIR/logs/trpo.pid"

echo "=== TRPO trial on Ascend NPUs: $TRAIN_NPUS (max_steps=$MAX_STEPS) ==="
echo "Log: $LOG_FILE"

nohup env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
  --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
  scripts/train_trpo.py \
  --verifier-name-or-path "$MODEL" \
  --data-path "$DATA_PATH" \
  --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
  --save-path "$OUTPUT_DIR/checkpoints" \
  --log-dir "$OUTPUT_DIR/logs" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --logger "$LOGGER" \
  --total-seq-len "$SEQ_LENGTH" \
  --speculator-type dspark \
  --from-pretrained "$FROM_PRETRAINED" \
  --block-size "$BLOCK_SIZE" \
  --max-anchors "$MAX_ANCHORS" \
  --max-trpo-anchors "$MAX_TRPO_ANCHORS" \
  --tree-top-k "$TREE_TOP_K" \
  --tree-max-nodes "$TREE_MAX_NODES" \
  --tree-pack-mode "$TREE_PACK_MODE" \
  --lambda-ce "$LAMBDA_CE" \
  --max-steps "$MAX_STEPS" \
  --draft-attn-impl "$DRAFT_ATTN_IMPL" \
  --target-layer-ids $TARGET_LAYER_IDS \
  --markov-rank "$MARKOV_RANK" \
  --markov-head-type "$MARKOV_HEAD_TYPE" \
  --no-enable-confidence-head \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --on-missing generate \
  --on-generate delete \
  --num-workers 2 \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "PID file: $PID_FILE"
echo "Follow: tail -f $LOG_FILE"
echo "Stop:   kill \$(cat $PID_FILE)"
