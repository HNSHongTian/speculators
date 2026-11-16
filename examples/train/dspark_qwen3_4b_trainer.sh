#!/bin/bash
# Train DSpark draft for local Qwen3-4B on Ascend (online hidden states).
# Prerequisites: vLLM server already running (dspark_qwen3_4b_server.sh).
# Run inside container: dspark-qwen34b-train
#
# Usage:
#   bash examples/train/dspark_qwen3_4b_trainer.sh

set -euo pipefail
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1

MODEL="/home/z50062551/weights/Qwen3-4B"
# Speculators-format draft (converted from DeepSpec). Leave empty to train from scratch.
FROM_PRETRAINED="${FROM_PRETRAINED:-/home/z50062551/weights/dspark_qwen3_4b_block7_speculators}"
DATA_PATH="/home/z50062551/datasets/perfectblend_qwen3_4b_5k_processed"
OUTPUT_DIR="/home/z50062551/speculators/output/dspark_qwen3_4b_perfectblend_5k"
VLLM_PORT="${VLLM_PORT:-8000}"

SEQ_LENGTH=4096
EPOCHS=5
LR=6e-4
LOGGER="tensorboard"

SPECULATOR_TYPE="dspark"
BLOCK_SIZE=7
MAX_ANCHORS=512
NUM_LAYERS=5
# Full verifier vocab for Qwen3-4B; set lower only if you built a reduced mapping.
DRAFT_VOCAB_SIZE=""
TARGET_LAYER_IDS="1 9 17 25 33"
DRAFT_ATTN_IMPL="sdpa"

MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# Keep vLLM and trainer on disjoint NPUs (server default uses 1,2).
TRAIN_NPUS="${TRAIN_NPUS:-3,4,5,6}"
NUM_TRAIN_NPUS="${NUM_TRAIN_NPUS:-4}"

cd /home/z50062551/speculators
mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/checkpoints"
LOG_FILE="$OUTPUT_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$OUTPUT_DIR/logs/train.pid"

EXTRA_ARGS=()
if [[ -n "$FROM_PRETRAINED" ]]; then
  EXTRA_ARGS+=(--from-pretrained "$FROM_PRETRAINED")
fi
if [[ -n "$DRAFT_VOCAB_SIZE" ]]; then
  EXTRA_ARGS+=(--draft-vocab-size "$DRAFT_VOCAB_SIZE")
fi

echo "=== Training DSpark on Ascend NPUs: $TRAIN_NPUS ==="
echo "Log: $LOG_FILE"

nohup env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
  --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
  scripts/train.py \
  --verifier-name-or-path "$MODEL" \
  --data-path "$DATA_PATH" \
  --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
  --save-path "$OUTPUT_DIR/checkpoints" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --logger "$LOGGER" \
  --total-seq-len "$SEQ_LENGTH" \
  --speculator-type "$SPECULATOR_TYPE" \
  --block-size "$BLOCK_SIZE" \
  --max-anchors "$MAX_ANCHORS" \
  --num-layers "$NUM_LAYERS" \
  --draft-attn-impl "$DRAFT_ATTN_IMPL" \
  --target-layer-ids $TARGET_LAYER_IDS \
  --markov-rank "$MARKOV_RANK" \
  --markov-head-type "$MARKOV_HEAD_TYPE" \
  --enable-confidence-head \
  --confidence-head-with-markov \
  --loss-fn "$LOSS_FN" \
  --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
  --on-missing generate \
  --on-generate delete \
  "${EXTRA_ARGS[@]}" \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "PID file: $PID_FILE"
echo "Follow: tail -f $LOG_FILE"
echo "Stop:   kill \$(cat $PID_FILE)"
