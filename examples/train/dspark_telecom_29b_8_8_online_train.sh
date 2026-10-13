#!/bin/bash
# Online DSpark training for Qwen3-4B on Ascend NPU.

set -euo pipefail

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000
export NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1,80.5.5.45,80.5.5.44,80.5.5.54
export no_proxy="$NO_PROXY"
export GLOO_SOCKET_IFNAME=lo

# General configuration
MODEL="/hpfs/nlp/yangzhihao/ckpts/TeleChat4-29B/29b_12.05T_wodsa_id20_s3_256k_8_260903_1000_final_hf"
DATA_PATH="/hpfs/huawei-2607/shangxt/tmpDataTest/arrow_open_perfected_dehup_80w_no_thinking"
OUTPUT_DIR="/hpfs/huawei-2607/shangxt/Telecom-29B/telecom_80w_no_thinking_0910_8_8"
CODE_PATH="/hpfs/huawei-2607/shangxt/speculators"
VLLM_SERVER_IP=localhost
VLLM_PORT=8000
SEQ_LENGTH=4096
EPOCHS=10
LR=6e-4

# DSpark configuration
BLOCK_SIZE=7
MAX_ANCHORS=512
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=32000
TARGET_LAYER_IDS="1 10 19 28 37"
DRAFT_ATTN_IMPL="sdpa"
MASK_TOKEN_ID=139
MARKOV_RANK=256
LOSS_FN='{"ce": 0.3, "tv": 0.7}'

# Ascend training devices. Online vLLM must run on separate available devices.
TRAIN_NPUS="0,1,2,3,4,5,6,7"
NUM_TRAIN_NPUS=8

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_telecom_40w_no_thinking_8_8_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train.pid"

echo "=== Training on Ascend NPU(s): $TRAIN_NPUS ==="
nohup env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    ${CODE_PATH}/scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --trust-remote-code \
    --data-path "$DATA_PATH" \
    --vllm-endpoint "http://${VLLM_SERVER_IP}:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type dspark \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --draft-attn-impl "$DRAFT_ATTN_IMPL" \
    --mask-token-id "$MASK_TOKEN_ID" \
    --markov-rank "$MARKOV_RANK" \
    --loss-fn "$LOSS_FN" \
    --dflash-decay-gamma 6.0 \
    --hidden-states-dtype bfloat16 \
    --on-missing generate \
    --on-generate delete \
    --optimizer adamw \
    --weight-decay 0.01 \
    --scheduler-type cosine \
    --scheduler-warmup-ratio 0.04 \
    --num-workers 4 \
    --prefetch-factor 4 \
    --log-freq 20 \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Log file: $LOG_FILE"
echo "TensorBoard: tensorboard --logdir ./logs --host 0.0.0.0 --port 6006"
echo "View log with: tail -f $LOG_FILE"
echo "Stop with: kill \$(cat $PID_FILE)"
