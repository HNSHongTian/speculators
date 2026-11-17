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
MODEL="/hpfs/huawei/zwj/inference/qwen3_4b"
DATA_PATH="/hpfs/nlp/wuyuxuan/data/DSpark_gpu_data"
OUTPUT_DIR="/hpfs/huawei-2607/shangxt/Telecom-29B/open_perfect_80w__full_vocab_1110_b7_treespark_markov_new_all_qwen3_shuaishuai"
CODE_PATH="/hpfs/huawei-2607/xiaoyf/code/shuaishuai/speculators"
FROM_PRETRAINED="${FROM_PRETRAINED:?Set FROM_PRETRAINED to your Qwen3-4B draft checkpoint directory}"
VLLM_SERVER_IP=localhost
VLLM_PORT=8000
SEQ_LENGTH=4096
EPOCHS=5
LR=6e-4

# DSpark configuration
BLOCK_SIZE=7
MAX_ANCHORS=256
TARGET_LAYER_IDS="1 9 17 25 33"
DRAFT_ATTN_IMPL="sdpa"
MASK_TOKEN_ID=151669
MARKOV_RANK=256
LOSS_FN='{"ce": 0.1, "tv": 1.8}'

# TreeSpark training configuration (aligned with trial script)
TREE_TOP_K=4
TREE_MAX_NODES=16
TREE_PACK_MODE="leaf-up"
MAX_TRPO_ANCHORS=256
LAMBDA_CE=0.1

# Ascend training devices. Online vLLM must run on separate available devices.
TRAIN_NPUS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
NUM_TRAIN_NPUS=16

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_open_perfect_80w_no_thinking_full_vocab_16_16_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train_16_16.pid"

echo "=== Training on Ascend NPU(s): $TRAIN_NPUS ==="
nohup env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
  --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
  "${CODE_PATH}/scripts/train_trpo.py" \
  --verifier-name-or-path "$MODEL" \
  --trust-remote-code \
  --data-path "$DATA_PATH" \
  --vllm-endpoint "http://10.127.10.139:8000/v1" \
  --save-path "$OUTPUT_DIR/checkpoints" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --total-seq-len "$SEQ_LENGTH" \
  --speculator-type dspark \
  --from-pretrained "$FROM_PRETRAINED" \
  --block-size "$BLOCK_SIZE" \
  --max-anchors "$MAX_ANCHORS" \
  --target-layer-ids $TARGET_LAYER_IDS \
  --draft-attn-impl "$DRAFT_ATTN_IMPL" \
  --mask-token-id "$MASK_TOKEN_ID" \
  --markov-rank "$MARKOV_RANK" \
  --loss-fn "$LOSS_FN" \
  --dflash-decay-gamma 6.0 \
  --tree-top-k "$TREE_TOP_K" \
  --tree-max-nodes "$TREE_MAX_NODES" \
  --tree-pack-mode "$TREE_PACK_MODE" \
  --max-trpo-anchors "$MAX_TRPO_ANCHORS" \
  --lambda-ce "$LAMBDA_CE" \
  --hidden-states-dtype bfloat16 \
  --on-missing generate \
  --on-generate delete \
  --optimizer adamw \
  --weight-decay 0.01 \
  --scheduler-type cosine \
  --scheduler-warmup-ratio 0.04 \
  --num-workers 4 \
  --prefetch-factor 4 \
  --log-freq 1 \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Log file: $LOG_FILE"
echo "TensorBoard: tensorboard --logdir ./logs --host 0.0.0.0 --port 6006"
echo "View log with: tail -f $LOG_FILE"
echo "Stop with: kill $(cat "$PID_FILE")"
