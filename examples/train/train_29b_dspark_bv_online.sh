#!/bin/bash

# Single-node online DSpark training for TeleChat4-29B on Ascend NPU.

set -euo pipefail

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000
export NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1,80.5.5.45,80.5.5.44,80.5.5.54,10.127.10.130,10.127.10.135,10.127.10.106,10.127.10.143,10.127.10.141,10.127.10.131,10.127.10.208
export no_proxy="$NO_PROXY"
#export GLOO_SOCKET_IFNAME=lo

NODE_RANK="${1:-0}"
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29500"
NNODES=1
NPROC_PER_NODE=16
if [[ "${NODE_RANK}" != "0" ]]; then
echo "node_rank must be 0 for single-node training, got: ${NODE_RANK}" >&2
exit 1
fi

# General configuration

MODEL="/hpfs/nlp/yangzhihao/ckpts/TeleChat4-29B/29b_12.05T_wodsa_id20_s3_256k_8_260903_1000_final_hf"
DATA_PATH="/hpfs/huawei-2607/shangxt/tmpDataTest/arrow_open_perfected_dehup_80w_no_thinking_full_vocab"
OUTPUT_DIR="/hpfs/huawei-2607/shangxt/Telecom-29B/open_perfect_80w_no_thinking_full_vocab_1014_32_96"
CODE_PATH="/hpfs/huawei-2607/shangxt/speculators"
export PYTHONPATH="$CODE_PATH/src${PYTHONPATH:+:$PYTHONPATH}"
VLLM_SERVER_IP=localhost
VLLM_PORT=8000
SEQ_LENGTH=3072
EPOCHS=10
LR=6e-4

# DSpark configuration

BLOCK_SIZE=15
MAX_ANCHORS=512
NUM_LAYERS=5
TARGET_LAYER_IDS="1 10 19 28 37"
DRAFT_ATTN_IMPL="sdpa"
MASK_TOKEN_ID=139
MARKOV_RANK=256
LOSS_FN='bv'

# Ascend training devices. Online vLLM must run on separate available devices.

TRAIN_NPUS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
NUM_TRAIN_NPUS=16

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_telecom_80w_no_thinking_32_96_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train.pid"

# Single-node rendezvous; remove inherited multi-node network bindings.

unset HCCL_IF_IP GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME HCCL_SOCKET_IFNAME
echo "[$(date '+%F %T')] starting single-node train: node_rank=${NODE_RANK} master=${MASTER_ADDR}:${MASTER_PORT}"

echo "=== Training on Ascend NPU(s): $TRAIN_NPUS ==="
nohup env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
--nnodes="${NNODES}" \
--nproc_per_node="${NPROC_PER_NODE}" \
--node_rank="${NODE_RANK}" \
--master_addr="${MASTER_ADDR}" \
--master_port="${MASTER_PORT}" \
${CODE_PATH}/scripts/train.py \
--verifier-name-or-path "$MODEL" \
--trust-remote-code \
--data-path "$DATA_PATH" \
--vllm-endpoint \
"http://10.127.10.130:8000/v1" \
"http://10.127.10.135:8000/v1" \
"http://10.127.10.106:8000/v1" \
"http://10.127.10.143:8000/v1" \
"http://10.127.10.141:8000/v1" \
"http://10.127.10.131:8000/v1" \
--save-path "$OUTPUT_DIR/checkpoints" \
--epochs "$EPOCHS" \
--draft-config draft_config_29b.json \
--lr "$LR" \
--total-seq-len "$SEQ_LENGTH" \
--speculator-type dspark \
--block-size "$BLOCK_SIZE" \
--max-anchors "$MAX_ANCHORS" \
--target-layer-ids $TARGET_LAYER_IDS \
--draft-attn-impl "$DRAFT_ATTN_IMPL" \
--mask-token-id "$MASK_TOKEN_ID" \
--markov-rank "$MARKOV_RANK" \
--loss-fn "$LOSS_FN" \
--enable-bv-loss \
--bv-anneal-epochs 3 \
--no-enable-confidence-head \
--hidden-states-dtype bfloat16 \
--noise-std 0 \
--on-missing generate \
--on-generate delete \
--optimizer adamw \
--weight-decay 0.01 \
--scheduler-type cosine \
--scheduler-warmup-ratio 0.04 \
--seed 42 \
--num-workers 4 \
--prefetch-factor 4 \
--log-freq 1 \
> "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Log file: $LOG_FILE"
echo "TensorBoard: tensorboard --logdir ./logs --host 0.0.0.0 --port 6006"
echo "View log with: tail -f $LOG_FILE"
echo "Stop with: kill $(cat $PID_FILE)"
