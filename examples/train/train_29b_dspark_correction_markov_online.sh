#!/bin/bash
# Online DSpark Correction + RNN Markov training for TeleChat4-29B on Ascend NPU.

set -euo pipefail

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000
export NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1,80.5.5.45,80.5.5.44,80.5.5.54
export no_proxy="$NO_PROXY"

NODE_RANK="${1:?usage: $0 <node_rank>}"
MASTER_ADDR="10.127.10.100"
MASTER_PORT="8000"
NNODES=2
NPROC_PER_NODE=16
NIC_NAME="eth0"
if [[ "${NODE_RANK}" == "0" ]]; then
  LOCAL_IP="10.127.10.100"
elif [[ "${NODE_RANK}" == "1" ]]; then
  LOCAL_IP="10.127.10.206"
else
  echo "node_rank must be 0 or 1, got: ${NODE_RANK}" >&2
  exit 1
fi

# General configuration
MODEL="/hpfs/nlp/yangzhihao/ckpts/TeleChat4-29B/29b_12.05T_wodsa_id20_s3_256k_8_260903_1000_final_hf"
DATA_PATH="/hpfs/huawei-2607/shangxt/tmpDataTest/arrow_open_perfected_dehup_80w_no_thinking_full_vocab"
OUTPUT_DIR="/hpfs/huawei-2607/shangxt/Telecom-29B/open_perfect_80w_no_thinking_full_vocab_1014_32_96_dspark_correction_markov_rnn"
CODE_PATH="/hpfs/huawei-2607/shangxt/speculators"
SEQ_LENGTH=4096
EPOCHS=10
LR=3e-4

# DSpark backbone
BLOCK_SIZE=7
MAX_ANCHORS=512
TARGET_LAYER_IDS="1 10 19 28 37"
DRAFT_ATTN_IMPL="sdpa"
MASK_TOKEN_ID=139

# Native DSpark Markov/RNN head
MARKOV_RANK=256
MARKOV_HEAD_TYPE="rnn"

# Causal Correction head
CORRECTION_OUTPUT_MODE="hidden"
CORRECTION_HIDDEN_SIZE=512
CORRECTION_RANK=256
CORRECTION_NUM_LAYERS=1
CORRECTION_NUM_HEADS=8
CORRECTION_GATE_BIAS=0.0

# MMuse-style Correction-Markov collaboration
CORRECTION_MARKOV_GATE_BIAS=-2.0
CORRECTION_MARKOV_SCALE_INIT=0.0
LOSS_FN='{"ce": 0.1, "tv": 0.9}'

# Ascend training devices. Online vLLM must run on separate available devices.
TRAIN_NPUS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_telecom_80w_no_thinking_32_96_dspark_correction_markov_rnn_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train.pid"

# Multi-node networking
export HCCL_IF_IP="${LOCAL_IP}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export TP_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"
echo "[$(date '+%F %T')] starting train: node_rank=${NODE_RANK} local_ip=${LOCAL_IP} master=${MASTER_ADDR}:${MASTER_PORT}"

echo "=== Training DSpark Correction + RNN Markov on Ascend NPU(s): $TRAIN_NPUS ==="
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
    --speculator-type dspark_correction_markov \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --draft-attn-impl "$DRAFT_ATTN_IMPL" \
    --mask-token-id "$MASK_TOKEN_ID" \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --correction-output-mode "$CORRECTION_OUTPUT_MODE" \
    --correction-hidden-size "$CORRECTION_HIDDEN_SIZE" \
    --correction-rank "$CORRECTION_RANK" \
    --correction-num-layers "$CORRECTION_NUM_LAYERS" \
    --correction-num-heads "$CORRECTION_NUM_HEADS" \
    --correction-gate-bias "$CORRECTION_GATE_BIAS" \
    --correction-markov-gate-bias "$CORRECTION_MARKOV_GATE_BIAS" \
    --correction-markov-scale-init "$CORRECTION_MARKOV_SCALE_INIT" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --confidence-head-with-correction \
    --no-correction-hidden-aux-loss \
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
    --log-freq 1 \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Log file: $LOG_FILE"
echo "TensorBoard: tensorboard --logdir $LOG_DIR --host 0.0.0.0 --port 6006"
echo "View log with: tail -f $LOG_FILE"
echo "Stop with: kill \$(cat $PID_FILE)"
