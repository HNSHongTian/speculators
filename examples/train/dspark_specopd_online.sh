#!/usr/bin/env bash
# Original DSpark + SpecOPD. Teacher service runs separately.
set -euo pipefail
: "${MODEL:?Set MODEL to the target model directory}"
: "${DATA_PATH:?Set DATA_PATH to the prepared Arrow dataset}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the training output directory}"
: "${VLLM_ENDPOINT:?Set VLLM_ENDPOINT to the hidden-state server /v1 URL}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}"
extra=()
if [[ -n "${DRAFT_CHECKPOINT:-}" ]]; then
  extra+=(--from-pretrained "$DRAFT_CHECKPOINT")
fi
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
  "$REPO_DIR/scripts/train.py" \
  --speculator-type dspark \
  --verifier-name-or-path "$MODEL" \
  --data-path "$DATA_PATH" \
  --save-path "$OUTPUT_DIR" \
  --vllm-endpoint "$VLLM_ENDPOINT" \
  --on-missing generate --on-generate delete \
  --train-e2e-loss-specopd \
  --specopd-num-anchors "${SPECOPD_ANCHORS:-8}" \
  --specopd-target-concurrency "${TARGET_CONCURRENCY:-4}" \
  --specopd-target-hidden-norm "${TARGET_HIDDEN_NORM:-pre}" \
  --specopd-eval-interval "${SPECOPD_EVAL_INTERVAL:-0}" \
  --specopd-eval-samples "${SPECOPD_EVAL_SAMPLES:-4}" \
  --specopd-eval-max-new-tokens "${SPECOPD_EVAL_MAX_NEW_TOKENS:-64}" \
  --specopd-eval-temperature "${SPECOPD_EVAL_TEMPERATURE:-0}" \
  --epochs "${EPOCHS:-3}" --lr "${LR:-1e-5}" \
  --checkpoint-freq 0.25 --log-freq 10 \
  "${extra[@]}" "$@"
