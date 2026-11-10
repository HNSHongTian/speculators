#!/bin/bash
set -euo pipefail
VERIFY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for step in 01_loss 02_alignment 03_training 04_compatibility 05_script 07_paper_audit 09_mixture 10_loss_schedule; do
  echo "Running BV verification: $step"
  bash "$VERIFY_DIR/$step.sh" "$@"
done
