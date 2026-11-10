#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_pytest \
  tests/unit/train/test_dspark_loss_weight_schedule.py \
  tests/unit/models/test_dspark_bv_mixture.py \
  tests/unit/train/test_bv_29b_script.py \
  "$@"
