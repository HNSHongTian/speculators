#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../bv/common.sh"
run_pytest \
  tests/unit/train/test_trust_remote_config.py \
  tests/unit/train/test_dspark_29b_original_script.py \
  "$@"
