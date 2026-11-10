#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_pytest tests/unit/train/test_trust_remote_config.py \
  tests/unit/train/test_draft_config_init.py tests/unit/train/test_bv_29b_script.py "$@"
