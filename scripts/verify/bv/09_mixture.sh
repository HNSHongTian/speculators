#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_pytest tests/unit/models/test_dspark_bv_mixture.py "$@"
