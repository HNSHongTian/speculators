#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_pytest tests/unit/models/test_dspark_bv_compatibility.py tests/unit/models/test_dspark_metrics.py tests/unit/models/test_dspark_model_definitions.py "$@"
