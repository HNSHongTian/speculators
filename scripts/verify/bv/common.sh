#!/bin/bash
set -euo pipefail
BV_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$BV_REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$BV_REPO_ROOT/src:$BV_REPO_ROOT/hs_connectors/src:$BV_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
run_pytest() { "$PYTHON_BIN" -m pytest -q --override-ini addopts='' "$@"; }
