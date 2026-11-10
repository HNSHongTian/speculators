#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
"$PYTHON_BIN" -m torch.distributed.run --rdzv-backend=c10d \
  --rdzv-endpoint=127.0.0.1:0 --rdzv-id=bv-verification --nproc_per_node=2 \
  scripts/verify/bv/distributed.py "$@"
