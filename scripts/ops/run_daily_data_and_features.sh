#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${QUANTMIND_CONTAINER:-quantmind}"
DOCKER_BIN="$(command -v docker)"

"${DOCKER_BIN}" exec "${CONTAINER}" \
  python /app/scripts/data/maintenance/sync_daily_from_baostock.py --apply

"${DOCKER_BIN}" exec "${CONTAINER}" \
  python /app/scripts/data/maintenance/build_model_qlib_features_incremental.py --apply
