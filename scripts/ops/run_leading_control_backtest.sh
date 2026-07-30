#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${QUANTMIND_PROJECT_DIR:-/opt/quantmind}"
CONTAINER="${QUANTMIND_CONTAINER:-quantmind}"
BASHRC_PATH="${QUANTMIND_ROOT_BASHRC:-/root/.bashrc}"

TOKEN_LINE="$(grep -m1 -E '^[[:space:]]*export[[:space:]]+TUSHARE_TOKEN=' "${BASHRC_PATH}" || true)"
if [[ -z "${TOKEN_LINE}" ]]; then
    echo "TUSHARE_TOKEN export was not found in ${BASHRC_PATH}" >&2
    exit 1
fi
source /dev/stdin <<<"${TOKEN_LINE}"
if [[ -z "${TUSHARE_TOKEN:-}" ]]; then
    echo "TUSHARE_TOKEN is empty" >&2
    exit 1
fi

export TUSHARE_TOKEN
DOCKER_BIN="$(command -v docker)"

exec "${DOCKER_BIN}" exec \
    -e TUSHARE_TOKEN \
    "${CONTAINER}" \
    python /app/scripts/analysis/leading_control_backtest.py \
    --snapshot-dir /app/db/feature_snapshots \
    --cache-dir /data/cache/leading-control-backtest \
    --output-dir /data/uploads/reports/leading-control-backtest \
    "$@"
