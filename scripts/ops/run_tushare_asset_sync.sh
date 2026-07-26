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

# Source only the single trusted export line.  This bypasses the standard
# non-interactive early return in /root/.bashrc without logging the secret.
source /dev/stdin <<<"${TOKEN_LINE}"
if [[ -z "${TUSHARE_TOKEN:-}" ]]; then
    echo "TUSHARE_TOKEN is empty" >&2
    exit 1
fi

exec docker exec -e TUSHARE_TOKEN "${CONTAINER}" \
    python /app/scripts/data/maintenance/sync_etf_futures_from_tushare.py \
    --apply "$@"
