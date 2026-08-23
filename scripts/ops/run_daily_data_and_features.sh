#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${QUANTMIND_CONTAINER:-quantmind}"
BASHRC_PATH="${QUANTMIND_ROOT_BASHRC:-/root/.bashrc}"
DOCKER_BIN="$(command -v docker)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
QUANTDB_SYNC_ENABLED="${QUANTDB_SYNC_ENABLED:-true}"
QUANTDB_SYNC_IMAGE="${QUANTDB_SYNC_IMAGE:-quantmind-oss:latest}"
QUANTDB_SYNC_SCRIPT="${PROJECT_ROOT}/sync_quantdb.py"
QUANTDB_DATA_ROOT="${PROJECT_ROOT}/data"

if [[ -z "${TUSHARE_TOKEN:-}" && -r "${BASHRC_PATH}" ]]; then
  TOKEN_LINE="$(grep -m1 -E '^[[:space:]]*export[[:space:]]+TUSHARE_TOKEN=' "${BASHRC_PATH}" || true)"
  if [[ -n "${TOKEN_LINE}" ]]; then
    # Source only the trusted export line; never print or persist the token.
    source /dev/stdin <<<"${TOKEN_LINE}"
  fi
fi

SYNC_ENV_ARGS=()
if [[ -n "${TUSHARE_TOKEN:-}" ]]; then
  export TUSHARE_TOKEN
  SYNC_ENV_ARGS=(-e TUSHARE_TOKEN)
fi

"${DOCKER_BIN}" exec "${SYNC_ENV_ARGS[@]}" "${CONTAINER}" \
  python /app/scripts/data/maintenance/sync_daily_from_baostock.py \
  --apply --refresh-days 1

"${DOCKER_BIN}" exec "${CONTAINER}" \
  python /app/scripts/data/maintenance/build_model_qlib_features_incremental.py --apply

if [[ "${QUANTDB_SYNC_ENABLED,,}" == "true" ]]; then
  if [[ ! -f "${QUANTDB_SYNC_SCRIPT}" ]]; then
    echo "QuantDB 同步脚本不存在: ${QUANTDB_SYNC_SCRIPT}" >&2
    exit 1
  fi

  mkdir -p "${QUANTDB_DATA_ROOT}/quantdb"
  "${DOCKER_BIN}" run --rm \
    --env-file "${PROJECT_ROOT}/.env" \
    -e QM_QUANTDB_DATA_DIR=/data/quantdb \
    -v "${QUANTDB_SYNC_SCRIPT}:/app/sync_quantdb.py:ro" \
    -v "${QUANTDB_DATA_ROOT}:/data" \
    "${QUANTDB_SYNC_IMAGE}" \
    python /app/sync_quantdb.py
fi
