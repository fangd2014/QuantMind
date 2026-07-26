#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${QUANTMIND_PROJECT_DIR:-/opt/quantmind}"
SCHEDULE="${QUANTMIND_DATA_UPDATE_SCHEDULE:-30 18 * * 1-5}"
TUSHARE_SCHEDULE="${QUANTMIND_TUSHARE_UPDATE_SCHEDULE:-15 19 * * 1-5}"
CONTAINER="${QUANTMIND_CONTAINER:-quantmind}"
TAG="# quantmind-data-update"
TUSHARE_TAG="# quantmind-tushare-asset-update"
LOG_FILE="${PROJECT_DIR}/logs/data_update.log"
TUSHARE_LOG_FILE="${PROJECT_DIR}/logs/tushare_asset_update.log"
LOCK_FILE="/var/lock/quantmind-data-update.lock"

mkdir -p "${PROJECT_DIR}/logs"

DOCKER_BIN="$(command -v docker)"
FLOCK_BIN="$(command -v flock)"
JOB="${SCHEDULE} ${FLOCK_BIN} -n ${LOCK_FILE} ${DOCKER_BIN} exec ${CONTAINER} python /app/scripts/data/maintenance/sync_daily_from_baostock.py --apply >> ${LOG_FILE} 2>&1 ${TAG}"
TUSHARE_JOB="${TUSHARE_SCHEDULE} ${PROJECT_DIR}/scripts/ops/run_tushare_asset_sync.sh >> ${TUSHARE_LOG_FILE} 2>&1 ${TUSHARE_TAG}"

EXISTING="$(crontab -l 2>/dev/null || true)"
FILTERED="$(printf '%s\n' "${EXISTING}" | grep -Fv "${TAG}" | grep -Fv "${TUSHARE_TAG}" || true)"
{
    printf '%s\n' "${FILTERED}"
    printf '%s\n' "${JOB}"
    printf '%s\n' "${TUSHARE_JOB}"
} | sed '/^[[:space:]]*$/d' | crontab -

echo "Installed: ${JOB}"
echo "Installed: ${TUSHARE_JOB}"
