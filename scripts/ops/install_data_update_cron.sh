#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${QUANTMIND_PROJECT_DIR:-/opt/quantmind}"
SCHEDULE="${QUANTMIND_DATA_UPDATE_SCHEDULE:-30 18 * * 1-5}"
TUSHARE_SCHEDULE="${QUANTMIND_TUSHARE_UPDATE_SCHEDULE:-15 19 * * 1-5}"
TAG="# quantmind-data-update"
TUSHARE_TAG="# quantmind-tushare-asset-update"
LOG_FILE="${PROJECT_DIR}/logs/data_update.log"
TUSHARE_LOG_FILE="${PROJECT_DIR}/logs/tushare_asset_update.log"
LOCK_FILE="/var/lock/quantmind-data-update.lock"
RUNNER="${PROJECT_DIR}/scripts/ops/run_daily_data_and_features.sh"

mkdir -p "${PROJECT_DIR}/logs"

FLOCK_BIN="$(command -v flock)"
if [[ ! -x "${RUNNER}" ]]; then
    echo "Daily data/feature runner is not executable: ${RUNNER}" >&2
    exit 1
fi
JOB="${SCHEDULE} ${FLOCK_BIN} -n ${LOCK_FILE} ${RUNNER} >> ${LOG_FILE} 2>&1 ${TAG}"
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
