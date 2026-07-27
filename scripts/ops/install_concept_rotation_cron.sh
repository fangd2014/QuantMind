#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${QUANTMIND_PROJECT_DIR:-/opt/quantmind}"
SCHEDULE="${QUANTMIND_CONCEPT_REPORT_SCHEDULE:-0 20 * * *}"
TAG="# quantmind-concept-rotation-report"
TIMEZONE_LINE="CRON_TZ=Asia/Shanghai"
LOG_FILE="${PROJECT_DIR}/logs/concept_rotation_report.log"
LOCK_FILE="/var/lock/quantmind-concept-rotation-report.lock"
RUNNER="${PROJECT_DIR}/scripts/ops/run_concept_rotation_report.sh"

if [[ ! -x "${RUNNER}" ]]; then
    echo "Report runner is not executable: ${RUNNER}" >&2
    exit 1
fi

mkdir -p "${PROJECT_DIR}/logs"
FLOCK_BIN="$(command -v flock)"
JOB="${SCHEDULE} ${FLOCK_BIN} -n ${LOCK_FILE} ${RUNNER} >> ${LOG_FILE} 2>&1 ${TAG}"
EXISTING="$(crontab -l 2>/dev/null || true)"
FILTERED="$(
    printf '%s\n' "${EXISTING}" \
        | grep -Fv "${TAG}" \
        | grep -Fvx "${TIMEZONE_LINE}" \
        || true
)"
{
    printf '%s\n' "${FILTERED}"
    printf '%s\n' "${TIMEZONE_LINE}"
    printf '%s\n' "${JOB}"
} | sed '/^[[:space:]]*$/d' | crontab -

echo "Installed daily concept-rotation report: ${JOB}"
echo "Cron timezone is pinned to Asia/Shanghai."
