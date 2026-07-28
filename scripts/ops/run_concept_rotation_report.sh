#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${QUANTMIND_PROJECT_DIR:-/opt/quantmind}"
CONTAINER="${QUANTMIND_CONTAINER:-quantmind}"
ENV_FILE="${QUANTMIND_ENV_FILE:-${PROJECT_DIR}/.env}"
BASHRC_PATH="${QUANTMIND_ROOT_BASHRC:-/root/.bashrc}"

if [[ ! -r "${ENV_FILE}" ]]; then
    echo "Environment file is not readable: ${ENV_FILE}" >&2
    exit 1
fi

# Read only WEB_HOOK instead of sourcing the whole .env file. This avoids
# executing shell syntax from configuration and keeps the secret out of logs
# and Docker CLI arguments.
WEB_HOOK="$(
    sed -n -E 's/^[[:space:]]*WEB_HOOK[[:space:]]*=[[:space:]]*(.*)$/\1/p' \
        "${ENV_FILE}" | tail -n 1 | tr -d '\r'
)"
WEB_HOOK="${WEB_HOOK#\"}"
WEB_HOOK="${WEB_HOOK%\"}"
WEB_HOOK="${WEB_HOOK#\'}"
WEB_HOOK="${WEB_HOOK%\'}"
if [[ -z "${WEB_HOOK}" ]]; then
    echo "WEB_HOOK is empty or absent in ${ENV_FILE}" >&2
    exit 1
fi

TOKEN_LINE="$(grep -m1 -E '^[[:space:]]*export[[:space:]]+TUSHARE_TOKEN=' "${BASHRC_PATH}" || true)"
if [[ -z "${TOKEN_LINE}" ]]; then
    echo "TUSHARE_TOKEN export was not found in ${BASHRC_PATH}" >&2
    exit 1
fi
# Source only the trusted token export, bypassing the non-interactive return in
# /root/.bashrc without logging or copying the credential into project files.
source /dev/stdin <<<"${TOKEN_LINE}"
if [[ -z "${TUSHARE_TOKEN:-}" ]]; then
    echo "TUSHARE_TOKEN is empty" >&2
    exit 1
fi

PUBLIC_BASE_FROM_ENV="$(
    sed -n -E \
        's/^[[:space:]]*CONCEPT_REPORT_PUBLIC_BASE_URL[[:space:]]*=[[:space:]]*(.*)$/\1/p' \
        "${ENV_FILE}" | tail -n 1 | tr -d '\r'
)"
PUBLIC_BASE_FROM_ENV="${PUBLIC_BASE_FROM_ENV#\"}"
PUBLIC_BASE_FROM_ENV="${PUBLIC_BASE_FROM_ENV%\"}"
PUBLIC_BASE_FROM_ENV="${PUBLIC_BASE_FROM_ENV#\'}"
PUBLIC_BASE_FROM_ENV="${PUBLIC_BASE_FROM_ENV%\'}"

export WEB_HOOK
export TUSHARE_TOKEN
export CONCEPT_REPORT_PUBLIC_BASE_URL="${CONCEPT_REPORT_PUBLIC_BASE_URL:-${PUBLIC_BASE_FROM_ENV:-http://192.168.5.10:18000}}"
DOCKER_BIN="$(command -v docker)"

exec "${DOCKER_BIN}" exec \
    -e WEB_HOOK \
    -e TUSHARE_TOKEN \
    -e CONCEPT_REPORT_PUBLIC_BASE_URL \
    "${CONTAINER}" \
    python /app/scripts/analysis/concept_rotation_report.py \
    --send-feishu "$@"
