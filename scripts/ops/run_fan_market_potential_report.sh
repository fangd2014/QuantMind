#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${QUANTMIND_PROJECT_DIR:-/opt/quantmind}"
CONTAINER="${QUANTMIND_CONTAINER:-quantmind}"
ENV_FILE="${QUANTMIND_ENV_FILE:-${PROJECT_DIR}/.env}"
BASHRC_PATH="${QUANTMIND_ROOT_BASHRC:-/root/.bashrc}"
REPORT_SCRIPT="/app/scripts/analysis/fan_market_potential_report.py"
PREFLIGHT_FILE="/data/cache/fan-market/preflight.json"

if [[ ! -r "${ENV_FILE}" ]]; then
    echo "Environment file is not readable: ${ENV_FILE}" >&2
    exit 1
fi

read_env_value() {
    local key="$1"
    local value
    value="$(
        sed -n -E \
            "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*(.*)$/\\1/p" \
            "${ENV_FILE}" | tail -n 1 | tr -d '\r'
    )"
    value="${value#\"}"
    value="${value%\"}"
    value="${value#\'}"
    value="${value%\'}"
    printf '%s' "${value}"
}

WEB_HOOK="$(read_env_value WEB_HOOK)"
FAN_MARKET_WEB_HOOK="$(read_env_value FAN_MARKET_WEB_HOOK)"
if [[ -z "${FAN_MARKET_WEB_HOOK}" && -z "${WEB_HOOK}" ]]; then
    echo "FAN_MARKET_WEB_HOOK and WEB_HOOK are both empty" >&2
    exit 1
fi

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

PUBLIC_BASE_FROM_ENV="$(read_env_value FAN_MARKET_PUBLIC_BASE_URL)"
export WEB_HOOK
export FAN_MARKET_WEB_HOOK
export TUSHARE_TOKEN
export FAN_MARKET_PUBLIC_BASE_URL="${FAN_MARKET_PUBLIC_BASE_URL:-${PUBLIC_BASE_FROM_ENV:-http://192.168.5.10:18000}}"
DOCKER_BIN="$(command -v docker)"

"${DOCKER_BIN}" exec \
    -e WEB_HOOK \
    "${CONTAINER}" \
    python /app/scripts/data/maintenance/concept_rotation_preflight.py \
    --skip-update \
    --status-file "${PREFLIGHT_FILE}" \
    --send-feishu-alerts

exec "${DOCKER_BIN}" exec \
    -e WEB_HOOK \
    -e FAN_MARKET_WEB_HOOK \
    -e TUSHARE_TOKEN \
    -e FAN_MARKET_PUBLIC_BASE_URL \
    "${CONTAINER}" \
    python "${REPORT_SCRIPT}" \
    --preflight-file "${PREFLIGHT_FILE}" \
    --send-feishu "$@"
