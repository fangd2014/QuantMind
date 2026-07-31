#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${QUANTMIND_CONTAINER:-quantmind}"
BASHRC_PATH="${QUANTMIND_ROOT_BASHRC:-/root/.bashrc}"
DOCKER_BIN="$(command -v docker)"

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
