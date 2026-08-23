#!/usr/bin/env bash
set -uo pipefail

CONTAINER="${QUANTMIND_CONTAINER:-quantmind}"
BASHRC_PATH="${QUANTMIND_ROOT_BASHRC:-/root/.bashrc}"
DOCKER_BIN="${QUANTMIND_DOCKER_BIN:-$(command -v docker || true)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
QUANTDB_SYNC_ENABLED="${QUANTDB_SYNC_ENABLED:-true}"
QUANTDB_SYNC_IMAGE="${QUANTDB_SYNC_IMAGE:-quantmind-oss:latest}"
QUANTDB_SYNC_SCRIPT="${PROJECT_ROOT}/sync_quantdb.py"
QUANTDB_DATA_ROOT="${PROJECT_ROOT}/data"
QUANTDB_ENV_FILE="${PROJECT_ROOT}/.env"
FINAL_STATUS=0

if [[ -z "${DOCKER_BIN}" ]]; then
  echo "未找到 docker 命令" >&2
  exit 127
fi

if [[ -z "${TUSHARE_TOKEN:-}" && -r "${BASHRC_PATH}" ]]; then
  TOKEN_LINE="$(grep -m1 -E '^[[:space:]]*export[[:space:]]+TUSHARE_TOKEN=' "${BASHRC_PATH}" || true)"
  if [[ -n "${TOKEN_LINE}" ]]; then
    # Source only the trusted export line; never print or persist the token.
    source /dev/stdin <<<"${TOKEN_LINE}"
  fi
fi

if [[ -n "${TUSHARE_TOKEN:-}" ]]; then
  export TUSHARE_TOKEN
fi

run_baostock_sync() {
  if [[ -n "${TUSHARE_TOKEN:-}" ]]; then
    "${DOCKER_BIN}" exec -e TUSHARE_TOKEN "${CONTAINER}" \
      python /app/scripts/data/maintenance/sync_daily_from_baostock.py \
      --apply --refresh-days 1
  else
    "${DOCKER_BIN}" exec "${CONTAINER}" \
      python /app/scripts/data/maintenance/sync_daily_from_baostock.py \
      --apply --refresh-days 1
  fi
}

case "${QUANTDB_SYNC_ENABLED}" in
true | TRUE | True | 1 | yes | YES | Yes)
  echo "[QuantDB] 开始同步本地数据目录"
  if [[ ! -f "${QUANTDB_SYNC_SCRIPT}" ]]; then
    echo "QuantDB 同步脚本不存在: ${QUANTDB_SYNC_SCRIPT}" >&2
    FINAL_STATUS=1
  else
    QUANTDB_ENV_ARGS=()
    if [[ -r "${QUANTDB_ENV_FILE}" ]]; then
      QUANTDB_ENV_ARGS=(--env-file "${QUANTDB_ENV_FILE}")
    elif [[ -n "${QUANTDB_API_KEY:-}" ]]; then
      QUANTDB_ENV_ARGS=(-e QUANTDB_API_KEY)
    else
      echo "QuantDB 同步失败: ${QUANTDB_ENV_FILE} 不可读且 QUANTDB_API_KEY 未配置" >&2
      FINAL_STATUS=1
    fi

    if [[ ${#QUANTDB_ENV_ARGS[@]} -gt 0 ]]; then
      mkdir -p "${QUANTDB_DATA_ROOT}/quantdb"
      if ! "${DOCKER_BIN}" run --rm \
        "${QUANTDB_ENV_ARGS[@]}" \
        -e QM_QUANTDB_DATA_DIR=/data/quantdb \
        -v "${QUANTDB_SYNC_SCRIPT}:/app/sync_quantdb.py:ro" \
        -v "${QUANTDB_DATA_ROOT}:/data" \
        "${QUANTDB_SYNC_IMAGE}" \
        python /app/sync_quantdb.py; then
        echo "[QuantDB] 同步失败，继续执行独立的 Baostock 数据链" >&2
        FINAL_STATUS=1
      fi
    fi
  fi
  ;;
*)
  echo "[QuantDB] 已通过 QUANTDB_SYNC_ENABLED 禁用"
  ;;
esac

echo "[Baostock] 开始同步 stock_daily_latest"
if run_baostock_sync; then
  echo "[Features] 开始增量构建 Qlib 特征"
  if ! "${DOCKER_BIN}" exec "${CONTAINER}" \
    python /app/scripts/data/maintenance/build_model_qlib_features_incremental.py --apply; then
    echo "[Features] 增量构建失败" >&2
    FINAL_STATUS=1
  fi
else
  echo "[Baostock] 同步失败，跳过依赖它的特征构建" >&2
  FINAL_STATUS=1
fi

if [[ ${FINAL_STATUS} -ne 0 ]]; then
  echo "每日数据任务存在失败步骤，请检查上述日志" >&2
fi
exit "${FINAL_STATUS}"
