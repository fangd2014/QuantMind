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
QUANTDB_RUNTIME_ENV_FILE="${PROJECT_ROOT}/config/runtime.env"
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
    # 即使当天没有 key 或远端暂不可用，也先建立固定的五类目录，
    # 让挂载点结构稳定，避免前端把“尚未同步”误判为“路径不存在”。
    mkdir -p "${QUANTDB_DATA_ROOT}/quantdb/1_kline_data" \
      "${QUANTDB_DATA_ROOT}/quantdb/2_base_sector" \
      "${QUANTDB_DATA_ROOT}/quantdb/3_financial_data" \
      "${QUANTDB_DATA_ROOT}/quantdb/5_technical_derived" \
      "${QUANTDB_DATA_ROOT}/quantdb/6_ml_datasets"
    QUANTDB_ENV_ARGS=()
    # 管理页保存的 QuantDB key 位于 config/runtime.env；只有确认文件中
    # 存在非空 key 才使用它，避免“文件存在但 key 为空”吞掉后续回退。
    if [[ -r "${QUANTDB_RUNTIME_ENV_FILE}" ]] &&
      grep -q -E '^[[:space:]]*QUANTDB_API_KEY=[^[:space:]]' "${QUANTDB_RUNTIME_ENV_FILE}"; then
      QUANTDB_ENV_ARGS=(--env-file "${QUANTDB_RUNTIME_ENV_FILE}")
    elif [[ -r "${QUANTDB_ENV_FILE}" ]] &&
      grep -q -E '^[[:space:]]*QUANTDB_API_KEY=[^[:space:]]' "${QUANTDB_ENV_FILE}"; then
      QUANTDB_ENV_ARGS=(--env-file "${QUANTDB_ENV_FILE}")
    elif [[ -n "${QUANTDB_API_KEY:-}" ]]; then
      export QUANTDB_API_KEY
      QUANTDB_ENV_ARGS=(-e QUANTDB_API_KEY)
    else
      # 独立临时同步容器不能读取 quantmind 容器的环境。部署期间若 key
      # 已注入运行中的服务，安全地继承它，但不把值写入日志或项目文件。
      CONTAINER_QUANTDB_API_KEY="$(${DOCKER_BIN} exec "${CONTAINER}" printenv QUANTDB_API_KEY 2>/dev/null || true)"
      if [[ -n "${CONTAINER_QUANTDB_API_KEY}" ]]; then
        export QUANTDB_API_KEY="${CONTAINER_QUANTDB_API_KEY}"
        QUANTDB_ENV_ARGS=(-e QUANTDB_API_KEY)
      else
        echo "QuantDB 同步失败: runtime.env/.env/服务容器均未配置 QUANTDB_API_KEY" >&2
        FINAL_STATUS=1
      fi
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
