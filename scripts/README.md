# Scripts Directory

Unified scripts directory for the QuantMind project, organized by functionality.

## Directory Structure

- `scripts/build/`: Build and deployment scripts.
  - `scripts/build/core/`: Docker image build scripts (e.g., `build_core_images.sh`).
  - `scripts/build/gke/`: GKE specific deployment scripts (moved from `backend/scripts/`).
- `scripts/training/`: AI Model training and promotion scripts (e.g., `train_custom_lgbm_duckdb.py`).
- `scripts/data/`: Data management.
  - `scripts/data/ingestion/`: Inbound data scripts (backfill, supplement, daily sync).
  - `scripts/data/processing/`: Feature extraction, generation, and formatting.
- `scripts/ops/`: Operations, maintenance, and CI.
  - `scripts/ops/db/`: Database schema audit, verification, and cleanup.
  - `scripts/ops/ci/`: CI quality gates and health checks.
- `scripts/pipeline/`: Production runners and daily inference pipelines.
- `scripts/frontend/`: Frontend build and asset management scripts.
- `scripts/redis/`: Redis and market data tasks.

## Key Scripts

### Training & Model Management
- `scripts/training/train_custom_lgbm_duckdb.py`: Custom LightGBM training with DuckDB features.
- `scripts/training/promote_model.py`: Promote a candidate model to production.

### Data Pipelines
- `scripts/pipeline/run_daily_fusion_pipeline.py`: Daily dual-model fusion pipeline.
- `scripts/pipeline/run_engine_margin_topk_2024.py`: 调用 `quantmind-engine` 的 Qlib 回测接口，执行 2024 年固定两融股票池的多空 TopK 回测，并导出 `summary.json / equity_curve.csv / trades.csv`。
- `scripts/analysis/concept_rotation_report.py`: 从本地 `stock_daily_latest` 计算申万 2021 版一级行业扩散度、RRG、龙头确认、次日条件候选，以及领先区内“控盘量价代理 + 洗盘/开始拉升”关注股（硬限制最多 10 只、单行业最多 2 只），输出 JSON/CSV/PDF/交互 HTML；行业分类和最新成分来自 Tushare 并带 7 日本地缓存。`--max-control-picks` 可调低数量但不会突破 10 只；该指标不代表真实机构持仓。
- `scripts/ops/run_concept_rotation_report.sh`: 从服务器 `.env` 安全读取 `WEB_HOOK`、从 `/root/.bashrc` 读取 `TUSHARE_TOKEN`，在 `quantmind` 容器内生成日报并推送飞书摘要、交互四象限和 PDF 下载链接。
- `scripts/data/maintenance/concept_rotation_preflight.py`: 每次轮动日报前先执行 Baostock 增量更新，并校验应有交易日、截面完整度、最近 26 日、代码/OHLC/成交量与极端涨跌；发现阻断异常时对最近 35 个日历日做同源幂等回刷并复检，仍失败则阻断推荐并发送飞书告警。不会自动删行、猜测复权因子或用前值填充价格。
- `scripts/analysis/leading_control_backtest.py`: 对“申万领先区 + 控盘量价代理 + 洗盘/开始拉升”执行最近 60 个完整自然月的点时回测。月末收盘出信号，次月首开买入、再下一月首开调仓；信号使用原始 OHLC，收益使用复权价格，历史申万成分按 `[in_date, out_date)` 过滤，沪深300为基准。正式运行若不是连续 60 月、任一月信号失败或缺年度快照会拒绝生成报告；输出 JSON、CSV、独立 HTML 和中文 PDF。
- `scripts/ops/run_leading_control_backtest.sh`: 在服务器读取 `/root/.bashrc` 中既有的 `TUSHARE_TOKEN`，通过 `quantmind` 容器运行上述回测，并将报告写入 Nginx 已暴露的 `/data/uploads/reports/leading-control-backtest`。
- `scripts/ops/install_concept_rotation_cron.sh`: 幂等安装每天 20:00 的申万行业轮动日报宿主机 cron 任务。
- `scripts/data/processing/sync_margin_instruments.py`: 将 [融资融券.xlsx](/Users/qusong/git/quantmind/data/融资融券.xlsx) 同步为 `db/qlib_data/instruments/margin.txt`，供回测直接复用固定两融股票池。
- `scripts/data/processing/generate_rolling_pred_online.py`: Generate rolling predictions for online use.
- `scripts/data/processing/merge_features_with_labels.py`: 将 `db/feature_snapshots/features_YYYY.parquet`（默认）或 `db/feature_snapshots/model_features_YYYY.parquet`（兼容）与 `db/csmar_data.duckdb` 的次日收益率标签做高性能合并，输出更新后的 `db/feature_snapshots/model_features_YYYY.parquet`，并自动执行去重与质量校验报告生成。
- `scripts/data/ingestion/update_qlib_sh_sz_from_baostock.py`: 使用 Baostock 对 `db/qlib_data` 做 SH/SZ 日线增量补数（默认 dry-run，`--apply` 才写入）。会同步更新 `calendars/day.txt` 与 `instruments/*.txt` 的 SH/SZ 截止日期。BJ 不在该脚本支持范围内。
- `scripts/data/ingestion/sync_market_data_daily_from_baostock.py`: 使用 Baostock 同步 `market_data_daily`，会自动按表结构选择“基础字段模式”或 `feature_0..N` 模式写入（默认 dry-run，`--apply` 写库）。
- `scripts/data/maintenance/sync_etf_futures_from_tushare.py`: 使用环境变量 `TUSHARE_TOKEN` 同步 ETF 日线/复权因子、CFFEX 的 IF/IH/IC/IM 合约日线与主力映射。数据写入独立 PostgreSQL 表并按年导出到 `db/market_ext`，不会混入股票表或股票 Qlib provider。
- `scripts/training/sync_feature_catalog_to_db.py`: 将 `config/features/model_training_feature_catalog_v1.json` 同步到 PostgreSQL 特征注册表（`qm_feature_category/qm_feature_definition/qm_feature_set_version/qm_feature_set_item`），默认会把同步版本置为 `active` 并将其他 active 版本置为 `inactive`；支持 `--dry-run` 预检。

### Redis & Market Data
- `scripts/redis/pull_remote_rdb.sh`: 拉取远端 Redis 的 RDB 快照到本地 `redis/data/dump.rdb`。
- `scripts/redis/market_data_to_redis.py`: 旧版外部采集脚本（已停用，不再作为服务器默认方案）。当前生产改为容器内 `market-redis`，由外部推送器按 `docs/行情快照写入规范.md` 写入 `market:snapshot:{symbol}`。

示例：

```bash
source .venv/bin/activate
python scripts/pipeline/run_engine_margin_topk_2024.py \
  --pred-path models/production/05_T5_Selected/pred.pkl \
  --topk 50 \
  --short-topk 50
```

说明：
- 该脚本现在会同时做两层约束：
  - 先按 `db/qlib_data/instruments/margin.txt` 过滤信号
  - 再把回测请求中的 `universe` 也固定为 `db/qlib_data/instruments/margin.txt`

同步两融股票池：

```bash
source .venv/bin/activate
python scripts/data/processing/sync_margin_instruments.py
```

增量补齐 SH/SZ 到 Qlib：

```bash
source .venv/bin/activate
python scripts/data/ingestion/update_qlib_sh_sz_from_baostock.py          # dry-run
python scripts/data/ingestion/update_qlib_sh_sz_from_baostock.py --apply  # 写入
python scripts/data/ingestion/sync_market_data_daily_from_baostock.py     # dry-run
python scripts/data/ingestion/sync_market_data_daily_from_baostock.py --apply

# Tushare ETF/股指期货权限冒烟检查（不写数据）
python scripts/data/maintenance/sync_etf_futures_from_tushare.py --smoke-test

# 首次从 2016 年回填；之后自动按数据库水位增量更新
python scripts/data/maintenance/sync_etf_futures_from_tushare.py --apply

# 合并特征与训练标签（默认读取 features_YYYY，覆盖生成 model_features_YYYY）
python scripts/data/processing/merge_features_with_labels.py --force

# 仅处理指定年份
python scripts/data/processing/merge_features_with_labels.py --years 2025 --force
```

## Operations & CI
- `scripts/ops/ci/p2_ci_quality_gate.py`: CI quality gate for isolation and smoke tests.
- `scripts/ops/health_check.py`: General system health check.
- `scripts/ops/db/cleanup_optimization_backtest_pollution.py`: One-time cleanup for optimization-generated sub-backtests accidentally mixed into normal backtest history (`dry-run` by default, add `--apply` to delete).
- `scripts/db_init/add_api_keys_secret_ciphertext.sql`: 为 `api_keys` 表补充 `secret_ciphertext` 字段（幂等），用于“用户本人可见的最新私钥缓存”。
- `scripts/postgres/research_candidate_incremental.sql`: 为投研平台建立 `qm_research_candidate_snapshot` 候选快照表，并提供基于 `qm_model_inference_runs + engine_signal_scores` 的增量导入函数；当 `stock_selection` 缺失时自动回退到 `stock_daily_latest`。

## Usage Guide

All scripts should be run from the project root directory using the unified virtual environment:

```bash
source .venv/bin/activate
python scripts/<category>/<script_name>.py
```

For more details on specific scripts, refer to the documentation in their respective functional directories (if available) or the comments within the script files.
