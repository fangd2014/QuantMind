#!/usr/bin/env python3
"""将 QuantDB 订阅数据增量同步到本地数据目录。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from quantdb_sdk import QuantDBClient

DEFAULT_DATA_DIR = "/data/quantdb"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10

V2_DATASETS = (
    {"category_id": "1", "sub_category": "daily_unadjusted", "dir": "1_kline_data"},
    {"category_id": "1", "sub_category": "daily_forward", "dir": "1_kline_data"},
    {"category_id": "1", "sub_category": "daily_backward", "dir": "1_kline_data"},
    {"category_id": "1", "sub_category": "index_daily", "dir": "1_kline_data"},
    {"category_id": "2", "sub_category": "margin_trading", "dir": "2_base_sector"},
    {"category_id": "5", "sub_category": "valuation", "dir": "5_technical_derived"},
    {"category_id": "5", "sub_category": "technical_indicators", "dir": "5_technical_derived"},
    {"category_id": "5", "sub_category": "market_sentiment", "dir": "5_technical_derived"},
    {"category_id": "6", "sub_category": "features_daily", "dir": "6_ml_datasets"},
    {"category_id": "6", "sub_category": "l1_factors", "dir": "6_ml_datasets"},
    {"category_id": "6", "sub_category": "l2_factors", "dir": "6_ml_datasets"},
)

V1_DATASETS = (
    {"category_id": "1", "sub_category": "min1_kline", "dir": "1_kline_data"},
    {"category_id": "1", "sub_category": "min5_kline", "dir": "1_kline_data"},
    {"category_id": "2", "sub_category": "sector_concept", "dir": "2_base_sector"},
    {"category_id": "2", "sub_category": "instrument_detail", "dir": "2_base_sector"},
    {"category_id": "2", "sub_category": "index_weights", "dir": "2_base_sector"},
    {"category_id": "2", "sub_category": "trading_calendar", "dir": "2_base_sector"},
    {"category_id": "3", "sub_category": "balance", "dir": "3_financial_data"},
    {"category_id": "3", "sub_category": "income", "dir": "3_financial_data"},
    {"category_id": "3", "sub_category": "cashflow", "dir": "3_financial_data"},
    {"category_id": "3", "sub_category": "capital", "dir": "3_financial_data"},
    {"category_id": "3", "sub_category": "pershare_index", "dir": "3_financial_data"},
    {"category_id": "3", "sub_category": "dividend_factors", "dir": "3_financial_data"},
    {"category_id": "3", "sub_category": "holder_num", "dir": "3_financial_data"},
)

ALL_DATASETS = V2_DATASETS + V1_DATASETS
REQUIRED_ROOT_DIRS = tuple(sorted({item["dir"] for item in ALL_DATASETS}))


def resolve_data_dir() -> Path:
    configured = os.environ.get("QM_QUANTDB_DATA_DIR", DEFAULT_DATA_DIR).strip()
    if not configured:
        raise RuntimeError("QM_QUANTDB_DATA_DIR 不能为空")
    return Path(configured)


def require_api_key() -> str:
    api_key = os.environ.get("QUANTDB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("QUANTDB_API_KEY 未配置")
    return api_key


def ensure_layout(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for directory in REQUIRED_ROOT_DIRS:
        (data_dir / directory).mkdir(parents=True, exist_ok=True)


def configure_logging(data_dir: Path) -> logging.Logger:
    logger = logging.getLogger("quantdb-sync")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        data_dir / "sync.log", mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def make_client(api_key: str) -> QuantDBClient:
    return QuantDBClient(api_key=api_key, timeout=(15, 300), max_retries=5)


def count_local_files(data_dir: Path, dataset: dict[str, str]) -> int:
    local_path = data_dir / dataset["dir"] / dataset["sub_category"]
    if not local_path.exists():
        return 0
    return sum(1 for _ in local_path.rglob("*.parquet"))


def get_remote_count(
    client: QuantDBClient, dataset: dict[str, str], logger: logging.Logger
) -> tuple[int | None, str | None]:
    try:
        manifest = client.query_manifest(
            category_id=dataset["category_id"],
            sub_category=dataset["sub_category"],
        )
        if manifest is None:
            raise RuntimeError("远端 manifest 为空")
        if isinstance(manifest, dict):
            for key in ("files", "items", "data"):
                entries = manifest.get(key)
                if isinstance(entries, list):
                    return len(entries), None
        return len(manifest), None
    except Exception as exc:  # SDK 将网络和服务端错误包装为多个异常类型
        logger.error(
            "[FAIL] 无法获取 %s/%s manifest: %s",
            dataset["category_id"],
            dataset["sub_category"],
            exc,
        )
        return None, str(exc)


def sync_single_dataset(
    client: QuantDBClient,
    api_key: str,
    data_dir: Path,
    dataset: dict[str, str],
    logger: logging.Logger,
    dry_run: bool,
) -> tuple[dict[str, Any], QuantDBClient]:
    sub_category = dataset["sub_category"]
    local_count = count_local_files(data_dir, dataset)
    remote_count, manifest_error = get_remote_count(client, dataset, logger)
    result: dict[str, Any] = {
        "dataset": f"{dataset['category_id']}/{sub_category}",
        "local": local_count,
        "remote": remote_count,
        "status": "error",
        "downloaded": 0,
        "error": manifest_error,
    }

    if remote_count is None or remote_count <= 0:
        result["error"] = manifest_error or "远端 manifest 没有文件"
        return result, client

    if local_count >= remote_count:
        result["status"] = "up_to_date"
        result["error"] = None
        logger.info("[OK] %s: %s/%s 已是最新", sub_category, local_count, remote_count)
        return result, client

    logger.info(
        "[SYNC] %s: %s/%s，缺少 %s 个文件",
        sub_category,
        local_count,
        remote_count,
        remote_count - local_count,
    )
    if dry_run:
        result["status"] = "dry_run"
        result["error"] = None
        return result, client

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            sync_result = client.sync_dataset(dataset=sub_category, save_dir=str(data_dir))
            downloaded = (
                len(sync_result.get("downloaded", []))
                if isinstance(sync_result, dict)
                else 0
            )
            final_count = count_local_files(data_dir, dataset)
            if final_count < remote_count:
                raise RuntimeError(
                    f"同步返回成功但本地仅有 {final_count}/{remote_count} 个文件"
                )
            result.update(
                {
                    "local": final_count,
                    "status": "synced",
                    "downloaded": downloaded,
                    "error": None,
                }
            )
            logger.info(
                "[DONE] %s: 下载 %s 个文件，本地共 %s 个",
                sub_category,
                downloaded,
                final_count,
            )
            return result, client
        except Exception as exc:  # SDK 异常类型随版本变化
            result["error"] = str(exc)
            if attempt == MAX_RETRIES:
                logger.error(
                    "[FAIL] %s: 重试 %s 次后仍失败: %s",
                    sub_category,
                    MAX_RETRIES,
                    exc,
                )
                break
            logger.warning(
                "[RETRY] %s: 第 %s 次失败，%s 秒后重试: %s",
                sub_category,
                attempt,
                RETRY_DELAY_SECONDS,
                exc,
            )
            time.sleep(RETRY_DELAY_SECONDS)
            client = make_client(api_key)

    return result, client


def show_status(
    client: QuantDBClient, data_dir: Path, logger: logging.Logger
) -> bool:
    logger.info("QuantDB 数据目录: %s", data_dir)
    logger.info("%-48s %12s %12s %s", "数据集", "本地", "远端", "状态")
    success = True
    for dataset in ALL_DATASETS:
        local_count = count_local_files(data_dir, dataset)
        remote_count, _ = get_remote_count(client, dataset, logger)
        if remote_count is None:
            state = "ERROR"
            success = False
        elif local_count >= remote_count > 0:
            state = "OK"
        else:
            state = f"缺 {max(remote_count - local_count, 0)}"
            success = False
        logger.info(
            "%-48s %12s %12s %s",
            f"{dataset['dir']}/{dataset['sub_category']}",
            local_count,
            remote_count if remote_count is not None else "-",
            state,
        )
    return success


def select_datasets(only: str | None, dataset_name: str | None) -> tuple[dict[str, str], ...]:
    if dataset_name:
        selected = tuple(
            item for item in ALL_DATASETS if item["sub_category"] == dataset_name
        )
        if not selected:
            raise RuntimeError(f"未找到数据集: {dataset_name}")
        return selected
    if only == "v1":
        return V1_DATASETS
    if only == "v2":
        return V2_DATASETS
    return ALL_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantDB 全量/增量数据同步工具")
    parser.add_argument("--only", choices=("v1", "v2"), help="仅同步 V1 或 V2 数据集")
    parser.add_argument("--dataset", help="仅同步指定的 sub_category")
    parser.add_argument("--dry-run", action="store_true", help="检查差异但不下载")
    parser.add_argument("--status", action="store_true", help="显示全部数据集状态")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data_dir = resolve_data_dir()
        ensure_layout(data_dir)
        logger = configure_logging(data_dir)
        api_key = require_api_key()
        client = make_client(api_key)
        datasets = select_datasets(args.only, args.dataset)
    except Exception as exc:
        logging.getLogger("quantdb-sync").error("QuantDB 初始化失败: %s", exc)
        return 2

    logger.info("QuantDB 数据目录: %s", data_dir)
    if args.status:
        return 0 if show_status(client, data_dir, logger) else 1

    started_at = time.time()
    results = []
    for index, dataset in enumerate(datasets, 1):
        logger.info(
            "--- [%s/%s] %s ---", index, len(datasets), dataset["sub_category"]
        )
        result, client = sync_single_dataset(
            client, api_key, data_dir, dataset, logger, args.dry_run
        )
        results.append(result)

    elapsed_seconds = round(time.time() - started_at, 1)
    errors = [result for result in results if result["status"] == "error"]
    record = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "results": results,
    }
    with (data_dir / "sync_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "同步结束: 数据集=%s，错误=%s，耗时=%.1fs",
        len(results),
        len(errors),
        elapsed_seconds,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
