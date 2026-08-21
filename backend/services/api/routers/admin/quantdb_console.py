"""管理员 - QuantDB SDK 控制台

GET  /api/v1/admin/data-platform/quantdb/info            SDK 安装/连接状态、账户与流量配额
GET  /api/v1/admin/data-platform/quantdb/config          API Key 配置状态（脱敏）
POST /api/v1/admin/data-platform/quantdb/config          写入 API Key（落盘 config/runtime.env）
GET  /api/v1/admin/data-platform/quantdb/catalog         数据集目录（6 大类 / 28 数据集）+ 本地落盘统计
GET  /api/v1/admin/data-platform/quantdb/preview         数据集预览（本地 parquet 优先，零流量）
GET  /api/v1/admin/data-platform/quantdb/remote-meta     远端各数据集日期范围/行数
GET  /api/v1/admin/data-platform/quantdb/diff            远端 vs 本地差异（哪些数据集有更新）
POST /api/v1/admin/data-platform/quantdb/sync-datasets   按数据集同步（后台任务，可选写 PG / 重建 Qlib）
GET  /api/v1/admin/data-platform/quantdb/sync-jobs       同步任务列表
GET  /api/v1/admin/data-platform/quantdb/sync-jobs/{id}  单个任务进度
POST /api/v1/admin/data-platform/quantdb/sync-jobs/{id}/cancel  取消同步任务
POST /api/v1/admin/data-platform/quantdb/query-kline     远端 K 线查询（消耗流量）
GET  /api/v1/admin/data-platform/quantdb/stock-list      远端股票列表
GET  /api/v1/admin/data-platform/quantdb/calendar        远端交易日历
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.services.api.user_app.middleware.auth import require_admin
from backend.shared.runtime_secrets import (
    QUANTDB_CONFIG_HINT,
    get_quantdb_api_key,
    mask_secret,
    set_secret,
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_admin)])  # 路由器级认证兜底

MAX_PREVIEW_ROWS = 200
MAX_MANIFEST_FILES = 500
MAX_SYMBOL_CHOICES = 500


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# 数据集目录
# ---------------------------------------------------------------------------
# layout 决定落盘形态与预览读法：
#   partition — dt=YYYYMMDD/data.parquet 按交易日分区
#   symbol    — 每标的一个 {SYMBOL}.parquet
#   single    — 整个数据集一个 parquet
Layout = Literal["partition", "symbol", "single"]


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    name: str
    category_id: str
    group: str
    rel_dir: str
    layout: Layout
    note: str = ""


GROUPS: list[dict[str, str]] = [
    {"id": "kline", "name": "K线行情", "category_id": "1"},
    {"id": "base_sector", "name": "基础板块", "category_id": "2"},
    {"id": "financial", "name": "财务数据", "category_id": "3"},
    {"id": "bond_etf", "name": "债券/ETF", "category_id": "4"},
    {"id": "technical", "name": "技术衍生", "category_id": "5"},
    {"id": "ml", "name": "ML数据集", "category_id": "6"},
]

DATASETS: tuple[DatasetSpec, ...] = (
    # 1 K线行情
    DatasetSpec("daily_forward", "日线前复权", "1", "kline", "1_kline_data/daily_forward", "partition", "训练/回测主用"),
    DatasetSpec("daily_backward", "日线后复权", "1", "kline", "1_kline_data/daily_backward", "partition"),
    DatasetSpec("daily_unadjusted", "日线不复权", "1", "kline", "1_kline_data/daily_unadjusted", "partition", "amount/volume 单位在 20260721 切换"),
    DatasetSpec("index_daily", "指数日线", "1", "kline", "1_kline_data/index_daily", "partition"),
    DatasetSpec("min5_kline", "5分钟线", "1", "kline", "1_kline_data/min5_kline", "symbol"),
    DatasetSpec("min1_kline", "1分钟线", "1", "kline", "1_kline_data/min1_kline", "symbol", "体积大，按需同步"),
    DatasetSpec("tick_data", "Tick逐笔", "1", "kline", "1_kline_data/tick_data", "partition", "流量消耗极高"),
    # 2 基础板块
    DatasetSpec("instrument_detail", "个股详情", "2", "base_sector", "2_base_sector/instrument_detail", "single", "152 列基本面快照"),
    DatasetSpec("sector_concept", "板块概念", "2", "base_sector", "2_base_sector/sector_concept", "single"),
    DatasetSpec("index_weights", "指数权重", "2", "base_sector", "2_base_sector/index_weights", "symbol", "沪深300/中证500/1000 等"),
    DatasetSpec("trading_calendar", "交易日历", "2", "base_sector", "2_base_sector/trading_calendar", "single"),
    DatasetSpec("margin_trading", "融资融券", "2", "base_sector", "2_base_sector/margin_trading", "partition"),
    DatasetSpec("hsgt_north", "北向资金(季度)", "2", "base_sector", "2_base_sector/hsgt_north", "partition", "2024-08 起北向个股改季度披露，每季度末+第5交易日抓取，symbol 6位格式"),
    DatasetSpec("hsgt_north_daily", "北向资金日频(akshare)", "2", "base_sector", "2_base_sector/hsgt_north/daily_freq", "symbol", "2017-03~2024-08 北向持股日频，akshare逐股拉取"),
    # 3 财务数据
    DatasetSpec("balance", "资产负债表", "3", "financial", "3_financial_data/balance", "symbol"),
    DatasetSpec("income", "利润表", "3", "financial", "3_financial_data/income", "symbol"),
    DatasetSpec("cashflow", "现金流量表", "3", "financial", "3_financial_data/cashflow", "symbol"),
    DatasetSpec("capital", "股本结构", "3", "financial", "3_financial_data/capital", "symbol"),
    DatasetSpec("pershare_index", "每股指标", "3", "financial", "3_financial_data/pershare_index", "symbol"),
    DatasetSpec("dividend_factors", "分红因子", "3", "financial", "3_financial_data/dividend_factors", "symbol"),
    DatasetSpec("holder_num", "股东户数", "3", "financial", "3_financial_data/holder_num", "symbol"),
    # 4 债券/ETF
    DatasetSpec("etf_pcf", "ETF申赎清单", "4", "bond_etf", "4_bond_etf/etf_pcf", "symbol"),
    DatasetSpec("convertible_bond", "可转债", "4", "bond_etf", "4_bond_etf/convertible_bond", "symbol"),
    # 5 技术衍生
    DatasetSpec("valuation", "估值", "5", "technical", "5_technical_derived/valuation", "partition", "PE/PB/市值"),
    DatasetSpec("technical_indicators", "技术指标", "5", "technical", "5_technical_derived/technical_indicators", "partition", "本地覆盖不全，优先用 features_daily"),
    DatasetSpec("market_sentiment", "市场情绪", "5", "technical", "5_technical_derived/market_sentiment", "partition"),
    # 6 ML数据集
    DatasetSpec("features_daily", "日频特征", "6", "ml", "6_ml_datasets/features_daily", "partition", "技术指标 + 估值合并，PG 填充主源"),
    DatasetSpec("l1_factors", "L1 因子", "6", "ml", "6_ml_datasets/l1_factors", "partition", "98 因子，因子挖掘核心"),
    DatasetSpec("l2_factors", "L2 因子", "6", "ml", "6_ml_datasets/l2_factors", "partition", "216 高频微观因子"),
    DatasetSpec("l1_l2_factors", "L1+L2 合并", "6", "ml", "6_ml_datasets/l1_l2_factors", "partition"),
)

_BY_NAME = {ds.dataset: ds for ds in DATASETS}


def _spec(dataset: str) -> DatasetSpec:
    spec = _BY_NAME.get(dataset)
    if spec is None:
        raise HTTPException(status_code=400, detail=f"未知数据集: {dataset}")
    return spec


def _data_dir() -> Path:
    from backend.services.engine.data_platform.quantdb_hub import _resolve_data_dir
    return _resolve_data_dir()


# ---------------------------------------------------------------------------
# 本地落盘统计
# ---------------------------------------------------------------------------
def _partition_dates(root: Path) -> list[str]:
    """分区目录可用日期/季度。支持 dt=YYYYMMDD（按日）与 quarter=YYYYQN（按季）。"""
    out = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("dt="):
            out.append(p.name[3:])
        elif p.is_dir() and p.name.startswith("quarter="):
            out.append(p.name[8:])  # 2026Q2
    out.sort()
    return out


_DATE_IN_NAME = re.compile(r"(20\d{6})")


def _dataset_dates(spec: DatasetSpec, d: Path) -> list[str]:
    """按日数据集的可用日期。

    l1_factors 等为混合布局：既有 dt=YYYYMMDD/ 分区目录，也有平铺的
    {dataset}_YYYYMMDD.parquet；两者都要计入，否则区间会严重低估。
    """
    dates = set(_partition_dates(d))
    for f in d.glob("*.parquet"):
        m = _DATE_IN_NAME.search(f.stem)
        if m:
            dates.add(m.group(1))
    return sorted(dates)


def _dataset_stats(spec: DatasetSpec, root: Path) -> dict[str, Any]:
    """统计单个数据集的本地落盘情况。"""
    d = root / spec.rel_dir
    if not d.is_dir():
        return {"synced": False, "files": 0, "size_mb": 0.0}

    files = [f for f in d.rglob("*.parquet") if f.is_file()]
    size_mb = round(sum(f.stat().st_size for f in files) / 1024 / 1024, 1)
    stats: dict[str, Any] = {"synced": bool(files), "files": len(files), "size_mb": size_mb}

    if spec.layout == "partition":
        dates = _dataset_dates(spec, d)
        if dates:
            stats["start_date"] = dates[0]
            stats["end_date"] = dates[-1]
            stats["partitions"] = len(dates)
    if files:
        latest = max(f.stat().st_mtime for f in files)
        stats["updated_at"] = datetime.fromtimestamp(latest, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return stats


@router.get("/catalog")
async def get_catalog(current_user: dict = Depends(require_admin)):
    """返回数据集目录（按大类分组）+ 本地落盘统计。"""
    try:
        root = _data_dir()
        items = []
        for spec in DATASETS:
            items.append({
                "dataset": spec.dataset,
                "name": spec.name,
                "group": spec.group,
                "category_id": spec.category_id,
                "layout": spec.layout,
                "rel_dir": spec.rel_dir,
                "note": spec.note,
                **_dataset_stats(spec, root),
            })

        groups = []
        for g in GROUPS:
            members = [it for it in items if it["group"] == g["id"]]
            groups.append({
                **g,
                "dataset_count": len(members),
                "synced_count": sum(1 for it in members if it["synced"]),
                "files": sum(it["files"] for it in members),
                "size_mb": round(sum(it["size_mb"] for it in members), 1),
            })

        return {
            "success": True,
            "data": {
                "data_dir": str(root),
                "groups": groups,
                "datasets": items,
                "timestamp": _now_iso(),
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("quantdb catalog failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"failed: {exc}")


# ---------------------------------------------------------------------------
# 数据源勾选配置
# ---------------------------------------------------------------------------
@router.get("/data-sources")
async def get_data_sources(current_user: dict = Depends(require_admin)):
    from backend.shared.data_source_config import list_sources

    return {
        "success": True,
        "data": {
            "market": "A",
            "sources": list_sources("A"),
            "timestamp": _now_iso(),
        },
    }


@router.post("/data-sources")
async def save_data_sources(payload: DataSourcesRequest, current_user: dict = Depends(require_admin)):
    from backend.shared.data_source_config import save_sources

    saved = save_sources("A", payload.sources)
    return {
        "success": True,
        "data": {
            "market": "A",
            "sources": saved,
            "timestamp": _now_iso(),
        },
    }


# ---------------------------------------------------------------------------
# 数据预览
# ---------------------------------------------------------------------------
def _pick_local_file(spec: DatasetSpec, root: Path, symbol: str | None) -> Path | None:
    """选定要预览的单个 parquet：分区取最新交易日，标的层取指定/首个标的。"""
    d = root / spec.rel_dir
    if not d.is_dir():
        return None

    if spec.layout == "partition":
        dates = _partition_dates(d)
        for dt in reversed(dates):
            files = sorted((d / f"dt={dt}").glob("*.parquet"))
            if files:
                return files[0]
        # l1_factors 等混合布局：分区目录缺失时回退平铺文件
        flat = sorted(d.glob("*.parquet"))
        return flat[-1] if flat else None

    files = sorted(f for f in d.glob("*.parquet") if f.is_file())
    if not files:
        return None
    if symbol:
        target = symbol.strip().upper()
        for f in files:
            if f.stem.upper() == target:
                return f
        raise HTTPException(status_code=404, detail=f"{spec.dataset} 无 {symbol} 的本地文件")
    # instrument_detail 数据集：SDK 新版落盘 instrument_list.parquet，优先新版
    if spec.dataset == "instrument_detail":
        for f in files:
            if f.stem == "instrument_list":
                return f
    return files[0]


def _symbol_choices(spec: DatasetSpec, root: Path) -> dict[str, Any]:
    if spec.layout != "symbol":
        return {}
    d = root / spec.rel_dir
    if not d.is_dir():
        return {}
    stems = sorted(f.stem for f in d.glob("*.parquet") if f.is_file())
    return {
        "symbol_total": len(stems),
        "symbol_choices": stems[:MAX_SYMBOL_CHOICES],
    }


def _json_safe(value: Any) -> Any:
    """把 parquet 单元格转成可 JSON 序列化的值。

    instrument_detail 等数据集含 ndarray/list 单元格，且浮点列可能有 NaN/Inf，
    两者都无法直接进 JSON，需要逐值处理而不能用 DataFrame.replace。
    """
    import numpy as np
    import pandas as pd

    if value is None:
        return None
    if isinstance(value, (np.ndarray, list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, bytes)):
        return value.decode("utf-8", "replace") if isinstance(value, bytes) else value
    return str(value)


def _json_safe_records(df: Any) -> list[dict[str, Any]]:
    return [
        {str(k): _json_safe(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


@router.get("/preview")
async def preview_dataset(
    dataset: str = Query(..., description="数据集名，如 daily_forward"),
    symbol: str | None = Query(None, description="标的层数据集指定标的，如 600519.SH"),
    limit: int = Query(50, ge=1, le=MAX_PREVIEW_ROWS),
    remote: bool = Query(False, description="强制走远端 SDK 预览"),
    current_user: dict = Depends(require_admin),
):
    """预览数据集样本。本地 parquet 优先（零流量），缺失时回退 SDK 远端预览。"""
    import pandas as pd

    spec = _spec(dataset)
    root = _data_dir()

    file_path = None if remote else _pick_local_file(spec, root, symbol)

    try:
        if file_path is not None:
            df = pd.read_parquet(file_path)
            source = "local"
            file_label = str(file_path.relative_to(root))
        else:
            from backend.services.engine.data_platform.adapters.quantdb_adapter import (
                _get_client,
                _to_qdb_symbol,
            )
            client = _get_client()
            df = client.preview_as_df(
                category_id=spec.category_id,
                sub_category=spec.dataset,
                symbol=_to_qdb_symbol(symbol) if symbol else None,
                limit=limit,
            )
            source = "remote"
            file_label = None
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("quantdb preview failed (%s): %s", dataset, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"预览失败: {exc}")

    if df is None or df.empty:
        return {
            "success": True,
            "data": {
                "dataset": dataset,
                "source": source,
                "rows_total": 0,
                "columns": [],
                "data": [],
                **_symbol_choices(spec, root),
                "timestamp": _now_iso(),
            },
        }

    columns = [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns]
    records = _json_safe_records(df.head(limit))

    return {
        "success": True,
        "data": {
            "dataset": dataset,
            "name": spec.name,
            "source": source,
            "file": file_label,
            "rows_total": int(len(df)),
            "column_count": len(columns),
            "columns": columns,
            "data": records,
            **_symbol_choices(spec, root),
            "timestamp": _now_iso(),
        },
    }


@router.get("/remote-meta")
async def get_remote_meta(current_user: dict = Depends(require_admin)):
    """查询远端各数据集的日期范围与行数（网关 JSON，不消耗流量）。"""
    try:
        from backend.services.engine.data_platform.adapters.quantdb_adapter import _get_client
        df = _get_client().query_meta()
        if df is None or df.empty:
            return {"success": True, "data": {"rows": 0, "columns": [], "data": []}}
        return {
            "success": True,
            "data": {
                "rows": int(len(df)),
                "columns": [str(c) for c in df.columns],
                "data": _json_safe_records(df),
                "timestamp": _now_iso(),
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("quantdb remote meta failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"failed: {exc}")


# ---------------------------------------------------------------------------
# 按数据集同步
# ---------------------------------------------------------------------------
class SyncDatasetsRequest(BaseModel):
    datasets: list[str] = Field(..., min_length=1, description="数据集名列表")
    with_pg: bool = Field(False, description="同步后从 parquet 填充 PG stock_daily_latest")
    with_qlib: bool = Field(False, description="同步后增量重建 Qlib 缓存")
    pg_full: bool = Field(False, description="PG 全量重灌（默认增量）")


class DataSourcesRequest(BaseModel):
    sources: dict[str, bool] = Field(..., description="数据源勾选状态 {source: enabled}")


_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_job_counter = itertools.count(1)
MAX_JOB_HISTORY = 20


def _job_update(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(fields)


def _sync_with_fallback(client: Any, dataset: str, save_dir: str, job_id: str | None = None) -> dict[str, Any]:
    """同步数据集，SHA-256/SSL 错误时自动回退到 manifest 逐文件下载。"""
    try:
        return client.sync_dataset(dataset=dataset, save_dir=save_dir)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if any(kw in msg for kw in ("sha-256", "sha256", "校验", "ssl", "eof", "connection")):
            logger.info("quantdb sync %s: sync_dataset failed (%s), falling back to manifest", dataset, type(exc).__name__)
            return _sync_manifest_fallback(client, dataset, save_dir, job_id=job_id)
        raise


SYNC_FALLBACK_WORKERS = 2
SYNC_FALLBACK_RETRIES = 3
# 同一数据集的 fallback 串行化：避免两个任务写同一批文件与 sqlite 抢锁
_fallback_locks: dict[str, threading.Lock] = {}
_fallback_locks_guard = threading.Lock()


def _fallback_lock(dataset: str) -> threading.Lock:
    with _fallback_locks_guard:
        return _fallback_locks.setdefault(dataset, threading.Lock())


def _normalise_object_key(key: str) -> str:
    """与 SDK 的 _normalise_release_key 保持一致地去掉历史 v2/ 前缀。

    自行实现而非调用 SDK 私有方法：若前缀规则变化，落盘路径不会被
    SDK 升级悄悄改掉，且 sqlite key 与 SDK 写入的保持同一形态。
    """
    return key.strip().lstrip("/").removeprefix("v2/")


def _is_valid_parquet(path: str) -> bool:
    """校验 parquet 首尾 magic bytes。

    由于服务端 size/SHA-256 元数据不可信而被跳过，尾部 PAR1 是唯一能
    发现 CDN 中断导致的截断的信号。
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"PAR1":
                return False
            if f.seek(0, os.SEEK_END) < 8:
                return False
            f.seek(-4, os.SEEK_END)
            return f.read(4) == b"PAR1"
    except OSError:
        return False


def _sync_manifest_fallback(client: Any, dataset: str, save_dir: str, job_id: str | None = None) -> dict[str, Any]:
    """用 V2 manifest 逐文件下载，跳过不可信的 SHA-256/size 校验，多线程并行。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from contextlib import closing, suppress
    from quantdb_sdk.client import SYNC_DATASET_CATEGORIES, safe_join
    import sqlite3 as _sqlite3

    category_id = SYNC_DATASET_CATEGORIES[dataset]
    # V2 manifest 包含所有文件元数据
    manifest = client.query_manifest(category_id, dataset, layout="v2")
    if not manifest:
        return {"dataset": dataset, "layout": "v2_manifest", "downloaded": [], "fallback": True}

    root = os.path.abspath(save_dir)
    with _fallback_lock(dataset), closing(
        _sqlite3.connect(os.path.join(root, "quantdb_sync.sqlite"), timeout=30.0)
    ) as state:
        return _run_manifest_fallback(
            client, dataset, category_id, manifest, root, state, safe_join,
            ThreadPoolExecutor, as_completed, suppress, job_id=job_id,
        )


def _run_manifest_fallback(
    client: Any,
    dataset: str,
    category_id: str,
    manifest: list[dict[str, Any]],
    root: str,
    state: Any,
    safe_join: Any,
    ThreadPoolExecutor: Any,
    as_completed: Any,
    suppress: Any,
    job_id: str | None = None,
) -> dict[str, Any]:
    state.execute(
        "CREATE TABLE IF NOT EXISTS objects "
        "(key TEXT PRIMARY KEY, etag TEXT, sha256 TEXT, size INTEGER, "
        "path TEXT, layout TEXT, dataset TEXT)"
    )
    cols = {row[1] for row in state.execute("PRAGMA table_info(objects)")}
    if "dataset" not in cols:
        state.execute("ALTER TABLE objects ADD COLUMN dataset TEXT")
    if "size" not in cols:
        state.execute("ALTER TABLE objects ADD COLUMN size INTEGER")

    # 筛出需要下载的文件：仅当 etag 未变且本地文件仍是有效 parquet 时跳过。
    # 不能只看"本地存在有效 parquet"，否则 single-layout（每日重写的
    # instrument_detail / trading_calendar）和被供应商回溯修正的分区永远不会刷新。
    to_download: list[dict[str, Any]] = []
    adopted = 0
    for obj in manifest:
        key = _normalise_object_key(obj["key"])
        etag = obj.get("etag")
        relative_path = obj.get("relative_path") or key
        target = safe_join(root, relative_path)
        old = state.execute(
            "SELECT etag, path FROM objects WHERE key=?", (key,)
        ).fetchone()
        if (
            old
            and old[0]
            and old[0] == etag
            and os.path.exists(old[1])
            and _is_valid_parquet(old[1])
        ):
            continue
        # 补记账：历史上经由其它途径落盘、sqlite 无记录的完好文件，
        # 登记当前 etag 而不重下。否则每次同步都要全量重拉（数千文件），
        # 且后续 etag 变化仍能被正常检出。
        if not old and os.path.exists(target) and _is_valid_parquet(target):
            state.execute(
                "INSERT OR REPLACE INTO objects(key,etag,sha256,size,path,layout,dataset) "
                "VALUES(?,?,?,?,?,?,?)",
                (key, etag, "", os.path.getsize(target), target, "v2_manifest", dataset),
            )
            adopted += 1
            continue
        to_download.append(obj)
    if adopted:
        state.commit()
        logger.info("quantdb sync %s: adopted %d existing valid files into sync state", dataset, adopted)

    logger.info("quantdb sync %s: manifest fallback — %d/%d files to download", dataset, len(to_download), len(manifest))

    def _download_one(obj: dict[str, Any]) -> tuple[str, bool]:
        # 请求用服务端原始 key；落盘路径与 sqlite 记账用归一化 key
        raw_key = obj["key"]
        key = _normalise_object_key(raw_key)
        relative_path = obj.get("relative_path") or key
        target = safe_join(root, relative_path)
        expected_size = obj.get("size")
        # 唯一临时名：同一 target 若被并发任务命中，不会互相覆写半成品
        tmp = f"{target}.{os.getpid()}.{threading.get_ident()}.part"
        for attempt in range(1, SYNC_FALLBACK_RETRIES + 1):
            try:
                resp = client._download_stream({
                    "category_id": category_id,
                    "sub_category": dataset,
                    "layout": "v2",
                    "object_key": raw_key,
                })
                try:
                    if resp.status_code != 200:
                        logger.warning(
                            "quantdb sync %s: download %s returned %s",
                            dataset, key, resp.status_code,
                        )
                    else:
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        with open(tmp, "wb") as fh:
                            for chunk in resp.iter_content(1024 * 1024):
                                if chunk:
                                    fh.write(chunk)
                        # 服务端 size/SHA-256 不可信，只校验 parquet 首尾 magic
                        if _is_valid_parquet(tmp):
                            os.replace(tmp, target)
                            return key, True
                        logger.warning(
                            "quantdb sync %s: invalid/truncated parquet %s (expected %s, got %s)",
                            dataset, key, expected_size, os.path.getsize(tmp),
                        )
                finally:
                    resp.close()
            except Exception as exc:  # noqa: BLE001
                if attempt >= SYNC_FALLBACK_RETRIES:
                    logger.warning(
                        "quantdb sync %s: download %s failed after %d attempts: %s",
                        dataset, key, attempt, exc,
                    )
            # 所有失败路径共用：清理半成品 + 退避后重试
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt < SYNC_FALLBACK_RETRIES:
                time.sleep(1.0 * attempt)
        return key, False

    downloaded: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=SYNC_FALLBACK_WORKERS) as pool:
            futures = {pool.submit(_download_one, obj): obj for obj in to_download}
            for future in as_completed(futures):
                # 协作式取消：每完成一个文件检查一次
                if job_id:
                    with _jobs_lock:
                        if _jobs.get(job_id, {}).get("cancel_requested"):
                            pool.shutdown(wait=False, cancel_futures=True)
                            state.commit()
                            logger.info("quantdb sync %s: cancelled during download (%d/%d done)", dataset, len(downloaded), len(to_download))
                            return {
                                "dataset": dataset,
                                "layout": "v2_manifest",
                                "downloaded": downloaded,
                                "fallback": True,
                                "cancelled": True,
                            }
                key, ok = future.result()
                if ok:
                    obj = futures[future]
                    relative_path = obj.get("relative_path") or key
                    target = safe_join(root, relative_path)
                    written_size = os.path.getsize(target) if os.path.exists(target) else 0
                    state.execute(
                        "INSERT OR REPLACE INTO objects(key,etag,sha256,size,path,layout,dataset) "
                        "VALUES(?,?,?,?,?,?,?)",
                        (key, obj.get("etag"), "", written_size, target, "v2_manifest", dataset),
                    )
                    downloaded.append(key)
        state.commit()
        return {
            "dataset": dataset,
            "layout": "v2_manifest",
            "downloaded": downloaded,
            "fallback": True,
        }
    except Exception:
        with suppress(Exception):
            state.rollback()
        raise


def _run_sync_job(job_id: str, req: SyncDatasetsRequest) -> None:
    from backend.scripts.quantdb_daily_sync import run_daily_sync

    # 取消检查辅助
    def _cancelled() -> bool:
        with _jobs_lock:
            return bool(_jobs.get(job_id, {}).get("cancel_requested"))

    # Phase 1: parquet 同步（使用 quantdb_daily_sync 的统一逻辑）
    _job_update(job_id, stage="sync_parquet")
    try:
        sync_result = run_daily_sync(
            datasets=req.datasets,
            skip_pg=True,       # PG 单独处理
            skip_qlib=True,     # Qlib 单独处理
            skip_snapshot=True,  # snapshot 单独处理
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("quantdb sync job %s: parquet sync failed: %s", job_id, exc, exc_info=True)
        _job_update(job_id, status="failed", error=str(exc), finished_at=_now_iso())
        return

    # 将 parquet 同步结果映射到 job results
    parquet_info = sync_result.get("parquet") or {}
    synced_count = parquet_info.get("synced", 0)
    parquet_info.get("up_to_date", 0)
    errors = parquet_info.get("errors", [])
    parquet_info.get("total_downloaded", 0)
    sources_info = sync_result.get("sources") or {}

    results = []
    for name in req.datasets:
        src = sources_info.get(name)
        if src is not None:
            if src.get("status") == "error":
                results.append({"dataset": name, "status": "failed", "downloaded": 0,
                                "error": str(src.get("error", "unknown"))})
            elif src.get("synced", 0) > 0 or src.get("status") in ("ok", "completed"):
                results.append({"dataset": name, "status": "synced", "downloaded": src.get("synced", 1)})
            else:
                results.append({"dataset": name, "status": "up_to_date", "downloaded": 0})
        elif any(name in str(e) for e in errors):
            results.append({"dataset": name, "status": "failed", "downloaded": 0, "error": next((e for e in errors if name in str(e)), "unknown")})
        elif synced_count > 0:
            results.append({"dataset": name, "status": "synced", "downloaded": 1})
        else:
            results.append({"dataset": name, "status": "up_to_date", "downloaded": 0})

    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["results"] = results
            job["done"] = len(req.datasets)

    if _cancelled():
        _job_update(job_id, status="cancelled", current=None, finished_at=_now_iso())
        return

    # Phase 2: PG 填充
    if req.with_pg:
        _job_update(job_id, stage="pg_fill")
        try:
            from backend.scripts.quantdb_daily_sync import fill_pg_from_parquet, QUANTDB_EPOCH
            start = QUANTDB_EPOCH if req.pg_full else None
            _job_update(job_id, pg_fill=fill_pg_from_parquet(start_date=start))
        except Exception as exc:  # noqa: BLE001
            logger.error("quantdb sync job %s: pg fill failed: %s", job_id, exc, exc_info=True)
            _job_update(job_id, pg_fill={"status": "error", "reason": str(exc)})

    if _cancelled():
        _job_update(job_id, status="cancelled", finished_at=_now_iso())
        return

    # Phase 3: Qlib 缓存
    if req.with_qlib:
        _job_update(job_id, stage="qlib_cache")
        try:
            from backend.scripts.quantdb_daily_sync import update_qlib_cache
            _job_update(job_id, qlib_cache=update_qlib_cache())
        except Exception as exc:  # noqa: BLE001
            logger.error("quantdb sync job %s: qlib cache failed: %s", job_id, exc, exc_info=True)
            _job_update(job_id, qlib_cache={"status": "error", "reason": str(exc)})

    _job_update(job_id, status="completed", stage="done", finished_at=_now_iso())


@router.post("/sync-datasets")
async def sync_datasets(
    payload: SyncDatasetsRequest,
    current_user: dict = Depends(require_admin),
):
    """按数据集触发增量同步（后台线程执行，返回 job_id 供轮询）。"""
    for name in payload.datasets:
        _spec(name)

    if not get_quantdb_api_key():
        raise HTTPException(status_code=422, detail=QUANTDB_CONFIG_HINT)

    job_id = f"qdb-{next(_job_counter)}"
    job = {
        "job_id": job_id,
        "status": "running",
        "stage": "sync_parquet",
        "datasets": list(payload.datasets),
        "total": len(payload.datasets),
        "done": 0,
        "current": None,
        "results": [],
        "with_pg": payload.with_pg,
        "with_qlib": payload.with_qlib,
        "cancel_requested": False,
        "started_at": _now_iso(),
        "started_by": current_user.get("username") or current_user.get("user_id"),
    }
    with _jobs_lock:
        _jobs[job_id] = job
        for stale in sorted(_jobs)[:-MAX_JOB_HISTORY]:
            if _jobs[stale]["status"] != "running":
                _jobs.pop(stale, None)

    threading.Thread(target=_run_sync_job, args=(job_id, payload), daemon=True).start()
    return {"success": True, "data": {"job": job}}


@router.get("/sync-jobs")
async def list_sync_jobs(current_user: dict = Depends(require_admin)):
    """列出同步任务（最近优先）。"""
    with _jobs_lock:
        jobs = [dict(j) for j in _jobs.values()]
    jobs.sort(key=lambda j: j["started_at"], reverse=True)
    return {"success": True, "data": {"jobs": jobs, "timestamp": _now_iso()}}


@router.get("/sync-jobs/{job_id}")
async def get_sync_job(job_id: str, current_user: dict = Depends(require_admin)):
    """查询单个同步任务进度。"""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
        return {"success": True, "data": {"job": dict(job)}}


@router.post("/sync-jobs/{job_id}/cancel")
async def cancel_sync_job(job_id: str, current_user: dict = Depends(require_admin)):
    """取消正在运行的同步任务（协作式，当前数据集完成后停止）。"""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
        if job["status"] != "running":
            raise HTTPException(status_code=400, detail=f"任务状态为 {job['status']}，无法取消")
        job["cancel_requested"] = True
    return {
        "success": True,
        "data": {
            "job_id": job_id,
            "status": "cancelling",
            "message": "取消信号已发送，当前数据集完成后将停止",
        },
    }


# ---------------------------------------------------------------------------
# 远端 vs 本地差异比对
# ---------------------------------------------------------------------------
@router.get("/diff")
async def get_remote_diff(
    datasets: str | None = Query(None, description="逗号分隔的数据集名，留空查全部"),
    current_user: dict = Depends(require_admin),
):
    """比较远端 QuantDB 数据与本地状态，返回哪些数据集有更新。

    使用 manifest 比对而非 query_meta()：manifest 是实时数据，且能检测
    单文件数据集（如 trading_calendar）的 ETag 变化，query_meta 仅反映
    日期范围差异。
    """
    try:
        from backend.services.engine.data_platform.adapters.quantdb_adapter import _get_client
        client = _get_client()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"QuantDB SDK 不可用: {exc}") from None

    filter_names = [s.strip() for s in datasets.split(",") if s.strip()] if datasets else None
    specs = [s for s in DATASETS if not filter_names or s.dataset in filter_names]

    root = _data_dir()

    items: list[dict[str, Any]] = []
    summary = {"total_datasets": len(specs), "up_to_date": 0, "updates_available": 0, "not_synced": 0, "unknown": 0}

    for spec in specs:
        local_stats = _dataset_stats(spec, root)
        local_synced = local_stats.get("synced", False)
        local_end = local_stats.get("end_date")
        local_files = local_stats.get("files", 0)

        item: dict[str, dict[str, Any] | Any] = {
            "dataset": spec.dataset,
            "name": spec.name,
            "category_id": spec.category_id,
            "layout": spec.layout,
            "local": {
                "synced": local_synced,
                "files": local_files,
                "size_mb": local_stats.get("size_mb", 0),
                "end_date": local_end,
                "partitions": local_stats.get("partitions"),
            },
            "remote": None,
            "status": "unknown",
            "new_files": 0,
        }

        # 通过 manifest 比对远端 vs 本地
        try:
            manifest = client.query_manifest(category_id=spec.category_id, sub_category=spec.dataset)
        except Exception:
            manifest = None

        if manifest:
            remote_end = None
            remote_files_count = len(manifest)
            etag_changed = 0

            # 从 manifest 提取远端最新日期和 ETag 变化数
            for obj in manifest:
                key = obj.get("key", "")
                trade_date = obj.get("trade_date")
                if trade_date and (remote_end is None or str(trade_date) > str(remote_end)):
                    remote_end = str(trade_date)

                # 比对 ETag：检查本地状态库
                rel_path = obj.get("relative_path") or key
                target = root / spec.rel_dir
                target / rel_path if spec.layout == "partition" else target / rel_path
                (obj.get("etag") or "").strip('"')
                remote_size = obj.get("size")

                # partition 布局：文件存在 + 大小一致才算通过（内容被污染但存在
                # 的文件通过大小差异暴露，如 daily_backward 复权因子错误）
                if spec.layout == "partition" and trade_date:
                    dt_dir = target / f"dt={trade_date.replace('-', '')}"
                    pq = next(dt_dir.glob("*.parquet"), None) if dt_dir.exists() else None
                    if pq is None:
                        etag_changed += 1
                    elif remote_size and abs(pq.stat().st_size - int(remote_size)) > 64:
                        etag_changed += 1
                elif spec.layout in ("single", "symbol"):
                    # 单文件/标的文件：检查本地文件是否存在且 ETag 未变
                    local_path = root / rel_path
                    if not local_path.exists():
                        etag_changed += 1

            item["remote"] = {
                "end_date": remote_end,
                "files": remote_files_count,
            }

            if not local_synced:
                item["status"] = "not_synced"
                summary["not_synced"] += 1
            elif etag_changed > 0:
                item["status"] = "updates_available"
                item["new_files"] = etag_changed
                summary["updates_available"] += 1
            elif remote_end and local_end and str(remote_end) > str(local_end):
                item["status"] = "updates_available"
                summary["updates_available"] += 1
            else:
                item["status"] = "up_to_date"
                summary["up_to_date"] += 1
        elif not local_synced:
            item["status"] = "not_synced"
            summary["not_synced"] += 1
        else:
            item["status"] = "unknown"
            summary["unknown"] += 1

        items.append(item)

    return {
        "success": True,
        "data": {
            "datasets": items,
            "summary": summary,
            "timestamp": _now_iso(),
        },
    }


# ---------------------------------------------------------------------------
# API Key 配置
# ---------------------------------------------------------------------------
class QuantDBConfigRequest(BaseModel):
    api_key: str = Field(..., min_length=8, description="QuantDB API Key")


@router.get("/config")
async def get_quantdb_config(current_user: dict = Depends(require_admin)):
    """返回 API Key 配置状态（脱敏），以及数据目录等运行时配置。"""
    from backend.shared.runtime_secrets import runtime_env_path

    api_key = get_quantdb_api_key()
    return {
        "success": True,
        "data": {
            "api_key_configured": bool(api_key),
            "api_key_masked": mask_secret(api_key),
            "data_dir": str(_data_dir()),
            "runtime_env_file": str(runtime_env_path()),
            "timestamp": _now_iso(),
        },
    }


@router.post("/config")
async def save_quantdb_config(
    payload: QuantDBConfigRequest,
    current_user: dict = Depends(require_admin),
):
    """保存 API Key 到 config/runtime.env，并立即在当前进程生效。"""
    api_key = payload.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key 不能为空")

    try:
        set_secret("QUANTDB_API_KEY", api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError as exc:
        logger.error("写入 runtime.env 失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"写入配置文件失败: {exc}")

    # 立刻验证一次，让用户当场知道 Key 是否可用
    verified, error = False, None
    try:
        from backend.services.engine.data_platform.adapters.quantdb_adapter import _get_client
        _get_client().get_me()
        verified = True
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        logger.warning("QuantDB API Key 校验失败: %s", exc)

    return {
        "success": True,
        "data": {
            "api_key_masked": mask_secret(api_key),
            "verified": verified,
            "error": error,
            "timestamp": _now_iso(),
        },
    }


# ---------------------------------------------------------------------------
# SDK 状态与远端查询
# ---------------------------------------------------------------------------
@router.get("/info")
async def get_quantdb_info(current_user: dict = Depends(require_admin)):
    """返回 QuantDB SDK 安装/连接状态、账户信息与流量配额。"""
    try:
        from backend.services.engine.data_platform.adapters.quantdb_adapter import get_sdk_info
        return {"success": True, "data": {"quantdb": get_sdk_info(), "timestamp": _now_iso()}}
    except Exception as exc:  # noqa: BLE001
        logger.error("get_quantdb_info failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"failed: {exc}")


class QuantDBKlineRequest(BaseModel):
    symbol: str
    adj_type: str = "forward"
    start_date: str | None = None
    end_date: str | None = None


class QuantDBTickRequest(BaseModel):
    symbol: str
    trade_date: str
    start_ts: str | None = None
    end_ts: str | None = None
    fields: str = "last_price,open,high,low,last_close,volume,amount"
    limit: int = 500


class QuantDBManifestRequest(BaseModel):
    category_id: str
    sub_category: str
    trade_date: str | None = None
    limit: int = Field(MAX_MANIFEST_FILES, ge=1, le=MAX_MANIFEST_FILES)


@router.post("/query-kline")
async def quantdb_query_kline(
    payload: QuantDBKlineRequest,
    current_user: dict = Depends(require_admin),
):
    """通过 QuantDB SDK 查询 K 线数据（消耗流量配额）。"""
    try:
        from backend.services.engine.data_platform.adapters.quantdb_adapter import (
            _get_client,
            _to_qdb_symbol,
        )
        qdb_symbol = _to_qdb_symbol(payload.symbol)
        df = _get_client().query_kline(
            qdb_symbol,
            adj_type=payload.adj_type,
            start_date=payload.start_date,
            end_date=payload.end_date,
        )
        if df is None or df.empty:
            return {"success": True, "data": {"rows": 0, "columns": [], "data": []}}
        return {
            "success": True,
            "data": {
                "symbol": qdb_symbol,
                "rows": int(len(df)),
                "columns": [str(c) for c in df.columns],
                "data": _json_safe_records(df.head(500)),
                "timestamp": _now_iso(),
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("quantdb_query_kline failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"failed: {exc}")


@router.get("/stock-list")
async def quantdb_stock_list(
    keyword: str | None = None,
    limit: int = Query(50, ge=1, le=10000),
    current_user: dict = Depends(require_admin),
):
    """通过 QuantDB SDK 查询股票列表。"""
    try:
        from backend.services.engine.data_platform.adapters.quantdb_adapter import _get_client
        df = _get_client().query_stock_list(keyword=keyword, limit=limit)
        if df is None or df.empty:
            return {"success": True, "data": {"rows": 0, "columns": [], "data": []}}
        return {
            "success": True,
            "data": {
                "rows": int(len(df)),
                "columns": [str(c) for c in df.columns],
                "data": _json_safe_records(df.head(limit)),
                "timestamp": _now_iso(),
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("quantdb_stock_list failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"failed: {exc}")


@router.get("/calendar")
async def quantdb_calendar(
    start_date: str = Query(..., description="开始日期 YYYY-MM-DD"),
    end_date: str = Query(..., description="结束日期 YYYY-MM-DD"),
    current_user: dict = Depends(require_admin),
):
    """通过 QuantDB SDK 查询交易日历。"""
    try:
        from backend.services.engine.data_platform.adapters.quantdb_adapter import _get_client
        df = _get_client().query_calendar(start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return {"success": True, "data": {"rows": 0, "columns": [], "data": []}}
        return {
            "success": True,
            "data": {
                "rows": int(len(df)),
                "columns": [str(c) for c in df.columns],
                "data": _json_safe_records(df),
                "timestamp": _now_iso(),
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("quantdb_calendar failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"failed: {exc}")


@router.post("/query-tick")
async def quantdb_query_tick(
    payload: QuantDBTickRequest,
    current_user: dict = Depends(require_admin),
):
    """通过 QuantDB SDK 查询 Tick 分笔数据（消耗流量配额，体积大）。"""
    try:
        from backend.services.engine.data_platform.adapters.quantdb_adapter import (
            _get_client,
            _to_qdb_symbol,
        )
        qdb_symbol = _to_qdb_symbol(payload.symbol)
        df = _get_client().query_tick(
            qdb_symbol,
            trade_date=payload.trade_date,
            start_ts=payload.start_ts,
            end_ts=payload.end_ts,
            fields=payload.fields,
            limit=payload.limit,
        )
        if df is None or df.empty:
            return {"success": True, "data": {"rows": 0, "columns": [], "data": []}}
        return {
            "success": True,
            "data": {
                "symbol": qdb_symbol,
                "trade_date": payload.trade_date,
                "rows": int(len(df)),
                "columns": [str(c) for c in df.columns],
                "data": _json_safe_records(df.head(payload.limit)),
                "timestamp": _now_iso(),
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("quantdb_query_tick failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"failed: {exc}")


@router.post("/query-manifest")
async def quantdb_query_manifest(
    payload: QuantDBManifestRequest,
    current_user: dict = Depends(require_admin),
):
    """查询远端 COS 可下载文件清单（不消耗流量）。"""
    try:
        from backend.services.engine.data_platform.adapters.quantdb_adapter import _get_client
        files = _get_client().query_manifest(
            category_id=payload.category_id,
            sub_category=payload.sub_category,
            trade_date=payload.trade_date,
        ) or []
        total = len(files)
        capped = [
            {k: _json_safe(v) for k, v in f.items()}
            for f in files[: payload.limit]
        ]
        return {
            "success": True,
            "data": {
                "files": capped,
                "count": len(capped),
                "total": total,
                "truncated": total > len(capped),
                "timestamp": _now_iso(),
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("quantdb_query_manifest failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"failed: {exc}")
