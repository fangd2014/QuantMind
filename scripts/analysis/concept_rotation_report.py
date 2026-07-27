#!/usr/bin/env python3
"""Generate the daily concept-rotation report and notify Feishu.

The report combines concept breadth, an RRG-style relative-strength model and
leader confirmation.  It reads only the local ``stock_daily_latest`` table,
uses Sina Finance for current concept membership, writes a PDF under the API's
``/uploads`` directory and optionally sends a summary plus the PDF URL through
a Feishu custom webhook.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import logging
import math
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2


LOGGER = logging.getLogger("concept_rotation_report")
SINA_CONCEPT_URL = "https://money.finance.sina.com.cn/q/view/newFLJK.php?param=class"
SINA_MEMBER_URL = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeData"
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
    ),
    "Referer": "https://money.finance.sina.com.cn/",
}
NON_THEMATIC_CONCEPT = re.compile(
    r"(?:ST|超大盘|含H股|融资融券|基金重仓|保险重仓|QFII|社保重仓|"
    r"央企50|业绩预|外资背景|信托重仓|券商重仓|未股改|转债标的|"
    r"MSCI|沪股通|深股通|高送转|低价股|高价股|破净股|新股)"
)
DEFAULT_OUTPUT_DIR = "/data/uploads/reports/concept-rotation"
DEFAULT_CACHE_DIR = "/data/cache/concept-rotation"
DEFAULT_PUBLIC_BASE_URL = "http://192.168.5.10:18000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a concept rotation PDF and notify Feishu"
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("CONCEPT_REPORT_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("CONCEPT_REPORT_CACHE_DIR", DEFAULT_CACHE_DIR),
    )
    parser.add_argument(
        "--public-base-url",
        default=os.getenv("CONCEPT_REPORT_PUBLIC_BASE_URL", DEFAULT_PUBLIC_BASE_URL),
    )
    parser.add_argument("--send-feishu", action="store_true")
    parser.add_argument("--max-buy", type=int, default=5)
    parser.add_argument("--max-watch", type=int, default=10)
    parser.add_argument("--member-cache-days", type=float, default=7.0)
    return parser.parse_args()


def normalize_symbol(symbol: str) -> str | None:
    raw = str(symbol or "").strip().upper()
    match = re.search(r"(\d{6})", raw)
    if not match:
        return None
    code = match.group(1)
    if raw.startswith("SH") or raw.endswith(".SH") or code.startswith(("6", "9")):
        return f"SH{code}"
    if raw.startswith("BJ") or raw.endswith(".BJ") or code.startswith(("4", "8")):
        return f"BJ{code}"
    return f"SZ{code}"


def classify_quadrant(rs_ratio: float, rs_momentum: float) -> str:
    if rs_ratio >= 100 and rs_momentum >= 100:
        return "领先区"
    if rs_ratio < 100 <= rs_momentum:
        return "改善区"
    if rs_ratio >= 100 > rs_momentum:
        return "转弱区"
    return "落后区"


def percentile(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average").fillna(0.5)


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return ((numeric - numeric.mean()) / std).clip(-3, 3).fillna(0.0)


def finite_number(value: Any) -> float | int | str | bool | None:
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fetch_bytes(url: str, retries: int = 4, timeout: int = 40) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"request failed after {retries} attempts: {last_error}")


def parse_concept_snapshot(text: str) -> list[dict[str, Any]]:
    match = re.search(r"=\s*(\{.*\})\s*;?\s*$", text, flags=re.S)
    if not match:
        raise ValueError("Sina concept response did not contain the expected object")
    raw = json.loads(match.group(1))
    concepts: list[dict[str, Any]] = []
    for code, value in raw.items():
        parts = str(value).split(",")
        if not str(code).startswith("gn_") or len(parts) < 13:
            continue
        try:
            concepts.append(
                {
                    "code": str(code),
                    "name": parts[1],
                    "reported_count": int(float(parts[2] or 0)),
                    "snapshot_pct": float(parts[5] or 0),
                    "source_leader_symbol": parts[8],
                    "source_leader_pct": float(parts[9] or 0),
                    "source_leader_name": parts[12],
                }
            )
        except (TypeError, ValueError):
            continue
    if len(concepts) < 30:
        raise ValueError(f"Sina concept response contained only {len(concepts)} rows")
    return concepts


def load_concepts(cache_dir: Path) -> list[dict[str, Any]]:
    cache_path = cache_dir / "sina-concepts.json"
    try:
        payload = fetch_bytes(SINA_CONCEPT_URL)
        text = payload.decode("gb18030", errors="replace")
        concepts = parse_concept_snapshot(text)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(concepts, ensure_ascii=False), encoding="utf-8"
        )
        return concepts
    except Exception:
        if cache_path.exists():
            LOGGER.exception("Concept snapshot fetch failed; using cached snapshot")
            return json.loads(cache_path.read_text(encoding="utf-8"))
        raise


def fetch_members(concept_code: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in range(1, 16):
        query = urllib.parse.urlencode(
            {
                "page": page,
                "num": 100,
                "sort": "symbol",
                "asc": 1,
                "node": concept_code,
                "symbol": "",
                "_s_r_a": "page",
            }
        )
        payload = fetch_bytes(f"{SINA_MEMBER_URL}?{query}")
        current = json.loads(payload.decode("gb18030", errors="replace"))
        if not isinstance(current, list):
            raise ValueError(f"Unexpected member schema for {concept_code}")
        if not current:
            break
        rows.extend(item for item in current if isinstance(item, dict))
        if len(current) < 100:
            break
    unique = {str(row.get("symbol")): row for row in rows if row.get("symbol")}
    return list(unique.values())


def load_memberships(
    concepts: list[dict[str, Any]], cache_dir: Path, cache_days: float = 7.0
) -> dict[str, list[dict[str, Any]]]:
    cache_path = cache_dir / "sina-concept-members.json.gz"
    stale_cache: dict[str, list[dict[str, Any]]] = {}
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            candidate = json.load(handle)
        if isinstance(candidate, dict):
            stale_cache = candidate
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days <= cache_days and len(stale_cache) >= len(concepts) * 0.90:
            LOGGER.info("Using %.1f-day-old Sina membership cache", age_days)
            return stale_cache

    memberships: dict[str, list[dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {
            pool.submit(fetch_members, concept["code"]): concept for concept in concepts
        }
        for index, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            concept = jobs[job]
            try:
                rows = job.result()
                if len(rows) < 2:
                    raise ValueError("too few members")
                memberships[concept["code"]] = rows
            except Exception as exc:
                cached = stale_cache.get(concept["code"], [])
                if cached:
                    memberships[concept["code"]] = cached
                    LOGGER.warning(
                        "Using stale members for %s after fetch error: %s",
                        concept["name"],
                        exc,
                    )
                else:
                    memberships[concept["code"]] = []
                    LOGGER.error(
                        "No members available for %s: %s", concept["name"], exc
                    )
            if index % 40 == 0:
                LOGGER.info("Downloaded memberships %s/%s", index, len(concepts))

    successful = sum(bool(rows) for rows in memberships.values())
    if successful < len(concepts) * 0.75:
        raise RuntimeError(
            f"Only {successful}/{len(concepts)} concept memberships are available"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(".tmp")
    with gzip.open(temp_path, "wt", encoding="utf-8") as handle:
        json.dump(memberships, handle, ensure_ascii=False)
    temp_path.replace(cache_path)
    return memberships


def database_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "quantmind"),
        user=os.getenv("DB_USER", "quantmind"),
        password=os.getenv("DB_PASSWORD", ""),
        connect_timeout=20,
        options="-c statement_timeout=180000",
    )


def load_stock_history() -> pd.DataFrame:
    sql = """
        SELECT
            trade_date, symbol, stock_name, is_st,
            open, high, low, close, volume, amount, pct_change, turnover_rate,
            float_mv, total_mv, ma5, ma20, amount_ma_5,
            limit_up_today, limit_down_today
        FROM stock_daily_latest
        WHERE trade_date >= (
            SELECT MAX(trade_date) - INTERVAL '140 days'
            FROM stock_daily_latest
        )
        ORDER BY symbol, trade_date
    """
    connection = database_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            columns = [item.name for item in cursor.description]
            rows = cursor.fetchall()
    finally:
        connection.close()
    if not rows:
        raise RuntimeError("stock_daily_latest contains no recent rows")
    return pd.DataFrame(rows, columns=columns)


def prepare_stock_history(stock: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "symbol", "open", "high", "low", "close", "amount"}
    missing = required - set(stock.columns)
    if missing:
        raise ValueError(f"stock history is missing columns: {sorted(missing)}")
    prepared = stock.copy()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"], errors="coerce")
    prepared["symbol"] = prepared["symbol"].map(normalize_symbol)
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pct_change",
        "turnover_rate",
        "float_mv",
        "total_mv",
        "ma5",
        "ma20",
        "amount_ma_5",
        "is_st",
        "limit_up_today",
        "limit_down_today",
    ]
    for column in numeric_columns:
        if column not in prepared:
            prepared[column] = np.nan
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=["trade_date", "symbol", "close"])
    prepared = prepared.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    grouped = prepared.groupby("symbol", group_keys=False)
    if "stock_name" not in prepared:
        prepared["stock_name"] = prepared["symbol"]
    prepared["stock_name"] = grouped["stock_name"].transform(
        lambda values: values.ffill().bfill()
    )

    raw_pct = prepared["pct_change"].abs().quantile(0.99)
    pct_divisor = 100.0 if pd.notna(raw_pct) and raw_pct > 1.0 else 1.0
    prepared["pct_return"] = prepared["pct_change"] / pct_divisor
    fallback_return = grouped["close"].pct_change(fill_method=None)
    prepared["pct_return"] = prepared["pct_return"].fillna(fallback_return)

    # BaoStock writes turnover as a ratio (0.08 means 8%) while older imports
    # used percentage points. Normalize once so the selection thresholds and
    # PDF display both consistently use percentage points.
    turnover_q99 = prepared["turnover_rate"].abs().quantile(0.99)
    if pd.notna(turnover_q99) and turnover_q99 <= 1.0:
        prepared["turnover_rate"] = prepared["turnover_rate"] * 100.0

    prepared["ret5_calc"] = grouped["close"].pct_change(5, fill_method=None)
    prepared["ret20_calc"] = grouped["close"].pct_change(20, fill_method=None)
    prepared["ma5_calc"] = grouped["close"].transform(
        lambda values: values.rolling(5, min_periods=4).mean()
    )
    prepared["ma20_calc"] = grouped["close"].transform(
        lambda values: values.rolling(20, min_periods=15).mean()
    )
    prepared["amount_ma5_calc"] = grouped["amount"].transform(
        lambda values: values.rolling(5, min_periods=3).mean()
    )
    prepared["ma5_calc"] = prepared["ma5_calc"].fillna(prepared["ma5"])
    prepared["ma20_calc"] = prepared["ma20_calc"].fillna(prepared["ma20"])
    prepared["amount_ma5_calc"] = prepared["amount_ma5_calc"].fillna(
        prepared["amount_ma_5"]
    )

    positive_float_mv = prepared["float_mv"].where(prepared["float_mv"] > 0)
    float_shares = positive_float_mv / prepared["close"]
    prepared["float_shares_proxy"] = float_shares.replace([np.inf, -np.inf], np.nan)
    prepared["float_shares_proxy"] = grouped["float_shares_proxy"].transform(
        lambda values: values.ffill().bfill()
    )
    prepared["market_weight"] = (
        prepared["float_shares_proxy"] * prepared["close"]
    ).where(lambda values: values > 0)
    prepared["market_weight"] = prepared["market_weight"].fillna(positive_float_mv)
    positive_total_mv = prepared["total_mv"].where(prepared["total_mv"] > 0)
    prepared["market_weight"] = prepared["market_weight"].fillna(positive_total_mv)
    prepared["market_weight"] = prepared["market_weight"].fillna(1.0)
    return prepared


def weighted_return(frame: pd.DataFrame) -> float:
    valid = frame.dropna(subset=["pct_return", "market_weight"])
    valid = valid[valid["market_weight"] > 0]
    if valid.empty:
        return math.nan
    return float(np.average(valid["pct_return"], weights=valid["market_weight"]))


def analyze_concepts(
    stock: pd.DataFrame,
    concepts: list[dict[str, Any]],
    raw_memberships: dict[str, list[dict[str, Any]]],
) -> tuple[pd.DataFrame, dict[str, set[str]], dict[str, Any]]:
    dates = pd.Index(sorted(stock["trade_date"].dropna().unique()))
    if len(dates) < 26:
        raise RuntimeError("At least 26 trading days are required for the report")
    latest_date = pd.Timestamp(dates[-1])
    latest = stock[stock["trade_date"] == latest_date].copy()
    latest_by_symbol = latest.set_index("symbol", drop=False)

    benchmark_values: dict[pd.Timestamp, float] = {}
    for trade_date, frame in stock.groupby("trade_date"):
        benchmark_values[pd.Timestamp(trade_date)] = weighted_return(frame)
    benchmark_ret = pd.Series(benchmark_values).sort_index()
    benchmark_cum = (1 + benchmark_ret.fillna(0)).cumprod()

    member_sets: dict[str, set[str]] = {}
    results: list[dict[str, Any]] = []
    for concept in concepts:
        symbols = {
            normalized
            for row in raw_memberships.get(concept["code"], [])
            if (normalized := normalize_symbol(str(row.get("symbol", ""))))
        }
        member_sets[concept["code"]] = symbols
        mapped = sorted(symbols.intersection(latest_by_symbol.index))
        if len(mapped) < 5:
            continue
        subset = stock[stock["symbol"].isin(mapped)]
        daily_rows: list[tuple[pd.Timestamp, float, float, int]] = []
        for trade_date, frame in subset.groupby("trade_date"):
            valid = frame.dropna(
                subset=["market_weight", "close", "ma20_calc", "ret5_calc"]
            )
            valid = valid[valid["market_weight"] > 0]
            if valid.empty:
                continue
            rising = (valid["close"] > valid["ma20_calc"]) & (valid["ret5_calc"] > 0)
            breadth = float(
                np.average(rising.astype(float), weights=valid["market_weight"])
            )
            daily_rows.append(
                (
                    pd.Timestamp(trade_date),
                    breadth,
                    weighted_return(frame),
                    len(valid),
                )
            )
        daily = (
            pd.DataFrame(
                daily_rows,
                columns=["trade_date", "breadth", "board_ret", "covered"],
            )
            .set_index("trade_date")
            .sort_index()
        )
        if len(daily) < 25:
            continue
        daily["breadth_ma20"] = daily["breadth"].rolling(20, min_periods=15).mean()
        if len(daily) < 6 or pd.isna(daily["breadth_ma20"].iloc[-6]):
            continue
        breadth_latest = float(daily["breadth_ma20"].iloc[-1])
        breadth_delta5 = float(
            daily["breadth_ma20"].iloc[-1] - daily["breadth_ma20"].iloc[-6]
        )

        board_cum = (1 + daily["board_ret"].fillna(0)).cumprod()
        common = board_cum.index.intersection(benchmark_cum.index)
        relative = board_cum.loc[common] / benchmark_cum.loc[common]
        relative_mean = relative.rolling(20, min_periods=15).mean()
        ratio_series = relative / relative_mean - 1
        if len(ratio_series) < 6 or pd.isna(ratio_series.iloc[-6]):
            continue
        ratio_raw = float(ratio_series.iloc[-1])
        momentum_raw = float(ratio_series.iloc[-1] - ratio_series.iloc[-6])

        leaders = latest_by_symbol.loc[mapped].copy()
        if isinstance(leaders, pd.Series):
            leaders = leaders.to_frame().T
        leaders["z5"] = zscore(leaders["ret5_calc"])
        leaders["amount_ratio"] = (
            leaders["amount"] / leaders["amount_ma5_calc"]
        ).replace([np.inf, -np.inf], np.nan)
        leaders["close_pos"] = (
            ((leaders["close"] - leaders["low"]) / (leaders["high"] - leaders["low"]))
            .replace([np.inf, -np.inf], 0.5)
            .fillna(0.5)
            .clip(0, 1)
        )
        amount_sum = leaders["amount"].sum()
        leaders["amount_share"] = (
            leaders["amount"] / amount_sum if amount_sum > 0 else 0.0
        )
        leaders["leader_score"] = (
            0.30 * percentile(leaders["ret5_calc"])
            + 0.20 * percentile(leaders["amount_ratio"])
            + 0.15 * percentile(leaders["turnover_rate"])
            + 0.15 * leaders["close_pos"]
            + 0.10 * percentile(leaders["amount_share"])
            + 0.10 * percentile(leaders["ret20_calc"])
        )
        leaders["confirmed"] = (
            (leaders["z5"] > 1)
            & (leaders["amount_ratio"] > 1)
            & (leaders["close_pos"] >= 0.70)
            & (leaders["close"] > leaders["ma5_calc"])
        )
        confirmed = leaders[leaders["confirmed"]]
        pool = confirmed if not confirmed.empty else leaders
        leader = pool.sort_values("leader_score", ascending=False).iloc[0]
        unweighted = float(
            (
                (leaders["close"] > leaders["ma20_calc"]) & (leaders["ret5_calc"] > 0)
            ).mean()
        )
        results.append(
            {
                **concept,
                "member_count": len(symbols),
                "mapped_count": len(mapped),
                "coverage": len(mapped) / max(len(symbols), 1),
                "breadth_raw": float(daily["breadth"].iloc[-1]),
                "breadth_unweighted": unweighted,
                "breadth20": breadth_latest,
                "breadth_delta5": breadth_delta5,
                "ratio_raw": ratio_raw,
                "momentum_raw": momentum_raw,
                "leader_symbol": str(leader["symbol"]),
                "leader_name": str(leader["stock_name"]),
                "leader_confirmed": bool(leader["confirmed"]),
                "leader_score": float(leader["leader_score"]),
            }
        )

    result = pd.DataFrame(results)
    if result.empty:
        raise RuntimeError("No concepts passed the coverage and history checks")
    result = result[(result["member_count"] >= 8) & (result["coverage"] >= 0.60)]
    result = result[~result["name"].str.contains(NON_THEMATIC_CONCEPT, na=False)]
    if len(result) < 4:
        raise RuntimeError("Too few thematic concepts remain after validation")
    result = result.copy()
    result["rs_ratio"] = 100 + 10 * zscore(result["ratio_raw"])
    result["rs_momentum"] = 100 + 10 * zscore(result["momentum_raw"])
    result["quadrant"] = [
        classify_quadrant(ratio, momentum)
        for ratio, momentum in zip(
            result["rs_ratio"], result["rs_momentum"], strict=False
        )
    ]
    result["score"] = 100 * (
        0.30 * percentile(result["breadth20"])
        + 0.20 * percentile(result["breadth_delta5"])
        + 0.15 * percentile(result["ratio_raw"])
        + 0.15 * percentile(result["momentum_raw"])
        + 0.20 * percentile(result["leader_score"])
    )
    expansion_cut = float(result["breadth20"].quantile(0.60))
    result["eligible"] = (
        result["quadrant"].isin(["领先区", "改善区"])
        & (result["breadth20"] >= expansion_cut)
        & (result["breadth_delta5"] > 0)
    )

    latest_valid = latest.dropna(subset=["pct_return"])
    previous = stock[stock["trade_date"] == pd.Timestamp(dates[-2])]
    previous_count = int(previous["close"].notna().sum())
    snapshot_ratio = len(latest_valid) / max(previous_count, 1)
    snapshot_complete = len(latest_valid) >= 1000 and snapshot_ratio >= 0.85
    win_rate = float((latest_valid["pct_return"] > 0).mean())
    if win_rate < 0.40:
        regime = "退潮"
    elif win_rate <= 0.55:
        regime = "分化"
    elif win_rate <= 0.75:
        regime = "活跃"
    else:
        regime = "过热"
    market = {
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "stock_count": int(len(latest_valid)),
        "previous_stock_count": previous_count,
        "snapshot_ratio": snapshot_ratio,
        "snapshot_complete": snapshot_complete,
        "win_rate": win_rate,
        "median_pct": float(latest_valid["pct_return"].median()),
        "limit_up": int(
            (
                (latest_valid["limit_up_today"] == 1)
                | (latest_valid["pct_return"] >= 0.098)
            ).sum()
        ),
        "limit_down": int(
            (
                (latest_valid["limit_down_today"] == 1)
                | (latest_valid["pct_return"] <= -0.098)
            ).sum()
        ),
        "benchmark_5d": float(benchmark_cum.iloc[-1] / benchmark_cum.iloc[-6] - 1),
        "benchmark_20d": float(benchmark_cum.iloc[-1] / benchmark_cum.iloc[-21] - 1),
        "concept_count": int(len(result)),
        "expansion_cut": expansion_cut,
        "regime": regime,
    }
    return result, member_sets, market


def _stock_candidate_row(
    row: pd.Series, concept: pd.Series, z5: float
) -> dict[str, Any]:
    amount_ratio = float(row["amount"] / row["amount_ma5_calc"])
    price_range = float(row["high"] - row["low"])
    close_pos = (
        float((row["close"] - row["low"]) / price_range) if price_range > 0 else 0.5
    )
    return {
        "symbol": str(row["symbol"]),
        "stock_name": str(row.get("stock_name") or row["symbol"]),
        "concept_code": str(concept["code"]),
        "concept_name": str(concept["name"]),
        "concept_score": float(concept["score"]),
        "quadrant": str(concept["quadrant"]),
        "close": float(row["close"]),
        "previous_low": float(row["low"]),
        "ma5": float(row["ma5_calc"]),
        "ma20": float(row["ma20_calc"]),
        "ret5": float(row["ret5_calc"]),
        "ret20": float(row["ret20_calc"]),
        "pct_change": float(row["pct_return"]),
        "amount_ratio": amount_ratio,
        "turnover_rate": float(row.get("turnover_rate") or 0),
        "close_pos": close_pos,
        "z5": float(z5),
    }


def build_stock_recommendations(
    boards: pd.DataFrame,
    latest: pd.DataFrame,
    memberships: dict[str, set[str]],
    max_buy: int = 5,
    max_watch: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = boards[
        boards["quadrant"].isin(["领先区", "改善区"]) & (boards["breadth_delta5"] > 0)
    ].copy()
    if "eligible" in eligible:
        eligible = eligible[eligible["eligible"]]
    eligible = eligible.sort_values("score", ascending=False)
    latest_by_symbol = latest.set_index("symbol", drop=False)
    buy_pool: list[dict[str, Any]] = []
    watch_pool: list[dict[str, Any]] = []

    for _, concept in eligible.iterrows():
        symbols = sorted(memberships.get(str(concept["code"]), set()))
        mapped = [symbol for symbol in symbols if symbol in latest_by_symbol.index]
        if not mapped:
            continue
        concept_stocks = latest_by_symbol.loc[mapped].copy()
        if isinstance(concept_stocks, pd.Series):
            concept_stocks = concept_stocks.to_frame().T
        z5_values = zscore(concept_stocks["ret5_calc"])
        for index, row in concept_stocks.iterrows():
            name = str(row.get("stock_name") or "")
            is_st_value = pd.to_numeric(row.get("is_st"), errors="coerce")
            limit_up = pd.to_numeric(row.get("limit_up_today"), errors="coerce")
            limit_down = pd.to_numeric(row.get("limit_down_today"), errors="coerce")
            is_st = (pd.notna(is_st_value) and is_st_value != 0) or (
                "ST" in name.upper()
            )
            is_limit = (pd.notna(limit_up) and limit_up != 0) or (
                pd.notna(limit_down) and limit_down != 0
            )
            required = [
                "close",
                "low",
                "high",
                "ma5_calc",
                "ma20_calc",
                "ret5_calc",
                "ret20_calc",
                "pct_return",
                "amount",
                "amount_ma5_calc",
                "turnover_rate",
            ]
            if is_st or is_limit or row[required].isna().any():
                continue
            if (
                row["close"] <= 0
                or row["high"] < row["low"]
                or row["amount_ma5_calc"] <= 0
                or row["amount"] < 0
            ):
                continue
            candidate = _stock_candidate_row(row, concept, z5_values.loc[index])
            numeric_values = [
                candidate["amount_ratio"],
                candidate["close_pos"],
                candidate["turnover_rate"],
                candidate["ret5"],
                candidate["ret20"],
                candidate["pct_change"],
            ]
            if not all(math.isfinite(value) for value in numeric_values):
                continue
            conditions = {
                "五日强度Z分数未超过1": candidate["z5"] > 1.0,
                "量能未达到1.05倍": candidate["amount_ratio"] >= 1.05,
                "收盘位置低于当日振幅70%": candidate["close_pos"] >= 0.70,
                "未同时站上MA5和MA20": (
                    candidate["close"] > candidate["ma5"]
                    and candidate["close"] > candidate["ma20"]
                ),
                "当日涨跌幅不在-2%至7%": -0.02 <= candidate["pct_change"] <= 0.07,
                "五日涨幅不在3%至25%": 0.03 <= candidate["ret5"] <= 0.25,
                "换手率不在1%至30%": 1.0 <= candidate["turnover_rate"] <= 30.0,
            }
            unmet = [message for message, passed in conditions.items() if not passed]
            candidate["stock_score"] = float(
                0.55 * candidate["concept_score"]
                + 8.0 * min(max(candidate["z5"], 0), 3)
                + 8.0 * min(candidate["amount_ratio"], 2.5)
                + 8.0 * candidate["close_pos"]
            )
            candidate["trigger"] = (
                "次日开盘涨幅不超过3%，且价格不跌破前一日低点或MA5；"
                "盘中放量转强时再触发"
            )
            candidate["invalidation"] = (
                "高开超过5%、跌破前一日低点或MA5时取消买入，只保留观察"
            )
            candidate["unmet_conditions"] = "；".join(unmet)
            if not unmet:
                buy_pool.append(candidate)
                continue
            watch_ok = (
                candidate["close"] > candidate["ma20"]
                and candidate["ret5"] >= -0.02
                and candidate["ret5"] <= 0.30
                and candidate["amount_ratio"] >= 0.75
                and candidate["close_pos"] >= 0.45
                and -0.05 <= candidate["pct_change"] <= 0.08
            )
            if watch_ok:
                watch_pool.append(candidate)

    def select_unique(
        pool: list[dict[str, Any]], limit: int, excluded: set[str] | None = None
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        symbols = set(excluded or set())
        concept_counts: dict[str, int] = {}
        for item in sorted(pool, key=lambda value: value["stock_score"], reverse=True):
            symbol = item["symbol"]
            concept_code = item["concept_code"]
            if symbol in symbols or concept_counts.get(concept_code, 0) >= 2:
                continue
            selected.append(item)
            symbols.add(symbol)
            concept_counts[concept_code] = concept_counts.get(concept_code, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    buys = select_unique(buy_pool, max_buy)
    watches = select_unique(
        watch_pool, max_watch, excluded={item["symbol"] for item in buys}
    )
    return buys, watches


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    return [
        {key: finite_number(value) for key, value in row.items()}
        for row in frame[columns].to_dict(orient="records")
    ]


def build_report(
    stock: pd.DataFrame,
    concepts: list[dict[str, Any]],
    memberships: dict[str, list[dict[str, Any]]],
    max_buy: int = 5,
    max_watch: int = 10,
) -> tuple[dict[str, Any], pd.DataFrame]:
    prepared = prepare_stock_history(stock)
    boards, member_sets, market = analyze_concepts(prepared, concepts, memberships)
    latest_date = prepared["trade_date"].max()
    latest = prepared[prepared["trade_date"] == latest_date].copy()
    buys, watches = build_stock_recommendations(
        boards, latest, member_sets, max_buy=max_buy, max_watch=max_watch
    )
    today = datetime.now().astimezone().date()
    data_age_days = (today - pd.Timestamp(latest_date).date()).days
    date_fresh = (
        data_age_days <= 3
        if today.weekday() >= 5
        else pd.Timestamp(latest_date).date() == today
    )
    data_fresh = date_fresh and bool(market["snapshot_complete"])
    market["data_age_days"] = data_age_days
    market["data_fresh"] = data_fresh
    market["freshness_note"] = (
        "数据日期和最新截面完整性检查通过"
        if data_fresh
        else (
            "行情日期未更新至当前工作日，条件候选已全部降级为观察"
            if not date_fresh
            else "最新行情截面不足前一交易日的85%，条件候选已全部降级为观察"
        )
    )
    if not data_fresh:
        market["regime"] = "数据滞后"
    if market["regime"] in {"退潮", "过热", "数据滞后"}:
        for item in buys:
            item["unmet_conditions"] = f"市场环境为{market['regime']}，降级为观察"
        watches = (buys + watches)[:max_watch]
        buys = []

    ranked = boards.sort_values("score", ascending=False)
    eligible = ranked[ranked["eligible"]]
    if len(eligible) < 10:
        eligible = ranked[
            ranked["quadrant"].isin(["领先区", "改善区"])
            & (ranked["breadth20"] >= market["expansion_cut"])
        ]
    selected_codes: list[str] = []
    for code in eligible["code"]:
        candidate_set = member_sets.get(code, set())
        duplicate = False
        for chosen in selected_codes:
            chosen_set = member_sets.get(chosen, set())
            union = candidate_set | chosen_set
            if union and len(candidate_set & chosen_set) / len(union) >= 0.55:
                duplicate = True
                break
        if not duplicate:
            selected_codes.append(code)
        if len(selected_codes) >= 10:
            break
    top = ranked.set_index("code").loc[selected_codes].reset_index()
    top_columns = [
        "code",
        "name",
        "score",
        "quadrant",
        "rs_ratio",
        "rs_momentum",
        "breadth20",
        "breadth_delta5",
        "member_count",
        "coverage",
        "leader_symbol",
        "leader_name",
        "leader_confirmed",
    ]
    plot_columns = [
        "code",
        "name",
        "score",
        "quadrant",
        "rs_ratio",
        "rs_momentum",
        "breadth20",
        "breadth_delta5",
    ]
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "market": market,
        "top": _records(top, top_columns),
        "plot": _records(ranked, plot_columns),
        "quadrant_counts": {
            str(key): int(value)
            for key, value in boards["quadrant"].value_counts().items()
        },
        "buy_candidates": buys,
        "watchlist": watches,
        "method": {
            "breadth_state": "收盘价高于MA20且近5日收益为正",
            "breadth_weight": (
                "自由流通市值加权；缺失时依次回退流通市值、总市值、等权"
            ),
            "breadth_smoothing": "20个交易日移动平均，趋势为相对5日前变化",
            "rrg_benchmark": "全A自由流通市值加权组合",
            "leader_confirmation": (
                "板块内5日收益Z分数>1、成交额放大、收盘位置>=70%、站上MA5"
            ),
            "candidate_policy": (
                "仅输出次日条件触发候选；退潮或过热环境自动降级为观察"
            ),
            "membership": "新浪财经当前概念成分，使用7日缓存并做覆盖率校验",
        },
    }
    return report, boards


def _find_chinese_font() -> str | None:
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    return next((path for path in candidates if Path(path).exists()), None)


def _rrg_chart(plot: list[dict[str, Any]]) -> BytesIO:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    font_path = _find_chinese_font()
    font = font_manager.FontProperties(fname=font_path) if font_path else None
    frame = pd.DataFrame(plot)
    fig, axis = plt.subplots(figsize=(9.2, 5.4), dpi=150)
    x_delta = max(float((frame["rs_ratio"] - 100).abs().max()), 5.0) * 1.08
    y_delta = max(float((frame["rs_momentum"] - 100).abs().max()), 5.0) * 1.08
    axis.set_xlim(100 - x_delta, 100 + x_delta)
    axis.set_ylim(100 - y_delta, 100 + y_delta)
    axis.axvspan(100 - x_delta, 100, ymin=0.5, color="#fff4cc")
    axis.axvspan(100, 100 + x_delta, ymin=0.5, color="#e6f5e9")
    axis.axvspan(100 - x_delta, 100, ymax=0.5, color="#f4f4f4")
    axis.axvspan(100, 100 + x_delta, ymax=0.5, color="#fde8e7")
    axis.axvline(100, color="#5b6472", linewidth=1)
    axis.axhline(100, color="#5b6472", linewidth=1)
    colors = {
        "领先区": "#1a8f5d",
        "改善区": "#d69e00",
        "转弱区": "#d95040",
        "落后区": "#7b8492",
    }
    for quadrant, group in frame.groupby("quadrant"):
        axis.scatter(
            group["rs_ratio"],
            group["rs_momentum"],
            s=18 + group["score"].clip(lower=0) * 0.65,
            alpha=0.68,
            color=colors.get(str(quadrant), "#40566f"),
            edgecolors="white",
            linewidths=0.5,
            label=str(quadrant),
        )
    for item in frame.nlargest(min(14, len(frame)), "score").itertuples():
        axis.annotate(
            item.name,
            (item.rs_ratio, item.rs_momentum),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            fontproperties=font,
        )
    axis.set_xlabel("RS-Ratio（相对强度）", fontproperties=font)
    axis.set_ylabel("RS-Momentum（相对动量）", fontproperties=font)
    axis.set_title("概念板块 RRG 四象限", fontproperties=font, fontsize=13)
    legend = axis.legend(loc="best", fontsize=8)
    if font:
        for label in legend.get_texts():
            label.set_fontproperties(font)
    axis.grid(alpha=0.18)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def render_report_pdf(report: dict[str, Any], target: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    font_path = _find_chinese_font()
    font_name = "QMChinese"
    try:
        if font_path:
            pdfmetrics.registerFont(TTFont(font_name, font_path, subfontIndex=0))
        else:
            raise ValueError("no local Chinese font")
    except Exception:
        font_name = "STSong-Light"
        try:
            pdfmetrics.getFont(font_name)
        except KeyError:
            pdfmetrics.registerFont(UnicodeCIDFont(font_name))

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "QMTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=20,
        leading=28,
        textColor=colors.HexColor("#172235"),
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "QMHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#173a5e"),
        spaceBefore=8,
        spaceAfter=7,
    )
    body_style = ParagraphStyle(
        "QMBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=8.5,
        leading=13,
        textColor=colors.HexColor("#303846"),
        alignment=TA_LEFT,
    )
    small_style = ParagraphStyle(
        "QMSmall",
        parent=body_style,
        fontSize=7.2,
        leading=10,
    )
    header_cell_style = ParagraphStyle(
        "QMHeaderCell",
        parent=small_style,
        textColor=colors.white,
    )

    def paragraph(value: Any, small: bool = False) -> Paragraph:
        safe = str(value if value is not None else "-").replace("&", "&amp;")
        return Paragraph(safe, small_style if small else body_style)

    def table(data: list[list[Any]], widths: list[float]) -> Table:
        converted: list[list[Paragraph]] = []
        for row_index, row in enumerate(data):
            style = header_cell_style if row_index == 0 else small_style
            converted.append(
                [
                    Paragraph(
                        str(value if value is not None else "-").replace("&", "&amp;"),
                        style,
                    )
                    for value in row
                ]
            )
        result = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173a5e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c7d0da")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f4f7fa")],
                    ),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        return result

    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont(font_name, 7)
        canvas.setFillColor(colors.HexColor("#697586"))
        canvas.drawString(18 * mm, 10 * mm, "QuantMind 概念轮动日报 · 仅供研究")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"第 {document.page} 页")
        canvas.restoreState()

    market = report["market"]
    temp_target = target.with_suffix(".tmp.pdf")
    document = SimpleDocTemplate(
        str(temp_target),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=17 * mm,
        title=f"QuantMind 概念轮动日报 {market['latest_date']}",
        author="QuantMind",
    )
    story: list[Any] = [
        Paragraph("QuantMind 概念轮动与次日候选", title_style),
        Paragraph(
            f"数据日期：{market['latest_date']}　市场状态：{market['regime']}　"
            f"生成时间：{report.get('generated_at', datetime.now().isoformat(timespec='minutes'))}",
            body_style,
        ),
        Spacer(1, 6),
        Paragraph("市场环境", heading_style),
    ]
    market_data = [
        ["上涨占比", "中位涨跌幅", "涨停/跌停", "全A 5日", "全A 20日"],
        [
            f"{market['win_rate']:.1%}",
            f"{market['median_pct']:.2%}",
            f"{market['limit_up']} / {market['limit_down']}",
            f"{market['benchmark_5d']:.2%}",
            f"{market['benchmark_20d']:.2%}",
        ],
    ]
    story.extend(
        [
            table(market_data, [34 * mm] * 5),
            Spacer(1, 7),
            Paragraph(
                f"{market.get('freshness_note', '')}。市场退潮、过热或数据滞后时，"
                "系统会自动取消条件买入列表并降级到观察池；"
                "列表为空是风险过滤的正常结果。",
                small_style,
            ),
            Paragraph("RRG 四象限", heading_style),
            Image(_rrg_chart(report["plot"]), width=176 * mm, height=102 * mm),
            PageBreak(),
            Paragraph("热门概念 Top 10", heading_style),
        ]
    )
    board_rows = [["概念", "象限", "得分", "扩散度", "5日变化", "龙头", "确认"]]
    for item in report.get("top", []):
        board_rows.append(
            [
                item["name"],
                item["quadrant"],
                f"{item['score']:.1f}",
                f"{item['breadth20']:.1%}",
                f"{item['breadth_delta5']:+.1%}",
                item["leader_name"],
                "是" if item["leader_confirmed"] else "待确认",
            ]
        )
    story.extend(
        [
            table(
                board_rows,
                [31 * mm, 19 * mm, 15 * mm, 21 * mm, 21 * mm, 36 * mm, 19 * mm],
            ),
            Paragraph("次日条件买入候选", heading_style),
        ]
    )
    buy_rows = [
        ["代码/名称", "概念", "5/20日", "量比/换手", "收盘位", "次日触发与失效"]
    ]
    for item in report.get("buy_candidates", []):
        buy_rows.append(
            [
                f"{item['symbol']}\n{item['stock_name']}",
                item["concept_name"],
                f"{item['ret5']:.1%} / {item['ret20']:.1%}",
                f"{item['amount_ratio']:.2f}x / {item['turnover_rate']:.1f}%",
                f"{item['close_pos']:.0%}",
                f"触发：{item['trigger']}\n失效：{item['invalidation']}",
            ]
        )
    if len(buy_rows) == 1:
        buy_rows.append(["无", "-", "-", "-", "-", "风险过滤后无合格标的"])
    story.extend(
        [
            table(
                buy_rows,
                [30 * mm, 24 * mm, 23 * mm, 29 * mm, 17 * mm, 50 * mm],
            ),
            Paragraph("观察池", heading_style),
        ]
    )
    watch_rows = [["代码/名称", "概念", "5日", "量比", "收盘位", "尚未满足"]]
    for item in report.get("watchlist", []):
        watch_rows.append(
            [
                f"{item['symbol']}\n{item['stock_name']}",
                item["concept_name"],
                f"{item['ret5']:.1%}",
                f"{item['amount_ratio']:.2f}x",
                f"{item['close_pos']:.0%}",
                item.get("unmet_conditions") or "等待次日确认",
            ]
        )
    if len(watch_rows) == 1:
        watch_rows.append(["无", "-", "-", "-", "-", "无趋势有效标的"])
    story.extend(
        [
            table(
                watch_rows,
                [30 * mm, 25 * mm, 18 * mm, 18 * mm, 18 * mm, 62 * mm],
            ),
            Spacer(1, 7),
            KeepTogether(
                [
                    Paragraph("口径与风险提示", heading_style),
                    Paragraph(
                        "扩散度使用自由流通市值加权；RRG以全A加权组合为基准。"
                        "当前概念成分来自新浪财经并应用于历史窗口，可能存在幸存者偏差。"
                        "候选仅代表量价条件满足，不代表次日一定上涨；次日必须等待触发条件，"
                        "禁止追高，并结合仓位、止损、流动性和公告风险独立决策。"
                        "本报告仅用于量化研究，不构成投资建议。",
                        body_style,
                    ),
                ]
            ),
        ]
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    temp_target.replace(target)


def build_feishu_payload(report: dict[str, Any], pdf_url: str) -> dict[str, Any]:
    market = report["market"]
    top = (
        "、".join(
            f"{item['name']}({item['quadrant']})" for item in report.get("top", [])[:5]
        )
        or "无"
    )
    buys = (
        "、".join(
            f"{item['symbol']} {item['stock_name']}[{item['concept_name']}]"
            for item in report.get("buy_candidates", [])
        )
        or "无（风险过滤后为空）"
    )
    watches = (
        "、".join(
            f"{item['symbol']} {item['stock_name']}"
            for item in report.get("watchlist", [])[:5]
        )
        or "无"
    )
    content = [
        [
            {
                "tag": "text",
                "text": f"数据日期：{market['latest_date']}　市场：{market['regime']}",
            }
        ],
        [{"tag": "text", "text": f"热门概念：{top}"}],
        [{"tag": "text", "text": f"次日条件候选：{buys}"}],
        [{"tag": "text", "text": f"观察池：{watches}"}],
        [
            {"tag": "a", "text": "下载完整 PDF 报告", "href": pdf_url},
        ],
        [
            {
                "tag": "text",
                "text": "候选需等待次日触发，禁止无条件追高；仅供研究，不构成投资建议。",
            }
        ],
    ]
    return {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": f"QuantMind 概念轮动日报 {market['latest_date']}",
                    "content": content,
                }
            }
        },
    }


def send_feishu(
    webhook: str, payload: dict[str, Any], retries: int = 3
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                webhook,
                data=body,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            if "code" in result:
                code = result["code"]
            elif "StatusCode" in result:
                code = result["StatusCode"]
            else:
                raise ValueError("Feishu response did not contain a status code")
            if code not in (0, "0"):
                raise ValueError(
                    "Feishu webhook rejected the message: "
                    f"{result.get('msg') or result.get('StatusMessage')}"
                )
            return result
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
        except ValueError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(
        f"Feishu webhook request failed after {retries} attempts: {last_error}"
    )


def write_outputs(
    report: dict[str, Any], boards: pd.DataFrame, output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    date_token = str(report["market"]["latest_date"]).replace("-", "")
    json_path = output_dir / f"concept_rotation_{date_token}.json"
    csv_path = output_dir / f"concept_rotation_all_{date_token}.csv"
    pdf_path = output_dir / f"concept_rotation_{date_token}.pdf"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    boards.sort_values("score", ascending=False).to_csv(csv_path, index=False)
    render_report_pdf(report, pdf_path)
    shutil.copyfile(pdf_path, output_dir / "latest.pdf")
    shutil.copyfile(json_path, output_dir / "latest.json")
    return pdf_path, output_dir / "latest.pdf"


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    LOGGER.info("Loading local stock history")
    stock = load_stock_history()
    LOGGER.info("Loading concept list and membership")
    concepts = load_concepts(cache_dir)
    memberships = load_memberships(
        concepts, cache_dir, cache_days=args.member_cache_days
    )
    LOGGER.info("Analyzing %s concepts", len(concepts))
    report, boards = build_report(
        stock,
        concepts,
        memberships,
        max_buy=args.max_buy,
        max_watch=args.max_watch,
    )
    pdf_path, _latest_path = write_outputs(report, boards, output_dir)
    base_url = args.public_base_url.rstrip("/")
    uploads_root = output_dir.parents[1]
    relative = pdf_path.relative_to(uploads_root)
    pdf_url = f"{base_url}/uploads/{relative.as_posix()}"
    LOGGER.info("Report generated: %s", pdf_path)
    if args.send_feishu:
        webhook = os.getenv("WEB_HOOK", "").strip()
        if not webhook:
            raise RuntimeError("WEB_HOOK is empty; PDF was generated but not sent")
        send_feishu(webhook, build_feishu_payload(report, pdf_url))
        LOGGER.info("Feishu notification sent with PDF URL: %s", pdf_url)
    print(
        json.dumps(
            {
                "data_date": report["market"]["latest_date"],
                "pdf": str(pdf_path),
                "pdf_url": pdf_url,
                "buy_candidates": len(report["buy_candidates"]),
                "watchlist": len(report["watchlist"]),
                "feishu_sent": bool(args.send_feishu),
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOGGER.exception("Daily concept rotation report failed: %s", exc)
        raise SystemExit(1) from exc
