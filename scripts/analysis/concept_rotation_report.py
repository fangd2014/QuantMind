#!/usr/bin/env python3
"""Generate the daily Shenwan industry-rotation report and notify Feishu.

The report combines SW2021 level-one industry breadth, an RRG-style
relative-strength model and leader confirmation. It reads prices only from the
local ``stock_daily_latest`` table, caches Tushare industry membership, writes
PDF and interactive HTML reports under the API's ``/uploads`` directory and
optionally sends both report URLs through a Feishu custom webhook.
"""

from __future__ import annotations

import argparse
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
import urllib.request
from collections.abc import Callable
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2


LOGGER = logging.getLogger("sw_industry_rotation_report")
TUSHARE_API_URL = "http://api.tushare.pro"
SW_INDUSTRY_VERSION = "SW2021"
DEFAULT_OUTPUT_DIR = "/data/uploads/reports/concept-rotation"
DEFAULT_CACHE_DIR = "/data/cache/concept-rotation"
DEFAULT_PUBLIC_BASE_URL = "http://192.168.5.10:18000"
DEFAULT_PREFLIGHT_STATUS_FILE = "/data/cache/concept-rotation/preflight.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SW industry rotation PDF/HTML reports and notify Feishu"
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
    parser.add_argument("--max-control-picks", type=int, default=10)
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


def query_tushare(
    api_name: str,
    params: dict[str, str],
    fields: tuple[str, ...],
    retries: int = 4,
) -> list[dict[str, Any]]:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not configured")
    body = json.dumps(
        {
            "api_name": api_name,
            "token": token,
            "params": params,
            "fields": ",".join(fields),
        }
    ).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries):
        if attempt == 0:
            time.sleep(0.12)
        try:
            request = urllib.request.Request(
                TUSHARE_API_URL,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                result = json.loads(response.read().decode("utf-8"))
            code = int(result.get("code") or 0)
            if code != 0:
                message = str(result.get("msg") or f"Tushare error {code}")
                if any(word in message for word in ("频率", "每分钟", "稍后")):
                    raise RuntimeError(message)
                raise ValueError(f"{api_name}: {message}")
            data = result.get("data") or {}
            names = list(data.get("fields") or [])
            return [
                dict(zip(names, row, strict=False)) for row in data.get("items") or []
            ]
        except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(
        f"{api_name} request failed after {retries} attempts: {last_error}"
    )


TushareQuery = Callable[[str, dict[str, str], tuple[str, ...]], list[dict[str, Any]]]


def load_sw_industry_universe(
    cache_dir: Path,
    cache_days: float = 7.0,
    query: TushareQuery | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Load SW2021 level-one industries and current constituents."""
    cache_path = cache_dir / "tushare-sw2021-l1-universe.json.gz"
    stale_payload: dict[str, Any] = {}
    if cache_path.exists():
        try:
            with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
                candidate = json.load(handle)
            if (
                isinstance(candidate, dict)
                and candidate.get("version") == SW_INDUSTRY_VERSION
                and isinstance(candidate.get("industries"), list)
                and isinstance(candidate.get("memberships"), dict)
            ):
                stale_payload = candidate
            age_days = (time.time() - cache_path.stat().st_mtime) / 86400
            if (
                age_days <= cache_days
                and len(stale_payload.get("industries", [])) >= 28
            ):
                LOGGER.info("Using %.1f-day-old SW2021 industry cache", age_days)
                return stale_payload["industries"], stale_payload["memberships"]
        except (OSError, ValueError, json.JSONDecodeError):
            LOGGER.exception("Ignoring invalid SW2021 industry cache")

    query_api = query or query_tushare
    try:
        classification = query_api(
            "index_classify",
            {"level": "L1", "src": SW_INDUSTRY_VERSION},
            ("index_code", "industry_name", "level", "src"),
        )
        industries = sorted(
            [
                {
                    "code": str(item["index_code"]),
                    "name": str(item["industry_name"]),
                    "level": str(item.get("level") or "L1"),
                    "source": str(item.get("src") or SW_INDUSTRY_VERSION),
                }
                for item in classification
                if item.get("index_code") and item.get("industry_name")
            ],
            key=lambda item: item["code"],
        )
        if not 28 <= len(industries) <= 40:
            raise ValueError(
                f"Expected 28-40 SW2021 L1 industries, received {len(industries)}"
            )

        memberships: dict[str, list[dict[str, Any]]] = {}
        for index, industry in enumerate(industries, 1):
            code = industry["code"]
            rows = query_api(
                "index_member_all",
                {"l1_code": code},
                (
                    "l1_code",
                    "l1_name",
                    "ts_code",
                    "name",
                    "in_date",
                    "out_date",
                    "is_new",
                ),
            )
            current = [
                {
                    "symbol": str(item["ts_code"]),
                    "name": str(item.get("name") or item["ts_code"]),
                    "in_date": item.get("in_date"),
                    "out_date": item.get("out_date"),
                }
                for item in rows
                if item.get("ts_code") and str(item.get("is_new") or "Y").upper() == "Y"
            ]
            unique = {item["symbol"]: item for item in current}
            memberships[code] = sorted(unique.values(), key=lambda item: item["symbol"])
            industry["reported_count"] = len(memberships[code])
            if index % 10 == 0:
                LOGGER.info(
                    "Downloaded SW2021 memberships %s/%s", index, len(industries)
                )

        successful = sum(len(rows) >= 5 for rows in memberships.values())
        if successful < len(industries) * 0.90:
            raise RuntimeError(
                f"Only {successful}/{len(industries)} SW2021 memberships are usable"
            )
        payload = {
            "version": SW_INDUSTRY_VERSION,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "industries": industries,
            "memberships": memberships,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(".tmp")
        with gzip.open(temp_path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        temp_path.replace(cache_path)
        return industries, memberships
    except Exception:
        if stale_payload:
            LOGGER.exception("SW2021 fetch failed; using stale industry cache")
            return stale_payload["industries"], stale_payload["memberships"]
        raise


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
    prepared["high20_calc"] = grouped["high"].transform(
        lambda values: values.rolling(20, min_periods=15).max()
    )
    prepared["low20_calc"] = grouped["low"].transform(
        lambda values: values.rolling(20, min_periods=15).min()
    )
    prepared["volatility20"] = grouped["pct_return"].transform(
        lambda values: values.rolling(20, min_periods=15).std(ddof=0)
    )
    prepared["absolute_return20"] = grouped["pct_return"].transform(
        lambda values: values.abs().rolling(20, min_periods=15).sum()
    )
    prepared["trend_efficiency20"] = (
        prepared["ret20_calc"].abs() / prepared["absolute_return20"]
    ).replace([np.inf, -np.inf], np.nan)
    up_amount = prepared["amount"].where(prepared["pct_return"] > 0, 0.0)
    down_amount = prepared["amount"].where(prepared["pct_return"] < 0, 0.0)
    prepared["up_amount20"] = up_amount.groupby(prepared["symbol"]).transform(
        lambda values: values.rolling(20, min_periods=15).sum()
    )
    prepared["down_amount20"] = down_amount.groupby(prepared["symbol"]).transform(
        lambda values: values.rolling(20, min_periods=15).sum()
    )
    prepared["up_down_amount_ratio20"] = (
        prepared["up_amount20"] / prepared["down_amount20"]
    ).replace([np.inf, -np.inf], np.nan)
    range20 = prepared["high20_calc"] - prepared["low20_calc"]
    prepared["range_pos20"] = (
        (prepared["close"] - prepared["low20_calc"]) / range20
    ).replace([np.inf, -np.inf], np.nan)
    prepared["drawdown20"] = (prepared["close"] / prepared["high20_calc"] - 1).replace(
        [np.inf, -np.inf], np.nan
    )
    intraday_range = prepared["high"] - prepared["low"]
    prepared["close_pos"] = (
        (prepared["close"] - prepared["low"]) / intraday_range
    ).replace([np.inf, -np.inf], 0.5)

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


def analyze_industries(
    stock: pd.DataFrame,
    industries: list[dict[str, Any]],
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
    for industry in industries:
        symbols = {
            normalized
            for row in raw_memberships.get(industry["code"], [])
            if (normalized := normalize_symbol(str(row.get("symbol", ""))))
        }
        member_sets[industry["code"]] = symbols
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
                **industry,
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
        raise RuntimeError("No SW industries passed the coverage and history checks")
    result = result[(result["member_count"] >= 8) & (result["coverage"] >= 0.60)]
    if len(result) < 4:
        raise RuntimeError("Too few SW industries remain after validation")
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
        "industry_count": int(len(result)),
        "expansion_cut": expansion_cut,
        "regime": regime,
    }
    return result, member_sets, market


def _stock_candidate_row(
    row: pd.Series, industry: pd.Series, z5: float
) -> dict[str, Any]:
    amount_ratio = float(row["amount"] / row["amount_ma5_calc"])
    price_range = float(row["high"] - row["low"])
    close_pos = (
        float((row["close"] - row["low"]) / price_range) if price_range > 0 else 0.5
    )
    return {
        "symbol": str(row["symbol"]),
        "stock_name": str(row.get("stock_name") or row["symbol"]),
        "industry_code": str(industry["code"]),
        "industry_name": str(industry["name"]),
        "industry_score": float(industry["score"]),
        "quadrant": str(industry["quadrant"]),
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

    for _, industry in eligible.iterrows():
        symbols = sorted(memberships.get(str(industry["code"]), set()))
        mapped = [symbol for symbol in symbols if symbol in latest_by_symbol.index]
        if not mapped:
            continue
        industry_stocks = latest_by_symbol.loc[mapped].copy()
        if isinstance(industry_stocks, pd.Series):
            industry_stocks = industry_stocks.to_frame().T
        z5_values = zscore(industry_stocks["ret5_calc"])
        for index, row in industry_stocks.iterrows():
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
            candidate = _stock_candidate_row(row, industry, z5_values.loc[index])
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
                0.55 * candidate["industry_score"]
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
        industry_counts: dict[str, int] = {}
        for item in sorted(pool, key=lambda value: value["stock_score"], reverse=True):
            symbol = item["symbol"]
            industry_code = item["industry_code"]
            if symbol in symbols or industry_counts.get(industry_code, 0) >= 2:
                continue
            selected.append(item)
            symbols.add(symbol)
            industry_counts[industry_code] = industry_counts.get(industry_code, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    buys = select_unique(buy_pool, max_buy)
    watches = select_unique(
        watch_pool, max_watch, excluded={item["symbol"] for item in buys}
    )
    return buys, watches


def build_leading_control_picks(
    boards: pd.DataFrame,
    latest: pd.DataFrame,
    memberships: dict[str, set[str]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Select explainable washout/breakout setups from leading SW industries.

    ``control`` is deliberately a price/volume proxy. Public OHLCV data cannot
    identify the true positions of a particular class of market participant.
    """
    hard_limit = max(0, min(int(limit), 10))
    if hard_limit == 0 or boards.empty or latest.empty:
        return []
    leading = boards[boards["quadrant"] == "领先区"].sort_values(
        "score", ascending=False
    )
    if leading.empty:
        return []
    latest_by_symbol = latest.set_index("symbol", drop=False)
    pool: list[dict[str, Any]] = []
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
        "high20_calc",
        "low20_calc",
        "up_down_amount_ratio20",
        "trend_efficiency20",
        "volatility20",
        "range_pos20",
        "drawdown20",
        "close_pos",
    ]
    for _, industry in leading.iterrows():
        symbols = sorted(memberships.get(str(industry["code"]), set()))
        mapped = [symbol for symbol in symbols if symbol in latest_by_symbol.index]
        for symbol in mapped:
            row = latest_by_symbol.loc[symbol]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            name = str(row.get("stock_name") or symbol)
            is_st_value = pd.to_numeric(row.get("is_st"), errors="coerce")
            limit_up = pd.to_numeric(row.get("limit_up_today"), errors="coerce")
            limit_down = pd.to_numeric(row.get("limit_down_today"), errors="coerce")
            is_st = (pd.notna(is_st_value) and is_st_value != 0) or (
                "ST" in name.upper()
            )
            is_limit = (
                (pd.notna(limit_up) and limit_up != 0)
                or (pd.notna(limit_down) and limit_down != 0)
                or abs(float(row.get("pct_return") or 0)) >= 0.095
            )
            if (
                is_st
                or is_limit
                or any(key not in row.index or pd.isna(row[key]) for key in required)
            ):
                continue
            values = {key: float(row[key]) for key in required}
            if not all(math.isfinite(value) for value in values.values()):
                continue
            if (
                values["close"] <= 0
                or values["high"] < values["low"]
                or values["amount_ma5_calc"] <= 0
                or values["amount"] < 0
            ):
                continue
            amount_ratio = values["amount"] / values["amount_ma5_calc"]
            controlled = (
                values["close"] > values["ma20_calc"]
                and values["ma5_calc"] >= values["ma20_calc"]
                and values["ret20_calc"] > 0
                and values["up_down_amount_ratio20"] >= 0.95
                and values["trend_efficiency20"] >= 0.12
                and values["volatility20"] <= 0.07
                and values["range_pos20"] >= 0.55
            )
            if not controlled:
                continue
            washout = (
                -0.06 <= values["ret5_calc"] <= 0.03
                and -0.12 <= values["drawdown20"] <= -0.02
                and 0.55 <= amount_ratio <= 1.05
                and values["close_pos"] >= 0.45
            )
            breakout = (
                values["close"] > values["ma5_calc"] > values["ma20_calc"]
                and 0.02 <= values["ret5_calc"] <= 0.15
                and -0.03 <= values["drawdown20"] <= 0
                and 1.05 <= amount_ratio <= 2.50
                and values["close_pos"] >= 0.65
                and 0 <= values["pct_return"] <= 0.07
            )
            if not washout and not breakout:
                continue
            stage = "开始拉升" if breakout else "洗盘"
            pool.append(
                {
                    "symbol": str(symbol),
                    "stock_name": name,
                    "industry_code": str(industry["code"]),
                    "industry_name": str(industry["name"]),
                    "industry_score": float(industry["score"]),
                    "quadrant": "领先区",
                    "rs_ratio": float(industry.get("rs_ratio") or 100.0),
                    "stage": stage,
                    "close": values["close"],
                    "ma20": values["ma20_calc"],
                    "ret5": values["ret5_calc"],
                    "ret20": values["ret20_calc"],
                    "pct_change": values["pct_return"],
                    "amount_ratio": amount_ratio,
                    "close_pos": values["close_pos"],
                    "range_pos20": values["range_pos20"],
                    "drawdown20": values["drawdown20"],
                    "up_down_amount_ratio20": values["up_down_amount_ratio20"],
                    "trend_efficiency20": values["trend_efficiency20"],
                    "volatility20": values["volatility20"],
                }
            )
    if not pool:
        return []

    scored = pd.DataFrame(pool)
    scored["control_score"] = 100 * (
        0.30 * percentile(scored["up_down_amount_ratio20"])
        + 0.25 * percentile(scored["trend_efficiency20"])
        + 0.15 * percentile(scored["range_pos20"])
        + 0.15 * percentile(-scored["volatility20"])
        + 0.15 * percentile(scored["ret20"])
    )
    scored["selection_score"] = (
        0.50 * scored["industry_score"]
        + 0.45 * scored["control_score"]
        + scored["stage"].map({"开始拉升": 5.0, "洗盘": 2.0}).fillna(0)
    )
    selected: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()
    industry_counts: dict[str, int] = {}
    for item in scored.sort_values("selection_score", ascending=False).to_dict(
        orient="records"
    ):
        symbol = str(item["symbol"])
        industry_code = str(item["industry_code"])
        if symbol in seen_symbols or industry_counts.get(industry_code, 0) >= 2:
            continue
        if item["stage"] == "洗盘":
            stage_evidence = (
                f"洗盘：距20日高点{item['drawdown20']:.1%}，量能缩至"
                f"5日均额{item['amount_ratio']:.2f}倍，仍守MA20"
            )
            invalidation = "收盘跌破MA20或前低、出现放量长阴时失效"
        else:
            stage_evidence = (
                f"开始拉升：距20日高点{item['drawdown20']:.1%}，量能为"
                f"5日均额{item['amount_ratio']:.2f}倍，近5日{item['ret5']:+.1%}"
            )
            invalidation = "收盘跌破MA20、放量长阴或次日高开超过5%时失效"
        item["reason"] = (
            f"{item['industry_name']}位于领先区（行业得分{item['industry_score']:.1f}，"
            f"RS {item['rs_ratio']:.1f}）；控盘量价代理：20日上涨/下跌成交额比"
            f"{item['up_down_amount_ratio20']:.2f}、趋势效率"
            f"{item['trend_efficiency20']:.2f}；{stage_evidence}。"
        )
        item["invalidation"] = invalidation
        item["action_label"] = "次日关注"
        selected.append({key: finite_number(value) for key, value in item.items()})
        seen_symbols.add(symbol)
        industry_counts[industry_code] = industry_counts.get(industry_code, 0) + 1
        if len(selected) >= hard_limit:
            break
    return selected


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    return [
        {key: finite_number(value) for key, value in row.items()}
        for row in frame[columns].to_dict(orient="records")
    ]


def build_report(
    stock: pd.DataFrame,
    industries: list[dict[str, Any]],
    memberships: dict[str, list[dict[str, Any]]],
    max_buy: int = 5,
    max_watch: int = 10,
    max_control_picks: int = 10,
) -> tuple[dict[str, Any], pd.DataFrame]:
    prepared = prepare_stock_history(stock)
    boards, member_sets, market = analyze_industries(prepared, industries, memberships)
    latest_date = prepared["trade_date"].max()
    latest = prepared[prepared["trade_date"] == latest_date].copy()
    buys, watches = build_stock_recommendations(
        boards, latest, member_sets, max_buy=max_buy, max_watch=max_watch
    )
    control_picks = build_leading_control_picks(
        boards, latest, member_sets, limit=max_control_picks
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
        for item in control_picks:
            item["action_label"] = "仅观察"
            item["risk_note"] = f"当前市场状态为{market['regime']}，不作为条件买入信号"

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
        "member_count",
        "mapped_count",
        "coverage",
        "leader_symbol",
        "leader_name",
        "leader_confirmed",
        "eligible",
    ]
    focus_columns = [
        "code",
        "name",
        "quadrant",
        "score",
        "rs_ratio",
        "rs_momentum",
        "breadth20",
        "breadth_delta5",
        "member_count",
        "mapped_count",
        "coverage",
        "leader_symbol",
        "leader_name",
        "leader_confirmed",
        "eligible",
    ]
    focus = ranked[ranked["quadrant"].isin(["领先区", "改善区"])].copy()
    focus["quadrant_order"] = focus["quadrant"].map({"领先区": 0, "改善区": 1})
    focus = focus.sort_values(["quadrant_order", "score"], ascending=[True, False])
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "universe": "申万2021版一级行业",
        "market": market,
        "top": _records(top, top_columns),
        "focus_industries": _records(focus, focus_columns),
        "plot": _records(ranked, plot_columns),
        "quadrant_counts": {
            str(key): int(value)
            for key, value in boards["quadrant"].value_counts().items()
        },
        "leading_control_picks": control_picks,
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
            "control_proxy": (
                "主力控盘仅为量价代理：综合20日上涨/下跌成交额结构、趋势效率、"
                "区间位置与波动率；不代表真实机构持仓"
            ),
            "control_stage": (
                "仅从领先区筛选守住MA20的缩量洗盘，或接近20日高点且温和放量的开始拉升"
            ),
            "membership": (
                "Tushare申万2021版一级行业及最新成分，使用7日缓存并做覆盖率校验"
            ),
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
    axis.set_title("申万一级行业 RRG 四象限", fontproperties=font, fontsize=13)
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
        canvas.drawString(18 * mm, 10 * mm, "QuantMind 申万行业轮动日报 · 仅供研究")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"第 {document.page} 页")
        canvas.restoreState()

    market = report["market"]
    preflight = report.get("data_preflight") or {}
    temp_target = target.with_suffix(".tmp.pdf")
    document = SimpleDocTemplate(
        str(temp_target),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=17 * mm,
        title=f"QuantMind 申万行业轮动日报 {market['latest_date']}",
        author="QuantMind",
    )
    story: list[Any] = [
        Paragraph("QuantMind 申万行业轮动与次日候选", title_style),
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
                (
                    "数据门禁：自动修复后复检通过。"
                    if preflight.get("repaired")
                    else "数据门禁：更新与质量检查通过。"
                )
                if preflight
                else "数据门禁状态：未随本次报告提供。",
                small_style,
            ),
            Paragraph(
                f"{market.get('freshness_note', '')}。市场退潮、过热或数据滞后时，"
                "系统会自动取消条件买入列表并降级到观察池；"
                "列表为空是风险过滤的正常结果。",
                small_style,
            ),
            Paragraph("RRG 四象限", heading_style),
            Image(_rrg_chart(report["plot"]), width=176 * mm, height=102 * mm),
            PageBreak(),
            Paragraph("领先区控盘阶段关注（最多10只）", heading_style),
        ]
    )
    control_rows = [["代码/名称", "申万行业", "阶段", "控盘分", "推荐理由 / 失效条件"]]
    for item in report.get("leading_control_picks", []):
        risk_note = item.get("risk_note")
        reason = f"{item['reason']}\n失效：{item['invalidation']}"
        if risk_note:
            reason = f"{reason}\n风险：{risk_note}"
        control_rows.append(
            [
                f"{item['symbol']}\n{item['stock_name']}",
                item["industry_name"],
                f"{item['stage']}\n{item.get('action_label', '次日关注')}",
                f"{item['control_score']:.1f}",
                reason,
            ]
        )
    if len(control_rows) == 1:
        control_rows.append(
            ["无", "-", "-", "-", "领先区暂无同时满足控盘代理与阶段条件的股票"]
        )
    story.extend(
        [
            Paragraph(
                "“主力控盘”为公开量价数据构建的代理信号，并非真实机构持仓识别；"
                "仅筛选洗盘或开始拉升阶段，行情滞后时一律仅观察。",
                small_style,
            ),
            Spacer(1, 5),
            table(
                control_rows,
                [29 * mm, 22 * mm, 20 * mm, 17 * mm, 85 * mm],
            ),
            PageBreak(),
            Paragraph("领先区 / 改善区行业清单", heading_style),
        ]
    )
    board_rows = [["申万行业", "区域", "得分", "扩散度", "5日变化", "板块龙头", "确认"]]
    for item in report.get("focus_industries", []):
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
        ["代码/名称", "申万行业", "5/20日", "量比/换手", "收盘位", "次日触发与失效"]
    ]
    for item in report.get("buy_candidates", []):
        buy_rows.append(
            [
                f"{item['symbol']}\n{item['stock_name']}",
                item["industry_name"],
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
    watch_rows = [["代码/名称", "申万行业", "5日", "量比", "收盘位", "尚未满足"]]
    for item in report.get("watchlist", []):
        watch_rows.append(
            [
                f"{item['symbol']}\n{item['stock_name']}",
                item["industry_name"],
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
                        "行业口径为Tushare申万2021版一级行业，最新成分应用于历史窗口，"
                        "可能存在成分调整带来的幸存者偏差。"
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


def render_interactive_html(report: dict[str, Any], target: Path) -> None:
    """Render a self-contained, searchable RRG report for static hosting."""
    import plotly.graph_objects as go
    import plotly.io as pio

    target.parent.mkdir(parents=True, exist_ok=True)
    boards = sorted(
        report.get("plot", []),
        key=lambda item: float(item.get("score") or 0),
        reverse=True,
    )
    colors = {
        "领先区": "#16835d",
        "改善区": "#ca8a04",
        "转弱区": "#dc5a4d",
        "落后区": "#77808f",
    }

    def number(item: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            value = float(item.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    x_values = [number(item, "rs_ratio", 100.0) for item in boards]
    y_values = [number(item, "rs_momentum", 100.0) for item in boards]
    x_delta = max([abs(value - 100) for value in x_values] + [5.0]) * 1.08
    y_delta = max([abs(value - 100) for value in y_values] + [5.0]) * 1.08

    def custom_data(item: dict[str, Any]) -> list[Any]:
        return [
            item.get("code", ""),
            item.get("quadrant", "未知"),
            number(item, "score"),
            number(item, "breadth20"),
            number(item, "breadth_delta5"),
            int(number(item, "member_count")),
            int(number(item, "mapped_count")),
            number(item, "coverage"),
            item.get("leader_name") or "待识别",
            item.get("leader_symbol") or "-",
            bool(item.get("leader_confirmed")),
            bool(item.get("eligible")),
        ]

    figure = go.Figure(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            text=[item.get("name") or item.get("code") or "未命名" for item in boards],
            customdata=[custom_data(item) for item in boards],
            marker={
                "size": [
                    14 + min(max(number(item, "score"), 0), 100) * 0.32
                    for item in boards
                ],
                "color": [
                    colors.get(str(item.get("quadrant")), "#45627d") for item in boards
                ],
                "opacity": 0.82,
                "line": {"color": "#ffffff", "width": 1},
            },
            hovertemplate=(
                "<b>%{text}</b><br>"
                "象限：%{customdata[1]}<br>"
                "综合得分：%{customdata[2]:.1f}<br>"
                "RS-Ratio：%{x:.2f}<br>"
                "RS-Momentum：%{y:.2f}<br>"
                "扩散度：%{customdata[3]:.1%}<br>"
                "5日扩散变化：%{customdata[4]:+.1%}<br>"
                "覆盖：%{customdata[6]}/%{customdata[5]} (%{customdata[7]:.1%})<br>"
                "龙头：%{customdata[8]} %{customdata[9]}"
                "<extra>点击查看完整信息</extra>"
            ),
        )
    )
    figure.update_layout(
        autosize=True,
        height=570,
        margin={"l": 62, "r": 24, "t": 42, "b": 58},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "-apple-system,BlinkMacSystemFont,'Segoe UI','Microsoft YaHei',sans-serif"
        },
        showlegend=False,
        hoverlabel={"align": "left"},
        xaxis={
            "title": "RS-Ratio（相对强度）",
            "range": [100 - x_delta, 100 + x_delta],
            "gridcolor": "rgba(113,128,150,.18)",
            "zeroline": False,
        },
        yaxis={
            "title": "RS-Momentum（相对动量）",
            "range": [100 - y_delta, 100 + y_delta],
            "gridcolor": "rgba(113,128,150,.18)",
            "zeroline": False,
        },
        shapes=[
            {
                "type": "rect",
                "xref": "x",
                "yref": "y",
                "x0": 100,
                "x1": 100 + x_delta,
                "y0": 100,
                "y1": 100 + y_delta,
                "fillcolor": "rgba(22,131,93,.10)",
                "line": {"width": 0},
                "layer": "below",
            },
            {
                "type": "rect",
                "xref": "x",
                "yref": "y",
                "x0": 100 - x_delta,
                "x1": 100,
                "y0": 100,
                "y1": 100 + y_delta,
                "fillcolor": "rgba(202,138,4,.10)",
                "line": {"width": 0},
                "layer": "below",
            },
            {
                "type": "rect",
                "xref": "x",
                "yref": "y",
                "x0": 100,
                "x1": 100 + x_delta,
                "y0": 100 - y_delta,
                "y1": 100,
                "fillcolor": "rgba(220,90,77,.09)",
                "line": {"width": 0},
                "layer": "below",
            },
            {
                "type": "rect",
                "xref": "x",
                "yref": "y",
                "x0": 100 - x_delta,
                "x1": 100,
                "y0": 100 - y_delta,
                "y1": 100,
                "fillcolor": "rgba(119,128,143,.09)",
                "line": {"width": 0},
                "layer": "below",
            },
            {
                "type": "line",
                "x0": 100,
                "x1": 100,
                "y0": 100 - y_delta,
                "y1": 100 + y_delta,
                "line": {"color": "#738095", "width": 1},
            },
            {
                "type": "line",
                "x0": 100 - x_delta,
                "x1": 100 + x_delta,
                "y0": 100,
                "y1": 100,
                "line": {"color": "#738095", "width": 1},
            },
        ],
        annotations=[
            {
                "x": 0.98,
                "y": 0.97,
                "xref": "paper",
                "yref": "paper",
                "text": "领先区",
                "showarrow": False,
            },
            {
                "x": 0.02,
                "y": 0.97,
                "xref": "paper",
                "yref": "paper",
                "text": "改善区",
                "showarrow": False,
            },
            {
                "x": 0.98,
                "y": 0.03,
                "xref": "paper",
                "yref": "paper",
                "text": "转弱区",
                "showarrow": False,
            },
            {
                "x": 0.02,
                "y": 0.03,
                "xref": "paper",
                "yref": "paper",
                "text": "落后区",
                "showarrow": False,
            },
        ],
    )
    chart = pio.to_html(
        figure,
        include_plotlyjs=True,
        full_html=False,
        div_id="rrg-chart",
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    embedded_data = (
        json.dumps(boards, ensure_ascii=False, allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    market = report.get("market", {})
    report_date = escape(str(market.get("latest_date", "-")))
    regime = escape(str(market.get("regime", "未知")))
    generated_at = escape(str(report.get("generated_at", "")))
    focus_rows: list[str] = []
    for item in report.get("focus_industries", []):
        code = escape(str(item.get("code") or "-"), quote=True)
        name = escape(str(item.get("name") or "未命名"))
        quadrant = escape(str(item.get("quadrant") or "未知"))
        leader = escape(
            f"{item.get('leader_name') or '待识别'} · "
            f"{item.get('leader_symbol') or '-'}"
        )
        confirmation = "已确认" if item.get("leader_confirmed") else "待确认"
        focus_rows.append(
            "<tr>"
            f'<td><button type="button" class="industry-link" data-code="{code}">'
            f"{name}</button><small>{code}</small></td>"
            f'<td><span class="quadrant-tag">{quadrant}</span></td>'
            f'<td class="numeric">{number(item, "score"):.1f}</td>'
            f'<td class="numeric">{number(item, "breadth20"):.1%}</td>'
            f'<td class="numeric">{number(item, "breadth_delta5"):+.1%}</td>'
            f"<td>{leader}<small>{confirmation}</small></td>"
            "</tr>"
        )
    focus_table_rows = "".join(focus_rows) or (
        '<tr><td colspan="6" class="empty-row">当前没有位于领先区或改善区的行业</td></tr>'
    )
    control_rows: list[str] = []
    for item in report.get("leading_control_picks", []):
        symbol = escape(str(item.get("symbol") or "-"))
        stock_name = escape(str(item.get("stock_name") or "未命名"))
        industry_name = escape(str(item.get("industry_name") or "-"))
        stage = escape(str(item.get("stage") or "-"))
        action_label = escape(str(item.get("action_label") or "次日关注"))
        reason = escape(str(item.get("reason") or "-"))
        invalidation = escape(str(item.get("invalidation") or "-"))
        risk_note = escape(str(item.get("risk_note") or ""))
        risk_html = f"<small>风险：{risk_note}</small>" if risk_note else ""
        control_rows.append(
            "<tr>"
            f"<td><strong>{stock_name}</strong><small>{symbol}</small></td>"
            f"<td>{industry_name}</td>"
            f'<td><span class="stage-tag">{stage}</span><small>{action_label}</small></td>'
            f'<td class="numeric">{number(item, "control_score"):.1f}</td>'
            f"<td>{reason}<small>失效：{invalidation}</small>{risk_html}</td>"
            "</tr>"
        )
    control_table_rows = "".join(control_rows) or (
        '<tr><td colspan="5" class="empty-row">领先区暂无同时满足控盘代理与阶段条件的股票</td></tr>'
    )
    html_document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="QuantMind 申万一级行业 RRG 四象限日报">
  <title>QuantMind 申万行业轮动 {report_date}</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#f3f6f9; --surface:#fff; --text:#172235; --muted:#647085; --border:#dce3ea; --accent:#173a5e; --soft:#eaf0f5; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; }}
    main {{ width:min(1480px,100%); margin:0 auto; padding:26px; }}
    header {{ display:flex; justify-content:space-between; gap:20px; align-items:flex-end; margin-bottom:20px; }}
    h1 {{ margin:0 0 7px; font-size:clamp(24px,3vw,38px); font-weight:650; letter-spacing:-.025em; }}
    .subtitle,.hint,.meta {{ color:var(--muted); }}
    .subtitle {{ margin:0; }}
    .status {{ padding:8px 12px; border:1px solid var(--border); border-radius:999px; background:var(--surface); white-space:nowrap; }}
    .toolbar {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px; align-items:center; }}
    label {{ position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); }}
    input,select,button {{ min-height:42px; border:1px solid var(--border); border-radius:9px; background:var(--surface); color:var(--text); padding:9px 12px; font:inherit; }}
    input {{ flex:1 1 260px; }}
    select {{ flex:0 1 180px; }}
    button {{ cursor:pointer; font-weight:600; }}
    button:hover {{ border-color:var(--accent); }}
    input:focus-visible,select:focus-visible,button:focus-visible {{ outline:3px solid color-mix(in srgb,var(--accent) 28%,transparent); outline-offset:2px; }}
    .count {{ margin-left:auto; color:var(--muted); }}
    .workspace {{ display:grid; grid-template-columns:minmax(0,2.25fr) minmax(300px,.75fr); gap:16px; align-items:start; }}
    .panel {{ background:var(--surface); border:1px solid var(--border); border-radius:14px; box-shadow:0 8px 30px rgba(24,42,64,.06); }}
    .chart-panel {{ padding:8px; min-width:0; }}
    #rrg-chart {{ width:100%; min-height:540px; }}
    .legend {{ display:flex; flex-wrap:wrap; gap:14px; padding:0 16px 14px; color:var(--muted); font-size:14px; }}
    .legend span::before {{ content:""; display:inline-block; width:9px; height:9px; margin-right:6px; border-radius:50%; background:var(--dot); }}
    .detail {{ padding:22px; position:sticky; top:16px; }}
    .eyebrow {{ color:var(--muted); font-size:13px; text-transform:uppercase; letter-spacing:.08em; }}
    h2 {{ margin:6px 0 3px; font-size:26px; }}
    .code {{ color:var(--muted); font-variant-numeric:tabular-nums; }}
    dl {{ display:grid; grid-template-columns:1fr 1fr; gap:0; margin:18px 0; }}
    .metric {{ padding:13px 0; border-top:1px solid var(--border); }}
    .metric:nth-child(odd) {{ padding-right:12px; }}
    dt {{ color:var(--muted); font-size:13px; }}
    dd {{ margin:5px 0 0; font-size:18px; font-weight:650; font-variant-numeric:tabular-nums; }}
    .leader {{ padding:14px; border-radius:10px; background:var(--soft); }}
    .leader strong {{ display:block; margin-top:5px; }}
    .badge {{ display:inline-block; margin-top:12px; padding:5px 9px; border-radius:999px; background:var(--soft); color:var(--text); font-size:13px; }}
    .table-panel {{ margin-top:16px; padding:22px; }}
    .table-panel h2 {{ margin:0 0 5px; }}
    .table-panel > p {{ margin:0 0 16px; color:var(--muted); }}
    .table-responsive {{ width:100%; overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; min-width:760px; }}
    th,td {{ padding:12px 10px; border-bottom:1px solid var(--border); text-align:left; vertical-align:middle; }}
    th {{ color:var(--muted); font-size:13px; font-weight:600; }}
    td small {{ display:block; margin-top:3px; color:var(--muted); }}
    .numeric {{ text-align:right; font-variant-numeric:tabular-nums; }}
    .industry-link {{ min-height:0; padding:0; border:0; border-radius:0; background:transparent; color:var(--accent); font-weight:650; }}
    .industry-link:hover {{ text-decoration:underline; }}
    .quadrant-tag {{ display:inline-block; padding:4px 8px; border-radius:999px; background:var(--soft); white-space:nowrap; }}
    .stage-tag {{ display:inline-block; padding:4px 8px; border-radius:999px; background:color-mix(in srgb,#16835d 14%,var(--surface)); color:#16835d; white-space:nowrap; font-weight:650; }}
    .control-note {{ padding:12px 14px; border-left:3px solid #ca8a04; background:var(--soft); border-radius:8px; line-height:1.6; }}
    #leading-control-picks td:last-child {{ min-width:420px; line-height:1.55; }}
    .empty-row {{ color:var(--muted); text-align:center; }}
    footer {{ margin-top:17px; color:var(--muted); font-size:13px; line-height:1.65; }}
    @media (prefers-color-scheme:dark) {{ :root {{ --bg:#101722; --surface:#172130; --text:#edf3f8; --muted:#aab6c4; --border:#2d3a4a; --accent:#85baf0; --soft:#202d3e; }} .panel {{ box-shadow:none; }} }}
    @media (max-width:900px) {{ main {{ padding:18px; }} header {{ align-items:flex-start; flex-direction:column; }} .workspace {{ grid-template-columns:1fr; }} .detail {{ position:static; }} .count {{ width:100%; margin-left:0; }} }}
    @media (max-width:520px) {{ main {{ padding:12px; }} .toolbar > * {{ flex:1 1 100%; }} #rrg-chart {{ min-height:430px; }} dl {{ grid-template-columns:1fr; }} .metric:nth-child(odd) {{ padding-right:0; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>申万一级行业 RRG 四象限</h1>
      <p class="subtitle">申万2021版一级行业 · 数据日期 {report_date} · 悬停查看指标，点击行业查看完整信息</p>
    </div>
    <div class="status">市场状态：<strong>{regime}</strong></div>
  </header>
  <section class="toolbar" aria-label="行业筛选">
    <label for="board-search">搜索申万行业</label>
    <input id="board-search" type="search" placeholder="搜索行业名称或代码" autocomplete="off">
    <label for="quadrant-filter">筛选象限</label>
    <select id="quadrant-filter">
      <option value="">全部象限</option>
      <option value="领先区">领先区</option>
      <option value="改善区">改善区</option>
      <option value="转弱区">转弱区</option>
      <option value="落后区">落后区</option>
    </select>
    <button id="reset-filter" type="button">重置筛选</button>
    <span id="visible-count" class="count" aria-live="polite"></span>
  </section>
  <div class="workspace">
    <section class="panel chart-panel" aria-label="申万一级行业四象限散点图">
      {chart}
      <div class="legend" aria-label="象限图例">
        <span style="--dot:#16835d">领先区</span><span style="--dot:#ca8a04">改善区</span>
        <span style="--dot:#dc5a4d">转弱区</span><span style="--dot:#77808f">落后区</span>
      </div>
    </section>
    <aside id="board-detail" class="panel detail" aria-live="polite">
      <div class="eyebrow" id="detail-quadrant">选择行业</div>
      <h2 id="detail-name">暂无数据</h2>
      <div id="detail-code" class="code">-</div>
      <dl>
        <div class="metric"><dt>综合得分</dt><dd id="detail-score">-</dd></div>
        <div class="metric"><dt>RS-Ratio</dt><dd id="detail-ratio">-</dd></div>
        <div class="metric"><dt>RS-Momentum</dt><dd id="detail-momentum">-</dd></div>
        <div class="metric"><dt>扩散度</dt><dd id="detail-breadth">-</dd></div>
        <div class="metric"><dt>5日扩散变化</dt><dd id="detail-delta">-</dd></div>
        <div class="metric"><dt>成分覆盖</dt><dd id="detail-coverage">-</dd></div>
      </dl>
      <div class="leader"><span class="meta">行业龙头</span><strong id="detail-leader">-</strong><span id="detail-confirmation" class="badge">待确认</span></div>
      <div id="detail-eligible" class="badge">未进入严格候选</div>
    </aside>
  </div>
  <section class="panel table-panel" aria-labelledby="control-picks-title">
    <h2 id="control-picks-title">领先区控盘阶段关注（最多10只）</h2>
    <p class="control-note">“主力控盘”是基于20日成交额结构、趋势效率、区间位置和波动率的量价代理，不代表真实机构持仓；仅保留洗盘或开始拉升阶段，市场退潮、过热或数据滞后时自动降级为仅观察。</p>
    <div class="table-responsive">
      <table id="leading-control-picks">
        <thead><tr><th>股票</th><th>申万行业</th><th>阶段</th><th class="numeric">控盘分</th><th>推荐理由 / 失效条件</th></tr></thead>
        <tbody>{control_table_rows}</tbody>
      </table>
    </div>
  </section>
  <section class="panel table-panel" aria-labelledby="industry-list-title">
    <h2 id="industry-list-title">领先区 / 改善区行业清单</h2>
    <p>按所在区域及综合得分排序；点击行业名称可同步查看上方详情。</p>
    <div class="table-responsive">
      <table id="industry-list">
        <thead><tr><th>申万行业</th><th>所在区域</th><th class="numeric">得分</th><th class="numeric">扩散度</th><th class="numeric">5日变化</th><th>板块龙头</th></tr></thead>
        <tbody>{focus_table_rows}</tbody>
      </table>
    </div>
  </section>
  <footer>点位大小代表综合得分；缩放可通过图表工具栏重置。行业口径为申万2021版一级行业，扩散度使用自由流通市值加权，RRG 以全 A 加权组合为基准。页面仅供量化研究，不构成投资建议。<br><span class="meta">生成时间：{generated_at or "-"}</span></footer>
</main>
<script>
  const allBoards = {embedded_data};
  const quadrantColors = {{"领先区":"#16835d","改善区":"#ca8a04","转弱区":"#dc5a4d","落后区":"#77808f"}};
  const graph = document.getElementById("rrg-chart");
  const search = document.getElementById("board-search");
  const quadrant = document.getElementById("quadrant-filter");
  const count = document.getElementById("visible-count");
  const value = (item, key, fallback = 0) => Number.isFinite(Number(item[key])) ? Number(item[key]) : fallback;
  const percent = number => `${{(Number(number) * 100).toFixed(1)}}%`;
  const signedPercent = number => `${{number >= 0 ? "+" : ""}}${{percent(number)}}`;
  const custom = item => [item.code || "", item.quadrant || "未知", value(item,"score"), value(item,"breadth20"), value(item,"breadth_delta5"), value(item,"member_count"), value(item,"mapped_count"), value(item,"coverage"), item.leader_name || "待识别", item.leader_symbol || "-", Boolean(item.leader_confirmed), Boolean(item.eligible)];

  function showDetail(item) {{
    if (!item) return;
    document.getElementById("detail-quadrant").textContent = item.quadrant || "未知象限";
    document.getElementById("detail-name").textContent = item.name || "未命名";
    document.getElementById("detail-code").textContent = item.code || "-";
    document.getElementById("detail-score").textContent = value(item,"score").toFixed(1);
    document.getElementById("detail-ratio").textContent = value(item,"rs_ratio",100).toFixed(2);
    document.getElementById("detail-momentum").textContent = value(item,"rs_momentum",100).toFixed(2);
    document.getElementById("detail-breadth").textContent = percent(value(item,"breadth20"));
    document.getElementById("detail-delta").textContent = signedPercent(value(item,"breadth_delta5"));
    document.getElementById("detail-coverage").textContent = `${{Math.round(value(item,"mapped_count"))}} / ${{Math.round(value(item,"member_count"))}} (${{percent(value(item,"coverage"))}})`;
    document.getElementById("detail-leader").textContent = `${{item.leader_name || "待识别"}} · ${{item.leader_symbol || "-"}}`;
    document.getElementById("detail-confirmation").textContent = item.leader_confirmed ? "龙头已确认" : "龙头待确认";
    document.getElementById("detail-eligible").textContent = item.eligible ? "进入严格候选" : "未进入严格候选";
  }}

  function applyFilters() {{
    const query = search.value.trim().toLocaleLowerCase("zh-CN");
    const selectedQuadrant = quadrant.value;
    const filtered = allBoards.filter(item => (!selectedQuadrant || item.quadrant === selectedQuadrant) && (!query || `${{item.name || ""}} ${{item.code || ""}}`.toLocaleLowerCase("zh-CN").includes(query)));
    Plotly.restyle(graph, {{
      x: [filtered.map(item => value(item,"rs_ratio",100))],
      y: [filtered.map(item => value(item,"rs_momentum",100))],
      text: [filtered.map(item => item.name || item.code || "未命名")],
      customdata: [filtered.map(custom)],
      "marker.size": [filtered.map(item => 14 + Math.min(Math.max(value(item,"score"),0),100) * .32)],
      "marker.color": [filtered.map(item => quadrantColors[item.quadrant] || "#45627d")]
    }}, [0]);
    count.textContent = `显示 ${{filtered.length}} / ${{allBoards.length}} 个行业`;
    if (filtered.length) showDetail(filtered[0]);
    return filtered;
  }}

  graph.on("plotly_click", event => {{
    const code = event.points?.[0]?.customdata?.[0];
    showDetail(allBoards.find(item => item.code === code));
  }});
  search.addEventListener("input", applyFilters);
  quadrant.addEventListener("change", applyFilters);
  document.getElementById("reset-filter").addEventListener("click", () => {{ search.value = ""; quadrant.value = ""; applyFilters(); search.focus(); }});
  document.querySelectorAll(".industry-link").forEach(button => button.addEventListener("click", () => {{
    showDetail(allBoards.find(item => item.code === button.dataset.code));
    document.getElementById("board-detail").scrollIntoView({{behavior:"smooth",block:"center"}});
  }}));
  count.textContent = `显示 ${{allBoards.length}} / ${{allBoards.length}} 个行业`;
  if (allBoards.length) showDetail(allBoards[0]);
</script>
</body>
</html>
"""
    temp_target = target.with_suffix(".tmp.html")
    temp_target.write_text(html_document, encoding="utf-8")
    temp_target.replace(target)


def build_feishu_payload(
    report: dict[str, Any], pdf_url: str, html_url: str
) -> dict[str, Any]:
    market = report["market"]
    preflight = report.get("data_preflight") or {}
    preflight_audit = preflight.get("final_audit") or {}
    preflight_warnings = preflight_audit.get("warnings") or []
    preflight_text = (
        "数据门禁：自动修复后复检通过"
        if preflight.get("repaired")
        else "数据门禁：更新与质量检查通过"
    )
    if preflight_warnings:
        preflight_text += f"；提醒：{'；'.join(preflight_warnings[:3])}"
    top = (
        "、".join(
            f"{item['name']}({item['quadrant']})" for item in report.get("top", [])[:5]
        )
        or "无"
    )
    buys = (
        "、".join(
            f"{item['symbol']} {item['stock_name']}[{item['industry_name']}]"
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
    control_picks = report.get("leading_control_picks", [])[:10]
    control_content: list[list[dict[str, str]]] = [
        [
            {
                "tag": "text",
                "text": (
                    "领先区控盘阶段关注（量价代理，最多10只）："
                    if control_picks
                    else "领先区控盘阶段关注：暂无同时满足控盘代理与阶段条件的股票"
                ),
            }
        ]
    ]
    for index, item in enumerate(control_picks, 1):
        risk_note = f"；{item['risk_note']}" if item.get("risk_note") else ""
        control_content.append(
            [
                {
                    "tag": "text",
                    "text": (
                        f"{index}. {item['symbol']} {item['stock_name']}"
                        f"［{item['industry_name']}｜{item['stage']}｜"
                        f"{item.get('action_label', '次日关注')}］\n"
                        f"理由：{item['reason']}\n"
                        f"失效：{item['invalidation']}{risk_note}"
                    ),
                }
            ]
        )
    content = [
        [
            {
                "tag": "text",
                "text": f"数据日期：{market['latest_date']}　市场：{market['regime']}",
            }
        ],
        *([[{"tag": "text", "text": preflight_text}]] if preflight else []),
        [{"tag": "text", "text": f"强势申万行业：{top}"}],
        *control_content,
        [{"tag": "text", "text": f"次日条件候选：{buys}"}],
        [{"tag": "text", "text": f"观察池：{watches}"}],
        [
            {"tag": "a", "text": "查看交互四象限", "href": html_url},
            {"tag": "text", "text": "　|　"},
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
                    "title": f"QuantMind 申万行业轮动日报 {market['latest_date']}",
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
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    date_token = str(report["market"]["latest_date"]).replace("-", "")
    json_path = output_dir / f"concept_rotation_{date_token}.json"
    csv_path = output_dir / f"concept_rotation_all_{date_token}.csv"
    pdf_path = output_dir / f"concept_rotation_{date_token}.pdf"
    html_path = output_dir / f"concept_rotation_{date_token}.html"
    latest_pdf_path = output_dir / "latest.pdf"
    latest_html_path = output_dir / "latest.html"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    boards.sort_values("score", ascending=False).to_csv(csv_path, index=False)
    render_report_pdf(report, pdf_path)
    render_interactive_html(report, html_path)
    shutil.copyfile(pdf_path, latest_pdf_path)
    shutil.copyfile(html_path, latest_html_path)
    shutil.copyfile(json_path, output_dir / "latest.json")
    return pdf_path, latest_pdf_path, html_path, latest_html_path


def load_preflight_status(report_date: str) -> dict[str, Any] | None:
    path = Path(
        os.getenv("CONCEPT_REPORT_PREFLIGHT_STATUS_FILE", DEFAULT_PREFLIGHT_STATUS_FILE)
    )
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.exception("Ignoring unreadable data preflight status: %s", path)
        return None
    audit = payload.get("final_audit") or {}
    if not payload.get("passed") or str(audit.get("latest_date")) != report_date:
        raise RuntimeError(
            "Data preflight status is missing, failed, or does not match report date"
        )
    return payload


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
    LOGGER.info("Loading SW2021 level-one industries and membership")
    industries, memberships = load_sw_industry_universe(
        cache_dir, cache_days=args.member_cache_days
    )
    LOGGER.info("Analyzing %s SW industries", len(industries))
    report, boards = build_report(
        stock,
        industries,
        memberships,
        max_buy=args.max_buy,
        max_watch=args.max_watch,
        max_control_picks=args.max_control_picks,
    )
    report["data_preflight"] = load_preflight_status(
        str(report["market"]["latest_date"])
    )
    pdf_path, _latest_pdf_path, html_path, _latest_html_path = write_outputs(
        report, boards, output_dir
    )
    base_url = args.public_base_url.rstrip("/")
    uploads_root = output_dir.parents[1]
    pdf_relative = pdf_path.relative_to(uploads_root)
    html_relative = html_path.relative_to(uploads_root)
    pdf_url = f"{base_url}/uploads/{pdf_relative.as_posix()}"
    html_url = f"{base_url}/uploads/{html_relative.as_posix()}"
    LOGGER.info("Reports generated: PDF=%s HTML=%s", pdf_path, html_path)
    if args.send_feishu:
        webhook = os.getenv("WEB_HOOK", "").strip()
        if not webhook:
            raise RuntimeError("WEB_HOOK is empty; PDF was generated but not sent")
        send_feishu(webhook, build_feishu_payload(report, pdf_url, html_url))
        LOGGER.info("Feishu notification sent with PDF=%s HTML=%s", pdf_url, html_url)
    print(
        json.dumps(
            {
                "data_date": report["market"]["latest_date"],
                "pdf": str(pdf_path),
                "pdf_url": pdf_url,
                "html": str(html_path),
                "html_url": html_url,
                "buy_candidates": len(report["buy_candidates"]),
                "leading_control_picks": len(report["leading_control_picks"]),
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
        LOGGER.exception("Daily SW industry rotation report failed: %s", exc)
        raise SystemExit(1) from exc
