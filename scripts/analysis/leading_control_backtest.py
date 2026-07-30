#!/usr/bin/env python3
"""Five-year monthly backtest for the leading-industry control-stage signal.

The signal is evaluated at each calendar month's final trading close. Orders
are assumed to execute at the next month's first trading open and rebalance at
the following month's first trading open. Historical SW2021 membership is filtered by
``in_date``/``out_date`` at every signal date; no price after the signal date is
passed to the selector.

The script deliberately reuses :mod:`concept_rotation_report` for industry
quadrants, market regimes and the public-price/volume "control" proxy. The
proxy is not evidence of real institutional positions and this program does
not provide investment advice.
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
from collections.abc import Callable
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("leading_control_backtest")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "db" / "feature_snapshots"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "leading-control-backtest"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "leading-control-backtest"
FORBIDDEN_REGIMES = {"退潮", "过热", "数据滞后", "数据异常"}
SW_VERSION = "SW2021"


SignalBuilder = Callable[
    [pd.DataFrame, list[dict[str, Any]], dict[str, list[dict[str, Any]]], int],
    dict[str, Any],
]


def normalize_symbol(symbol: Any) -> str | None:
    """Normalize any common A-share spelling to QuantMind prefix format."""
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


def _parse_membership_date(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "nat"}:
        return None
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


def point_in_time_memberships(
    memberships: dict[str, list[dict[str, Any]]], as_of: Any
) -> dict[str, list[dict[str, Any]]]:
    """Return constituents effective on ``as_of`` using [in_date, out_date)."""
    signal_date = pd.Timestamp(as_of).normalize()
    result: dict[str, list[dict[str, Any]]] = {}
    for industry_code, rows in memberships.items():
        unique: dict[str, dict[str, Any]] = {}
        for row in rows:
            symbol = normalize_symbol(row.get("symbol") or row.get("ts_code"))
            if not symbol:
                continue
            in_date = _parse_membership_date(row.get("in_date"))
            out_date = _parse_membership_date(row.get("out_date"))
            if in_date is not None and in_date > signal_date:
                continue
            # Tushare's out_date is the first date on which the constituent is
            # no longer effective, so membership is [in_date, out_date).
            if out_date is not None and out_date <= signal_date:
                continue
            unique[symbol] = {**row, "symbol": symbol}
        result[str(industry_code)] = sorted(
            unique.values(), key=lambda item: str(item["symbol"])
        )
    return result


def validate_market_data(
    stock: pd.DataFrame,
    *,
    min_daily_count: int = 50,
    min_daily_ratio: float = 0.50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate, repair and audit market data before signal evaluation.

    Deterministic repairs are limited to normalization, last-write-wins key
    deduplication and removal of unusable price rows/dates. Every repair is
    exposed in the returned audit dictionary.
    """
    required = {"trade_date", "symbol", "open", "high", "low", "close"}
    missing = sorted(required - set(stock.columns))
    if missing:
        raise ValueError(f"market data is missing required columns: {missing}")
    frame = stock.copy()
    input_rows = len(frame)
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="coerce"
    ).dt.normalize()
    frame["symbol"] = frame["symbol"].map(normalize_symbol)
    bad_key = frame["trade_date"].isna() | frame["symbol"].isna()
    bad_key_count = int(bad_key.sum())
    frame = frame.loc[~bad_key].copy()

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
        "factor",
        "adj_open",
        "adj_close",
        "is_st",
        "limit_up_today",
        "limit_down_today",
    ]
    for column in numeric_columns:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values(["trade_date", "symbol"], kind="stable")
    duplicate_count = int(frame.duplicated(["trade_date", "symbol"], keep="last").sum())
    frame = frame.drop_duplicates(["trade_date", "symbol"], keep="last")
    prices = frame[["open", "high", "low", "close"]]
    invalid_price = (
        ~np.isfinite(prices).all(axis=1)
        | (prices <= 0).any(axis=1)
        | (frame["high"] < frame["low"])
    )
    invalid_price_count = int(invalid_price.sum())
    frame = frame.loc[~invalid_price].copy()

    daily_counts = frame.groupby("trade_date")["symbol"].nunique().sort_index()
    rolling_reference = daily_counts.rolling(20, min_periods=1).median()
    thin_mask = (daily_counts < int(min_daily_count)) | (
        daily_counts < rolling_reference * float(min_daily_ratio)
    )
    # Small synthetic/unit-test universes can opt in with min_daily_count=1.
    thin_dates = list(daily_counts.index[thin_mask])
    thin_rows = int(frame["trade_date"].isin(thin_dates).sum())
    if thin_dates:
        frame = frame.loc[~frame["trade_date"].isin(thin_dates)].copy()

    frame = frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    issues: list[str] = []
    if bad_key_count:
        issues.append(f"移除 {bad_key_count} 行无效日期或证券代码")
    if duplicate_count:
        issues.append(f"按日期+证券代码去重 {duplicate_count} 行，保留最后记录")
    if invalid_price_count:
        issues.append(f"移除 {invalid_price_count} 行非正数、非有限或高低价倒置数据")
    if thin_dates:
        issues.append(
            f"移除 {len(thin_dates)} 个截面显著不完整交易日（{thin_rows} 行）"
        )
    audit = {
        "status": "warning" if issues else "ok",
        "input_rows": input_rows,
        "output_rows": len(frame),
        "invalid_key_rows_removed": bad_key_count,
        "duplicate_rows_removed": duplicate_count,
        "invalid_price_rows_removed": invalid_price_count,
        "thin_dates_removed": [
            pd.Timestamp(value).date().isoformat() for value in thin_dates
        ],
        "date_min": (
            frame["trade_date"].min().date().isoformat() if not frame.empty else None
        ),
        "date_max": (
            frame["trade_date"].max().date().isoformat() if not frame.empty else None
        ),
        "symbols": int(frame["symbol"].nunique()) if not frame.empty else 0,
        "issues": issues,
    }
    if frame.empty:
        raise RuntimeError("market data is empty after quality checks")
    return frame, audit


def _default_signal_builder(
    history: pd.DataFrame,
    industries: list[dict[str, Any]],
    memberships: dict[str, list[dict[str, Any]]],
    max_stocks: int,
) -> dict[str, Any]:
    """Reuse the production daily-report signal without exposing future rows."""
    from scripts.analysis.concept_rotation_report import (
        analyze_industries,
        build_leading_control_picks,
        prepare_stock_history,
    )

    prepared = prepare_stock_history(history)
    boards, member_sets, market = analyze_industries(prepared, industries, memberships)
    latest_date = prepared["trade_date"].max()
    latest = prepared[prepared["trade_date"] == latest_date].copy()
    picks: list[dict[str, Any]] = []
    if market.get("regime") not in FORBIDDEN_REGIMES:
        picks = build_leading_control_picks(
            boards, latest, member_sets, limit=min(max_stocks, 10)
        )
    return {
        "picks": picks,
        "market": market,
        "boards": boards.to_dict(orient="records"),
    }


def _enforce_caps(picks: list[dict[str, Any]], max_stocks: int) -> list[dict[str, Any]]:
    limit = min(max(int(max_stocks), 0), 10)
    ordered = sorted(
        enumerate(picks),
        key=lambda pair: (
            -float(pair[1].get("selection_score") or 0.0),
            pair[0],
        ),
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    industry_counts: dict[str, int] = {}
    for _, raw in ordered:
        symbol = normalize_symbol(raw.get("symbol"))
        if not symbol or symbol in seen:
            continue
        industry_code = str(raw.get("industry_code") or "未知")
        if industry_counts.get(industry_code, 0) >= 2:
            continue
        selected.append({**raw, "symbol": symbol, "industry_code": industry_code})
        seen.add(symbol)
        industry_counts[industry_code] = industry_counts.get(industry_code, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def _paired_returns(
    stock: pd.DataFrame, buy_date: pd.Timestamp, sell_date: pd.Timestamp
) -> pd.DataFrame:
    buy_column = "adj_open" if "adj_open" in stock.columns else "open"
    sell_column = "adj_open" if "adj_open" in stock.columns else "open"
    buy = stock[stock["trade_date"] == buy_date][["symbol", buy_column]].rename(
        columns={buy_column: "buy_open"}
    )
    sell = stock[stock["trade_date"] == sell_date][["symbol", sell_column]].rename(
        columns={sell_column: "sell_open"}
    )
    paired = buy.merge(sell, on="symbol", how="outer")
    paired["gross_return"] = paired["sell_open"] / paired["buy_open"] - 1
    return paired.set_index("symbol", drop=False)


def _build_periods(
    stock: pd.DataFrame, start_date: Any, end_date: Any
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, str]]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    dates = pd.Series(stock["trade_date"].dropna().unique()).sort_values()
    groups = {
        period: [pd.Timestamp(value) for value in group.tolist()]
        for period, group in dates.groupby(dates.dt.to_period("M"))
    }
    periods: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, str]] = []
    keys = sorted(groups)
    for index in range(len(keys) - 2):
        signal_period, holding_period, exit_period = (
            keys[index],
            keys[index + 1],
            keys[index + 2],
        )
        if (
            holding_period.ordinal != signal_period.ordinal + 1
            or exit_period.ordinal != holding_period.ordinal + 1
        ):
            continue
        signal_date = max(groups[signal_period])
        buy_date = min(groups[holding_period])
        sell_date = min(groups[exit_period])
        if signal_date < start or sell_date > end:
            continue
        periods.append((signal_date, buy_date, sell_date, str(holding_period)))
    return periods


def _finite(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: _finite(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def calculate_performance_metrics(
    monthly: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate monthly-frequency portfolio and benchmark statistics."""
    if monthly.empty:
        curve = pd.DataFrame(
            columns=["strategy_equity", "benchmark_equity", "drawdown"]
        )
        return {
            "months": 0,
            "invested_months": 0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown": 0.0,
            "calmar_ratio": None,
            "win_rate": None,
            "profit_loss_ratio": None,
            "benchmark_total_return": 0.0,
            "excess_total_return": 0.0,
            "annualized_excess_return": 0.0,
            "tracking_error": 0.0,
            "information_ratio": None,
            "alpha_annualized": None,
            "beta": None,
            "cash_months": 0,
            "cash_month_ratio": 0.0,
            "average_monthly_turnover": 0.0,
        }, curve
    returns = pd.to_numeric(monthly["net_return"], errors="coerce").fillna(0.0)
    benchmark = pd.to_numeric(
        monthly.get("benchmark_return", pd.Series(0.0, index=monthly.index)),
        errors="coerce",
    ).fillna(0.0)
    strategy_equity = (1 + returns).cumprod()
    benchmark_equity = (1 + benchmark).cumprod()
    running_peak = strategy_equity.cummax().clip(lower=1.0)
    drawdown = strategy_equity / running_peak - 1
    months = len(returns)
    total_return = float(strategy_equity.iloc[-1] - 1)
    benchmark_total = float(benchmark_equity.iloc[-1] - 1)
    annualized = float(strategy_equity.iloc[-1] ** (12 / months) - 1)
    volatility = float(returns.std(ddof=0) * math.sqrt(12))
    sharpe = (
        float(returns.mean() / returns.std(ddof=0) * math.sqrt(12))
        if returns.std(ddof=0) > 0
        else None
    )
    downside = returns[returns < 0]
    downside_deviation = (
        float(np.sqrt(np.mean(np.square(downside))) * math.sqrt(12))
        if not downside.empty
        else 0.0
    )
    sortino = (
        float(returns.mean() * 12 / downside_deviation)
        if downside_deviation > 0
        else None
    )
    maximum_drawdown = float(drawdown.min())
    calmar = float(annualized / abs(maximum_drawdown)) if maximum_drawdown < 0 else None
    positive = returns[returns > 0]
    negative = returns[returns < 0]
    profit_loss = (
        float(positive.mean() / abs(negative.mean()))
        if not positive.empty and not negative.empty
        else None
    )
    holding_count = pd.to_numeric(
        monthly.get("holding_count", pd.Series(0, index=monthly.index)),
        errors="coerce",
    ).fillna(0)
    excess = returns - benchmark
    tracking_error = float(excess.std(ddof=0) * math.sqrt(12))
    information_ratio = (
        float(excess.mean() / excess.std(ddof=0) * math.sqrt(12))
        if excess.std(ddof=0) > 0
        else None
    )
    benchmark_variance = float(benchmark.var(ddof=0))
    beta = (
        float(np.cov(returns, benchmark, ddof=0)[0, 1] / benchmark_variance)
        if benchmark_variance > 0
        else None
    )
    alpha = (
        float((returns.mean() - beta * benchmark.mean()) * 12)
        if beta is not None
        else None
    )
    invested = holding_count > 0
    invested_returns = returns[invested]
    turnover = pd.to_numeric(
        monthly.get("turnover", pd.Series(0.0, index=monthly.index)),
        errors="coerce",
    ).fillna(0.0)
    metrics = {
        "months": months,
        "invested_months": int((holding_count > 0).sum()),
        "total_return": total_return,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": maximum_drawdown,
        "calmar_ratio": calmar,
        "win_rate": (
            float((invested_returns > 0).mean()) if not invested_returns.empty else None
        ),
        "profit_loss_ratio": profit_loss,
        "best_month": float(returns.max()),
        "worst_month": float(returns.min()),
        "average_monthly_return": float(returns.mean()),
        "average_holdings": float(holding_count.mean()),
        "cash_months": int((~invested).sum()),
        "cash_month_ratio": float((~invested).mean()),
        "average_monthly_turnover": float(turnover.mean()),
        "benchmark_total_return": benchmark_total,
        "excess_total_return": float(
            strategy_equity.iloc[-1] / benchmark_equity.iloc[-1] - 1
        ),
        "annualized_excess_return": float(
            (strategy_equity.iloc[-1] / benchmark_equity.iloc[-1]) ** (12 / months) - 1
        ),
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "alpha_annualized": alpha,
        "beta": beta,
    }
    curve = pd.DataFrame(
        {
            "strategy_equity": strategy_equity,
            "benchmark_equity": benchmark_equity,
            "drawdown": drawdown,
        },
        index=monthly.index,
    )
    return metrics, curve


def _aggregate_report_tables(
    monthly: pd.DataFrame, holdings: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if monthly.empty:
        return [], [], []
    monthly = monthly.copy()
    monthly["year"] = monthly["holding_month"].str[:4].astype(int)
    monthly["month_number"] = monthly["holding_month"].str[5:7]
    annual_rows = []
    for year, group in monthly.groupby("year", sort=True):
        annual_rows.append(
            {
                "year": int(year),
                "return": float((1 + group["net_return"]).prod() - 1),
                "benchmark_return": float((1 + group["benchmark_return"]).prod() - 1),
                "months": int(len(group)),
                "invested_months": int((group["holding_count"] > 0).sum()),
            }
        )
    matrix_rows = []
    for year, group in monthly.groupby("year", sort=True):
        row: dict[str, Any] = {"year": int(year)}
        row.update(
            {
                str(item.month_number): float(item.net_return)
                for item in group.itertuples()
            }
        )
        row["全年"] = float((1 + group["net_return"]).prod() - 1)
        matrix_rows.append(row)

    attribution: list[dict[str, Any]] = []
    if not holdings.empty:
        for (industry, stage), group in holdings.groupby(
            ["industry_name", "stage"], dropna=False, sort=True
        ):
            attribution.append(
                {
                    "industry_name": str(industry or "未知"),
                    "stage": str(stage or "未知"),
                    "holding_records": int(len(group)),
                    "months": int(group["holding_month"].nunique()),
                    "average_stock_return": float(group["net_return"].mean()),
                    "win_rate": float((group["net_return"] > 0).mean()),
                    "total_portfolio_contribution": float(group["contribution"].sum()),
                }
            )
        attribution.sort(
            key=lambda item: item["total_portfolio_contribution"], reverse=True
        )
    return annual_rows, matrix_rows, attribution


def _sensitivity_metrics(
    monthly: pd.DataFrame, holdings: pd.DataFrame
) -> list[dict[str, Any]]:
    """Compare cost, portfolio-size, stage and post-2021 specifications."""
    variants: list[tuple[str, pd.DataFrame]] = []
    gross = monthly.copy()
    gross["net_return"] = gross["gross_return"]
    variants.append(("10只上限（不计成本）", gross))
    variants.append(("10只上限（单边15bp）", monthly.copy()))

    if not holdings.empty:
        top5 = monthly.copy()
        top5_returns: dict[str, float] = {}
        ranked = holdings[
            pd.to_numeric(holdings["selection_rank"], errors="coerce") <= 5
        ]
        for month, group in ranked.groupby("holding_month"):
            signal_count = int(
                monthly.loc[monthly["holding_month"] == month, "signal_count"].iloc[0]
            )
            denominator = max(min(signal_count, 5), 1)
            top5_returns[str(month)] = float(group["net_return"].sum() / denominator)
        top5["net_return"] = top5["holding_month"].map(top5_returns).fillna(0.0)
        top5["holding_count"] = (
            top5["holding_month"]
            .map(ranked.groupby("holding_month")["symbol"].count())
            .fillna(0)
        )
        variants.append(("5只上限（成本后）", top5))

        for label, pattern in (("仅洗盘", "洗盘"), ("仅开始拉升", "拉升")):
            stage_monthly = monthly.copy()
            selected = holdings[
                holdings["stage"].astype(str).str.contains(pattern, na=False)
            ]
            stage_returns = selected.groupby("holding_month")["net_return"].mean()
            stage_counts = selected.groupby("holding_month")["symbol"].count()
            stage_monthly["net_return"] = (
                stage_monthly["holding_month"].map(stage_returns).fillna(0.0)
            )
            stage_monthly["holding_count"] = (
                stage_monthly["holding_month"].map(stage_counts).fillna(0)
            )
            variants.append((f"{label}（归因切片）", stage_monthly))

    since_2022 = monthly[monthly["holding_month"] >= "2022-01"].copy()
    if not since_2022.empty:
        variants.append(("2022年起始（成本后）", since_2022))

    result: list[dict[str, Any]] = []
    for label, frame in variants:
        metrics, _curve = calculate_performance_metrics(frame)
        result.append(
            {"scenario": label, **{k: _finite(v) for k, v in metrics.items()}}
        )
    return result


def run_monthly_backtest(
    stock: pd.DataFrame,
    industries: list[dict[str, Any]],
    memberships: dict[str, list[dict[str, Any]]],
    *,
    start_date: Any,
    end_date: Any,
    transaction_cost_bps: float = 15.0,
    max_stocks: int = 10,
    lookback_calendar_days: int = 180,
    signal_builder: SignalBuilder | None = None,
    data_quality: dict[str, Any] | None = None,
    benchmark: pd.DataFrame | None = None,
    benchmark_name: str = "沪深300",
) -> dict[str, Any]:
    """Run a close-to-next-month-open monthly signal backtest.

    Only ``history[trade_date <= signal_date]`` is supplied to the signal
    builder. Execution prices are looked up afterwards, keeping the boundary
    explicit and readily testable.
    """
    frame = stock.copy()
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="coerce"
    ).dt.normalize()
    frame["symbol"] = frame["symbol"].map(normalize_symbol)
    frame = frame.dropna(subset=["trade_date", "symbol"])
    frame = frame.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    periods = _build_periods(frame, start_date, end_date)
    builder = signal_builder or _default_signal_builder
    cost = max(float(transaction_cost_bps), 0.0) / 10_000
    monthly_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    previous_weights: dict[str, float] = {}
    benchmark_frame = pd.DataFrame() if benchmark is None else benchmark.copy()
    if not benchmark_frame.empty:
        benchmark_frame["trade_date"] = pd.to_datetime(
            benchmark_frame["trade_date"], errors="coerce"
        ).dt.normalize()
        benchmark_frame["open"] = pd.to_numeric(
            benchmark_frame["open"], errors="coerce"
        )
        benchmark_frame = benchmark_frame.dropna(subset=["trade_date", "open"])
        benchmark_frame = benchmark_frame.drop_duplicates("trade_date", keep="last")
        benchmark_frame = benchmark_frame.set_index("trade_date")

    for signal_date, buy_date, sell_date, holding_month in periods:
        history_start = signal_date - pd.Timedelta(days=lookback_calendar_days)
        history = frame[
            (frame["trade_date"] >= history_start)
            & (frame["trade_date"] <= signal_date)
        ].copy()
        pit_members = point_in_time_memberships(memberships, signal_date)
        try:
            signal = builder(history, industries, pit_members, min(max_stocks, 10))
            market = dict(signal.get("market") or {})
            regime = str(market.get("regime") or "未知")
            raw_picks = list(signal.get("picks") or [])
            picks = (
                []
                if regime in FORBIDDEN_REGIMES
                else _enforce_caps(raw_picks, max_stocks)
            )
        except Exception as exc:
            LOGGER.exception("Signal failed for %s", signal_date.date())
            regime = "数据异常"
            market = {"regime": regime}
            picks = []
            errors.append(
                {"signal_date": signal_date.date().isoformat(), "error": str(exc)}
            )

        paired = _paired_returns(frame, buy_date, sell_date)
        if (
            not benchmark_frame.empty
            and buy_date in benchmark_frame.index
            and sell_date in benchmark_frame.index
        ):
            benchmark_buy = float(benchmark_frame.loc[buy_date, "open"])
            benchmark_sell = float(benchmark_frame.loc[sell_date, "open"])
            benchmark_return = benchmark_sell / benchmark_buy - 1
        elif not benchmark_frame.empty:
            raise RuntimeError(
                f"沪深300基准缺少月度边界 {buy_date.date()} 或 {sell_date.date()}"
            )
        else:
            benchmark_return = (
                float(paired["gross_return"].mean()) if not paired.empty else 0.0
            )
        target_weight = 1 / len(picks) if picks else 0.0
        current_weights: dict[str, float] = {}
        gross_return = 0.0
        net_return = 0.0
        holding_count = 0
        for rank, pick in enumerate(picks, start=1):
            symbol = str(pick["symbol"])
            if symbol not in paired.index:
                continue
            price = paired.loc[symbol]
            if isinstance(price, pd.DataFrame):
                price = price.iloc[-1]
            buy_open = _finite(price.get("buy_open"))
            sell_open = _finite(price.get("sell_open"))
            if buy_open is None or float(buy_open) <= 0:
                continue
            holding_count += 1
            current_weights[symbol] = target_weight
            if sell_open is None or float(sell_open) <= 0:
                gross = -1.0
                net = -1.0
                exit_status = "退出日无报价，保守按全部损失"
            else:
                gross = float(sell_open) / float(buy_open) - 1
                net = float((1 + gross) * (1 - cost) ** 2 - 1)
                exit_status = "正常退出"
            gross_return += target_weight * gross
            net_return += target_weight * net
            holding_rows.append(
                {
                    "holding_month": holding_month,
                    "signal_date": signal_date.date().isoformat(),
                    "buy_date": buy_date.date().isoformat(),
                    "sell_date": sell_date.date().isoformat(),
                    "symbol": str(pick["symbol"]),
                    "stock_name": str(pick.get("stock_name") or pick["symbol"]),
                    "industry_code": str(pick.get("industry_code") or "未知"),
                    "industry_name": str(pick.get("industry_name") or "未知"),
                    "stage": str(pick.get("stage") or "未知"),
                    "selection_rank": rank,
                    "buy_open": float(buy_open),
                    "sell_open": (float(sell_open) if sell_open is not None else None),
                    "exit_status": exit_status,
                    "gross_return": gross,
                    "net_return": net,
                    "weight": target_weight,
                    "contribution": target_weight * net,
                    "reason": str(pick.get("reason") or ""),
                    "invalidation": str(pick.get("invalidation") or ""),
                }
            )
        symbols = set(previous_weights) | set(current_weights)
        previous_cash = 1.0 - sum(previous_weights.values())
        current_cash = 1.0 - sum(current_weights.values())
        turnover = 0.5 * (
            sum(
                abs(
                    current_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0)
                )
                for symbol in symbols
            )
            + abs(current_cash - previous_cash)
        )
        previous_weights = current_weights
        monthly_rows.append(
            {
                "holding_month": holding_month,
                "signal_date": signal_date.date().isoformat(),
                "buy_date": buy_date.date().isoformat(),
                "sell_date": sell_date.date().isoformat(),
                "regime": regime,
                "signal_count": len(picks),
                "holding_count": holding_count,
                "gross_return": gross_return,
                "net_return": net_return,
                "benchmark_return": benchmark_return,
                "excess_return": net_return - benchmark_return,
                "transaction_cost_bps_per_side": float(transaction_cost_bps),
                "turnover": float(turnover),
                "cash_reason": regime if regime in FORBIDDEN_REGIMES else None,
                "market_win_rate": _finite(market.get("win_rate")),
            }
        )

    monthly = pd.DataFrame(monthly_rows)
    holdings = pd.DataFrame(holding_rows)
    metrics, curve = calculate_performance_metrics(monthly)
    if not monthly.empty:
        curve = curve.copy()
        curve.insert(0, "month", monthly["holding_month"].to_numpy())
    annual, matrix, attribution = _aggregate_report_tables(monthly, holdings)
    sensitivity = _sensitivity_metrics(monthly, holdings)
    limitations = [
        "“控盘”是公开日线量价构造的代理信号，不代表真实机构持仓或资金身份。",
        "月末收盘生成信号，次月首个交易日开盘成交；信号日之后的数据不参与选股。",
        "历史申万成分按 in_date/out_date 点时过滤；SW2021 在 2021 年区间属于历史回溯分类口径。",
        "成交按月初开盘价一次完成，未模拟涨跌停无法成交、停牌排队、冲击成本和容量。",
        "默认单边成本为15bp，未单独建模印花税、滑点随成交额变化和分红税。",
        "最大回撤按月度调仓净值计算，可能低估月内暂时性回撤。",
        "特征快照缺少历史 ST 名称时按非 ST 处理；结果可能高估该部分样本的可交易性。",
        "回测结果仅供研究，不构成投资建议；历史收益不能保证未来表现。",
    ]
    return {
        "meta": {
            "title": "领先区控盘阶段五年回测报告",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "strategy": "申万领先区 + 控盘量价代理 + 洗盘/开始拉升",
        },
        "parameters": {
            "start_signal_date": pd.Timestamp(start_date).date().isoformat(),
            "end_date": pd.Timestamp(end_date).date().isoformat(),
            "frequency": "monthly",
            "signal_timing": "月末最后交易日收盘后",
            "entry_timing": "次月首个交易日开盘",
            "exit_timing": "再下一月首个交易日开盘调仓",
            "transaction_cost_bps_per_side": float(transaction_cost_bps),
            "max_stocks": min(max(int(max_stocks), 0), 10),
            "max_stocks_per_industry": 2,
            "cash_regimes": sorted(FORBIDDEN_REGIMES),
            "benchmark": (
                benchmark_name if not benchmark_frame.empty else "A股全市场等权替代基准"
            ),
        },
        "data_quality": data_quality or {"status": "not_provided", "issues": []},
        "metrics": {key: _finite(value) for key, value in metrics.items()},
        "equity_curve": _records(curve.reset_index(drop=True)),
        "annual": annual,
        "monthly_matrix": matrix,
        "industry_stage_attribution": attribution,
        "sensitivity": sensitivity,
        "monthly": _records(monthly),
        "holdings": _records(holdings),
        "signal_errors": errors,
        "limitations": limitations,
    }


def _default_tushare_query(
    api_name: str, params: dict[str, str], fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    from scripts.analysis.concept_rotation_report import query_tushare

    return query_tushare(api_name, params, fields)


def load_historical_sw_industry_universe(
    cache_dir: Path,
    *,
    cache_days: float = 30.0,
    query: Callable[[str, dict[str, str], tuple[str, ...]], list[dict[str, Any]]]
    | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Load all historical SW2021 L1 membership intervals, not just latest."""
    cache_path = cache_dir / "tushare-sw2021-l1-membership-history.json.gz"
    stale: dict[str, Any] | None = None
    if cache_path.exists():
        try:
            with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
                candidate = json.load(handle)
            if candidate.get("version") == SW_VERSION and candidate.get("historical"):
                stale = candidate
                age = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 86400
                if age <= cache_days:
                    return candidate["industries"], candidate["memberships"]
        except (OSError, ValueError, json.JSONDecodeError):
            LOGGER.exception("Invalid historical SW membership cache")
    query_api = query or _default_tushare_query
    try:
        classifications = query_api(
            "index_classify",
            {"level": "L1", "src": SW_VERSION},
            ("index_code", "industry_name", "level", "src"),
        )
        industries = sorted(
            [
                {
                    "code": str(row["index_code"]),
                    "name": str(row["industry_name"]),
                    "level": str(row.get("level") or "L1"),
                    "source": str(row.get("src") or SW_VERSION),
                }
                for row in classifications
                if row.get("index_code") and row.get("industry_name")
            ],
            key=lambda row: row["code"],
        )
        if not 28 <= len(industries) <= 40:
            raise RuntimeError(
                f"unexpected SW2021 L1 industry count: {len(industries)}"
            )
        memberships: dict[str, list[dict[str, Any]]] = {}
        for industry in industries:
            rows: list[dict[str, Any]] = []
            for is_new in ("Y", "N"):
                rows.extend(
                    query_api(
                        "index_member_all",
                        {"l1_code": industry["code"], "is_new": is_new},
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
                )
            intervals: dict[tuple[str, str, str], dict[str, Any]] = {}
            for row in rows:
                symbol = normalize_symbol(row.get("ts_code"))
                if not symbol:
                    continue
                item = {
                    "symbol": symbol,
                    "name": str(row.get("name") or row.get("ts_code") or ""),
                    "in_date": row.get("in_date"),
                    "out_date": row.get("out_date"),
                }
                key = (
                    symbol,
                    str(item["in_date"] or ""),
                    str(item["out_date"] or ""),
                )
                intervals[key] = item
            memberships[industry["code"]] = list(intervals.values())
        usable = sum(bool(rows) for rows in memberships.values())
        if usable < len(industries) * 0.90:
            raise RuntimeError(
                f"only {usable}/{len(industries)} industries have members"
            )
        payload = {
            "version": SW_VERSION,
            "historical": True,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "industries": industries,
            "memberships": memberships,
        }
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        temporary.replace(cache_path)
        return industries, memberships
    except Exception:
        if stale:
            LOGGER.exception(
                "SW membership refresh failed; using stale historical cache"
            )
            return stale["industries"], stale["memberships"]
        raise


def load_hs300_benchmark(
    cache_dir: Path,
    start_date: Any,
    end_date: Any,
    *,
    cache_days: float = 30.0,
    query: Callable[[str, dict[str, str], tuple[str, ...]], list[dict[str, Any]]]
    | None = None,
) -> pd.DataFrame:
    """Load a cached, yearly-chunked沪深300 daily series from Tushare."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    cache_path = cache_dir / "tushare-hs300-daily.json.gz"
    if cache_path.exists():
        try:
            age = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 86400
            with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
                cached = pd.DataFrame(json.load(handle).get("rows") or [])
            if not cached.empty:
                cached["trade_date"] = pd.to_datetime(
                    cached["trade_date"], errors="coerce"
                )
                if (
                    age <= cache_days
                    and cached["trade_date"].min() <= start
                    and cached["trade_date"].max() >= end - pd.Timedelta(days=7)
                ):
                    return cached[
                        (cached["trade_date"] >= start) & (cached["trade_date"] <= end)
                    ].copy()
        except (OSError, ValueError, json.JSONDecodeError):
            LOGGER.exception("Invalid HS300 benchmark cache")

    query_api = query or _default_tushare_query
    rows: list[dict[str, Any]] = []
    for year in range(start.year, end.year + 1):
        chunk_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        chunk_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        rows.extend(
            query_api(
                "index_daily",
                {
                    "ts_code": "000300.SH",
                    "start_date": chunk_start.strftime("%Y%m%d"),
                    "end_date": chunk_end.strftime("%Y%m%d"),
                },
                ("ts_code", "trade_date", "open", "high", "low", "close"),
            )
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Tushare index_daily returned no HS300 benchmark rows")
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], format="%Y%m%d", errors="coerce"
    )
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "open", "close"])
    frame = frame.drop_duplicates("trade_date", keep="last").sort_values("trade_date")
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rows": _records(frame),
    }
    temporary = cache_path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    temporary.replace(cache_path)
    return frame


def _read_feature_year(
    path: Path, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    columns = [
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "factor",
        "mom_ret_1d",
        "liq_turnover_os",
        "liq_amount",
        "style_ln_mv_total",
        "style_ln_mv_float",
    ]
    try:
        frame = pd.read_parquet(
            path,
            columns=columns,
            filters=[("trade_date", ">=", start), ("trade_date", "<=", end)],
        )
    except (TypeError, ValueError):
        frame = pd.read_parquet(path, columns=columns)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)].copy()
    # Production signals are defined on raw OHLC. Adjusted prices are kept in
    # separate execution columns so distributions do not create fake returns.
    frame["factor"] = pd.to_numeric(frame["factor"], errors="coerce")
    invalid_factor = ~np.isfinite(frame["factor"]) | (frame["factor"] <= 0)
    if invalid_factor.any():
        raise ValueError(f"{path} contains {int(invalid_factor.sum())} invalid factors")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["adj_open"] = frame["open"] * frame["factor"]
    frame["adj_close"] = frame["close"] * frame["factor"]
    frame["amount"] = pd.to_numeric(frame.pop("liq_amount"), errors="coerce")
    frame["pct_change"] = pd.to_numeric(frame.pop("mom_ret_1d"), errors="coerce")
    frame["turnover_rate"] = (
        pd.to_numeric(frame.pop("liq_turnover_os"), errors="coerce") * 100
    )
    frame["total_mv"] = np.exp(
        pd.to_numeric(frame.pop("style_ln_mv_total"), errors="coerce")
    )
    frame["float_mv"] = np.exp(
        pd.to_numeric(frame.pop("style_ln_mv_float"), errors="coerce")
    )
    frame["stock_name"] = frame["symbol"]
    frame["is_st"] = 0
    frame["limit_up_today"] = 0
    frame["limit_down_today"] = 0
    frame["data_source"] = "feature_snapshot"
    return frame


def load_feature_snapshots(
    snapshot_dir: Path, start_date: Any, end_date: Any
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only required columns from annual 2021-2026 feature snapshots."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    frames: list[pd.DataFrame] = []
    files: list[str] = []
    missing: list[str] = []
    for year in range(start.year, end.year + 1):
        path = snapshot_dir / f"model_features_{year}.parquet"
        if not path.exists():
            alternative = snapshot_dir / f"features_{year}.parquet"
            path = alternative if alternative.exists() else path
        if not path.exists():
            missing.append(str(path))
            continue
        files.append(str(path))
        frames.append(_read_feature_year(path, start, end))
    if missing:
        raise FileNotFoundError(
            "missing annual feature snapshots required for the requested range: "
            + ", ".join(missing)
        )
    if not frames:
        raise FileNotFoundError(f"no feature snapshots found under {snapshot_dir}")
    frame = pd.concat(frames, ignore_index=True)
    return frame, {
        "snapshot_files": files,
        "missing_snapshot_files": missing,
        "snapshot_rows": len(frame),
        "snapshot_date_max": frame["trade_date"].max().date().isoformat(),
    }


def _database_connection():
    import psycopg2

    return psycopg2.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "quantmind"),
        user=os.getenv("DB_USER", "quantmind"),
        password=os.getenv("DB_PASSWORD", ""),
        connect_timeout=20,
        options="-c statement_timeout=180000",
    )


def load_database_tail(
    after_date: Any,
    end_date: Any,
    *,
    connection_factory: Callable[[], Any] | None = None,
) -> pd.DataFrame:
    """Supplement snapshots from QuantMind's own local PostgreSQL table."""
    connection = (connection_factory or _database_connection)()
    sql = """
        SELECT trade_date, symbol, stock_name, is_st,
               COALESCE(raw_open, open) AS open,
               COALESCE(raw_high, high) AS high,
               COALESCE(raw_low, low) AS low,
               COALESCE(raw_close, close) AS close,
               volume, amount, pct_change, adj_factor AS factor,
               turnover_rate, float_mv, total_mv,
               limit_up_today, limit_down_today
        FROM stock_daily_latest
        WHERE trade_date > %s AND trade_date <= %s
        ORDER BY trade_date, symbol
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql,
                (pd.Timestamp(after_date).date(), pd.Timestamp(end_date).date()),
            )
            columns = [getattr(item, "name", item[0]) for item in cursor.description]
            rows = cursor.fetchall()
    finally:
        connection.close()
    tail = pd.DataFrame(rows, columns=columns)
    if not tail.empty:
        tail["trade_date"] = pd.to_datetime(tail["trade_date"], errors="coerce")
        factor = pd.to_numeric(tail["factor"], errors="coerce")
        invalid_factor = ~np.isfinite(factor) | (factor <= 0)
        if invalid_factor.any():
            raise ValueError(
                f"database tail contains {int(invalid_factor.sum())} invalid factors"
            )
        tail["factor"] = factor
        tail["adj_open"] = pd.to_numeric(tail["open"], errors="coerce") * factor
        tail["adj_close"] = pd.to_numeric(tail["close"], errors="coerce") * factor
        tail["data_source"] = "stock_daily_latest"
    return tail


def load_market_data(
    snapshot_dir: Path,
    start_date: Any,
    end_date: Any,
    *,
    supplement_database: bool = True,
    connection_factory: Callable[[], Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    snapshots, source_audit = load_feature_snapshots(snapshot_dir, start_date, end_date)
    latest_snapshot = pd.Timestamp(snapshots["trade_date"].max())
    tail = pd.DataFrame()
    tail_error: str | None = None
    if supplement_database and latest_snapshot < pd.Timestamp(end_date):
        try:
            tail = load_database_tail(
                latest_snapshot,
                end_date,
                connection_factory=connection_factory,
            )
        except Exception as exc:
            tail_error = str(exc)
            LOGGER.warning("Database tail unavailable: %s", exc)
    combined = pd.concat([snapshots, tail], ignore_index=True, sort=False)
    combined = combined.sort_values(["trade_date", "symbol", "data_source"])
    combined = combined.drop_duplicates(["trade_date", "symbol"], keep="last")
    source_audit.update(
        {
            "database_tail_rows": len(tail),
            "database_tail_error": tail_error,
            "combined_date_max": pd.Timestamp(combined["trade_date"].max())
            .date()
            .isoformat(),
        }
    )
    return combined, source_audit


def _percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:+.2%}"


def _svg_chart(report: dict[str, Any]) -> str:
    curve = report.get("equity_curve") or []
    if not curve:
        return '<div class="empty">无权益曲线</div>'
    width, height, pad = 900, 280, 42
    values = [
        float(row[key])
        for row in curve
        for key in ("strategy_equity", "benchmark_equity")
        if row.get(key) is not None
    ]
    low, high = min(values), max(values)
    span = max(high - low, 1e-9)
    benchmark_label = escape(
        str(report.get("parameters", {}).get("benchmark") or "基准")
    )

    def points(key: str) -> str:
        result = []
        for index, row in enumerate(curve):
            x = pad + index * (width - 2 * pad) / max(len(curve) - 1, 1)
            y = height - pad - (float(row[key]) - low) / span * (height - 2 * pad)
            result.append(f"{x:.1f},{y:.1f}")
        return " ".join(result)

    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="权益曲线">
      <rect width="100%" height="100%" fill="#fbfcfe" rx="10"/>
      <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#ccd5df"/>
      <polyline fill="none" stroke="#176b5b" stroke-width="3" points="{points("strategy_equity")}"/>
      <polyline fill="none" stroke="#8391a2" stroke-width="2" points="{points("benchmark_equity")}"/>
      <text x="{pad}" y="22" fill="#176b5b">策略</text><text x="{pad + 55}" y="22" fill="#8391a2">{benchmark_label}</text>
    </svg>"""


def render_interactive_html(report: dict[str, Any], target: Path) -> None:
    """Write a standalone HTML report (all CSS/SVG/data embedded)."""
    metrics = report.get("metrics") or {}
    monthly_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row.get('holding_month', '-')))}</td>"
        f"<td>{escape(str(row.get('regime', '-')))}</td>"
        f"<td>{row.get('holding_count', 0)}</td>"
        f"<td>{_percent(row.get('net_return'))}</td>"
        f"<td>{_percent(row.get('benchmark_return'))}</td>"
        "</tr>"
        for row in report.get("monthly", [])
    )
    holding_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row.get('holding_month', '-')))}</td>"
        f"<td>{escape(str(row.get('symbol', '-')))}</td>"
        f"<td>{escape(str(row.get('stock_name', '-')))}</td>"
        f"<td>{escape(str(row.get('industry_name', '-')))}</td>"
        f"<td>{escape(str(row.get('stage', '-')))}</td>"
        f"<td>{_percent(row.get('net_return'))}</td>"
        "</tr>"
        for row in report.get("holdings", [])
    )
    annual_rows = "".join(
        "<tr>"
        f"<td>{row.get('year', '-')}</td>"
        f"<td>{_percent(row.get('return'))}</td>"
        f"<td>{_percent(row.get('benchmark_return'))}</td>"
        f"<td>{row.get('invested_months', 0)}/{row.get('months', 0)}</td>"
        "</tr>"
        for row in report.get("annual", [])
    )
    attribution_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row.get('industry_name', '-')))}</td>"
        f"<td>{escape(str(row.get('stage', '-')))}</td>"
        f"<td>{row.get('holding_records', 0)}</td>"
        f"<td>{_percent(row.get('win_rate'))}</td>"
        f"<td>{_percent(row.get('total_portfolio_contribution'))}</td>"
        "</tr>"
        for row in report.get("industry_stage_attribution", [])
    )
    sensitivity_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row.get('scenario', '-')))}</td>"
        f"<td>{row.get('months', 0)}</td>"
        f"<td>{_percent(row.get('total_return'))}</td>"
        f"<td>{_percent(row.get('annualized_return'))}</td>"
        f"<td>{_percent(row.get('max_drawdown'))}</td>"
        f"<td>{row.get('sharpe_ratio') if row.get('sharpe_ratio') is not None else '-'}</td>"
        "</tr>"
        for row in report.get("sensitivity", [])
    )
    limitation_rows = "".join(
        f"<li>{escape(str(item))}</li>" for item in report.get("limitations", [])
    )
    payload = escape(json.dumps(report, ensure_ascii=False, allow_nan=False))
    title = escape(str(report.get("meta", {}).get("title") or "量化回测报告"))
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
:root{{--ink:#172235;--muted:#697586;--green:#176b5b;--line:#dbe2ea;--paper:#fff;--bg:#eef2f5}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1180px;margin:24px auto;padding:0 18px}}header,.panel{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:16px;box-shadow:0 7px 22px #1b2c3d0b}}
h1{{margin:0 0 5px;font-size:28px}}h2{{margin:0 0 14px;font-size:18px}}.sub{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin-top:18px}}.card{{background:#f7faf9;border-radius:10px;padding:13px}}.card b{{display:block;font-size:21px;color:var(--green)}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{background:#f3f6f8;position:sticky;top:0}}.scroll{{overflow:auto;max-height:540px}}svg{{width:100%;height:auto}}ul{{padding-left:20px}}code{{white-space:pre-wrap}}@media(max-width:640px){{h1{{font-size:22px}}}}
</style></head><body><main><header><h1>{title}</h1><div class="sub">{escape(str(report.get("meta", {}).get("strategy", "")))} · 生成 {escape(str(report.get("meta", {}).get("generated_at", "-")))}</div>
<div class="cards"><div class="card">累计收益<b>{_percent(metrics.get("total_return"))}</b></div><div class="card">年化收益<b>{_percent(metrics.get("annualized_return"))}</b></div><div class="card">最大回撤（月频）<b>{_percent(metrics.get("max_drawdown"))}</b></div><div class="card">夏普比率<b>{metrics.get("sharpe_ratio") if metrics.get("sharpe_ratio") is not None else "-"}</b></div><div class="card">Sortino<b>{metrics.get("sortino_ratio") if metrics.get("sortino_ratio") is not None else "-"}</b></div><div class="card">胜率<b>{_percent(metrics.get("win_rate"))}</b></div><div class="card">现金月占比<b>{_percent(metrics.get("cash_month_ratio"))}</b></div><div class="card">月均换手<b>{_percent(metrics.get("average_monthly_turnover"))}</b></div></div></header>
<section class="panel"><h2>权益与基准</h2>{_svg_chart(report)}</section>
<section class="panel"><h2>年度汇总</h2><table><thead><tr><th>年度</th><th>策略</th><th>基准</th><th>投资月份/月份</th></tr></thead><tbody>{annual_rows}</tbody></table></section>
<section class="panel"><h2>逐月表现</h2><div class="scroll"><table><thead><tr><th>月份</th><th>市场状态</th><th>持仓数</th><th>策略</th><th>基准</th></tr></thead><tbody>{monthly_rows}</tbody></table></div></section>
<section class="panel"><h2>行业 × 阶段归因</h2><div class="scroll"><table><thead><tr><th>行业</th><th>阶段</th><th>持仓记录</th><th>胜率</th><th>组合贡献</th></tr></thead><tbody>{attribution_rows}</tbody></table></div></section>
<section class="panel"><h2>敏感性分析</h2><div class="scroll"><table><thead><tr><th>情景</th><th>月份</th><th>累计收益</th><th>年化收益</th><th>最大回撤</th><th>夏普</th></tr></thead><tbody>{sensitivity_rows}</tbody></table></div></section>
<section class="panel"><h2>逐月持仓</h2><div class="scroll"><table><thead><tr><th>月份</th><th>代码</th><th>名称</th><th>行业</th><th>阶段</th><th>净收益</th></tr></thead><tbody>{holding_rows}</tbody></table></div></section>
<section class="panel"><h2>局限与风险</h2><ul>{limitation_rows}</ul></section>
<section class="panel"><details><summary>嵌入式完整 JSON 数据</summary><code id="report-data">{payload}</code></details></section>
</main></body></html>"""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")


def render_report_pdf(report: dict[str, Any], target: Path) -> None:
    """Render a Chinese PDF with metrics, monthly holdings and limitations."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    try:
        pdfmetrics.getFont("STSong-Light")
    except KeyError:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    font = "STSong-Light"
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "body-cn", parent=styles["BodyText"], fontName=font, fontSize=8.5, leading=13
    )
    heading = ParagraphStyle(
        "heading-cn",
        parent=styles["Heading2"],
        fontName=font,
        fontSize=14,
        leading=20,
        textColor=colors.HexColor("#173a5e"),
    )
    title_style = ParagraphStyle(
        "title-cn", parent=styles["Title"], fontName=font, fontSize=20, leading=28
    )

    def p(value: Any, style: ParagraphStyle = body) -> Paragraph:
        return Paragraph(escape(str(value if value is not None else "-")), style)

    def table(rows: list[list[Any]], widths: list[float]) -> Table:
        converted = [[p(cell) for cell in row] for row in rows]
        result = Table(converted, colWidths=widths, repeatRows=1)
        result.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173a5e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cad3dc")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f4f7fa")],
                    ),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        return result

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.pdf")
    document = SimpleDocTemplate(
        str(temporary),
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )
    metrics = report.get("metrics", {})
    story: list[Any] = [
        p(report.get("meta", {}).get("title", "量化回测报告"), title_style),
        p(f"策略：{report.get('meta', {}).get('strategy', '-')}"),
        Spacer(1, 5 * mm),
        p("核心指标", heading),
        table(
            [
                ["累计收益", "年化收益", "年化波动", "最大回撤", "夏普", "胜率"],
                [
                    _percent(metrics.get("total_return")),
                    _percent(metrics.get("annualized_return")),
                    _percent(metrics.get("annualized_volatility")),
                    _percent(metrics.get("max_drawdown")),
                    metrics.get("sharpe_ratio", "-"),
                    _percent(metrics.get("win_rate")),
                ],
            ],
            [28 * mm] * 6,
        ),
        Spacer(1, 5 * mm),
        p("年度表现", heading),
    ]
    annual = [["年度", "策略收益", "基准收益", "月份", "投资月份"]] + [
        [
            row.get("year"),
            _percent(row.get("return")),
            _percent(row.get("benchmark_return")),
            row.get("months"),
            row.get("invested_months"),
        ]
        for row in report.get("annual", [])
    ]
    equity = [["月份", "策略权益", "基准权益", "回撤"]] + [
        [
            row.get("month"),
            f"{float(row.get('strategy_equity', 1)):.4f}",
            f"{float(row.get('benchmark_equity', 1)):.4f}",
            _percent(row.get("drawdown")),
        ]
        for row in report.get("equity_curve", [])
    ]
    story.extend(
        [
            table(annual, [28 * mm, 34 * mm, 34 * mm, 25 * mm, 30 * mm]),
            Spacer(1, 5 * mm),
            p("权益与回撤序列", heading),
            table(equity, [34 * mm, 38 * mm, 38 * mm, 35 * mm]),
            PageBreak(),
            p("逐月表现", heading),
        ]
    )
    monthly = [["月份", "信号日", "市场", "持仓", "策略", "基准"]] + [
        [
            row.get("holding_month"),
            row.get("signal_date"),
            row.get("regime"),
            row.get("holding_count"),
            _percent(row.get("net_return")),
            _percent(row.get("benchmark_return")),
        ]
        for row in report.get("monthly", [])
    ]
    story.extend(
        [
            table(monthly, [25 * mm, 30 * mm, 24 * mm, 18 * mm, 26 * mm, 26 * mm]),
            PageBreak(),
            p("逐月持仓", heading),
        ]
    )
    holdings = [["月份", "代码/名称", "行业", "阶段", "净收益"]] + [
        [
            row.get("holding_month"),
            f"{row.get('symbol')} {row.get('stock_name')}",
            row.get("industry_name"),
            row.get("stage"),
            _percent(row.get("net_return")),
        ]
        for row in report.get("holdings", [])
    ]
    story.extend(
        [
            table(holdings, [23 * mm, 45 * mm, 34 * mm, 25 * mm, 25 * mm]),
            Spacer(1, 5 * mm),
            p("行业 × 阶段归因", heading),
        ]
    )
    attribution = [["行业", "阶段", "记录", "胜率", "贡献"]] + [
        [
            row.get("industry_name"),
            row.get("stage"),
            row.get("holding_records"),
            _percent(row.get("win_rate")),
            _percent(row.get("total_portfolio_contribution")),
        ]
        for row in report.get("industry_stage_attribution", [])
    ]
    sensitivity = [["敏感性情景", "月份", "累计", "年化", "回撤", "夏普"]] + [
        [
            row.get("scenario"),
            row.get("months"),
            _percent(row.get("total_return")),
            _percent(row.get("annualized_return")),
            _percent(row.get("max_drawdown")),
            row.get("sharpe_ratio"),
        ]
        for row in report.get("sensitivity", [])
    ]
    story.extend(
        [
            table(attribution, [42 * mm, 28 * mm, 22 * mm, 26 * mm, 32 * mm]),
            Spacer(1, 5 * mm),
            p("敏感性分析", heading),
            table(sensitivity, [46 * mm, 18 * mm, 24 * mm, 24 * mm, 24 * mm, 22 * mm]),
            Spacer(1, 5 * mm),
            p("局限与风险", heading),
        ]
    )
    story.extend(p(f"• {item}") for item in report.get("limitations", []))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#697586"))
        canvas.drawCentredString(A4[0] / 2, 7 * mm, f"第 {doc.page} 页")
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    temporary.replace(target)


def write_outputs(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write JSON, analysis CSV extracts, standalone HTML and Chinese PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    end_token = str(report.get("parameters", {}).get("end_date") or "latest").replace(
        "-", ""
    )
    stem = f"leading_control_backtest_{end_token}"
    paths = {
        "json": output_dir / f"{stem}.json",
        "equity_csv": output_dir / f"{stem}_equity.csv",
        "monthly_csv": output_dir / f"{stem}_monthly.csv",
        "holdings_csv": output_dir / f"{stem}_holdings.csv",
        "annual_csv": output_dir / f"{stem}_annual.csv",
        "sensitivity_csv": output_dir / f"{stem}_sensitivity.csv",
        "html": output_dir / f"{stem}.html",
        "pdf": output_dir / f"{stem}.pdf",
    }
    paths["json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(report.get("monthly") or []).to_csv(paths["monthly_csv"], index=False)
    pd.DataFrame(report.get("holdings") or []).to_csv(
        paths["holdings_csv"], index=False
    )
    pd.DataFrame(report.get("equity_curve") or []).to_csv(
        paths["equity_csv"], index=False
    )
    pd.DataFrame(report.get("annual") or []).to_csv(paths["annual_csv"], index=False)
    pd.DataFrame(report.get("sensitivity") or []).to_csv(
        paths["sensitivity_csv"], index=False
    )
    render_interactive_html(report, paths["html"])
    render_report_pdf(report, paths["pdf"])
    for kind in ("json", "html", "pdf"):
        shutil.copyfile(paths[kind], output_dir / f"latest.{kind}")
    return paths


def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now().normalize()
    default_start = (today.to_period("M") - 61).start_time
    parser = argparse.ArgumentParser(description="回测领先区控盘阶段月度选股策略")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=default_start.date().isoformat())
    parser.add_argument("--end-date", default=today.date().isoformat())
    parser.add_argument("--cost-bps", type=float, default=15.0)
    parser.add_argument("--max-stocks", type=int, default=10)
    parser.add_argument("--member-cache-days", type=float, default=30.0)
    parser.add_argument("--expected-months", type=int, default=60)
    parser.add_argument("--skip-db-tail", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    start = pd.Timestamp(args.start_date).normalize()
    end = pd.Timestamp(args.end_date).normalize()
    if start >= end:
        raise ValueError("start-date must be earlier than end-date")
    load_start = pd.Timestamp(year=start.year, month=1, day=1)
    LOGGER.info("Loading annual feature snapshots and local database tail")
    stock, source_audit = load_market_data(
        args.snapshot_dir,
        load_start,
        end,
        supplement_database=not args.skip_db_tail,
    )
    stock, quality = validate_market_data(stock)
    quality["sources"] = source_audit
    if pd.Timestamp(stock["trade_date"].max()) < end - pd.Timedelta(days=7):
        quality["status"] = "warning"
        quality["issues"].append(
            f"数据最新日期 {quality['date_max']} 早于请求结束日期 {end.date().isoformat()}"
        )
    LOGGER.info("Loading point-in-time SW2021 membership history")
    industries, memberships = load_historical_sw_industry_universe(
        args.cache_dir, cache_days=args.member_cache_days
    )
    LOGGER.info("Loading HS300 benchmark")
    benchmark = load_hs300_benchmark(
        args.cache_dir,
        start,
        min(end, pd.Timestamp(stock["trade_date"].max())),
        cache_days=args.member_cache_days,
    )
    report = run_monthly_backtest(
        stock,
        industries,
        memberships,
        start_date=start,
        end_date=min(end, pd.Timestamp(stock["trade_date"].max())),
        transaction_cost_bps=args.cost_bps,
        max_stocks=args.max_stocks,
        data_quality=quality,
        benchmark=benchmark,
    )
    actual_months = int(report["metrics"]["months"] or 0)
    if args.expected_months > 0 and actual_months != args.expected_months:
        raise RuntimeError(
            f"expected {args.expected_months} complete monthly periods, got {actual_months}"
        )
    if report.get("signal_errors"):
        raise RuntimeError(
            f"{len(report['signal_errors'])} monthly signals failed; refusing a formal report"
        )
    paths = write_outputs(report, args.output_dir)
    LOGGER.info(
        "Backtest complete: %s", ", ".join(str(path) for path in paths.values())
    )
    print(
        json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
