#!/usr/bin/env python3
"""Point-in-time backtest for an Alpha144 liquidity-breakout strategy.

Signals are evaluated after the close.  Orders execute at the next trading
day's open.  The stock universe is the historical CSI 500 membership returned
by Baostock for each refresh date, rather than a present-day constituent list
backfilled into the past.

The canonical GTJA Alpha144 used here is::

    rolling_sum_20(IF(ret < 0, ABS(ret) / amount, 0))
    -------------------------------------------------
             rolling_count_20(ret < 0)

Higher values represent worse downside liquidity.  Prices used for returns
are adjusted, while raw prices are retained in the trade ledger.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("alpha144_liquidity_breakout_backtest")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "db" / "feature_snapshots"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "alpha144-liquidity-breakout"
DEFAULT_MEMBERSHIP_CACHE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "alpha144-liquidity-breakout"
    / "csi500-pit-memberships.json"
)
DEFAULT_BENCHMARK_CACHE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "alpha144-liquidity-breakout"
    / "csi500-index-daily.csv"
)


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str = "2021-01-04"
    end_date: str = "2026-05-15"
    factor_window: int = 20
    breakout_window: int = 5
    refresh_days: int = 10
    factor_top_fraction: float = 0.15
    max_positions: int = 5
    holding_days: int = 20
    market_window: int = 20
    market_stop_return: float = -0.03
    initial_capital: float = 100_000_000.0
    buy_cost_bps: float = 8.0
    sell_cost_bps: float = 13.0
    limit_tolerance: float = 0.001

    def validate(self) -> None:
        if pd.Timestamp(self.start_date) > pd.Timestamp(self.end_date):
            raise ValueError("start_date must not be after end_date")
        for name in (
            "factor_window",
            "breakout_window",
            "refresh_days",
            "max_positions",
            "holding_days",
            "market_window",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < float(self.factor_top_fraction) <= 1:
            raise ValueError("factor_top_fraction must be in (0, 1]")
        if float(self.initial_capital) <= 0:
            raise ValueError("initial_capital must be positive")


@dataclass
class Position:
    symbol: str
    units: float
    entry_date: pd.Timestamp
    entry_calendar_index: int
    entry_adj_open: float
    entry_raw_open: float
    alpha144: float
    buy_cost: float


def normalize_symbol(value: Any) -> str | None:
    """Normalize common A-share spellings to QuantMind prefix codes."""
    raw = str(value or "").strip().upper()
    match = re.search(r"(\d{6})", raw)
    if not match:
        return None
    code = match.group(1)
    if raw.startswith("SH") or raw.endswith(".SH") or code.startswith(("6", "9")):
        return f"SH{code}"
    if raw.startswith("BJ") or raw.endswith(".BJ") or code.startswith(("4", "8")):
        return f"BJ{code}"
    return f"SZ{code}"


def _read_snapshot_year(
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
        "liq_amount",
    ]
    try:
        frame = pd.read_parquet(
            path,
            columns=columns,
            filters=[("trade_date", ">=", start), ("trade_date", "<=", end)],
        )
    except (TypeError, ValueError):
        frame = pd.read_parquet(path, columns=columns)
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="coerce"
    ).dt.normalize()
    return frame[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)].copy()


def load_market_snapshots(
    snapshot_dir: Path,
    start_date: Any,
    end_date: Any,
    *,
    lookback_calendar_days: int = 90,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the minimum annual-snapshot columns needed by the strategy."""
    requested_start = pd.Timestamp(start_date).normalize()
    load_start = requested_start - pd.Timedelta(days=lookback_calendar_days)
    end = pd.Timestamp(end_date).normalize()
    frames: list[pd.DataFrame] = []
    files: list[str] = []
    for year in range(load_start.year, end.year + 1):
        path = snapshot_dir / f"model_features_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing annual feature snapshot: {path}")
        files.append(str(path))
        frames.append(_read_snapshot_year(path, load_start, end))
    frame = pd.concat(frames, ignore_index=True)
    frame["symbol"] = frame["symbol"].map(normalize_symbol)
    frame = frame.dropna(subset=["trade_date", "symbol"])
    frame = frame.sort_values(["symbol", "trade_date"], kind="stable")
    frame = frame.drop_duplicates(["trade_date", "symbol"], keep="last")
    numeric = ["open", "high", "low", "close", "volume", "factor", "liq_amount"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid = (
        ~np.isfinite(frame[["open", "high", "low", "close", "factor"]]).all(axis=1)
        | (frame[["open", "high", "low", "close", "factor"]] <= 0).any(axis=1)
        | ~np.isfinite(frame["liq_amount"])
        | (frame["liq_amount"] <= 0)
        | (frame["high"] < frame["low"])
    )
    invalid_rows = int(invalid.sum())
    frame = frame.loc[~invalid].copy()
    return frame.reset_index(drop=True), {
        "snapshot_files": files,
        "loaded_rows": int(len(frame)),
        "invalid_rows_removed": invalid_rows,
        "date_min": frame["trade_date"].min().date().isoformat(),
        "date_max": frame["trade_date"].max().date().isoformat(),
        "symbol_count": int(frame["symbol"].nunique()),
    }


def prepare_alpha144_features(
    market: pd.DataFrame,
    *,
    factor_window: int = 20,
    breakout_window: int = 5,
) -> pd.DataFrame:
    """Calculate adjusted prices, canonical Alpha144 and prior-close breakout."""
    frame = market.copy()
    frame = frame.sort_values(["symbol", "trade_date"], kind="stable").reset_index(
        drop=True
    )
    frame["adj_open"] = frame["open"] * frame["factor"]
    frame["adj_high"] = frame["high"] * frame["factor"]
    frame["adj_low"] = frame["low"] * frame["factor"]
    frame["adj_close"] = frame["close"] * frame["factor"]
    grouped = frame.groupby("symbol", sort=False, group_keys=False)
    frame["return_1d"] = grouped["adj_close"].pct_change(fill_method=None)
    frame["previous_adj_close"] = grouped["adj_close"].shift(1)
    frame["price_impact"] = frame["return_1d"].abs() / frame["liq_amount"]
    frame["down_impact"] = frame["price_impact"].where(frame["return_1d"] < 0, 0.0)
    frame["down_day"] = (frame["return_1d"] < 0).astype(float)
    numerator = grouped["down_impact"].transform(
        lambda values: values.rolling(factor_window, min_periods=factor_window).sum()
    )
    denominator = grouped["down_day"].transform(
        lambda values: values.rolling(factor_window, min_periods=factor_window).sum()
    )
    frame["alpha144"] = numerator / denominator.replace(0.0, np.nan)
    frame["prior_breakout_close"] = grouped["adj_close"].transform(
        lambda values: values.shift(1)
        .rolling(breakout_window, min_periods=breakout_window)
        .max()
    )
    frame["breakout"] = frame["adj_close"] > frame["prior_breakout_close"]
    return frame


BenchmarkQuery = Callable[[str, str], pd.DataFrame]


def _baostock_benchmark_query(start_date: str, end_date: str) -> pd.DataFrame:
    import baostock as bs

    result = bs.query_history_k_data_plus(
        "sh.000905",
        "date,open,high,low,close,preclose,pctChg",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )
    if result.error_code != "0":
        raise RuntimeError(
            "Baostock CSI 500 history query failed: "
            f"{result.error_code} {result.error_msg}"
        )
    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(
        rows,
        columns=["trade_date", "open", "high", "low", "close", "preclose", "pct_chg"],
    )


def load_csi500_benchmark(
    cache_path: Path,
    start_date: Any,
    end_date: Any,
    *,
    lookback_calendar_days: int = 90,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load CSI 500 history from a local cache, filling it through Baostock."""
    return _load_csi500_benchmark(
        cache_path,
        start_date,
        end_date,
        lookback_calendar_days=lookback_calendar_days,
    )


def _load_csi500_benchmark(
    cache_path: Path,
    start_date: Any,
    end_date: Any,
    *,
    lookback_calendar_days: int = 90,
    query: BenchmarkQuery | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    requested_start = pd.Timestamp(start_date).normalize()
    start = requested_start - pd.Timedelta(days=lookback_calendar_days)
    end = pd.Timestamp(end_date).normalize()
    cached = pd.DataFrame()
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path)
            if not {"trade_date", "open", "close"}.issubset(cached.columns):
                raise ValueError("benchmark cache is missing required columns")
            cached["trade_date"] = pd.to_datetime(
                cached["trade_date"], errors="coerce"
            ).dt.normalize()
            cached = cached.dropna(subset=["trade_date"])
        except (OSError, ValueError, TypeError, KeyError):
            LOGGER.warning("Ignoring invalid benchmark cache: %s", cache_path)
            cached = pd.DataFrame()
    cache_covers_range = (
        not cached.empty
        and cached["trade_date"].min() <= start + pd.Timedelta(days=7)
        and cached["trade_date"].max() >= end
    )
    queried = False
    if cache_covers_range:
        frame = cached
    else:
        query_start = (
            min(start, cached["trade_date"].min()) if not cached.empty else start
        )
        query_end = max(end, cached["trade_date"].max()) if not cached.empty else end
        owns_session = query is None
        if owns_session:
            import baostock as bs

            login = bs.login()
            if login.error_code != "0":
                raise RuntimeError(
                    f"Baostock login failed: {login.error_code} {login.error_msg}"
                )
            query = _baostock_benchmark_query
        try:
            assert query is not None
            frame = query(query_start.date().isoformat(), query_end.date().isoformat())
            queried = True
        finally:
            if owns_session:
                import baostock as bs

                bs.logout()
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="coerce"
    ).dt.normalize()
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "open", "close"])
    frame = frame[(frame["open"] > 0) & (frame["close"] > 0)]
    frame = frame.drop_duplicates("trade_date", keep="last").sort_values("trade_date")
    frame = frame[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)].copy()
    if (
        frame.empty
        or frame["trade_date"].min() > requested_start
        or frame["trade_date"].max() < end
    ):
        raise RuntimeError(
            "Baostock CSI 500 benchmark does not cover the requested backtest range"
        )
    if queried:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index=False)
    frame["adj_open"] = frame["open"]
    frame["adj_close"] = frame["close"]
    return frame.reset_index(drop=True), {
        "benchmark_source": "baostock.query_history_k_data_plus(sh.000905)",
        "benchmark_cache": str(cache_path),
        "benchmark_cache_refreshed": queried,
        "benchmark_rows": int(len(frame)),
        "benchmark_date_min": frame["trade_date"].min().date().isoformat(),
        "benchmark_date_max": frame["trade_date"].max().date().isoformat(),
    }


MembershipQuery = Callable[[str], tuple[str, list[dict[str, Any]]]]


def _baostock_membership_query(trade_date: str) -> tuple[str, list[dict[str, Any]]]:
    """Query one point-in-time CSI 500 snapshot from an active Baostock session."""
    import baostock as bs

    requested = pd.Timestamp(trade_date).normalize()
    last_count = 0
    lag_candidates = list(range(8)) + list(range(14, 92, 7))
    for lag_days in lag_candidates:
        query_date = (requested - pd.Timedelta(days=lag_days)).date().isoformat()
        result = bs.query_zz500_stocks(date=query_date)
        if result.error_code != "0":
            raise RuntimeError(
                f"Baostock query_zz500_stocks failed for {query_date}: "
                f"{result.error_code} {result.error_msg}"
            )
        rows: list[dict[str, Any]] = []
        effective_dates: set[str] = set()
        while result.next():
            values = result.get_row_data()
            if len(values) < 2:
                continue
            effective_dates.add(str(values[0] or query_date))
            symbol = normalize_symbol(values[1])
            if symbol:
                rows.append(
                    {
                        "symbol": symbol,
                        "name": str(values[2] if len(values) > 2 else symbol),
                    }
                )
        unique_symbols = {row["symbol"] for row in rows}
        last_count = len(unique_symbols)
        if (
            last_count == 500
            and len(effective_dates) == 1
            and pd.Timestamp(next(iter(effective_dates))).normalize() <= requested
        ):
            if lag_days:
                LOGGER.warning(
                    "CSI 500 snapshot %s had incomplete data; "
                    "using complete prior snapshot queried at %s",
                    trade_date,
                    query_date,
                )
            return next(iter(effective_dates)), rows
    raise RuntimeError(
        f"Baostock returned no complete prior CSI 500 snapshot for {trade_date}; "
        f"last unique member count was {last_count}"
    )


def load_pit_csi500_memberships(
    signal_dates: list[pd.Timestamp],
    cache_path: Path,
    *,
    query: MembershipQuery | None = None,
) -> tuple[dict[pd.Timestamp, set[str]], dict[str, Any]]:
    """Load or query historical CSI 500 membership for every signal date."""
    cache: dict[str, Any] = {"version": 1, "snapshots": {}}
    if cache_path.exists():
        try:
            candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            if candidate.get("version") == 1 and isinstance(
                candidate.get("snapshots"), dict
            ):
                cache = candidate
        except (OSError, ValueError, TypeError):
            LOGGER.warning("Ignoring invalid membership cache: %s", cache_path)
    snapshots = cache["snapshots"]
    missing = [
        date for date in signal_dates if date.date().isoformat() not in snapshots
    ]
    owns_session = query is None and bool(missing)
    if owns_session:
        import baostock as bs

        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(
                f"Baostock login failed: {login.error_code} {login.error_msg}"
            )
        query = _baostock_membership_query
    cache_updated = False

    def persist_cache() -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(cache_path)

    try:
        for index, signal_date in enumerate(missing, start=1):
            key = signal_date.date().isoformat()
            assert query is not None
            effective_date, rows = query(key)
            snapshots[key] = {
                "effective_date": effective_date,
                "members": rows,
            }
            cache_updated = True
            if index == 1 or index % 20 == 0 or index == len(missing):
                LOGGER.info("CSI 500 membership %s/%s: %s", index, len(missing), key)
            if index % 20 == 0:
                persist_cache()
    finally:
        if cache_updated:
            persist_cache()
        if owns_session:
            import baostock as bs

            bs.logout()
    result: dict[pd.Timestamp, set[str]] = {}
    effective_dates: dict[str, str] = {}
    for signal_date in signal_dates:
        key = signal_date.date().isoformat()
        item = snapshots.get(key) or {}
        members = {
            symbol
            for symbol in (
                normalize_symbol(row.get("symbol")) for row in item.get("members", [])
            )
            if symbol
        }
        if len(members) != 500:
            raise RuntimeError(
                f"CSI 500 point-in-time membership has {len(members)} unique members for {key}"
            )
        effective_date = pd.Timestamp(item.get("effective_date") or key).normalize()
        if effective_date > signal_date:
            raise RuntimeError(
                f"CSI 500 membership effective date {effective_date.date()} "
                f"is after signal date {key}"
            )
        result[signal_date] = members
        effective_dates[key] = effective_date.date().isoformat()
    return result, {
        "source": "baostock.query_zz500_stocks",
        "snapshot_count": len(result),
        "cache_path": str(cache_path),
        "queried_snapshot_count": len(missing),
        "effective_dates": effective_dates,
    }


def _limit_ratio(symbol: str) -> float:
    if symbol.startswith("SZ30") or symbol.startswith("SH68"):
        return 0.20
    return 0.10


def _at_price_limit(row: pd.Series, direction: str, tolerance: float) -> bool:
    previous = float(row.get("previous_adj_close") or np.nan)
    current_open = float(row.get("adj_open") or np.nan)
    if not math.isfinite(previous) or not math.isfinite(current_open) or previous <= 0:
        return True
    open_return = current_open / previous - 1.0
    limit = _limit_ratio(str(row.get("symbol") or ""))
    if direction == "buy":
        return open_return >= limit - tolerance
    return open_return <= -limit + tolerance


def calculate_daily_metrics(curve: pd.DataFrame) -> dict[str, Any]:
    """Calculate daily strategy, benchmark, excess and CAPM statistics."""
    if curve.empty:
        raise ValueError("equity curve is empty")
    strategy_returns = pd.to_numeric(curve["strategy_return"], errors="coerce").fillna(
        0.0
    )
    benchmark_returns = pd.to_numeric(
        curve["benchmark_return"], errors="coerce"
    ).fillna(0.0)
    comparison_strategy_returns = strategy_returns.iloc[1:]
    comparison_benchmark_returns = benchmark_returns.iloc[1:]
    strategy_equity = pd.to_numeric(curve["equity"], errors="coerce")
    periods = max(len(curve) - 1, 1)
    years = periods / 252.0
    total_return = float(strategy_equity.iloc[-1] / strategy_equity.iloc[0] - 1.0)
    benchmark_total = float((1.0 + comparison_benchmark_returns).prod() - 1.0)
    annualized = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    benchmark_annualized = float((1.0 + benchmark_total) ** (1.0 / years) - 1.0)
    volatility = float(strategy_returns.std(ddof=0) * math.sqrt(252))
    sharpe = (
        float(strategy_returns.mean() / strategy_returns.std(ddof=0) * math.sqrt(252))
        if strategy_returns.std(ddof=0) > 0
        else None
    )
    peak = strategy_equity.cummax()
    drawdown = strategy_equity / peak - 1.0
    max_drawdown = float(drawdown.min())
    benchmark_variance = float(comparison_benchmark_returns.var(ddof=0))
    beta = (
        float(
            np.cov(
                comparison_strategy_returns,
                comparison_benchmark_returns,
                ddof=0,
            )[0, 1]
            / benchmark_variance
        )
        if benchmark_variance > 0
        else None
    )
    alpha = (
        float(
            (
                comparison_strategy_returns.mean()
                - beta * comparison_benchmark_returns.mean()
            )
            * 252
        )
        if beta is not None
        else None
    )
    excess_returns = comparison_strategy_returns - comparison_benchmark_returns
    return {
        "trading_days": int(len(curve)),
        "total_return": total_return,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": annualized / abs(max_drawdown) if max_drawdown < 0 else None,
        "benchmark_total_return": benchmark_total,
        "benchmark_annualized_return": benchmark_annualized,
        "excess_total_return": float(
            (1.0 + total_return) / (1.0 + benchmark_total) - 1.0
        ),
        "annualized_excess_return": annualized - benchmark_annualized,
        "beta": beta,
        "alpha_annualized": alpha,
        "tracking_error": float(excess_returns.std(ddof=0) * math.sqrt(252)),
        "information_ratio": (
            float(excess_returns.mean() / excess_returns.std(ddof=0) * math.sqrt(252))
            if excess_returns.std(ddof=0) > 0
            else None
        ),
        "cash_day_ratio": float((pd.to_numeric(curve["holding_count"]) == 0).mean()),
        "average_holdings": float(pd.to_numeric(curve["holding_count"]).mean()),
    }


def run_backtest(
    prepared_market: pd.DataFrame,
    benchmark: pd.DataFrame,
    memberships: dict[pd.Timestamp, set[str]],
    config: BacktestConfig,
) -> dict[str, Any]:
    """Run a daily event-driven simulation with next-open execution."""
    config.validate()
    start = pd.Timestamp(config.start_date).normalize()
    end = pd.Timestamp(config.end_date).normalize()
    benchmark_frame = benchmark.copy().sort_values("trade_date")
    benchmark_frame["benchmark_return"] = benchmark_frame["adj_close"].pct_change(
        fill_method=None
    )
    benchmark_frame["market_return_window"] = benchmark_frame["adj_close"].pct_change(
        config.market_window, fill_method=None
    )
    benchmark_frame = benchmark_frame.set_index("trade_date")
    calendar = [
        pd.Timestamp(value)
        for value in benchmark_frame.loc[start:end].index.unique().sort_values()
    ]
    if not calendar:
        raise RuntimeError("benchmark calendar is empty in requested range")
    signal_dates = calendar[:: config.refresh_days]
    missing_memberships = [date for date in signal_dates if date not in memberships]
    if missing_memberships:
        raise RuntimeError(
            "missing CSI 500 membership snapshots: "
            + ", ".join(date.date().isoformat() for date in missing_memberships[:5])
        )

    market = prepared_market.copy()
    market = market[(market["trade_date"] >= start) & (market["trade_date"] <= end)]
    daily_rows = {
        pd.Timestamp(date): group.set_index("symbol", drop=False)
        for date, group in market.groupby("trade_date", sort=False)
    }
    buy_cost_rate = config.buy_cost_bps / 10_000.0
    sell_cost_rate = config.sell_cost_bps / 10_000.0
    cash = float(config.initial_capital)
    positions: dict[str, Position] = {}
    pending_buys: list[dict[str, Any]] = []
    pending_sells: dict[str, str] = {}
    last_prices: dict[str, float] = {}
    curve_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    previous_equity = float(config.initial_capital)
    active_factor_pool: dict[str, float] = {}
    active_membership_count = 0

    for calendar_index, trade_date in enumerate(calendar):
        day = daily_rows.get(trade_date, pd.DataFrame())
        traded_notional = 0.0

        for symbol, reason in list(pending_sells.items()):
            position = positions.get(symbol)
            if position is None:
                pending_sells.pop(symbol, None)
                continue
            if day.empty or symbol not in day.index:
                rejection_rows.append(
                    {
                        "trade_date": trade_date,
                        "symbol": symbol,
                        "side": "sell",
                        "reason": "missing_open_retry",
                    }
                )
                continue
            row = day.loc[symbol]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            if _at_price_limit(row, "sell", config.limit_tolerance):
                rejection_rows.append(
                    {
                        "trade_date": trade_date,
                        "symbol": symbol,
                        "side": "sell",
                        "reason": "limit_down_retry",
                    }
                )
                continue
            adj_open = float(row["adj_open"])
            raw_open = float(row["open"])
            gross_proceeds = position.units * adj_open
            sell_cost = gross_proceeds * sell_cost_rate
            cash += gross_proceeds - sell_cost
            traded_notional += gross_proceeds
            trade_rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "side": "sell",
                    "reason": reason,
                    "raw_price": raw_open,
                    "adjusted_price": adj_open,
                    "units": position.units,
                    "gross_notional": gross_proceeds,
                    "cost": sell_cost,
                    "holding_days": calendar_index - position.entry_calendar_index,
                    "net_trade_return": (
                        adj_open
                        * (1.0 - sell_cost_rate)
                        / (position.entry_adj_open * (1.0 + buy_cost_rate))
                        - 1.0
                    ),
                    "entry_date": position.entry_date,
                }
            )
            positions.pop(symbol, None)
            pending_sells.pop(symbol, None)

        opening_value = cash
        for symbol, position in positions.items():
            if not day.empty and symbol in day.index:
                row = day.loc[symbol]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                price = float(row["adj_open"])
            else:
                price = last_prices.get(symbol, position.entry_adj_open)
            opening_value += position.units * price

        available_slots = max(config.max_positions - len(positions), 0)
        for order in pending_buys[:available_slots]:
            symbol = str(order["symbol"])
            if symbol in positions or day.empty or symbol not in day.index:
                rejection_rows.append(
                    {
                        "trade_date": trade_date,
                        "symbol": symbol,
                        "side": "buy",
                        "reason": "duplicate_or_missing_open_cancelled",
                    }
                )
                continue
            row = day.loc[symbol]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            if _at_price_limit(row, "buy", config.limit_tolerance):
                rejection_rows.append(
                    {
                        "trade_date": trade_date,
                        "symbol": symbol,
                        "side": "buy",
                        "reason": "limit_up_cancelled",
                    }
                )
                continue
            target_notional = opening_value / config.max_positions
            gross_notional = min(target_notional, cash / (1.0 + buy_cost_rate))
            if gross_notional <= 0:
                continue
            adj_open = float(row["adj_open"])
            raw_open = float(row["open"])
            units = gross_notional / adj_open
            buy_cost = gross_notional * buy_cost_rate
            cash -= gross_notional + buy_cost
            traded_notional += gross_notional
            positions[symbol] = Position(
                symbol=symbol,
                units=units,
                entry_date=trade_date,
                entry_calendar_index=calendar_index,
                entry_adj_open=adj_open,
                entry_raw_open=raw_open,
                alpha144=float(order["alpha144"]),
                buy_cost=buy_cost,
            )
            trade_rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "side": "buy",
                    "reason": "alpha144_top15_breakout",
                    "raw_price": raw_open,
                    "adjusted_price": adj_open,
                    "units": units,
                    "gross_notional": gross_notional,
                    "cost": buy_cost,
                    "holding_days": 0,
                    "net_trade_return": None,
                    "entry_date": trade_date,
                }
            )
        pending_buys = []

        equity = cash
        for symbol, position in positions.items():
            if not day.empty and symbol in day.index:
                row = day.loc[symbol]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                close_price = float(row["adj_close"])
                last_prices[symbol] = close_price
            else:
                close_price = last_prices.get(symbol, position.entry_adj_open)
            equity += position.units * close_price

        benchmark_row = benchmark_frame.loc[trade_date]
        if isinstance(benchmark_row, pd.DataFrame):
            benchmark_row = benchmark_row.iloc[-1]
        benchmark_return = float(benchmark_row.get("benchmark_return") or 0.0)
        market_window_return = float(
            benchmark_row.get("market_return_window") or np.nan
        )
        market_open = not math.isfinite(market_window_return) or (
            market_window_return >= config.market_stop_return
        )

        for symbol, position in positions.items():
            held_sessions = calendar_index - position.entry_calendar_index + 1
            if held_sessions >= config.holding_days:
                pending_sells.setdefault(symbol, "holding_period_20d")
        if not market_open:
            for symbol in positions:
                pending_sells[symbol] = "csi500_20d_below_threshold"

        is_refresh = trade_date in memberships
        refresh_eligible_count: int | None = None
        if is_refresh:
            members = memberships[trade_date]
            eligible = day.loc[day.index.intersection(sorted(members))].copy()
            eligible = eligible.replace([np.inf, -np.inf], np.nan)
            eligible = eligible.dropna(subset=["alpha144"])
            pool_size = int(math.ceil(len(eligible) * config.factor_top_fraction))
            factor_pool = eligible.sort_values("alpha144", ascending=False).head(
                pool_size
            )
            active_factor_pool = {
                str(row.symbol): float(row.alpha144) for row in factor_pool.itertuples()
            }
            active_membership_count = len(members)
            refresh_eligible_count = len(eligible)

        pool_rows = day.loc[day.index.intersection(sorted(active_factor_pool))].copy()
        if not pool_rows.empty:
            pool_rows = pool_rows.replace([np.inf, -np.inf], np.nan)
            pool_rows = pool_rows.dropna(subset=["prior_breakout_close"])
            pool_rows = pool_rows[pool_rows["breakout"]].copy()
            pool_rows["pool_alpha144"] = pool_rows["symbol"].map(active_factor_pool)
            pool_rows = pool_rows.sort_values("pool_alpha144", ascending=False)
        expected_survivors = len(set(positions) - set(pending_sells))
        slots = max(config.max_positions - expected_survivors, 0)
        selected = []
        if market_open and slots > 0:
            for row in pool_rows.itertuples():
                if row.symbol in positions:
                    continue
                selected.append(
                    {"symbol": row.symbol, "alpha144": float(row.pool_alpha144)}
                )
                if len(selected) >= slots:
                    break
        pending_buys = selected
        selection_rows.append(
            {
                "signal_date": trade_date,
                "pool_refresh": is_refresh,
                "membership_count": active_membership_count,
                "eligible_count": refresh_eligible_count,
                "factor_pool_count": len(active_factor_pool),
                "breakout_count": len(pool_rows),
                "selected_count": len(selected),
                "selected_symbols": ",".join(item["symbol"] for item in selected),
                "market_20d_return": market_window_return,
                "market_open": market_open,
            }
        )

        strategy_return = equity / previous_equity - 1.0
        curve_rows.append(
            {
                "trade_date": trade_date,
                "equity": equity,
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "benchmark_20d_return": market_window_return,
                "holding_count": len(positions),
                "cash_ratio": cash / equity if equity > 0 else 0.0,
                "turnover": (
                    traded_notional / previous_equity if previous_equity > 0 else 0.0
                ),
                "refresh": is_refresh,
                "market_open": market_open,
            }
        )
        previous_equity = equity

    curve = pd.DataFrame(curve_rows)
    trades = pd.DataFrame(trade_rows)
    rejections = pd.DataFrame(rejection_rows)
    selections = pd.DataFrame(selection_rows)
    metrics = calculate_daily_metrics(curve)
    sell_trades = trades[trades["side"] == "sell"] if not trades.empty else trades
    metrics.update(
        {
            "buy_count": (
                int((trades["side"] == "buy").sum()) if not trades.empty else 0
            ),
            "closed_trade_count": int(len(sell_trades)),
            "closed_trade_win_rate": (
                float((sell_trades["net_trade_return"] > 0).mean())
                if not sell_trades.empty
                else None
            ),
            "average_closed_trade_return": (
                float(sell_trades["net_trade_return"].mean())
                if not sell_trades.empty
                else None
            ),
            "limit_up_rejections": int(
                (
                    rejections.get("reason", pd.Series(dtype=str))
                    == "limit_up_cancelled"
                ).sum()
            ),
            "limit_down_retries": int(
                (
                    rejections.get("reason", pd.Series(dtype=str)) == "limit_down_retry"
                ).sum()
            ),
            "missing_open_retries": int(
                (
                    rejections.get("reason", pd.Series(dtype=str))
                    == "missing_open_retry"
                ).sum()
            ),
        }
    )
    return {
        "metrics": metrics,
        "equity_curve": curve,
        "trades": trades,
        "rejections": rejections,
        "selections": selections,
        "open_positions": [asdict(position) for position in positions.values()],
    }


def factor_quintile_diagnostic(
    prepared_market: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    memberships: dict[pd.Timestamp, set[str]],
    *,
    forward_days: int = 10,
) -> list[dict[str, Any]]:
    """Measure monotonic forward returns by Alpha144 quintile on refresh dates."""
    market = prepared_market.set_index(["trade_date", "symbol"]).sort_index()
    calendar = sorted(
        pd.Timestamp(value) for value in prepared_market["trade_date"].unique()
    )
    date_position = {date: index for index, date in enumerate(calendar)}
    rows: list[pd.DataFrame] = []
    for signal_date in signal_dates:
        index = date_position.get(signal_date)
        if index is None or index + forward_days >= len(calendar):
            continue
        future_date = calendar[index + forward_days]
        current = (
            market.loc[signal_date].copy()
            if signal_date in market.index.levels[0]
            else pd.DataFrame()
        )
        future = (
            market.loc[future_date].copy()
            if future_date in market.index.levels[0]
            else pd.DataFrame()
        )
        if current.empty or future.empty:
            continue
        members = memberships[signal_date]
        current = current.loc[current.index.intersection(sorted(members))]
        current = current.dropna(subset=["alpha144", "adj_close"])
        current["quintile"] = np.ceil(current["alpha144"].rank(pct=True) * 5).clip(1, 5)
        joined = current[["quintile", "adj_close"]].join(
            future[["adj_close"]].rename(columns={"adj_close": "future_adj_close"}),
            how="left",
        )
        if len(joined) < 100:
            continue
        joined["forward_return"] = (
            joined["future_adj_close"] / joined["adj_close"] - 1.0
        )
        rows.append(joined[["quintile", "forward_return"]])
    if not rows:
        return []
    combined = pd.concat(rows, ignore_index=True)
    return [
        {
            "quintile": f"Q{int(quintile)}",
            "average_forward_10d_return": float(group["forward_return"].mean()),
            "observations": int(group["forward_return"].notna().sum()),
            "missing_forward_price": int(group["forward_return"].isna().sum()),
            "forward_price_coverage": float(group["forward_return"].notna().mean()),
        }
        for quintile, group in combined.groupby("quintile", sort=True)
    ]


def write_report(
    result: dict[str, Any],
    config: BacktestConfig,
    data_audit: dict[str, Any],
    membership_audit: dict[str, Any],
    factor_quintiles: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    token = f"{config.start_date.replace('-', '')}_{config.end_date.replace('-', '')}"
    stem = f"alpha144_liquidity_breakout_{token}"
    curve_path = output_dir / f"{stem}_equity.csv"
    trades_path = output_dir / f"{stem}_trades.csv"
    selections_path = output_dir / f"{stem}_selections.csv"
    rejections_path = output_dir / f"{stem}_rejections.csv"
    report_path = output_dir / f"{stem}.json"
    result["equity_curve"].to_csv(curve_path, index=False)
    result["trades"].to_csv(trades_path, index=False)
    result["selections"].to_csv(selections_path, index=False)
    result["rejections"].to_csv(rejections_path, index=False)
    annual = []
    curve = result["equity_curve"].copy()
    curve["year"] = pd.to_datetime(curve["trade_date"]).dt.year
    for year, group in curve.groupby("year", sort=True):
        annual.append(
            {
                "year": int(year),
                "strategy_return": float((1.0 + group["strategy_return"]).prod() - 1.0),
                "benchmark_return": float(
                    (1.0 + group["benchmark_return"]).prod() - 1.0
                ),
                "average_holdings": float(group["holding_count"].mean()),
                "cash_day_ratio": float((group["holding_count"] == 0).mean()),
            }
        )
    report = {
        "meta": {
            "title": "Alpha144流动性突破策略回测",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "signal_timing": "T日收盘后",
            "execution_timing": "T+1交易日开盘",
            "universe": "中证500点时历史成分",
        },
        "parameters": asdict(config),
        "data_audit": data_audit,
        "membership_audit": membership_audit,
        "metrics": result["metrics"],
        "annual": annual,
        "factor_quintiles": factor_quintiles,
        "open_positions": [
            {
                **row,
                "entry_date": pd.Timestamp(row["entry_date"]).date().isoformat(),
            }
            for row in result["open_positions"]
        ],
        "files": {
            "equity_curve": str(curve_path),
            "trades": str(trades_path),
            "selections": str(selections_path),
            "rejections": str(rejections_path),
        },
        "limitations": [
            "中证500成分按每个10交易日信号日通过 Baostock 点时查询，"
            "且要求恰好500只、生效日不晚于信号日。",
            "Alpha144前15%候选池每10个交易日刷新，"
            "候选池内股票在其余交易日继续每日检查5日新高。",
            "因子和突破信号仅使用信号日及以前数据，" "订单统一在下一交易日开盘处理。",
            "收益使用复权价，交易流水同时保留原始开盘价；" "分红送转不会制造虚假收益。",
            "开盘达到主板10%或创业板/科创板20%阈值时"
            "保守视为无法成交；历史ST标签缺失。",
            "未设置个股止损；大盘20日收益低于-3%时下一交易日尝试清仓。",
            "停牌期间持仓按最后可用复权收盘价估值，"
            "卖单延迟到恢复交易后执行；缺少退市现金结算状态。",
            "未施加成交额参与率上限，" "低流动性股票的实盘容量可能显著低于回测。",
            "中证500大盘过滤及基准收益来自 Baostock 指数日线缓存，"
            "不使用已发现日期错位的本地Qlib指数序列。",
            "历史回测不代表未来收益，不构成投资建议。",
        ],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "report": str(report_path),
        "equity_curve": str(curve_path),
        "trades": str(trades_path),
        "selections": str(selections_path),
        "rejections": str(rejections_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the Alpha144 liquidity-breakout strategy"
    )
    parser.add_argument("--start-date", default="2021-01-04")
    parser.add_argument("--end-date", default="2026-05-15")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument(
        "--membership-cache", type=Path, default=DEFAULT_MEMBERSHIP_CACHE
    )
    parser.add_argument("--benchmark-cache", type=Path, default=DEFAULT_BENCHMARK_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--buy-cost-bps", type=float, default=8.0)
    parser.add_argument("--sell-cost-bps", type=float, default=13.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    config = BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        buy_cost_bps=args.buy_cost_bps,
        sell_cost_bps=args.sell_cost_bps,
    )
    config.validate()
    market, data_audit = load_market_snapshots(
        args.snapshot_dir,
        config.start_date,
        config.end_date,
    )
    prepared = prepare_alpha144_features(
        market,
        factor_window=config.factor_window,
        breakout_window=config.breakout_window,
    )
    benchmark, benchmark_audit = load_csi500_benchmark(
        args.benchmark_cache,
        config.start_date,
        config.end_date,
    )
    data_audit.update(benchmark_audit)
    calendar = [
        pd.Timestamp(value)
        for value in benchmark.loc[
            (benchmark["trade_date"] >= pd.Timestamp(config.start_date))
            & (benchmark["trade_date"] <= pd.Timestamp(config.end_date)),
            "trade_date",
        ].sort_values()
    ]
    signal_dates = calendar[:: config.refresh_days]
    memberships, membership_audit = load_pit_csi500_memberships(
        signal_dates,
        args.membership_cache,
    )
    result = run_backtest(prepared, benchmark, memberships, config)
    quintiles = factor_quintile_diagnostic(
        prepared,
        signal_dates,
        memberships,
        forward_days=config.refresh_days,
    )
    files = write_report(
        result,
        config,
        data_audit,
        membership_audit,
        quintiles,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "metrics": result["metrics"],
                "factor_quintiles": quintiles,
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
