#!/usr/bin/env python3
"""Incrementally materialize model_qlib's daily feature rows.

The OSS daily updater maintains ``stock_daily_latest`` and Qlib OHLCV, while
the production LightGBM model consumes the yearly feature parquet.  This job
bridges that boundary without look-ahead:

1. recompute deterministic daily/rolling features from data available at T;
2. carry the last known per-symbol value for features whose upstream L2/CSMAR
   source is not part of the OSS daily feed;
3. use the model-declared fill value only as a final fallback;
4. reject low-coverage days before atomically replacing the yearly parquet.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from psycopg2 import connect
from psycopg2 import sql


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "production" / "model_qlib"
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "db" / "feature_snapshots"
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "db" / "feature_audit_v2"
KEY_COLUMNS = ("trade_date", "symbol")
LOOKBACK_CALENDAR_DAYS = 260


DIRECT_COLUMN_MAP = {
    "bp": "style_bp",
    "ep_ttm": "style_ep_ttm",
    "micro_effective_spread": "micro_effective_spread",
    "micro_imbalance_volume": "micro_imbalance_volume",
    "micro_jump_flag": "micro_jump_flag",
}


@dataclass(frozen=True)
class ModelContract:
    feature_columns: list[str]
    fill_values: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build incremental feature rows for model_qlib"
    )
    parser.add_argument("--start-date", help="First candidate date YYYY-MM-DD")
    parser.add_argument("--end-date", help="Last candidate date YYYY-MM-DD")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--apply", action="store_true", help="Atomically update parquet")
    parser.add_argument("--backup", action="store_true", help="Keep a pre-update backup")
    parser.add_argument("--min-symbols", type=int, default=1000)
    parser.add_argument("--min-row-ratio", type=float, default=0.85)
    parser.add_argument("--min-source-coverage", type=float, default=0.75)
    parser.add_argument("--min-recomputed-coverage", type=float, default=0.35)
    return parser.parse_args()


def load_contract(model_dir: Path) -> ModelContract:
    metadata_path = model_dir / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = [str(value) for value in payload.get("feature_columns") or []]
    if not features:
        raise RuntimeError(f"feature_columns missing: {metadata_path}")
    raw_fill = payload.get("fill_values") or {}
    fill_values = {
        feature: float(raw_fill.get(feature, 0.0) or 0.0) for feature in features
    }
    return ModelContract(feature_columns=features, fill_values=fill_values)


def database_connection():
    return connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "quantmind"),
        user=os.getenv("DB_USER", "quantmind"),
        password=os.getenv("DB_PASSWORD", ""),
        connect_timeout=15,
    )


def _available_stock_columns(connection) -> set[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'stock_daily_latest'
            """
        )
        return {str(row[0]) for row in cursor.fetchall()}


def latest_stock_date(connection, *, not_after: date) -> pd.Timestamp:
    """Return the latest source date that is safe for the default daily run."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT MAX(trade_date)
            FROM stock_daily_latest
            WHERE trade_date <= %s
            """,
            (not_after,),
        )
        value = cursor.fetchone()[0]
    if value is None:
        raise RuntimeError(
            f"stock_daily_latest has no rows on or before {not_after.isoformat()}"
        )
    return pd.Timestamp(value).normalize()


def load_stock_rows(
    connection,
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    candidates = [
        "trade_date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "raw_volume",
        "raw_amount",
        "adj_factor",
        "turnover_rate",
        "pb",
        "pe_ttm",
        "bp",
        "ep_ttm",
        "float_mv",
        "flow_net_amount",
        "lrg_trd_tolbuynum",
        "lrg_trd_tolsellnum",
        "micro_effective_spread",
        "micro_imbalance_volume",
        "micro_jump_flag",
        "ind_code_l1",
        "ind_code_l2",
    ]
    available = _available_stock_columns(connection)
    columns = [column for column in candidates if column in available]
    if not set(KEY_COLUMNS).issubset(columns):
        raise RuntimeError("stock_daily_latest is missing trade_date/symbol")
    query = sql.SQL(
        "SELECT {} FROM stock_daily_latest "
        "WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date, symbol"
    ).format(sql.SQL(", ").join(sql.Identifier(column) for column in columns))
    with connection.cursor() as cursor:
        cursor.execute(query, (start_date, end_date))
        rows = cursor.fetchall()
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame = frame[frame["symbol"].str.match(r"^(SH|SZ|BJ)\d{6}$", na=False)]
    return frame.drop_duplicates(list(KEY_COLUMNS), keep="last")


def parquet_date_range(path: Path) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    dates = pd.read_parquet(path, columns=["trade_date"])["trade_date"]
    parsed = pd.to_datetime(dates).dt.normalize()
    return parsed.min(), parsed.max(), int(len(parsed))


def _existing_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def load_history(
    yearly_path: Path,
    *,
    start_date: pd.Timestamp,
    feature_columns: list[str],
) -> pd.DataFrame:
    schema_columns = set(_existing_columns(yearly_path))
    wanted = [
        "trade_date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "factor",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "raw_volume",
        "raw_amount",
        "liq_amount",
        "ind_code_l1",
        "ind_code_l2",
        *feature_columns,
    ]
    columns = list(dict.fromkeys(column for column in wanted if column in schema_columns))
    lower = start_date - pd.Timedelta(days=LOOKBACK_CALENDAR_DAYS)
    frame = pd.read_parquet(yearly_path, columns=columns)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame = frame[(frame["trade_date"] >= lower) & (frame["trade_date"] < start_date)]
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    return frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def normalize_market_panel(history: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    hist = history.copy()
    new = fresh.copy()
    for target, preferred in (
        ("raw_open", ("raw_open", "open")),
        ("raw_high", ("raw_high", "high")),
        ("raw_low", ("raw_low", "low")),
        ("raw_close", ("raw_close", "close")),
        ("raw_volume", ("raw_volume", "volume")),
        ("raw_amount", ("raw_amount", "amount", "liq_amount")),
        ("factor", ("adj_factor", "factor")),
    ):
        for frame in (hist, new):
            value = pd.Series(np.nan, index=frame.index, dtype="float64")
            for source in preferred:
                if source in frame.columns:
                    value = value.combine_first(_numeric(frame, source))
            frame[target] = value
    hist["factor"] = _numeric(hist, "factor", 1.0).fillna(1.0)
    new["factor"] = _numeric(new, "factor", 1.0).fillna(1.0)
    new["turnover_ratio"] = _numeric(new, "turnover_rate") / 100.0
    hist["turnover_ratio"] = _numeric(hist, "liq_turnover_tl")
    combined = pd.concat([hist, new], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(list(KEY_COLUMNS), keep="last")
    combined = combined.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    for column in ("ind_code_l1", "ind_code_l2"):
        if column in combined.columns:
            combined[column] = combined.groupby("symbol", observed=True)[column].ffill()
    return combined


def _group_rolling(frame: pd.DataFrame, column: str, window: int, operation: str) -> pd.Series:
    grouped = frame.groupby("symbol", observed=True)[column]
    if operation == "mean":
        return grouped.transform(lambda values: values.rolling(window, min_periods=window).mean())
    if operation == "std":
        return grouped.transform(lambda values: values.rolling(window, min_periods=window).std())
    if operation == "max":
        return grouped.transform(lambda values: values.rolling(window, min_periods=window).max())
    if operation == "sum":
        return grouped.transform(lambda values: values.rolling(window, min_periods=window).sum())
    raise ValueError(operation)


def _cross_section_zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if not math.isfinite(float(std or 0.0)) or float(std or 0.0) == 0:
        return pd.Series(0.0, index=values.index)
    return (numeric - numeric.mean()) / std


def compute_reliable_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy().sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    grouped = frame.groupby("symbol", observed=True)
    raw_close = _numeric(frame, "raw_close")
    factor = _numeric(frame, "factor", 1.0).fillna(1.0)
    adj_high = _numeric(frame, "raw_high") * factor
    adj_low = _numeric(frame, "raw_low") * factor
    adj_close = raw_close * factor
    volume = _numeric(frame, "raw_volume")
    amount = _numeric(frame, "raw_amount")
    frame["_ret1"] = grouped["raw_close"].pct_change(fill_method=None)

    for window in (1, 5, 20, 60, 120):
        frame[f"mom_ret_{window}d"] = raw_close / grouped["raw_close"].shift(window) - 1.0
    for window in (5, 20, 60):
        moving_average = adj_close.groupby(frame["symbol"], observed=True).transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
        frame[f"mom_ma_gap_{window}"] = adj_close / moving_average - 1.0

    ema12 = adj_close.groupby(frame["symbol"], observed=True).transform(
        lambda values: values.ewm(span=12, adjust=False, min_periods=12).mean()
    )
    ema26 = adj_close.groupby(frame["symbol"], observed=True).transform(
        lambda values: values.ewm(span=26, adjust=False, min_periods=26).mean()
    )
    dif = ema12 - ema26
    dea = dif.groupby(frame["symbol"], observed=True).transform(
        lambda values: values.ewm(span=9, adjust=False, min_periods=9).mean()
    )
    frame["mom_macd_hist"] = 2.0 * (dif - dea)

    def rsi(values: pd.Series) -> pd.Series:
        gains = values.clip(lower=0)
        losses = -values.clip(upper=0)
        avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss.replace(0, np.nan))

    frame["mom_rsi_14"] = frame.groupby("symbol", observed=True)["_ret1"].transform(rsi)
    rolling_high = adj_high.groupby(frame["symbol"], observed=True).transform(
        lambda values: values.rolling(20, min_periods=20).max()
    )
    frame["mom_breakout_20d"] = adj_close / rolling_high - 1.0

    for window in (10, 20, 60):
        frame[f"vol_std_{window}"] = _group_rolling(frame, "_ret1", window, "std")
    previous_close = adj_close.groupby(frame["symbol"], observed=True).shift(1)
    true_range = pd.concat(
        [(adj_high - adj_low), (adj_high - previous_close).abs(), (adj_low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    frame["_true_range"] = true_range
    frame["vol_atr_14"] = _group_rolling(frame, "_true_range", 14, "mean")
    log_hl_sq = np.log((adj_high / adj_low).where((adj_high > 0) & (adj_low > 0))) ** 2
    frame["_log_hl_sq"] = log_hl_sq
    frame["vol_parkinson_20"] = np.sqrt(
        _group_rolling(frame, "_log_hl_sq", 20, "mean") / (4.0 * math.log(2.0))
    )
    frame["_downside_sq"] = frame["_ret1"].clip(upper=0) ** 2
    frame["vol_downside_20"] = np.sqrt(_group_rolling(frame, "_downside_sq", 20, "mean"))
    frame["_ret_sq"] = frame["_ret1"] ** 2
    frame["vol_realized_rv"] = np.sqrt(_group_rolling(frame, "_ret_sq", 20, "sum"))

    frame["liq_turnover_os"] = _numeric(frame, "turnover_ratio")
    frame["liq_turnover_tl"] = _numeric(frame, "turnover_ratio")
    frame["liq_volume"] = volume
    frame["liq_amount"] = amount
    for window in (5, 20):
        rolling_volume = volume.groupby(frame["symbol"], observed=True).transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
        frame[f"liq_volume_ratio_{window}"] = volume / rolling_volume
    frame["liq_amount_ma_20"] = amount.groupby(frame["symbol"], observed=True).transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    signed_volume = np.sign(frame["_ret1"].fillna(0.0)) * volume.fillna(0.0)
    frame["_signed_volume"] = signed_volume
    frame["liq_obv_20"] = _group_rolling(frame, "_signed_volume", 20, "sum")
    amihud = frame["_ret1"].abs() / amount.replace(0, np.nan) * 1_000_000.0
    frame["_amihud"] = amihud
    frame["liq_amihud_20"] = _group_rolling(frame, "_amihud", 20, "mean")
    frame["liq_amihud_60"] = _group_rolling(frame, "_amihud", 60, "mean")
    typical_price = (adj_high + adj_low + adj_close) / 3.0
    money_flow = typical_price * volume
    direction = np.sign(typical_price.groupby(frame["symbol"], observed=True).diff())
    frame["_positive_flow"] = money_flow.where(direction >= 0, 0.0)
    frame["_negative_flow"] = money_flow.where(direction < 0, 0.0)
    pos14 = _group_rolling(frame, "_positive_flow", 14, "sum")
    neg14 = _group_rolling(frame, "_negative_flow", 14, "sum")
    frame["liq_mfi_14"] = 100.0 - 100.0 / (1.0 + pos14 / neg14.replace(0, np.nan))

    flow = _numeric(frame, "flow_net_amount")
    frame["flow_net_amount_ratio"] = flow / amount.replace(0, np.nan)
    large_buy = _numeric(frame, "lrg_trd_tolbuynum")
    large_sell = _numeric(frame, "lrg_trd_tolsellnum")
    large_total = (large_buy.abs() + large_sell.abs()).replace(0, np.nan)
    frame["flow_large_net_ratio"] = (large_buy - large_sell) / large_total
    frame["flow_net_order_ratio"] = (large_buy - large_sell) / large_total
    frame["micro_pressure_score"] = _numeric(frame, "micro_imbalance_volume")

    for source, target in DIRECT_COLUMN_MAP.items():
        frame[target] = _numeric(frame, source)
    frame["style_ln_mv_float"] = np.log(_numeric(frame, "float_mv").where(_numeric(frame, "float_mv") > 0))
    frame["style_valuation_composite"] = frame.groupby("trade_date", observed=True)[
        ["style_bp", "style_ep_ttm"]
    ].transform(_cross_section_zscore).mean(axis=1)
    frame["style_size_percentile"] = frame.groupby("trade_date", observed=True)[
        "style_ln_mv_float"
    ].rank(pct=True)
    frame["style_value_percentile"] = frame.groupby("trade_date", observed=True)[
        "style_valuation_composite"
    ].rank(pct=True)

    if "ind_code_l1" in frame.columns:
        valid_industry = frame["ind_code_l1"].notna()
        industry_keys = [frame["trade_date"], frame["ind_code_l1"]]
        industry_volume = volume.groupby(industry_keys, observed=True).transform("mean")
        industry_volatility = frame["vol_std_20"].groupby(industry_keys, observed=True).transform("mean")
        industry_momentum = frame["mom_ret_20d"].groupby(industry_keys, observed=True).transform("mean")
        frame["ind_relative_volume_20"] = (volume / industry_volume).where(valid_industry)
        frame["ind_relative_volatility_20"] = (
            frame["vol_std_20"] / industry_volatility
        ).where(valid_industry)
        frame["ind_strength_20"] = (frame["mom_ret_20d"] - industry_momentum).where(valid_industry)
        industry_daily = (
            frame.loc[valid_industry]
            .groupby(["trade_date", "ind_code_l1"], observed=True)
            .agg(
                ind_momentum=("mom_ret_20d", "mean"),
                ind_value=("style_valuation_composite", "mean"),
            )
            .reset_index()
        )
        industry_daily["ind_momentum_rank_20"] = industry_daily.groupby("trade_date")[
            "ind_momentum"
        ].rank(pct=True)
        industry_daily["ind_value_rank"] = industry_daily.groupby("trade_date")[
            "ind_value"
        ].rank(pct=True)
        frame = frame.merge(
            industry_daily[
                ["trade_date", "ind_code_l1", "ind_momentum_rank_20", "ind_value_rank"]
            ],
            on=["trade_date", "ind_code_l1"],
            how="left",
        )
    return frame.replace([np.inf, -np.inf], np.nan)


def _latest_carry_state(history: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in history.columns]
    latest = history.sort_values(["symbol", "trade_date"]).groupby("symbol", observed=True).tail(1)
    state = latest.set_index("symbol").reindex(columns=available)
    return state.reindex(columns=columns)


def materialize_days(
    *,
    computed_panel: pd.DataFrame,
    history: pd.DataFrame,
    fresh: pd.DataFrame,
    contract: ModelContract,
    baseline_rows: int,
    min_symbols: int,
    min_row_ratio: float,
    min_source_coverage: float,
    min_recomputed_coverage: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    feature_columns = contract.feature_columns
    state = _latest_carry_state(history, feature_columns)
    history_dimensions = [column for column in ("ind_code_l1", "ind_code_l2") if column in history.columns]
    dimension_state = _latest_carry_state(history, history_dimensions) if history_dimensions else pd.DataFrame()
    target_dates = sorted(pd.to_datetime(fresh["trade_date"].dropna().unique()))
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    required_rows = max(int(min_symbols), int(math.ceil(baseline_rows * min_row_ratio)))

    for trade_date in target_dates:
        day = computed_panel[computed_panel["trade_date"] == trade_date].copy()
        day = day.drop_duplicates("symbol", keep="last").sort_values("symbol")
        if not dimension_state.empty:
            for column in history_dimensions:
                carried = day["symbol"].map(dimension_state[column])
                if column not in day.columns:
                    day[column] = carried
                else:
                    day[column] = day[column].combine_first(carried)

        total_cells = max(len(day) * len(feature_columns), 1)
        recomputed_cells = 0
        carried_cells = 0
        filled_cells = 0
        for feature in feature_columns:
            if feature not in day.columns:
                day[feature] = np.nan
            values = pd.to_numeric(day[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            recomputed = values.notna()
            recomputed_cells += int(recomputed.sum())
            carried_values = pd.to_numeric(day["symbol"].map(state[feature]), errors="coerce")
            use_carry = values.isna() & carried_values.notna()
            values = values.where(~use_carry, carried_values)
            carried_cells += int(use_carry.sum())
            use_fill = values.isna()
            values = values.fillna(contract.fill_values[feature])
            filled_cells += int(use_fill.sum())
            day[feature] = values.astype("float64")

        source_coverage = (recomputed_cells + carried_cells) / total_cells
        recomputed_coverage = recomputed_cells / total_cells
        passed = (
            len(day) >= required_rows
            and source_coverage >= min_source_coverage
            and recomputed_coverage >= min_recomputed_coverage
        )
        audit = {
            "trade_date": pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
            "rows": int(len(day)),
            "required_rows": int(required_rows),
            "feature_count": len(feature_columns),
            "recomputed_cells": recomputed_cells,
            "carried_cells": carried_cells,
            "model_fill_cells": filled_cells,
            "recomputed_coverage": round(recomputed_coverage, 6),
            "source_coverage": round(source_coverage, 6),
            "model_fill_ratio": round(filled_cells / total_cells, 6),
            "passed": passed,
        }
        audits.append(audit)
        if not passed:
            raise RuntimeError(f"feature quality gate failed: {json.dumps(audit, ensure_ascii=False)}")

        day["trade_date"] = pd.Timestamp(trade_date)
        outputs.append(day)
        state.update(day.set_index("symbol")[feature_columns])
        new_symbols = day.set_index("symbol")[feature_columns].index.difference(state.index)
        if len(new_symbols):
            state = pd.concat([state, day.set_index("symbol").loc[new_symbols, feature_columns]])
        if history_dimensions:
            dimension_day = day.set_index("symbol")[history_dimensions]
            dimension_state.update(dimension_day)
            missing_dimensions = dimension_day.index.difference(dimension_state.index)
            if len(missing_dimensions):
                dimension_state = pd.concat([dimension_state, dimension_day.loc[missing_dimensions]])

    if not outputs:
        return pd.DataFrame(), audits
    output = pd.concat(outputs, ignore_index=True, sort=False)
    return output.sort_values(["trade_date", "symbol"]).reset_index(drop=True), audits


def atomic_replace_yearly_parquet(
    *,
    yearly_path: Path,
    new_rows: pd.DataFrame,
    backup: bool,
) -> None:
    parquet_file = pq.ParquetFile(yearly_path)
    schema = parquet_file.schema_arrow
    target_dates = set(pd.to_datetime(new_rows["trade_date"]).dt.strftime("%Y-%m-%d"))
    temp_path = yearly_path.with_suffix(yearly_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    writer = pq.ParquetWriter(temp_path, schema, compression="snappy")
    try:
        for row_group in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group)
            frame = table.to_pandas()
            dates = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
            frame = frame[~dates.isin(target_dates)]
            if not frame.empty:
                writer.write_table(
                    pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=False)
                )

        append_frame = new_rows.reindex(columns=schema.names).copy()
        append_frame["trade_date"] = pd.to_datetime(append_frame["trade_date"])
        writer.write_table(
            pa.Table.from_pandas(append_frame, schema=schema, preserve_index=False, safe=False)
        )
    finally:
        writer.close()

    if backup:
        suffix = max(target_dates).replace("-", "")
        backup_path = yearly_path.with_suffix(f".before_{suffix}.parquet")
        if not backup_path.exists():
            shutil.copy2(yearly_path, backup_path)
    os.replace(temp_path, yearly_path)


def update_metadata_sidecar(
    yearly_path: Path,
    *,
    audits: list[dict[str, Any]],
    rows_added: int,
) -> None:
    metadata_path = yearly_path.with_suffix(".metadata.json")
    payload: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    payload.update(
        {
            "date_max": audits[-1]["trade_date"],
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "incremental_update": {
                "generator": Path(__file__).name,
                "date_min": audits[0]["trade_date"],
                "date_max": audits[-1]["trade_date"],
                "rows_added": int(rows_added),
                "audits": audits,
            },
        }
    )
    temp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, metadata_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    contract = load_contract(args.model_dir)
    source_last_date: pd.Timestamp | None = None
    if args.end_date:
        requested_end = pd.Timestamp(args.end_date).normalize()
    else:
        connection = database_connection()
        try:
            source_last_date = latest_stock_date(connection, not_after=date.today())
        finally:
            connection.close()
        requested_end = source_last_date

    yearly_path = args.snapshot_dir / f"model_features_{requested_end.year}.parquet"
    if not yearly_path.exists():
        raise FileNotFoundError(yearly_path)
    existing_min, existing_max, existing_rows = parquet_date_range(yearly_path)
    requested_start = (
        pd.Timestamp(args.start_date).normalize()
        if args.start_date
        else existing_max + pd.Timedelta(days=1)
    )
    if requested_start > requested_end:
        result = {
            "success": True,
            "skipped": True,
            "reason": "feature_parquet_already_current",
            "parquet_last_date": existing_max.strftime("%Y-%m-%d"),
        }
        if source_last_date is not None:
            result["source_last_date"] = source_last_date.strftime("%Y-%m-%d")
        return result

    connection = database_connection()
    try:
        fresh = load_stock_rows(
            connection,
            start_date=requested_start.date(),
            end_date=requested_end.date(),
        )
    finally:
        connection.close()
    if fresh.empty:
        raise RuntimeError(
            f"stock_daily_latest has no rows in [{requested_start.date()}, {requested_end.date()}]"
        )

    history = load_history(
        yearly_path,
        start_date=requested_start,
        feature_columns=contract.feature_columns,
    )
    if history.empty:
        raise RuntimeError("feature history is empty; cannot build rolling/carry state")
    baseline_date = history["trade_date"].max()
    baseline_rows = int((history["trade_date"] == baseline_date).sum())
    panel = normalize_market_panel(history, fresh)
    computed = compute_reliable_features(panel)
    new_rows, audits = materialize_days(
        computed_panel=computed,
        history=history,
        fresh=fresh,
        contract=contract,
        baseline_rows=baseline_rows,
        min_symbols=args.min_symbols,
        min_row_ratio=args.min_row_ratio,
        min_source_coverage=args.min_source_coverage,
        min_recomputed_coverage=args.min_recomputed_coverage,
    )
    result = {
        "success": True,
        "apply": bool(args.apply),
        "yearly_path": str(yearly_path),
        "existing_rows": existing_rows,
        "existing_date_min": existing_min.strftime("%Y-%m-%d"),
        "existing_date_max": existing_max.strftime("%Y-%m-%d"),
        "rows_materialized": int(len(new_rows)),
        "date_min": audits[0]["trade_date"],
        "date_max": audits[-1]["trade_date"],
        "audits": audits,
    }
    if not args.apply:
        return result

    atomic_replace_yearly_parquet(
        yearly_path=yearly_path,
        new_rows=new_rows,
        backup=bool(args.backup),
    )
    update_metadata_sidecar(yearly_path, audits=audits, rows_added=len(new_rows))
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.audit_dir / (
        f"model_qlib_incremental_{audits[0]['trade_date'].replace('-', '')}_"
        f"{audits[-1]['trade_date'].replace('-', '')}.json"
    )
    audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["audit_path"] = str(audit_path)
    return result


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        raise
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
