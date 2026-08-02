"""Point-in-time sector momentum and leader/core signal calculations.

This module deliberately contains no database writes and no Qlib index shifting.  It
accepts historical membership intervals and stock daily bars, then emits signals on
the signal date.  The runtime integration is the sole owner of the t+1 execution lag.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

MODEL_SPEC_V1: dict[str, Any] = {
    "version": "1",
    "board_weights": (0.25, 0.15, 0.15, 0.15, 0.15, 0.10, 0.05),
    "crowding_weights": (0.50, 0.50),
    "leader_weights": (0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05),
    "core_weights": (0.20, 0.20, 0.20, 0.15, 0.15, 0.10),
    "combined_weights": (0.60, 0.40),
    "phase_priority": ("retreat", "overheated", "diffusion", "launch", "watch"),
}

SW_VERSION = "SW2021"

DEFAULT_PARAMS: dict[str, Any] = {
    "board_universe": "sw_l1",
    "topk_sectors": 10,
    "topk_stocks": 10,
    "lookback_days": 80,
    "min_board_members": 10,
    "min_board_coverage": 0.60,
    "crowding_warning_quantile": 0.90,
    "crowding_overheat_quantile": 0.95,
    "crowding_penalty_max": 15.0,
    "launch_breadth_min": 0.30,
    "launch_breadth_max": 0.60,
    "launch_breadth_delta_min": 0.08,
    "launch_limit_count_min": 1,
    "launch_limit_count_max": 2,
    "launch_limit_ratio_min": 0.01,
    "launch_limit_ratio_max": 0.03,
    "launch_amount_ratio_min": 1.15,
    "launch_amount_ratio_max": 2.50,
    "launch_relative_return_min": 0.00,
    "launch_relative_return_max": 0.10,
    "diffusion_breadth_min": 0.60,
    "diffusion_limit_count_min": 3,
    "diffusion_limit_ratio_min": 0.03,
    "diffusion_amount_quantile_min": 0.70,
    "core_start_amount_ratio_min": 1.20,
    "overheat_breadth_min": 0.80,
    "overheat_relative_return_min": 0.12,
    "retreat_breadth_max": 0.40,
    "retreat_breadth_delta_max": -0.10,
    "retreat_relative_return_max": -0.03,
    "leader_float_mv_min": 5e9,
    "leader_float_mv_max": 3e10,
    "leader_amount_min": 3e8,
    "leader_turnover_min": 0.05,
    "leader_turnover_max": 0.25,
    "leader_rps_min": 0.85,
    "leader_amount_ratio_min": 1.20,
    "leader_amount_ratio_max": 3.00,
    "leader_return_3d_min": 0.03,
    "leader_return_3d_max": 0.25,
    "leader_upper_shadow_amount_ratio": 2.00,
    "leader_upper_shadow_ratio": 0.50,
    "core_float_mv_min": 1e10,
    "core_mv_top_quantile": 0.20,
    "core_amount_min": 1e9,
    "core_volatility_min": 0.25,
    "core_volatility_max": 0.50,
    "core_drawdown_min": -0.12,
    "core_amount_ratio_min": 1.10,
    "core_amount_ratio_max": 2.50,
    "core_return_5d_max": 0.20,
}

CORE_COLUMNS = {
    "trade_date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "adj_factor",
    "is_st",
}


class SectorCoreError(ValueError):
    """Base error for deterministic sector-core validation failures."""


class StrictDataError(SectorCoreError):
    """Raised when strict backtest inputs would introduce ambiguity or bias."""


@dataclass(frozen=True)
class SectorUniverse:
    boards: list[dict[str, Any]]
    memberships: dict[str, list[dict[str, Any]]]
    metadata: dict[str, Any]

    def __iter__(self):
        """Allow compatibility unpacking as ``boards, memberships``."""
        yield self.boards
        yield self.memberships


@dataclass(frozen=True)
class SectorSignalResult:
    signals: pd.DataFrame
    board_states: dict[pd.Timestamp, dict[str, dict[str, Any]]]
    metadata: dict[str, Any]


TushareQuery = Callable[[str, dict[str, str], tuple[str, ...]], list[dict[str, Any]]]


def normalize_stock_code(value: Any) -> str | None:
    """Return the mandatory upper-case market-prefix A-share code."""
    text = str(value or "").strip().upper()
    if not text:
        return None
    digits = "".join(char for char in text if char.isdigit())
    if len(digits) != 6:
        return None
    explicit = next(
        (market for market in ("SH", "SZ", "BJ") if text.startswith(market)),
        None,
    )
    if explicit is None:
        explicit = next(
            (market for market in ("SH", "SZ", "BJ") if text.endswith(f".{market}")),
            None,
        )
    if explicit is None:
        if digits.startswith(("6", "9")):
            explicit = "SH"
        elif digits.startswith(("4", "8")):
            explicit = "BJ"
        elif digits.startswith(("0", "2", "3")):
            explicit = "SZ"
    return f"{explicit}{digits}" if explicit else None


def _date(value: Any, *, field: str, required: bool = False) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        if required:
            raise StrictDataError(f"historical membership is missing {field}")
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "nat"}:
        if required:
            raise StrictDataError(f"historical membership is missing {field}")
        return None
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise StrictDataError(f"historical membership has invalid {field}: {value!r}")
    return pd.Timestamp(parsed).normalize()


def validate_membership_universe(
    boards: Sequence[Mapping[str, Any]],
    memberships: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    strict: bool = True,
) -> SectorUniverse:
    """Normalize and validate historical ``[in_date, out_date)`` intervals."""
    normalized_boards: list[dict[str, Any]] = []
    seen_boards: set[str] = set()
    issues: list[str] = []
    for raw in boards:
        code = str(raw.get("code") or raw.get("ts_code") or "").strip()
        if not code or code in seen_boards:
            if code:
                issues.append(f"duplicate board ignored: {code}")
            continue
        seen_boards.add(code)
        normalized_boards.append(
            {
                **dict(raw),
                "code": code,
                "name": str(raw.get("name") or raw.get("industry_name") or code),
                "source": str(raw.get("source") or "unknown"),
            }
        )
    if strict and not normalized_boards:
        raise StrictDataError("historical sector membership source is empty")

    output: dict[str, list[dict[str, Any]]] = {}
    for board in normalized_boards:
        code = board["code"]
        rows: dict[tuple[str, pd.Timestamp, pd.Timestamp | None], dict[str, Any]] = {}
        for raw in memberships.get(code, ()):
            symbol = normalize_stock_code(
                raw.get("symbol") or raw.get("con_code") or raw.get("ts_code")
            )
            if symbol is None:
                issues.append(f"{code}: membership row without a valid symbol ignored")
                continue
            in_date = _date(raw.get("in_date"), field="in_date", required=strict)
            if in_date is None:
                in_date = pd.Timestamp.min.normalize()
            out_date = _date(raw.get("out_date"), field="out_date")
            if out_date is not None and out_date <= in_date:
                raise StrictDataError(
                    f"{code}/{symbol}: out_date must be after in_date"
                )
            key = (symbol, in_date, out_date)
            if key in rows:
                issues.append(f"{code}/{symbol}: duplicate interval ignored")
                continue
            rows[key] = {
                **dict(raw),
                "symbol": symbol,
                "name": str(raw.get("name") or raw.get("con_name") or symbol),
                "in_date": in_date,
                "out_date": out_date,
            }
        by_symbol: dict[str, list[dict[str, Any]]] = {}
        for item in rows.values():
            by_symbol.setdefault(item["symbol"], []).append(item)
        for symbol, intervals in by_symbol.items():
            intervals.sort(key=lambda item: item["in_date"])
            for previous, current in zip(intervals, intervals[1:], strict=False):
                previous_end = previous["out_date"]
                if previous_end is None or current["in_date"] < previous_end:
                    raise StrictDataError(
                        f"{code}/{symbol}: overlapping historical membership intervals"
                    )
        output[code] = sorted(
            rows.values(), key=lambda item: (item["symbol"], item["in_date"])
        )
    if strict and not any(output.values()):
        raise StrictDataError("historical sector membership source has no usable rows")
    return SectorUniverse(
        boards=sorted(normalized_boards, key=lambda item: item["code"]),
        memberships=output,
        metadata={"strict": strict, "issues": issues},
    )


def point_in_time_members(
    memberships: Mapping[str, Sequence[Mapping[str, Any]]], as_of: Any
) -> dict[str, list[dict[str, Any]]]:
    """Filter already-validated intervals with exact ``[in_date, out_date)`` rules."""
    signal_date = _date(as_of, field="as_of", required=True)
    assert signal_date is not None
    result: dict[str, list[dict[str, Any]]] = {}
    for board_code, rows in memberships.items():
        active: dict[str, dict[str, Any]] = {}
        for raw in rows:
            symbol = normalize_stock_code(raw.get("symbol"))
            if symbol is None:
                continue
            in_date = _date(raw.get("in_date"), field="in_date", required=True)
            out_date = _date(raw.get("out_date"), field="out_date")
            if in_date <= signal_date and (out_date is None or signal_date < out_date):
                active[symbol] = {**dict(raw), "symbol": symbol}
        result[str(board_code)] = [active[key] for key in sorted(active)]
    return result


def _read_gzip_json(path: Path) -> dict[str, Any] | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_gzip_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, default=str)
    temporary.replace(path)


def _load_ths_history(cache_dir: Path, query: TushareQuery | None) -> SectorUniverse:
    cache_path = cache_dir / "tushare-ths-concept-membership-history.json.gz"
    cached = _read_gzip_json(cache_path)
    if cached and cached.get("historical") is True:
        universe = validate_membership_universe(
            cached.get("boards") or cached.get("industries") or [],
            cached.get("memberships") or {},
            strict=True,
        )
        return SectorUniverse(
            universe.boards,
            universe.memberships,
            {
                **universe.metadata,
                "survivorship_warning": (
                    "同花顺概念成员按点时区间过滤，但概念目录层面仍可能存在幸存者偏差"
                ),
            },
        )
    query_api = query
    if query_api is None and os.getenv("TUSHARE_TOKEN", "").strip():
        from scripts.analysis.concept_rotation_report import query_tushare

        query_api = query_tushare
    if query_api is None:
        raise StrictDataError(
            "strict ths_concept backtest requires dated Tushare ths_member history; "
            "public-page fallback has no in_date/out_date"
        )
    directory = query_api(
        "ths_index",
        {"exchange": "A", "type": "N"},
        ("ts_code", "name", "count", "exchange", "list_date", "type"),
    )
    boards = [
        {
            "code": str(row["ts_code"]),
            "name": str(row.get("name") or row["ts_code"]),
            "source": "THS_TUSHARE",
            "list_date": row.get("list_date"),
        }
        for row in directory
        if row.get("ts_code")
        and str(row.get("exchange") or "A").upper() == "A"
        and str(row.get("type") or "N").upper() == "N"
    ]
    memberships: dict[str, list[dict[str, Any]]] = {}
    for board in boards:
        rows = query_api(
            "ths_member",
            {"ts_code": board["code"]},
            ("ts_code", "con_code", "con_name", "in_date", "out_date", "is_new"),
        )
        # Keep both Y and N: former members are essential to point-in-time history.
        memberships[board["code"]] = [
            {
                "symbol": row.get("con_code"),
                "name": row.get("con_name"),
                "in_date": row.get("in_date"),
                "out_date": row.get("out_date"),
                "is_new": row.get("is_new"),
            }
            for row in rows
        ]
    universe = validate_membership_universe(boards, memberships, strict=True)
    _write_gzip_json(
        cache_path,
        {
            "historical": True,
            "generated_at": datetime.now().astimezone().isoformat(),
            "boards": universe.boards,
            "memberships": universe.memberships,
        },
    )
    return SectorUniverse(
        universe.boards,
        universe.memberships,
        {
            **universe.metadata,
            "survivorship_warning": (
                "同花顺概念成员按点时区间过滤，但概念目录层面仍可能存在幸存者偏差"
            ),
        },
    )


def _load_sw_history(cache_dir: Path, query: TushareQuery | None) -> SectorUniverse:
    """Load strict SW-L1 history without depending on the scripts package."""
    cache_path = cache_dir / "tushare-sw2021-l1-membership-history.json.gz"
    cached = _read_gzip_json(cache_path)
    stale: SectorUniverse | None = None
    if (
        cached
        and cached.get("version") == SW_VERSION
        and cached.get("historical") is True
    ):
        try:
            stale = _validate_sw_coverage(
                cached.get("industries") or cached.get("boards") or [],
                cached.get("memberships") or {},
            )
        except StrictDataError:
            LOGGER.exception("Invalid historical SW membership cache")
        if stale is not None:
            cache_days = float(os.getenv("SECTOR_MEMBERSHIP_CACHE_DAYS", "30"))
            age_days = (
                datetime.now().timestamp() - cache_path.stat().st_mtime
            ) / 86400
            if age_days <= cache_days:
                return stale

    query_api = query
    if query_api is None and os.getenv("TUSHARE_TOKEN", "").strip():
        from scripts.analysis.concept_rotation_report import query_tushare

        query_api = query_tushare
    if query_api is None:
        if stale is not None:
            LOGGER.warning(
                "SW membership cache is stale and cannot be refreshed; using stale "
                "validated history"
            )
            return stale
        raise StrictDataError(
            "strict sw_l1 backtest requires a historical membership cache or "
            "TUSHARE_TOKEN"
        )
    try:
        classifications = query_api(
            "index_classify",
            {"level": "L1", "src": SW_VERSION},
            ("index_code", "industry_name", "level", "src"),
        )
        boards = sorted(
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
        if not 28 <= len(boards) <= 40:
            raise StrictDataError(
                f"unexpected SW2021 L1 industry count: {len(boards)}"
            )

        memberships: dict[str, list[dict[str, Any]]] = {}
        for board in boards:
            intervals: dict[tuple[str, str, str], dict[str, Any]] = {}
            for is_new in ("Y", "N"):
                rows = query_api(
                    "index_member_all",
                    {"l1_code": board["code"], "is_new": is_new},
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
                for row in rows:
                    symbol = normalize_stock_code(row.get("ts_code"))
                    if symbol is None:
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
            memberships[board["code"]] = list(intervals.values())

        universe = _validate_sw_coverage(boards, memberships)
        _write_gzip_json(
            cache_path,
            {
                "version": SW_VERSION,
                "historical": True,
                "generated_at": datetime.now().astimezone().isoformat(),
                "industries": universe.boards,
                "memberships": universe.memberships,
            },
        )
        return universe
    except Exception:
        if stale is None:
            raise
        LOGGER.exception("SW membership refresh failed; using stale historical cache")
        return stale


def _validate_sw_coverage(
    boards: Sequence[Mapping[str, Any]],
    memberships: Mapping[str, Sequence[Mapping[str, Any]]],
) -> SectorUniverse:
    universe = validate_membership_universe(boards, memberships, strict=True)
    usable = sum(bool(rows) for rows in universe.memberships.values())
    if usable < len(universe.boards) * 0.90:
        raise StrictDataError(
            f"only {usable}/{len(universe.boards)} SW industries have members"
        )
    return universe


def load_sector_universe(
    board_universe: str,
    cache_dir: str | Path,
    *,
    strict: bool = True,
    query: TushareQuery | None = None,
) -> SectorUniverse:
    """Load a strict historical SW-L1 or THS-concept universe."""
    kind = str(board_universe).strip().lower()
    if kind not in {"sw_l1", "ths_concept"}:
        raise SectorCoreError(f"unsupported board_universe: {board_universe!r}")
    if not strict:
        raise SectorCoreError(
            "sector momentum backtests require strict membership mode"
        )
    root = Path(cache_dir)
    if kind == "ths_concept":
        return _load_ths_history(root, query)
    return _load_sw_history(root, query)


def validate_stock_daily(
    stock_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate core fields and derive adjusted OHLC without future information."""
    missing = sorted(CORE_COLUMNS - set(stock_daily.columns))
    if missing:
        raise StrictDataError(f"stock daily data is missing core columns: {missing}")
    frame = stock_daily.copy()
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="coerce"
    ).dt.normalize()
    frame["symbol"] = frame["symbol"].map(normalize_stock_code)
    if frame[["trade_date", "symbol"]].isna().any(axis=None):
        raise StrictDataError("stock daily data contains invalid dates or symbols")
    if frame.duplicated(["trade_date", "symbol"]).any():
        raise StrictDataError("stock daily data contains duplicate date/symbol rows")
    numeric = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "adj_factor",
        "is_st",
        "turnover_rate",
        "float_mv",
        "limit_up_today",
        "limit_down_today",
        "is_suspended",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
    }
    for column in numeric & set(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in (
        "turnover_rate",
        "float_mv",
        "limit_up_today",
        "limit_down_today",
    ):
        if column not in frame:
            frame[column] = np.nan

    audit: dict[str, Any] = {"adjusted_prices": "derived_or_verified", "issues": []}
    liquidity_missing = frame["volume"].isna() | frame["amount"].isna()
    if liquidity_missing.any():
        flat_price = (
            frame["open"].eq(frame["high"])
            & frame["high"].eq(frame["low"])
            & frame["low"].eq(frame["close"])
        )
        suspended_missing = (
            frame["volume"].isna() & frame["amount"].isna() & flat_price
        )
        if not suspended_missing.equals(liquidity_missing):
            raise StrictDataError(
                "volume/amount may be missing only together on flat-price "
                "suspension rows"
            )
        frame.loc[suspended_missing, ["volume", "amount"]] = 0.0
        audit["suspension_liquidity_filled"] = int(suspended_missing.sum())

    missing_is_st = frame["is_st"].isna()
    if missing_is_st.any():
        frame.loc[missing_is_st, "is_st"] = 1.0
        audit["missing_is_st_excluded"] = int(missing_is_st.sum())

    if "is_suspended" not in frame:
        frame["is_suspended"] = frame["volume"] <= 0
    core_numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "adj_factor",
        "is_st",
    ]
    values = frame[core_numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise StrictDataError(
            "stock daily core columns contain missing/non-finite values"
        )
    if (frame[["open", "high", "low", "close", "adj_factor"]] <= 0).any(axis=None):
        raise StrictDataError("prices and adj_factor must be finite positive numbers")
    if (frame["high"] < frame["low"]).any() or (frame[["volume", "amount"]] < 0).any(
        axis=None
    ):
        raise StrictDataError("stock daily OHLC/volume/amount values are invalid")
    derived = {
        f"adj_{column}": frame[column] * frame["adj_factor"]
        for column in ("open", "high", "low", "close")
    }
    for column, expected in derived.items():
        if column in frame:
            actual = frame[column]
            mismatch = actual.isna() | ~np.isclose(
                actual, expected, rtol=1e-6, atol=1e-8
            )
            if mismatch.any():
                raise StrictDataError(
                    f"explicit {column} is inconsistent with raw price * adj_factor"
                )
        else:
            frame[column] = expected
    if "turnover_rate" in frame and frame["turnover_rate"].notna().any():
        finite = frame.loc[frame["turnover_rate"].notna(), "turnover_rate"]
        if (finite < 0).any() or not np.isfinite(finite).all():
            raise StrictDataError("turnover_rate has invalid units or values")
        if finite.quantile(0.95) > 1.0:
            if finite.max() > 100.0:
                raise StrictDataError("turnover_rate unit audit failed")
            frame["turnover_rate"] /= 100.0
            audit["turnover_unit_conversion"] = "percent_to_decimal"
    for column in ("amount", "float_mv"):
        if column in frame and frame[column].notna().any():
            finite = frame.loc[frame[column].notna(), column]
            if (finite < 0).any() or not np.isfinite(finite).all():
                raise StrictDataError(f"{column} unit audit failed")
            positive = finite[finite > 0]
            minimum_typical = 1e5 if column == "amount" else 1e8
            if not positive.empty and positive.quantile(0.50) < minimum_typical:
                raise StrictDataError(
                    f"{column} unit audit failed: expected values in Chinese yuan"
                )
    frame = frame.sort_values(["trade_date", "symbol"], kind="stable").reset_index(
        drop=True
    )
    return frame, audit


def capped_cap_weights(float_mv: Sequence[float], cap: float = 0.10) -> np.ndarray:
    """Solve ``sum(min(cap, lambda * mv_i)) == 1`` by water filling."""
    values = np.asarray(float_mv, dtype=float)
    if values.ndim != 1 or not len(values):
        raise SectorCoreError("float_mv must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise SectorCoreError("float_mv must contain finite positive values")
    if not 0 < cap <= 1 or len(values) * cap < 1 - 1e-12:
        raise SectorCoreError("the requested cap is infeasible for this member count")
    low, high = 0.0, 1.0 / values.min()
    for _ in range(200):
        middle = (low + high) / 2
        if np.minimum(cap, middle * values).sum() < 1:
            low = middle
        else:
            high = middle
    weights = np.minimum(cap, high * values)
    # Numerical residue is assigned only to members still below the cap.
    residue = 1.0 - weights.sum()
    uncapped = np.flatnonzero(weights < cap - 1e-12)
    if abs(residue) > 1e-12 and len(uncapped):
        weights[uncapped] += residue * values[uncapped] / values[uncapped].sum()
    return weights


def chain_board_index(returns: Sequence[float], *, base: float = 100.0) -> np.ndarray:
    """Chain daily board returns without re-basing the first valid observation."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise SectorCoreError("returns must be a one-dimensional sequence")
    if not np.isfinite(base) or base <= 0:
        raise SectorCoreError("base must be finite and positive")
    output = np.full(len(values), np.nan, dtype=float)
    level: float | None = None
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if level is None:
            level = float(base)
        else:
            level *= 1 + value
        output[index] = level
    return output


def close_position(close: float, low: float, high: float) -> float:
    values = np.asarray([close, low, high], dtype=float)
    if not np.isfinite(values).all():
        return math.nan
    if high == low:
        return 0.5
    return float(np.clip((close - low) / (high - low), 0.0, 1.0))


def upper_shadow_ratio(open_: float, high: float, low: float, close: float) -> float:
    values = np.asarray([open_, high, low, close], dtype=float)
    if not np.isfinite(values).all():
        return math.nan
    if high == low:
        return 0.0
    return float(np.clip((high - max(open_, close)) / (high - low), 0.0, 1.0))


def healthy_volume(amount_ratio: float) -> float:
    if not np.isfinite(amount_ratio) or amount_ratio <= 0:
        return math.nan
    return float(np.clip(1 - abs(math.log(amount_ratio / 1.5)) / math.log(2), 0, 1))


def anti_fall(stock_returns: Sequence[float], board_returns: Sequence[float]) -> float:
    stocks = np.asarray(stock_returns, dtype=float)
    boards = np.asarray(board_returns, dtype=float)
    if stocks.shape != boards.shape:
        raise SectorCoreError("stock_returns and board_returns must have equal shape")
    mask = np.isfinite(stocks) & np.isfinite(boards) & (boards < 0)
    return float(np.mean(stocks[mask] - boards[mask])) if mask.any() else 0.0


def size_fit(float_mv: float, min_mv: float, max_mv: float) -> float:
    if (
        not all(np.isfinite([float_mv, min_mv, max_mv]))
        or not 0 < min_mv < max_mv
        or float_mv <= 0
    ):
        return math.nan
    middle = math.sqrt(min_mv * max_mv)
    denominator = max(abs(math.log(min_mv / middle)), abs(math.log(max_mv / middle)))
    return float(np.clip(1 - abs(math.log(float_mv / middle)) / denominator, 0, 1))


def trend_stability(ma5: float, ma20: float, ma20_lag5: float) -> float:
    if not np.isfinite([ma5, ma20, ma20_lag5]).all() or ma20 <= 0 or ma20_lag5 <= 0:
        return math.nan
    fast = np.clip((ma5 / ma20 - 1) / 0.10, 0, 1)
    slow = np.clip((ma20 / ma20_lag5 - 1) / 0.10, 0, 1)
    return float(0.5 * fast + 0.5 * slow)


def _finite_metric(metrics: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(metrics[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def classify_board_phase(
    metrics: Mapping[str, Any], params: Mapping[str, Any] | None = None
) -> tuple[str, list[str]]:
    """Classify one board with the fixed retreat-first priority."""
    config = {**DEFAULT_PARAMS, **dict(params or {})}
    required = {
        key: _finite_metric(metrics, key)
        for key in (
            "board_close",
            "ma5",
            "ma20",
            "breadth_ma20",
            "breadth_change_3d",
            "relative_return_3d",
            "relative_return_5d",
            "limit_up_count",
            "limit_up_ratio",
            "amount_share_vs_median20",
            "amount_share_quantile",
            "top3_return_median",
            "top3_amount_ratio_median",
            "crowding_quantile",
        )
    }
    missing = sorted(key for key, value in required.items() if value is None)
    diagnostic = [f"missing phase inputs: {', '.join(missing)}"] if missing else []
    close = required["board_close"]
    ma5 = required["ma5"]
    ma20 = required["ma20"]
    breadth = required["breadth_ma20"]
    breadth_delta = required["breadth_change_3d"]
    relative3 = required["relative_return_3d"]
    relative5 = required["relative_return_5d"]
    crowding = required["crowding_quantile"]
    retreat = (
        close is not None
        and ma20 is not None
        and breadth is not None
        and close < ma20
        and breadth < config["retreat_breadth_max"]
    ) or (
        breadth_delta is not None
        and relative3 is not None
        and breadth_delta <= config["retreat_breadth_delta_max"]
        and relative3 <= config["retreat_relative_return_max"]
    )
    if retreat:
        return "retreat", diagnostic
    overheated = (
        breadth is not None
        and relative5 is not None
        and breadth >= config["overheat_breadth_min"]
        and relative5 > config["overheat_relative_return_min"]
    ) or (crowding is not None and crowding >= config["crowding_overheat_quantile"])
    if overheated:
        return "overheated", diagnostic
    limit_count = required["limit_up_count"]
    limit_ratio = required["limit_up_ratio"]
    limit_launch = (
        limit_count is not None
        and config["launch_limit_count_min"]
        <= limit_count
        <= config["launch_limit_count_max"]
    ) or (
        limit_ratio is not None
        and config["launch_limit_ratio_min"]
        <= limit_ratio
        <= config["launch_limit_ratio_max"]
    )
    launch = (
        bool(metrics.get("crossed_ma20_within_3d", False))
        and close is not None
        and ma20 is not None
        and close > ma20
        and breadth is not None
        and config["launch_breadth_min"] <= breadth <= config["launch_breadth_max"]
        and breadth_delta is not None
        and breadth_delta >= config["launch_breadth_delta_min"]
        and limit_launch
        and required["amount_share_vs_median20"] is not None
        and config["launch_amount_ratio_min"]
        <= required["amount_share_vs_median20"]
        <= config["launch_amount_ratio_max"]
        and relative5 is not None
        and config["launch_relative_return_min"]
        <= relative5
        <= config["launch_relative_return_max"]
        and crowding is not None
        and crowding < config["crowding_overheat_quantile"]
    )
    amount_quantile = required["amount_share_quantile"]
    top3_return = required["top3_return_median"]
    top3_amount = required["top3_amount_ratio_median"]
    diffusion = (
        close is not None
        and ma5 is not None
        and ma20 is not None
        and close > ma5 > ma20
        and breadth is not None
        and breadth >= config["diffusion_breadth_min"]
        and breadth_delta is not None
        and breadth_delta > 0
        and (
            (
                limit_count is not None
                and limit_count >= config["diffusion_limit_count_min"]
            )
            or (
                limit_ratio is not None
                and limit_ratio >= config["diffusion_limit_ratio_min"]
            )
        )
        and amount_quantile is not None
        and amount_quantile >= config["diffusion_amount_quantile_min"]
        and top3_return is not None
        and top3_return > 0
        and top3_amount is not None
        and top3_amount >= config["core_start_amount_ratio_min"]
        and crowding is not None
        and crowding < config["crowding_overheat_quantile"]
    )
    if diffusion:
        return "diffusion", diagnostic
    if launch:
        return "launch", diagnostic
    return "watch", diagnostic


def _rolling_compound(series: pd.Series, window: int) -> pd.Series:
    return (1 + series).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1


def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values(["symbol", "trade_date"], kind="stable").copy()
    grouped = data.groupby("symbol", sort=False)
    data["stock_return"] = grouped["adj_close"].pct_change(fill_method=None)
    data["ma5"] = grouped["adj_close"].transform(
        lambda values: values.rolling(5).mean()
    )
    data["ma20"] = grouped["adj_close"].transform(
        lambda values: values.rolling(20).mean()
    )
    data["ma20_lag5"] = grouped["ma20"].shift(5)
    data["return_3d"] = grouped["stock_return"].transform(
        lambda values: _rolling_compound(values, 3)
    )
    data["return_5d"] = grouped["stock_return"].transform(
        lambda values: _rolling_compound(values, 5)
    )
    data["return_20d"] = grouped["stock_return"].transform(
        lambda values: _rolling_compound(values, 20)
    )
    data["amount_ma5"] = grouped["amount"].transform(
        lambda values: values.rolling(5).mean()
    )
    data["amount_ratio"] = data["amount"] / data["amount_ma5"]
    data["volatility_20d"] = grouped["stock_return"].transform(
        lambda values: values.rolling(20).std() * math.sqrt(252)
    )

    def rolling_drawdown(values: pd.Series) -> pd.Series:
        return values.rolling(20).apply(
            lambda window: float((window / np.maximum.accumulate(window) - 1).min()),
            raw=True,
        )

    data["max_drawdown_20d"] = grouped["adj_close"].transform(rolling_drawdown)
    by_date = data.groupby("trade_date", sort=False)
    data["rps20"] = by_date["return_20d"].rank(pct=True, method="average")
    return data


def _board_daily_rows(
    data: pd.DataFrame,
    boards: Sequence[Mapping[str, Any]],
    memberships: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
    diagnostics: list[dict[str, Any]],
) -> pd.DataFrame:
    board_names = {
        str(row["code"]): str(row.get("name") or row["code"]) for row in boards
    }
    dates = sorted(data["trade_date"].unique())
    by_date = {
        pd.Timestamp(day): part.set_index("symbol", drop=False)
        for day, part in data.groupby("trade_date")
    }
    rows: list[dict[str, Any]] = []
    index_levels: dict[str, float] = {}
    market_returns = data.groupby("trade_date")["stock_return"].mean()
    market_amount = data.groupby("trade_date")["amount"].sum()

    def members_with_limit_flags(members: pd.DataFrame) -> pd.DataFrame:
        """Return members whose daily price-limit flags are deterministic.

        A confirmed limit-up and limit-down are mutually exclusive, so a true
        flag can safely derive the missing opposite flag as false.  A lone
        false flag cannot determine the other side and is therefore excluded.
        """
        flagged = members.copy()
        up = pd.to_numeric(flagged["limit_up_today"], errors="coerce")
        down = pd.to_numeric(flagged["limit_down_today"], errors="coerce")
        up = up.where(up.isin([0, 1]))
        down = down.where(down.isin([0, 1]))
        up = up.mask(up.isna() & down.eq(1), 0)
        down = down.mask(down.isna() & up.eq(1), 0)
        flagged["limit_up_today"] = up
        flagged["limit_down_today"] = down
        valid = up.notna() & down.notna() & ~(up.eq(1) & down.eq(1))
        return flagged.loc[valid].copy()

    for day_index, raw_day in enumerate(dates):
        day = pd.Timestamp(raw_day)
        previous_day = pd.Timestamp(dates[day_index - 1]) if day_index else None
        active = point_in_time_members(memberships, day)
        today = by_date[day]
        previous = by_date.get(previous_day) if previous_day is not None else None
        for board_code in sorted(board_names):
            symbols = [item["symbol"] for item in active.get(board_code, [])]
            member_count = len(symbols)
            available = [symbol for symbol in symbols if symbol in today.index]
            coverage = len(available) / member_count if member_count else 0.0
            reason = None
            if member_count < int(config["min_board_members"]):
                reason = "insufficient_members"
            elif coverage < float(config["min_board_coverage"]):
                reason = "insufficient_coverage"
            if reason:
                diagnostics.append(
                    {
                        "date": day,
                        "board_code": board_code,
                        "reason": reason,
                        "soft": True,
                    }
                )
                rows.append(
                    {
                        "trade_date": day,
                        "board_code": board_code,
                        "board_name": board_names[board_code],
                        "evaluable": False,
                        "reason": reason,
                        "member_count": member_count,
                        "available_count": len(available),
                        "coverage": coverage,
                    }
                )
                continue
            members_today = members_with_limit_flags(today.loc[available])
            enhanced_coverage = (
                len(members_today) / member_count if member_count else 0.0
            )
            if (
                len(members_today) < int(config["min_board_members"])
                or enhanced_coverage < float(config["min_board_coverage"])
            ):
                reason = "insufficient_enhanced_coverage"
                diagnostics.append(
                    {
                        "date": day,
                        "board_code": board_code,
                        "reason": reason,
                        "soft": True,
                    }
                )
                rows.append(
                    {
                        "trade_date": day,
                        "board_code": board_code,
                        "board_name": board_names[board_code],
                        "evaluable": False,
                        "reason": reason,
                        "member_count": member_count,
                        "available_count": len(members_today),
                        "coverage": enhanced_coverage,
                    }
                )
                continue
            coverage = enhanced_coverage
            valid_returns = members_today["stock_return"].notna()
            return_symbols = list(members_today.index[valid_returns])
            if len(return_symbols) < int(config["min_board_members"]):
                reason = "insufficient_return_coverage"
                diagnostics.append(
                    {
                        "date": day,
                        "board_code": board_code,
                        "reason": reason,
                        "soft": True,
                    }
                )
                rows.append(
                    {
                        "trade_date": day,
                        "board_code": board_code,
                        "board_name": board_names[board_code],
                        "evaluable": False,
                        "reason": reason,
                        "member_count": member_count,
                        "available_count": len(return_symbols),
                        "coverage": coverage,
                    }
                )
                continue
            weighting = "capped_cap"
            weights: np.ndarray
            if previous is None or any(
                symbol not in previous.index for symbol in return_symbols
            ):
                weights = np.repeat(1 / len(return_symbols), len(return_symbols))
                weighting = "equal_missing_previous_float_mv"
            else:
                previous_mv = (
                    previous.loc[return_symbols, "float_mv"]
                    if "float_mv" in previous
                    else pd.Series(dtype=float)
                )
                if (
                    len(previous_mv) != len(return_symbols)
                    or previous_mv.isna().any()
                    or (previous_mv <= 0).any()
                ):
                    weights = np.repeat(1 / len(return_symbols), len(return_symbols))
                    weighting = "equal_missing_previous_float_mv"
                else:
                    weights = capped_cap_weights(previous_mv.to_numpy(dtype=float))
            if weighting != "capped_cap":
                diagnostics.append(
                    {
                        "date": day,
                        "board_code": board_code,
                        "reason": weighting,
                        "soft": True,
                    }
                )
            board_return = float(
                np.dot(members_today.loc[return_symbols, "stock_return"], weights)
            )
            board_index = index_levels.get(board_code, 100.0)
            if board_code in index_levels:
                board_index *= 1 + board_return
            index_levels[board_code] = board_index
            breadth = float((members_today["adj_close"] > members_today["ma20"]).mean())
            limit_count = int(members_today["limit_up_today"].astype(bool).sum())
            limit_ratio = limit_count / len(members_today)
            top3 = members_today.sort_values(
                "float_mv", ascending=False, na_position="last"
            ).head(3)
            top3_returns = top3["stock_return"].dropna()
            top3_amount_ratios = top3["amount_ratio"].dropna()
            rows.append(
                {
                    "trade_date": day,
                    "board_code": board_code,
                    "board_name": board_names[board_code],
                    "evaluable": True,
                    "reason": None,
                    "member_count": member_count,
                    "available_count": len(members_today),
                    "coverage": coverage,
                    "weighting": weighting,
                    "board_return": board_return,
                    "market_return": float(market_returns.get(day, np.nan)),
                    "board_close": board_index,
                    "breadth_ma20": breadth,
                    "amount_share": float(
                        members_today["amount"].sum() / market_amount.loc[day]
                    ),
                    "turnover": float(members_today["turnover_rate"].mean())
                    if "turnover_rate" in members_today
                    else np.nan,
                    "limit_up_count": limit_count,
                    "limit_up_ratio": limit_ratio,
                    "top3_return_median": float(top3_returns.median())
                    if not top3_returns.empty
                    else np.nan,
                    "top3_amount_ratio_median": float(top3_amount_ratios.median())
                    if not top3_amount_ratios.empty
                    else np.nan,
                }
            )
    result = pd.DataFrame(rows)
    for column in (
        "weighting",
        "board_return",
        "market_return",
        "board_close",
        "breadth_ma20",
        "amount_share",
        "turnover",
        "limit_up_count",
        "limit_up_ratio",
        "top3_return_median",
        "top3_amount_ratio_median",
    ):
        if column not in result:
            result[column] = np.nan
    return result


def _score_boards(daily: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    output = daily.sort_values(["board_code", "trade_date"], kind="stable").copy()
    evaluable = output["evaluable"].fillna(False)
    grouped = output.groupby("board_code", sort=False)
    output["ma5"] = grouped["board_close"].transform(
        lambda values: values.rolling(5).mean()
    )
    output["ma20"] = grouped["board_close"].transform(
        lambda values: values.rolling(20).mean()
    )
    relative = output["board_return"] - output["market_return"]
    output["_relative"] = relative
    grouped = output.groupby("board_code", sort=False)
    output["relative_return_3d"] = grouped["_relative"].transform(
        lambda values: _rolling_compound(values, 3)
    )
    output["relative_return_5d"] = grouped["_relative"].transform(
        lambda values: _rolling_compound(values, 5)
    )
    recent3 = output["relative_return_3d"]
    previous3 = grouped["_relative"].transform(
        lambda values: _rolling_compound(values.shift(3), 3)
    )
    output["relative_acceleration_3d"] = recent3 - previous3
    output["breadth_change_3d"] = output["breadth_ma20"] - grouped[
        "breadth_ma20"
    ].shift(3)

    def rolling_median20(values: pd.Series) -> pd.Series:
        if not values.notna().any():
            return pd.Series(np.nan, index=values.index)
        return values.rolling(20).median()

    output["amount_share_median20"] = grouped["amount_share"].transform(
        rolling_median20
    )
    output["amount_share_vs_median20"] = (
        output["amount_share"] / output["amount_share_median20"]
    )
    output["turnover_median20"] = grouped["turnover"].transform(rolling_median20)
    output["turnover_ratio20"] = output["turnover"] / output["turnover_median20"]
    output["volatility20"] = grouped["board_return"].transform(
        lambda values: values.rolling(20).std()
    )
    output["outperform_days_3d"] = grouped["_relative"].transform(
        lambda values: (values > 0).rolling(3).sum()
    )
    above = output["board_close"] > output["ma20"]
    previous_above = (
        above.groupby(output["board_code"])
        .shift(1)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    crossed = above & ~previous_above
    output["crossed_ma20_within_3d"] = (
        crossed.groupby(output["board_code"])
        .transform(lambda values: values.rolling(3).max())
        .astype(bool)
    )
    factor_columns = (
        "relative_return_5d",
        "relative_acceleration_3d",
        "breadth_ma20",
        "breadth_change_3d",
        "amount_share_vs_median20",
        "limit_up_ratio",
        "outperform_days_3d",
    )
    output["amount_share_quantile"] = output.groupby("trade_date")["amount_share"].rank(
        pct=True, method="average"
    )
    output["turnover_crowding_rank"] = output.groupby("trade_date")[
        "turnover_ratio20"
    ].rank(pct=True, method="average")
    output["volatility_crowding_rank"] = output.groupby("trade_date")[
        "volatility20"
    ].rank(pct=True, method="average")
    output["crowding_quantile"] = 0.5 * (
        output["turnover_crowding_rank"] + output["volatility_crowding_rank"]
    )
    warning = float(config["crowding_warning_quantile"])
    overheat = float(config["crowding_overheat_quantile"])
    span = max(overheat - warning, 1e-12)
    output["crowding_penalty"] = ((output["crowding_quantile"] - warning) / span).clip(
        0, 1
    ) * float(config["crowding_penalty_max"])
    ranks = []
    for column in factor_columns:
        rank_column = f"{column}_rank"
        output[rank_column] = output.groupby("trade_date")[column].rank(
            pct=True, method="average"
        )
        ranks.append(rank_column)
    weights = np.asarray(MODEL_SPEC_V1["board_weights"])
    output["board_score"] = (
        output[ranks].mul(weights, axis=1).sum(axis=1, min_count=len(ranks)) * 100
        - output["crowding_penalty"]
    )
    output.loc[~evaluable, "board_score"] = np.nan
    output = output.drop(columns=["_relative"])
    return output


def _percentile(series: pd.Series, *, ascending: bool = True) -> pd.Series:
    ranks = series.rank(pct=True, method="average", ascending=ascending)
    return ranks.where(series.notna())


def _role_candidate(
    members: pd.DataFrame,
    phase: str,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    candidates = members.copy().sort_values("symbol", kind="stable")
    candidates["close_position"] = [
        close_position(row.close, row.low, row.high) for row in candidates.itertuples()
    ]
    candidates["upper_shadow_ratio"] = [
        upper_shadow_ratio(row.open, row.high, row.low, row.close)
        for row in candidates.itertuples()
    ]
    candidates["healthy_volume"] = candidates["amount_ratio"].map(healthy_volume)
    if phase == "launch":
        top_return_cutoff = candidates["return_5d"].quantile(0.90)
        last5_limit = candidates.get(
            "limit_up_last5", pd.Series(False, index=candidates.index)
        ).fillna(False)
        hard = (
            (candidates["is_st"] == 0)
            & ~candidates.get("is_suspended", pd.Series(False, index=candidates.index))
            .fillna(True)
            .astype(bool)
            & ~candidates.get(
                "limit_down_today", pd.Series(np.nan, index=candidates.index)
            )
            .fillna(True)
            .astype(bool)
            & candidates["float_mv"].between(
                config["leader_float_mv_min"], config["leader_float_mv_max"]
            )
            & (candidates["amount_ma5"] >= config["leader_amount_min"])
            & candidates["turnover_rate"].between(
                config["leader_turnover_min"], config["leader_turnover_max"]
            )
            & (candidates["adj_close"] > candidates["ma20"])
            & (candidates["rps20"] >= config["leader_rps_min"])
            & candidates["amount_ratio"].between(
                config["leader_amount_ratio_min"], config["leader_amount_ratio_max"]
            )
            & candidates["return_3d"].between(
                config["leader_return_3d_min"], config["leader_return_3d_max"]
            )
            & (last5_limit | (candidates["return_5d"] >= top_return_cutoff))
            & ~(
                (
                    candidates["amount_ratio"]
                    >= config["leader_upper_shadow_amount_ratio"]
                )
                & (
                    candidates["upper_shadow_ratio"]
                    >= config["leader_upper_shadow_ratio"]
                )
            )
        )
        candidates = candidates.loc[hard].copy()
        if candidates.empty:
            return None
        candidates["limit_proxy"] = _percentile(candidates["return_5d"])
        candidates["rps_score"] = _percentile(candidates["rps20"])
        candidates["return_score"] = _percentile(candidates["return_3d"])
        candidates["volume_score"] = _percentile(candidates["healthy_volume"])
        candidates["position_score"] = _percentile(candidates["close_position"])
        candidates["anti_fall"] = candidates.get(
            "anti_fall", pd.Series(0.0, index=candidates.index)
        )
        candidates["anti_fall_score"] = _percentile(candidates["anti_fall"])
        candidates["size_fit"] = candidates["float_mv"].map(
            lambda value: size_fit(
                value, config["leader_float_mv_min"], config["leader_float_mv_max"]
            )
        )
        candidates["size_score"] = _percentile(candidates["size_fit"])
        columns = [
            "limit_proxy",
            "rps_score",
            "return_score",
            "volume_score",
            "position_score",
            "anti_fall_score",
            "size_score",
        ]
        weights = np.asarray(MODEL_SPEC_V1["leader_weights"])
        role = "leader"
    else:
        mv_rank = candidates["float_mv"].rank(method="min", ascending=False)
        mv_percentile = candidates["float_mv"].rank(pct=True, ascending=False)
        hard = (
            (candidates["is_st"] == 0)
            & ~candidates.get("is_suspended", pd.Series(False, index=candidates.index))
            .fillna(True)
            .astype(bool)
            & ~candidates.get(
                "limit_up_today", pd.Series(np.nan, index=candidates.index)
            )
            .fillna(True)
            .astype(bool)
            & ~candidates.get(
                "limit_down_today", pd.Series(np.nan, index=candidates.index)
            )
            .fillna(True)
            .astype(bool)
            & (candidates["float_mv"] >= config["core_float_mv_min"])
            & ((mv_rank <= 3) | (mv_percentile <= config["core_mv_top_quantile"]))
            & (candidates["amount_ma5"] >= config["core_amount_min"])
            & (candidates["adj_close"] > candidates["ma20"])
            & (candidates["ma5"] >= candidates["ma20"])
            & candidates["volatility_20d"].between(
                config["core_volatility_min"], config["core_volatility_max"]
            )
            & (candidates["max_drawdown_20d"] >= config["core_drawdown_min"])
            & candidates["amount_ratio"].between(
                config["core_amount_ratio_min"], config["core_amount_ratio_max"]
            )
            & (candidates["return_5d"] <= config["core_return_5d_max"])
        )
        candidates = candidates.loc[hard].copy()
        if candidates.empty:
            return None
        candidates["trend_stability"] = [
            trend_stability(row.ma5, row.ma20, row.ma20_lag5)
            for row in candidates.itertuples()
        ]
        candidates = candidates[candidates["trend_stability"].notna()].copy()
        if candidates.empty:
            return None
        candidates["mv_score"] = _percentile(candidates["float_mv"])
        candidates["amount_score"] = _percentile(candidates["amount_ma5"])
        candidates["trend_score"] = _percentile(candidates["trend_stability"])
        candidates["volume_score"] = _percentile(candidates["healthy_volume"])
        candidates["drawdown_score"] = _percentile(candidates["max_drawdown_20d"])
        candidates["rps_score"] = _percentile(candidates["rps20"])
        columns = [
            "mv_score",
            "amount_score",
            "trend_score",
            "volume_score",
            "drawdown_score",
            "rps_score",
        ]
        weights = np.asarray(MODEL_SPEC_V1["core_weights"])
        role = "core"
    candidates["role_score"] = (
        candidates[columns].mul(weights, axis=1).sum(axis=1, min_count=len(columns))
        * 100
    )
    winner = candidates.sort_values(
        ["role_score", "symbol"], ascending=[False, True], kind="stable"
    ).iloc[0]
    return {
        "symbol": winner["symbol"],
        "role": role,
        "role_score": float(winner["role_score"]),
    }


def select_role_candidate(
    members: pd.DataFrame,
    phase: str,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Select at most one deterministic leader or core from a board snapshot."""
    if phase not in {"launch", "diffusion"}:
        return None
    return _role_candidate(
        members,
        phase,
        {**DEFAULT_PARAMS, **dict(params or {})},
    )


def build_sector_signals(
    stock_daily: pd.DataFrame,
    boards: Sequence[Mapping[str, Any]],
    memberships: Mapping[str, Sequence[Mapping[str, Any]]],
    start_date: Any,
    end_date: Any,
    *,
    params: Mapping[str, Any] | None = None,
    strict: bool = True,
) -> SectorSignalResult:
    """Build unshifted signal-date candidates and full board-state mappings."""
    config = {**DEFAULT_PARAMS, **dict(params or {})}
    universe = validate_membership_universe(boards, memberships, strict=strict)
    clean, stock_audit = validate_stock_daily(stock_daily)
    start = _date(start_date, field="start_date", required=True)
    end = _date(end_date, field="end_date", required=True)
    assert start is not None and end is not None
    if start > end:
        raise SectorCoreError("start_date must not be after end_date")
    # This explicit bound is the core no-future-data guarantee.
    clean = clean.loc[clean["trade_date"] <= end].copy()
    if clean.empty or clean["trade_date"].max() < start:
        raise StrictDataError(
            "no stock daily rows are available in the requested range"
        )
    featured = _prepare_features(clean)
    featured["limit_up_last5"] = (
        featured.groupby("symbol")["limit_up_today"].transform(
            lambda values: values.rolling(5).max()
        )
        if "limit_up_today" in featured
        else np.nan
    )
    diagnostics: list[dict[str, Any]] = []
    daily = _board_daily_rows(
        featured, universe.boards, universe.memberships, config, diagnostics
    )
    if daily.empty:
        raise StrictDataError("no board-state rows could be calculated")
    scored = _score_boards(daily, config)
    scored = scored[
        (scored["trade_date"] >= start) & (scored["trade_date"] <= end)
    ].copy()
    board_states: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    candidate_rows: list[dict[str, Any]] = []
    featured_by_date = {
        pd.Timestamp(day): part.set_index("symbol", drop=False)
        for day, part in featured.groupby("trade_date")
    }
    active_by_date: dict[pd.Timestamp, dict[str, list[dict[str, Any]]]] = {}
    warm_dates = 0
    for day, section in scored.groupby("trade_date", sort=True):
        day = pd.Timestamp(day)
        evaluable = section[section["board_score"].notna()].copy()
        warmed = section["ma20"].notna().any()
        if warmed:
            warm_dates += 1
        if strict and warmed and len(evaluable) < int(config["topk_sectors"]):
            raise StrictDataError(
                f"{day.date()}: only {len(evaluable)} evaluable boards after warm-up; "
                f"topk_sectors={config['topk_sectors']}"
            )
        evaluable = evaluable.sort_values(
            ["board_score", "relative_return_5d", "board_code"],
            ascending=[False, False, True],
            kind="stable",
        )
        rank_by_code = {
            code: rank for rank, code in enumerate(evaluable["board_code"], 1)
        }
        states: dict[str, dict[str, Any]] = {}
        for _, raw in section.sort_values("board_code").iterrows():
            record = raw.to_dict()
            record["rank"] = rank_by_code.get(raw["board_code"])
            record["signal_date"] = day
            if bool(raw.get("evaluable")) and pd.notna(raw.get("board_score")):
                phase, phase_diagnostic = classify_board_phase(record, config)
                record["phase"] = phase
                if phase_diagnostic:
                    record["phase_diagnostic"] = phase_diagnostic
            else:
                record["phase"] = None
            states[str(raw["board_code"])] = record
        board_states[day] = states
        top_codes = list(evaluable.head(int(config["topk_sectors"]))["board_code"])
        if day not in active_by_date:
            active_by_date[day] = point_in_time_members(universe.memberships, day)
        today = featured_by_date.get(day)
        if today is None:
            continue
        for board_code in top_codes:
            state = states[board_code]
            phase = state["phase"]
            if phase not in {"launch", "diffusion"}:
                continue
            symbols = [
                item["symbol"] for item in active_by_date[day].get(board_code, [])
            ]
            rows = today.reindex(symbols).dropna(subset=["symbol"])
            needed = {"float_mv", "turnover_rate", "limit_up_today", "limit_down_today"}
            if not needed.issubset(rows.columns):
                diagnostics.append(
                    {
                        "date": day,
                        "board_code": board_code,
                        "reason": "missing_role_columns",
                        "soft": True,
                    }
                )
                continue
            board_recent = scored.loc[
                (scored["board_code"] == board_code) & (scored["trade_date"] <= day),
                ["trade_date", "board_return"],
            ].tail(5)
            board_return_by_date = board_recent.set_index("trade_date")["board_return"]
            anti_fall_values: list[float] = []
            for symbol in rows["symbol"]:
                stock_recent = featured.loc[
                    (featured["symbol"] == symbol)
                    & (featured["trade_date"].isin(board_return_by_date.index)),
                    ["trade_date", "stock_return"],
                ].set_index("trade_date")["stock_return"]
                aligned = pd.concat(
                    [stock_recent, board_return_by_date], axis=1, join="inner"
                ).dropna()
                anti_fall_values.append(
                    anti_fall(aligned["stock_return"], aligned["board_return"])
                    if not aligned.empty
                    else 0.0
                )
            rows = rows.copy()
            rows["anti_fall"] = anti_fall_values
            role = select_role_candidate(rows, phase, config)
            if role is None:
                continue
            combined = (
                MODEL_SPEC_V1["combined_weights"][0] * float(state["board_score"])
                + MODEL_SPEC_V1["combined_weights"][1] * role["role_score"]
            )
            candidate_rows.append(
                {
                    "datetime": day,
                    "instrument": role["symbol"],
                    "score": combined,
                    "board_code": board_code,
                    "board_name": state["board_name"],
                    "phase": phase,
                    "role": role["role"],
                    "role_score": role["role_score"],
                    "board_score": float(state["board_score"]),
                    "signal_date": day,
                }
            )
    if candidate_rows:
        signals = pd.DataFrame(candidate_rows).sort_values(
            ["datetime", "score", "board_code", "instrument"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        signals = signals.drop_duplicates(["datetime", "instrument"], keep="first")
        signals = signals.groupby("datetime", group_keys=False).head(
            int(config["topk_stocks"])
        )
        signals = signals.set_index(["datetime", "instrument"]).sort_index()
    else:
        signals = pd.DataFrame(
            columns=[
                "score",
                "board_code",
                "board_name",
                "phase",
                "role",
                "role_score",
                "board_score",
                "signal_date",
            ]
        )
        signals.index = pd.MultiIndex.from_arrays(
            [[], []], names=["datetime", "instrument"]
        )
    metadata = {
        "source": "sector_momentum_leader_core",
        "model_spec_version": MODEL_SPEC_V1["version"],
        "raw_signal_date_min": start.date().isoformat(),
        "raw_signal_date_max": end.date().isoformat(),
        "signal_lag_days": 0,
        "stock_audit": stock_audit,
        "membership_audit": universe.metadata,
        "board_universe": config["board_universe"],
        "diagnostics": diagnostics,
        "warm_dates": warm_dates,
    }
    if config["board_universe"] == "ths_concept":
        metadata["survivorship_warning"] = (
            "同花顺概念成员按点时区间过滤，但概念目录层面仍可能存在幸存者偏差"
        )
    return SectorSignalResult(
        signals=signals, board_states=board_states, metadata=metadata
    )


__all__ = [
    "DEFAULT_PARAMS",
    "MODEL_SPEC_V1",
    "SectorCoreError",
    "SectorSignalResult",
    "SectorUniverse",
    "StrictDataError",
    "anti_fall",
    "build_sector_signals",
    "capped_cap_weights",
    "chain_board_index",
    "classify_board_phase",
    "close_position",
    "healthy_volume",
    "load_sector_universe",
    "normalize_stock_code",
    "point_in_time_members",
    "select_role_candidate",
    "size_fit",
    "trend_stability",
    "upper_shadow_ratio",
    "validate_membership_universe",
    "validate_stock_daily",
]
