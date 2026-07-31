#!/usr/bin/env python3
"""Incrementally update QuantMind daily OHLCV data from Baostock.

This is the credential-free fallback for OSS deployments.  It updates the
shared ``stock_daily_latest`` table and the existing Qlib binary dataset.  It
does not fabricate the 152-dimensional model feature snapshots; those remain
the responsibility of the signed official data bundle updater.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from psycopg2 import connect
from psycopg2.extras import execute_values


QLIB_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "factor",
    "adjclose",
    "change",
    "vwap",
)
TUSHARE_API_URL = "http://api.tushare.pro"


@dataclass(frozen=True)
class DailyRow:
    trade_date: str
    symbol: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    preclose: float | None
    volume: float | None
    amount: float | None
    pct_change: float | None
    turnover_rate: float | None
    pe_ttm: float | None
    pb: float | None
    is_st: int


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Incrementally sync daily A-share OHLCV from Baostock"
    )
    parser.add_argument("--target-date", default=date.today().isoformat())
    parser.add_argument("--qlib-dir", default=str(root / "db" / "qlib_data"))
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--skip-database", action="store_true")
    parser.add_argument("--skip-qlib", action="store_true")
    parser.add_argument(
        "--refresh-days",
        type=int,
        default=0,
        help=(
            "Re-fetch and idempotently upsert this many recent calendar days; "
            "used by the report preflight to repair incomplete or invalid rows"
        ),
    )
    parser.add_argument(
        "--minimum-success-ratio",
        type=float,
        default=0.90,
        help="Abort writes when fewer instruments return data",
    )
    parser.add_argument(
        "--lock-file",
        default=os.getenv(
            "QUANTMIND_DATA_UPDATE_LOCK", "/data/quantmind-data-update.lock"
        ),
    )
    return parser.parse_args()


def _optional_float(value: Any) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def normalize_symbol(code: str) -> str | None:
    normalized = str(code or "").strip().lower()
    if normalized.startswith("sh."):
        return f"SH{normalized[3:]}"
    if normalized.startswith("sz."):
        return f"SZ{normalized[3:]}"
    if normalized.startswith("bj."):
        return f"BJ{normalized[3:]}"
    return None


def normalize_tushare_symbol(code: str) -> str | None:
    normalized = str(code or "").strip().upper()
    if "." not in normalized:
        return None
    digits, market = normalized.split(".", 1)
    if market not in {"SH", "SZ", "BJ"} or not digits.isdigit():
        return None
    return f"{market}{digits}"


def query_tushare(
    token: str,
    api_name: str,
    *,
    params: dict[str, Any] | None = None,
    fields: tuple[str, ...] = (),
    retries: int = 4,
) -> list[dict[str, Any]]:
    """Query the raw Tushare HTTP API without adding a package dependency."""
    payload = {
        "api_name": api_name,
        "token": token,
        "params": params or {},
        "fields": ",".join(fields),
    }
    body = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries):
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
                dict(zip(names, item, strict=False))
                for item in data.get("items") or []
            ]
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 >= retries:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"{api_name} request failed: {last_error}")


def _tushare_date(value: Any) -> str | None:
    raw = str(value or "").strip()
    if len(raw) != 8 or not raw.isdigit():
        return None
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def build_tushare_rows(
    daily_rows: list[dict[str, Any]],
    daily_basic_rows: list[dict[str, Any]],
    stock_basic_rows: list[dict[str, Any]],
    requested_date: str,
    instruments: set[str],
) -> dict[str, dict[str, DailyRow]]:
    """Convert one Tushare trading-day snapshot to the shared row model."""
    basics = {
        (str(row.get("ts_code") or "").upper(), str(row.get("trade_date") or "")): row
        for row in daily_basic_rows
    }
    names = {
        str(row.get("ts_code") or "").upper(): str(row.get("name") or "")
        for row in stock_basic_rows
    }
    result: dict[str, dict[str, DailyRow]] = {}
    for item in daily_rows:
        ts_code = str(item.get("ts_code") or "").strip().upper()
        symbol = normalize_tushare_symbol(ts_code)
        trade_date = _tushare_date(item.get("trade_date"))
        if (
            symbol is None
            or symbol not in instruments
            or trade_date != requested_date
        ):
            continue
        raw_trade_date = str(item.get("trade_date") or "")
        basic = basics.get((ts_code, raw_trade_date), {})
        volume = _optional_float(item.get("vol"))
        amount = _optional_float(item.get("amount"))
        turnover = _optional_float(basic.get("turnover_rate"))
        name = names.get(ts_code, "").upper()
        result.setdefault(symbol, {})[trade_date] = DailyRow(
            trade_date=trade_date,
            symbol=symbol,
            open=_optional_float(item.get("open")),
            high=_optional_float(item.get("high")),
            low=_optional_float(item.get("low")),
            close=_optional_float(item.get("close")),
            preclose=_optional_float(item.get("pre_close")),
            volume=(volume * 100.0 if volume is not None else None),
            amount=(amount * 1000.0 if amount is not None else None),
            pct_change=_optional_float(item.get("pct_chg")),
            turnover_rate=(turnover / 100.0 if turnover is not None else None),
            pe_ttm=_optional_float(basic.get("pe_ttm")),
            pb=_optional_float(basic.get("pb")),
            is_st=1 if "ST" in name else 0,
        )
    return result


def fetch_tushare_daily_rows(
    token: str,
    trade_dates: list[str],
    instruments: set[str],
) -> dict[str, dict[str, DailyRow]]:
    stock_basic_rows = query_tushare(
        token,
        "stock_basic",
        params={"list_status": "L"},
        fields=("ts_code", "name"),
    )
    result: dict[str, dict[str, DailyRow]] = {}
    for trade_date in trade_dates:
        compact_date = trade_date.replace("-", "")
        daily_rows = query_tushare(
            token,
            "daily",
            params={"trade_date": compact_date},
            fields=(
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "pct_chg",
                "vol",
                "amount",
            ),
        )
        if not daily_rows:
            raise RuntimeError(f"Tushare daily has no rows for {trade_date}")
        daily_basic_rows = query_tushare(
            token,
            "daily_basic",
            params={"trade_date": compact_date},
            fields=(
                "ts_code",
                "trade_date",
                "turnover_rate",
                "pe_ttm",
                "pb",
            ),
        )
        day_rows = build_tushare_rows(
            daily_rows,
            daily_basic_rows,
            stock_basic_rows,
            trade_date,
            instruments,
        )
        if not day_rows:
            raise RuntimeError(
                f"Tushare daily has no matching A-share rows for {trade_date}"
            )
        for symbol, symbol_rows in day_rows.items():
            result.setdefault(symbol, {}).update(symbol_rows)
    return result


def to_baostock_symbol(symbol: str) -> str | None:
    normalized = str(symbol or "").strip().upper()
    if normalized.startswith("SH"):
        return f"sh.{normalized[2:]}"
    if normalized.startswith("SZ"):
        return f"sz.{normalized[2:]}"
    if normalized.startswith("BJ"):
        return f"bj.{normalized[2:]}"
    return None


def load_calendar(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Qlib calendar not found: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_instruments(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Qlib instruments not found: {path}")
    symbols: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        symbol = parts[0].upper()
        if symbol.startswith(("SH", "SZ")):
            symbols.append(symbol)
    return sorted(set(symbols))


def _query_trade_dates(bs: Any, start_date: str, end_date: str) -> list[str]:
    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    if rs.error_code != "0":
        raise RuntimeError(
            f"Baostock trade calendar failed: {rs.error_code} {rs.error_msg}"
        )
    result: list[str] = []
    while rs.next():
        row = rs.get_row_data()
        if len(row) >= 2 and row[1] == "1":
            result.append(row[0])
    return result


def _benchmark_has_data(bs: Any, trade_date: str) -> bool:
    rs = bs.query_history_k_data_plus(
        "sh.600000",
        "date,close",
        start_date=trade_date,
        end_date=trade_date,
        frequency="d",
        adjustflag="3",
    )
    return rs.error_code == "0" and rs.next() and bool(rs.get_row_data())


def resolve_new_trade_dates(
    bs: Any, existing_calendar: list[str], target_date: str
) -> list[str]:
    if not existing_calendar:
        raise RuntimeError("Qlib calendar is empty")
    start = (date.fromisoformat(existing_calendar[-1]) + timedelta(days=1)).isoformat()
    if start > target_date:
        return []
    return _query_trade_dates(bs, start, target_date)


def resolve_missing_dates(
    bs: Any, existing_calendar: list[str], target_date: str
) -> list[str]:
    candidates = resolve_new_trade_dates(bs, existing_calendar, target_date)
    return [candidate for candidate in candidates if _benchmark_has_data(bs, candidate)]


def resolve_candidate_dates(
    bs: Any,
    existing_calendar: list[str],
    target_date: str,
    refresh_days: int = 0,
) -> list[str]:
    """Return open dates that must be sourced, before provider availability checks."""
    requested = set(resolve_new_trade_dates(bs, existing_calendar, target_date))
    if refresh_days > 0:
        refresh_start = (
            date.fromisoformat(target_date) - timedelta(days=refresh_days - 1)
        ).isoformat()
        refresh_start = max(refresh_start, existing_calendar[0])
        requested.update(_query_trade_dates(bs, refresh_start, target_date))
    return sorted(requested)


def resolve_requested_dates(
    bs: Any,
    existing_calendar: list[str],
    target_date: str,
    refresh_days: int = 0,
) -> list[str]:
    """Resolve new dates plus an optional bounded repair window."""
    missing_dates = resolve_missing_dates(bs, existing_calendar, target_date)
    requested = set(missing_dates)
    if refresh_days > 0:
        refresh_start = (
            date.fromisoformat(target_date) - timedelta(days=refresh_days - 1)
        ).isoformat()
        refresh_start = max(refresh_start, existing_calendar[0])
        for candidate in _query_trade_dates(bs, refresh_start, target_date):
            if candidate <= existing_calendar[-1] or _benchmark_has_data(bs, candidate):
                requested.add(candidate)
    return sorted(requested)


def fetch_symbol_rows(
    bs: Any,
    symbol: str,
    start_date: str,
    end_date: str,
) -> dict[str, DailyRow]:
    bs_symbol = to_baostock_symbol(symbol)
    if not bs_symbol:
        return {}
    fields = (
        "date,code,open,high,low,close,preclose,volume,amount,pctChg,"
        "turn,peTTM,pbMRQ,isST"
    )
    rs = bs.query_history_k_data_plus(
        bs_symbol,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )
    if rs.error_code != "0":
        raise RuntimeError(f"{symbol}: {rs.error_code} {rs.error_msg}")

    result: dict[str, DailyRow] = {}
    while rs.next():
        values = rs.get_row_data()
        if len(values) < 14:
            continue
        normalized = normalize_symbol(values[1])
        if normalized is None:
            continue
        pct_change = _optional_float(values[9])
        turnover = _optional_float(values[10])
        result[values[0]] = DailyRow(
            trade_date=values[0],
            symbol=normalized,
            open=_optional_float(values[2]),
            high=_optional_float(values[3]),
            low=_optional_float(values[4]),
            close=_optional_float(values[5]),
            preclose=_optional_float(values[6]),
            volume=_optional_float(values[7]),
            amount=_optional_float(values[8]),
            pct_change=pct_change,
            turnover_rate=(turnover / 100.0 if turnover is not None else None),
            pe_ttm=_optional_float(values[11]),
            pb=_optional_float(values[12]),
            is_st=1 if str(values[13]).strip() == "1" else 0,
        )
    return result


def read_last_factor(symbol_dir: Path) -> float:
    factor_path = symbol_dir / "factor.day.bin"
    if not factor_path.exists():
        return 1.0
    values = np.fromfile(factor_path, dtype="<f4")
    finite = values[1:][np.isfinite(values[1:])] if len(values) > 1 else np.array([])
    return float(finite[-1]) if len(finite) else 1.0


def qlib_value(row: DailyRow | None, field: str, factor: float) -> float:
    if row is None:
        return 0.0 if field in {"volume", "amount"} else np.nan
    if field in {"open", "high", "low", "close"}:
        value = getattr(row, field)
        return float(value) if value is not None else np.nan
    if field == "factor":
        return factor
    if field == "adjclose":
        return float(row.close * factor) if row.close is not None else np.nan
    if field == "volume":
        return (
            float(row.volume / (100.0 * factor))
            if row.volume is not None and factor > 0
            else 0.0
        )
    if field == "amount":
        return float(row.amount / 1000.0) if row.amount is not None else 0.0
    if field == "change":
        return float(row.pct_change / 100.0) if row.pct_change is not None else np.nan
    if field == "vwap":
        if row.amount is None or row.volume in (None, 0):
            return np.nan
        return float((row.amount / row.volume) * factor)
    return np.nan


def extend_bin_file(
    bin_path: Path,
    full_calendar: list[str],
    symbol_rows: dict[str, DailyRow],
    factor: float,
    apply: bool,
) -> bool:
    if not bin_path.exists():
        return False
    raw = np.fromfile(bin_path, dtype="<f4")
    if len(raw) < 1:
        return False
    start_index = int(raw[0])
    old_values = raw[1:]
    new_length = len(full_calendar) - start_index
    if new_length <= len(old_values):
        return False

    field = bin_path.name.split(".", 1)[0]
    fill = 0.0 if field in {"volume", "amount"} else np.nan
    new_values = np.full(new_length, fill, dtype="<f4")
    new_values[: len(old_values)] = old_values
    for offset in range(len(old_values), new_length):
        calendar_index = start_index + offset
        if calendar_index >= len(full_calendar):
            break
        trade_date = full_calendar[calendar_index]
        new_values[offset] = qlib_value(symbol_rows.get(trade_date), field, factor)

    if apply:
        temp_path = bin_path.with_suffix(bin_path.suffix + ".tmp")
        output = np.concatenate(([np.float32(start_index)], new_values)).astype("<f4")
        output.tofile(temp_path)
        os.replace(temp_path, bin_path)
    return True


def update_qlib(
    qlib_dir: Path,
    existing_calendar: list[str],
    missing_dates: list[str],
    rows_by_symbol: dict[str, dict[str, DailyRow]],
    factors_by_symbol: dict[str, float],
    apply: bool,
) -> dict[str, int]:
    full_calendar = existing_calendar + [
        d for d in missing_dates if d not in existing_calendar
    ]
    features_root = qlib_dir / "features"
    changed_files = 0
    changed_symbols = 0
    for symbol, rows in rows_by_symbol.items():
        symbol_dir = features_root / symbol.lower()
        if not symbol_dir.is_dir():
            continue
        factor = factors_by_symbol.get(symbol, 1.0)
        symbol_changed = False
        for field in QLIB_FIELDS:
            changed = extend_bin_file(
                symbol_dir / f"{field}.day.bin",
                full_calendar,
                rows,
                factor,
                apply,
            )
            changed_files += int(changed)
            symbol_changed = symbol_changed or changed
        changed_symbols += int(symbol_changed)

    if apply:
        calendar_path = qlib_dir / "calendars" / "day.txt"
        shutil.copy2(calendar_path, calendar_path.with_suffix(".txt.bak"))
        temp_path = calendar_path.with_suffix(".txt.tmp")
        temp_path.write_text("\n".join(full_calendar) + "\n", encoding="utf-8")
        os.replace(temp_path, calendar_path)
        latest = full_calendar[-1]
        for instrument_path in (qlib_dir / "instruments").glob("*.txt"):
            output: list[str] = []
            for line in instrument_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[0].upper() in rows_by_symbol:
                    parts[2] = latest
                    output.append("\t".join(parts))
                else:
                    output.append(line)
            instrument_path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return {"symbols": changed_symbols, "files": changed_files}


def _database_connection():
    return connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "quantmind"),
        user=os.getenv("DB_USER", "quantmind"),
        password=os.getenv("DB_PASSWORD", ""),
        connect_timeout=15,
    )


def upsert_database(
    rows_by_symbol: dict[str, dict[str, DailyRow]],
    factors_by_symbol: dict[str, float],
    apply: bool,
) -> int:
    columns = (
        "trade_date",
        "symbol",
        "is_st",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pct_change",
        "turnover_rate",
        "adj_factor",
        "pe_ttm",
        "pb",
        "bp",
        "ep_ttm",
        "return_1d",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "raw_volume",
        "raw_amount",
    )
    records: list[tuple[Any, ...]] = []
    for symbol_rows in rows_by_symbol.values():
        for row in symbol_rows.values():
            bp = (1.0 / row.pb) if row.pb not in (None, 0) else None
            ep = (1.0 / row.pe_ttm) if row.pe_ttm not in (None, 0) else None
            records.append(
                (
                    row.trade_date,
                    row.symbol,
                    row.is_st,
                    row.open,
                    row.high,
                    row.low,
                    row.close,
                    row.volume,
                    row.amount,
                    row.pct_change,
                    row.turnover_rate,
                    factors_by_symbol.get(row.symbol, 1.0),
                    row.pe_ttm,
                    row.pb,
                    bp,
                    ep,
                    (row.pct_change / 100.0 if row.pct_change is not None else None),
                    row.open,
                    row.high,
                    row.low,
                    row.close,
                    row.volume,
                    row.amount,
                )
            )
    if not apply or not records:
        return len(records)

    update_columns = [
        column for column in columns if column not in {"trade_date", "symbol"}
    ]
    sql = f"""
        INSERT INTO stock_daily_latest ({", ".join(columns)}) VALUES %s
        ON CONFLICT (trade_date, symbol) DO UPDATE SET
        {", ".join(f"{column}=EXCLUDED.{column}" for column in update_columns)}
    """
    connection = _database_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                execute_values(cursor, sql, records, page_size=2000)
    finally:
        connection.close()
    return len(records)


def invalidate_status_cache() -> None:
    try:
        from backend.shared.redis_sentinel_client import get_redis_sentinel_client

        get_redis_sentinel_client().delete("qm:admin:data_status")
    except Exception as exc:
        print(f"[WARN] failed to invalidate data status cache: {exc}", file=sys.stderr)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_symbols and args.apply:
        raise RuntimeError("--max-symbols is only allowed for dry-run validation")
    if not 0 < args.minimum_success_ratio <= 1:
        raise ValueError("--minimum-success-ratio must be in (0, 1]")
    if args.refresh_days < 0:
        raise ValueError("--refresh-days must be non-negative")

    qlib_dir = Path(args.qlib_dir).expanduser().resolve()
    calendar_path = qlib_dir / "calendars" / "day.txt"
    instruments_path = qlib_dir / "instruments" / "all.txt"
    existing_calendar = load_calendar(calendar_path)
    instruments = load_instruments(instruments_path)
    if args.max_symbols:
        instruments = instruments[: args.max_symbols]

    try:
        import baostock as bs
    except ImportError as exc:
        raise RuntimeError("baostock is not installed") from exc

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(
            f"Baostock login failed: {login.error_code} {login.error_msg}"
        )
    try:
        requested_dates = resolve_candidate_dates(
            bs,
            existing_calendar,
            args.target_date,
            refresh_days=args.refresh_days,
        )
        if not requested_dates:
            return {
                "success": True,
                "source": "baostock",
                "skipped": True,
                "reason": "qlib_calendar_already_current",
                "calendar_last_date": existing_calendar[-1],
            }

        baostock_dates: list[str] = []
        tushare_dates: list[str] = []
        for trade_date in requested_dates:
            if _benchmark_has_data(bs, trade_date):
                baostock_dates.append(trade_date)
            else:
                tushare_dates.append(trade_date)

        tushare_token = os.getenv("TUSHARE_TOKEN", "").strip()
        if tushare_dates and not tushare_token:
            joined_dates = ", ".join(tushare_dates)
            raise RuntimeError(
                "Baostock has no daily bars for open trading date(s) "
                f"{joined_dates}; TUSHARE_TOKEN is required for fallback"
            )

        rows_by_symbol: dict[str, dict[str, DailyRow]] = {}
        failures: list[dict[str, str]] = []
        if baostock_dates:
            for index, symbol in enumerate(instruments, start=1):
                try:
                    rows = fetch_symbol_rows(
                        bs, symbol, baostock_dates[0], baostock_dates[-1]
                    )
                    rows = {
                        trade_date: row
                        for trade_date, row in rows.items()
                        if trade_date in baostock_dates
                    }
                    if rows:
                        rows_by_symbol[symbol] = rows
                except Exception as exc:
                    failures.append({"symbol": symbol, "error": str(exc)[:200]})
                if index % 250 == 0 or index == len(instruments):
                    print(
                        f"[PROGRESS] baostock {index}/{len(instruments)} "
                        f"ok={len(rows_by_symbol)} failed={len(failures)}",
                        flush=True,
                    )

            incomplete_baostock_dates = []
            for trade_date in baostock_dates:
                date_coverage = (
                    sum(
                        trade_date in rows
                        for rows in rows_by_symbol.values()
                    )
                    / len(instruments)
                    if instruments
                    else 0.0
                )
                if date_coverage < args.minimum_success_ratio:
                    incomplete_baostock_dates.append(trade_date)
            if incomplete_baostock_dates:
                if not tushare_token:
                    joined_dates = ", ".join(incomplete_baostock_dates)
                    raise RuntimeError(
                        "Baostock coverage is incomplete for open trading date(s) "
                        f"{joined_dates}; TUSHARE_TOKEN is required for fallback"
                    )
                tushare_dates = sorted(
                    set(tushare_dates).union(incomplete_baostock_dates)
                )

        if tushare_dates:
            fallback_rows = fetch_tushare_daily_rows(
                tushare_token,
                tushare_dates,
                set(instruments),
            )
            for symbol, symbol_rows in fallback_rows.items():
                rows_by_symbol.setdefault(symbol, {}).update(symbol_rows)

        complete_symbols = {
            symbol
            for symbol, rows in rows_by_symbol.items()
            if all(trade_date in rows for trade_date in requested_dates)
        }
        for symbol in instruments:
            missing = [
                trade_date
                for trade_date in requested_dates
                if trade_date not in rows_by_symbol.get(symbol, {})
            ]
            if missing:
                failures.append(
                    {"symbol": symbol, "error": f"no rows for {', '.join(missing)}"}
                )

        success_ratio = len(complete_symbols) / len(instruments) if instruments else 0.0
        if success_ratio < args.minimum_success_ratio:
            raise RuntimeError(
                f"Market-data coverage too low: {success_ratio:.2%} "
                f"<{args.minimum_success_ratio:.2%}"
            )

        factors_by_symbol = {
            symbol: read_last_factor(qlib_dir / "features" / symbol.lower())
            for symbol in rows_by_symbol
        }
        database_rows = 0
        if not args.skip_database:
            database_rows = upsert_database(
                rows_by_symbol, factors_by_symbol, args.apply
            )
        qlib_result = {"symbols": 0, "files": 0}
        if not args.skip_qlib:
            qlib_result = update_qlib(
                qlib_dir,
                existing_calendar,
                requested_dates,
                rows_by_symbol,
                factors_by_symbol,
                args.apply,
            )
        if args.apply:
            invalidate_status_cache()
        if baostock_dates and tushare_dates:
            source = "baostock+tushare"
        elif tushare_dates:
            source = "tushare"
        else:
            source = "baostock"
        return {
            "success": True,
            "source": source,
            "apply": bool(args.apply),
            "date_start": requested_dates[0],
            "date_end": requested_dates[-1],
            "trading_days": len(requested_dates),
            "baostock_dates": baostock_dates,
            "tushare_dates": tushare_dates,
            "refresh_days": args.refresh_days,
            "instruments_total": len(instruments),
            "instruments_ok": len(complete_symbols),
            "instruments_failed": len(instruments) - len(complete_symbols),
            "success_ratio": round(success_ratio, 6),
            "database_rows": database_rows,
            "qlib": qlib_result,
            "failure_samples": failures[:20],
        }
    finally:
        bs.logout()


def main() -> int:
    args = parse_args()
    load_dotenv(project_root() / ".env", override=False)
    lock_path = Path(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                json.dumps(
                    {"success": True, "skipped": True, "reason": "already_running"}
                )
            )
            return 0
        try:
            result = run(args)
            print(json.dumps(result, ensure_ascii=False, default=str))
            return 0
        except Exception as exc:
            print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
