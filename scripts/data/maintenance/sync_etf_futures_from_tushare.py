#!/usr/bin/env python3
"""Download ETF and CFFEX equity-index futures data from Tushare.

The job deliberately keeps ETFs and futures outside ``stock_daily_latest`` and
the stock Qlib provider.  It writes dedicated PostgreSQL tables and optional
year-partitioned Parquet snapshots under ``db/market_ext``.

``TUSHARE_TOKEN`` is read only from the process environment.  The token is
never accepted as a command-line argument or included in logs.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from psycopg2 import connect
from psycopg2.extras import execute_values


PRODUCTS = ("IF", "IH", "IC", "IM")
DEFAULT_START_DATE = "20160101"
API_URL = "http://api.tushare.pro"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Sync ETF and IF/IH/IC/IM daily data from Tushare"
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--asset", choices=("all", "etf", "futures"), default="all")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-parquet", action="store_true")
    parser.add_argument("--output-dir", default=str(root / "db" / "market_ext"))
    parser.add_argument("--max-etfs", type=int, default=0)
    parser.add_argument("--max-contracts", type=int, default=0)
    parser.add_argument("--minimum-success-ratio", type=float, default=0.90)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=float(os.getenv("TUSHARE_REQUEST_INTERVAL", "0.12")),
    )
    parser.add_argument(
        "--lock-file",
        default=os.getenv(
            "QUANTMIND_TUSHARE_UPDATE_LOCK",
            "/data/quantmind-tushare-update.lock",
        ),
    )
    return parser.parse_args()


def normalize_date(value: str) -> str:
    raw = str(value or "").replace("-", "").strip()
    datetime.strptime(raw, "%Y%m%d")
    return raw


def iso_date(value: Any) -> str | None:
    raw = str(value or "").replace("-", "").strip()
    if len(raw) != 8 or not raw.isdigit():
        return None
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def next_date(value: str) -> str:
    return (datetime.strptime(value, "%Y%m%d").date() + timedelta(days=1)).strftime(
        "%Y%m%d"
    )


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def optional_int(value: Any) -> int | None:
    parsed = optional_float(value)
    return int(parsed) if parsed is not None else None


def normalize_etf_symbol(ts_code: str) -> str | None:
    raw = str(ts_code or "").strip().upper()
    if raw.endswith(".SH") and raw[:-3].isdigit():
        return f"SH{raw[:-3]}"
    if raw.endswith(".SZ") and raw[:-3].isdigit():
        return f"SZ{raw[:-3]}"
    return None


def futures_product(ts_code: str, fut_code: str = "") -> str | None:
    candidate = str(fut_code or "").strip().upper()
    if candidate in PRODUCTS:
        return candidate
    contract = str(ts_code or "").strip().upper().split(".", 1)[0]
    return next((product for product in PRODUCTS if contract.startswith(product)), None)


def yearly_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = datetime.strptime(start_date, "%Y%m%d").date()
    end = datetime.strptime(end_date, "%Y%m%d").date()
    ranges: list[tuple[str, str]] = []
    for year in range(start.year, end.year + 1):
        left = max(start, date(year, 1, 1))
        right = min(end, date(year, 12, 31))
        ranges.append((left.strftime("%Y%m%d"), right.strftime("%Y%m%d")))
    return ranges


@dataclass
class TushareClient:
    token: str
    interval: float = 0.12
    retries: int = 4
    api_url: str = API_URL

    def query(
        self,
        api_name: str,
        params: dict[str, Any] | None = None,
        fields: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        payload = {
            "api_name": api_name,
            "token": self.token,
            "params": params or {},
            "fields": ",".join(fields),
        }
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries):
            if self.interval > 0:
                time.sleep(self.interval)
            try:
                request = urllib.request.Request(
                    self.api_url,
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
                if attempt + 1 >= self.retries:
                    break
                time.sleep((2**attempt) + random.random())
        raise RuntimeError(f"{api_name} request failed: {last_error}")


def database_connection():
    return connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "quantmind"),
        user=os.getenv("DB_USER", "quantmind"),
        password=os.getenv("DB_PASSWORD", ""),
        connect_timeout=15,
    )


def ensure_schema(connection: Any) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS etf_instruments (
        symbol TEXT PRIMARY KEY,
        ts_code TEXT NOT NULL UNIQUE,
        name TEXT,
        management TEXT,
        custodian TEXT,
        benchmark TEXT,
        fund_type TEXT,
        list_date DATE,
        delist_date DATE,
        source TEXT NOT NULL DEFAULT 'tushare',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS etf_daily (
        trade_date DATE NOT NULL,
        symbol TEXT NOT NULL,
        ts_code TEXT NOT NULL,
        open DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        close DOUBLE PRECISION,
        pre_close DOUBLE PRECISION,
        change DOUBLE PRECISION,
        pct_chg DOUBLE PRECISION,
        volume DOUBLE PRECISION,
        amount DOUBLE PRECISION,
        adj_factor DOUBLE PRECISION,
        source TEXT NOT NULL DEFAULT 'tushare',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (trade_date, symbol)
    );
    CREATE INDEX IF NOT EXISTS idx_etf_daily_symbol_date
        ON etf_daily (symbol, trade_date DESC);

    CREATE TABLE IF NOT EXISTS futures_contracts (
        contract_code TEXT PRIMARY KEY,
        product_code TEXT NOT NULL,
        exchange TEXT NOT NULL,
        name TEXT,
        multiplier DOUBLE PRECISION,
        trade_unit TEXT,
        per_unit DOUBLE PRECISION,
        quote_unit TEXT,
        quote_unit_desc TEXT,
        delivery_month TEXT,
        list_date DATE,
        delist_date DATE,
        delivery_mode_desc TEXT,
        trade_time_desc TEXT,
        source TEXT NOT NULL DEFAULT 'tushare',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_futures_contracts_product
        ON futures_contracts (product_code, list_date, delist_date);

    CREATE TABLE IF NOT EXISTS futures_daily (
        trade_date DATE NOT NULL,
        contract_code TEXT NOT NULL,
        product_code TEXT NOT NULL,
        exchange TEXT NOT NULL DEFAULT 'CFFEX',
        open DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        close DOUBLE PRECISION,
        pre_close DOUBLE PRECISION,
        settle DOUBLE PRECISION,
        pre_settle DOUBLE PRECISION,
        change1 DOUBLE PRECISION,
        change2 DOUBLE PRECISION,
        volume DOUBLE PRECISION,
        amount DOUBLE PRECISION,
        open_interest DOUBLE PRECISION,
        oi_change DOUBLE PRECISION,
        source TEXT NOT NULL DEFAULT 'tushare',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (trade_date, contract_code)
    );
    CREATE INDEX IF NOT EXISTS idx_futures_daily_product_date
        ON futures_daily (product_code, trade_date DESC);

    CREATE TABLE IF NOT EXISTS futures_mapping (
        trade_date DATE NOT NULL,
        continuous_code TEXT NOT NULL,
        product_code TEXT NOT NULL,
        mapping_contract_code TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'tushare',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (trade_date, continuous_code)
    );
    CREATE INDEX IF NOT EXISTS idx_futures_mapping_product_date
        ON futures_mapping (product_code, trade_date DESC);
    """
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(ddl)


def table_watermark(connection: Any, table: str) -> str | None:
    allowed = {"etf_daily", "futures_daily", "futures_mapping"}
    if table not in allowed:
        raise ValueError(f"Unsupported table: {table}")
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT MAX(trade_date) FROM {table}")
        value = cursor.fetchone()[0]
    return value.strftime("%Y%m%d") if value else None


def effective_start(requested: str, watermark: str | None) -> str:
    return max(requested, next_date(watermark)) if watermark else requested


def upsert_values(
    connection: Any,
    table: str,
    columns: list[str],
    rows: list[tuple[Any, ...]],
    conflict_columns: tuple[str, ...],
) -> int:
    if not rows:
        return 0
    updates = [column for column in columns if column not in conflict_columns]
    sql = f"""
        INSERT INTO {table} ({', '.join(columns)}) VALUES %s
        ON CONFLICT ({', '.join(conflict_columns)}) DO UPDATE SET
        {', '.join(f'{column}=EXCLUDED.{column}' for column in updates)},
        updated_at=NOW()
    """
    with connection:
        with connection.cursor() as cursor:
            execute_values(cursor, sql, rows, page_size=2000)
    return len(rows)


def fetch_etf_instruments(client: TushareClient) -> list[dict[str, Any]]:
    fields = (
        "ts_code",
        "csname",
        "cname",
        "extname",
        "index_code",
        "index_name",
        "setup_date",
        "list_date",
        "list_status",
        "exchange",
        "mgt",
        "custod",
        "fund_type",
    )
    etf_basic_available = True
    try:
        rows = client.query("etf_basic", {}, fields)
    except ValueError:
        etf_basic_available = False
        rows = client.query(
            "fund_basic",
            {"market": "E"},
            (
                "ts_code",
                "name",
                "management",
                "custodian",
                "benchmark",
                "fund_type",
                "status",
                "list_date",
                "delist_date",
            ),
        )
    result: list[dict[str, Any]] = []
    for row in rows:
        public_name = str(
            row.get("csname") or row.get("cname") or row.get("name") or ""
        ).upper()
        if not etf_basic_available and "ETF" not in public_name:
            continue
        symbol = normalize_etf_symbol(row.get("ts_code", ""))
        if not symbol:
            continue
        item = dict(row)
        item["symbol"] = symbol
        item["name"] = row.get("csname") or row.get("cname") or row.get("name")
        item["management"] = row.get("mgt") or row.get("management")
        item["custodian"] = row.get("custod") or row.get("custodian")
        item["benchmark"] = row.get("index_name") or row.get("benchmark")
        item["list_date"] = row.get("list_date") or row.get("setup_date")
        result.append(item)
    return sorted(result, key=lambda item: str(item["ts_code"]))


def fetch_etf_daily(
    client: TushareClient,
    instruments: list[dict[str, Any]],
    start_date: str,
    end_date: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], int]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    if (datetime.strptime(end_date, "%Y%m%d") - datetime.strptime(start_date, "%Y%m%d")).days <= 62:
        calendar = client.query(
            "trade_cal",
            {"exchange": "SSE", "start_date": start_date, "end_date": end_date},
        )
        trade_dates = [
            str(item["cal_date"])
            for item in calendar
            if str(item.get("is_open")) == "1"
        ]
        allowed_codes = {str(item["ts_code"]).upper() for item in instruments}
        for trade_date in trade_dates:
            try:
                daily = client.query("fund_daily", {"trade_date": trade_date})
                factors = client.query(
                    "fund_adj",
                    {"trade_date": trade_date},
                    ("ts_code", "trade_date", "adj_factor"),
                )
                factor_map = {
                    str(item.get("ts_code") or "").upper(): item.get("adj_factor")
                    for item in factors
                }
                for item in daily:
                    ts_code = str(item.get("ts_code") or "").upper()
                    if ts_code not in allowed_codes:
                        continue
                    item["symbol"] = normalize_etf_symbol(ts_code)
                    item["adj_factor"] = factor_map.get(ts_code)
                    if item["symbol"]:
                        rows.append(item)
            except Exception as exc:
                failures.append({"trade_date": trade_date, "error": str(exc)[:240]})
        return rows, failures, len(trade_dates)

    attempted = 0
    for index, instrument in enumerate(instruments, start=1):
        ts_code = str(instrument["ts_code"])
        listed = str(instrument.get("list_date") or start_date).replace("-", "")
        delisted = str(instrument.get("delist_date") or end_date).replace("-", "")
        left, right = max(start_date, listed), min(end_date, delisted)
        if left > right:
            continue
        attempted += 1
        try:
            daily = client.query(
                "fund_daily",
                {"ts_code": ts_code, "start_date": left, "end_date": right},
            )
            factors = client.query(
                "fund_adj",
                {"ts_code": ts_code, "start_date": left, "end_date": right},
                ("ts_code", "trade_date", "adj_factor"),
            )
            factor_map = {
                (str(item.get("ts_code")), str(item.get("trade_date"))): item.get(
                    "adj_factor"
                )
                for item in factors
            }
            for item in daily:
                item["symbol"] = normalize_etf_symbol(str(item.get("ts_code") or ""))
                item["adj_factor"] = factor_map.get(
                    (str(item.get("ts_code")), str(item.get("trade_date")))
                )
                if item["symbol"]:
                    rows.append(item)
        except Exception as exc:
            failures.append({"ts_code": ts_code, "error": str(exc)[:240]})
        if index % 100 == 0 or index == len(instruments):
            print(
                f"[ETF] {index}/{len(instruments)} rows={len(rows)} "
                f"failures={len(failures)}",
                flush=True,
            )
    return rows, failures, attempted


def fetch_futures_contracts(client: TushareClient) -> list[dict[str, Any]]:
    rows = client.query("fut_basic", {"exchange": "CFFEX"})
    result: list[dict[str, Any]] = []
    for row in rows:
        product = futures_product(str(row.get("ts_code") or ""), str(row.get("fut_code") or ""))
        if product:
            item = dict(row)
            item["product_code"] = product
            result.append(item)
    return result


def fetch_futures_daily(
    client: TushareClient,
    contracts: list[dict[str, Any]],
    start_date: str,
    end_date: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], int]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    if (datetime.strptime(end_date, "%Y%m%d") - datetime.strptime(start_date, "%Y%m%d")).days <= 62:
        calendar = client.query(
            "trade_cal",
            {"exchange": "CFFEX", "start_date": start_date, "end_date": end_date},
        )
        trade_dates = [
            str(item["cal_date"])
            for item in calendar
            if str(item.get("is_open")) == "1"
        ]
        for trade_date in trade_dates:
            try:
                daily = client.query("fut_daily", {"trade_date": trade_date})
                for item in daily:
                    product = futures_product(str(item.get("ts_code") or ""))
                    if product:
                        item["product_code"] = product
                        rows.append(item)
            except Exception as exc:
                failures.append({"trade_date": trade_date, "error": str(exc)[:240]})
        return rows, failures, len(trade_dates)

    attempted = 0
    for index, contract in enumerate(contracts, start=1):
        ts_code = str(contract.get("ts_code") or "").upper()
        listed = str(contract.get("list_date") or start_date).replace("-", "")
        delisted = str(contract.get("delist_date") or end_date).replace("-", "")
        left, right = max(start_date, listed), min(end_date, delisted)
        if left > right:
            continue
        attempted += 1
        try:
            daily = client.query(
                "fut_daily",
                {"ts_code": ts_code, "start_date": left, "end_date": right},
            )
            product = str(contract["product_code"])
            for item in daily:
                item["product_code"] = product
                rows.append(item)
        except Exception as exc:
            failures.append({"ts_code": ts_code, "error": str(exc)[:240]})
        if index % 40 == 0 or index == len(contracts):
            print(
                f"[FUTURES] {index}/{len(contracts)} rows={len(rows)} "
                f"failures={len(failures)}",
                flush=True,
            )
    return rows, failures, attempted


def fetch_futures_mapping(
    client: TushareClient, start_date: str, end_date: str
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for product in PRODUCTS:
        continuous = f"{product}.CFX"
        for left, right in yearly_ranges(start_date, end_date):
            try:
                batch = client.query(
                    "fut_mapping",
                    {
                        "ts_code": continuous,
                        "start_date": left,
                        "end_date": right,
                    },
                )
                for item in batch:
                    item["product_code"] = product
                    item["continuous_code"] = continuous
                    rows.append(item)
            except Exception as exc:
                failures.append(
                    {
                        "ts_code": continuous,
                        "range": f"{left}-{right}",
                        "error": str(exc)[:240],
                    }
                )
    return rows, failures


def etf_instrument_records(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            row["symbol"],
            row["ts_code"],
            row.get("name"),
            row.get("management"),
            row.get("custodian"),
            row.get("benchmark"),
            row.get("fund_type"),
            iso_date(row.get("list_date")),
            iso_date(row.get("delist_date")),
            "tushare",
        )
        for row in rows
    ]


def etf_daily_records(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    result: list[tuple[Any, ...]] = []
    for row in rows:
        trade_date = iso_date(row.get("trade_date"))
        symbol = row.get("symbol") or normalize_etf_symbol(str(row.get("ts_code") or ""))
        if not trade_date or not symbol:
            continue
        result.append(
            (
                trade_date,
                symbol,
                row.get("ts_code"),
                optional_float(row.get("open")),
                optional_float(row.get("high")),
                optional_float(row.get("low")),
                optional_float(row.get("close")),
                optional_float(row.get("pre_close")),
                optional_float(row.get("change")),
                optional_float(row.get("pct_chg")),
                optional_float(row.get("vol")),
                optional_float(row.get("amount")),
                optional_float(row.get("adj_factor")),
                "tushare",
            )
        )
    return result


def futures_contract_records(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            str(row.get("ts_code") or "").upper(),
            row["product_code"],
            str(row.get("exchange") or "CFFEX").upper(),
            row.get("name"),
            optional_float(row.get("multiplier")),
            row.get("trade_unit"),
            optional_float(row.get("per_unit")),
            row.get("quote_unit"),
            row.get("quote_unit_desc"),
            row.get("d_month"),
            iso_date(row.get("list_date")),
            iso_date(row.get("delist_date")),
            row.get("d_mode_desc"),
            row.get("trade_time_desc"),
            "tushare",
        )
        for row in rows
        if row.get("ts_code")
    ]


def futures_daily_records(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    result: list[tuple[Any, ...]] = []
    for row in rows:
        trade_date = iso_date(row.get("trade_date"))
        contract = str(row.get("ts_code") or "").upper()
        product = futures_product(contract, str(row.get("product_code") or ""))
        if not trade_date or not contract or not product:
            continue
        result.append(
            (
                trade_date,
                contract,
                product,
                "CFFEX",
                optional_float(row.get("open")),
                optional_float(row.get("high")),
                optional_float(row.get("low")),
                optional_float(row.get("close")),
                optional_float(row.get("pre_close")),
                optional_float(row.get("settle")),
                optional_float(row.get("pre_settle")),
                optional_float(row.get("change1")),
                optional_float(row.get("change2")),
                optional_float(row.get("vol")),
                optional_float(row.get("amount")),
                optional_float(row.get("oi")),
                optional_float(row.get("oi_chg")),
                "tushare",
            )
        )
    return result


def futures_mapping_records(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    result: list[tuple[Any, ...]] = []
    for row in rows:
        trade_date = iso_date(row.get("trade_date"))
        continuous = str(row.get("continuous_code") or row.get("ts_code") or "").upper()
        mapped = str(row.get("mapping_ts_code") or "").upper()
        product = futures_product(continuous, str(row.get("product_code") or ""))
        if trade_date and continuous and mapped and product:
            result.append((trade_date, continuous, product, mapped, "tushare"))
    return result


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def export_parquet(connection: Any, output_dir: Path) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas/pyarrow are required for Parquet export") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"generated_at": datetime.now().isoformat(), "tables": {}}
    date_tables = ("etf_daily", "futures_daily", "futures_mapping")
    for table in date_tables:
        years = pd.read_sql_query(
            f"SELECT DISTINCT EXTRACT(YEAR FROM trade_date)::INT AS year FROM {table} ORDER BY year",
            connection,
        )["year"].tolist()
        table_rows = 0
        partitions: list[dict[str, Any]] = []
        for year in years:
            frame = pd.read_sql_query(
                f"SELECT * FROM {table} WHERE trade_date >= %s AND trade_date < %s ORDER BY trade_date",
                connection,
                params=(f"{int(year)}-01-01", f"{int(year) + 1}-01-01"),
            )
            target = output_dir / table / f"year={int(year)}" / "data.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(".parquet.tmp")
            frame.to_parquet(temp, index=False)
            os.replace(temp, target)
            table_rows += len(frame)
            partitions.append({"year": int(year), "rows": len(frame), "path": str(target)})
        manifest["tables"][table] = {"rows": table_rows, "partitions": partitions}

    for table in ("etf_instruments", "futures_contracts"):
        frame = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY 1", connection)
        target = output_dir / f"{table}.parquet"
        temp = target.with_suffix(".parquet.tmp")
        frame.to_parquet(temp, index=False)
        os.replace(temp, target)
        manifest["tables"][table] = {"rows": len(frame), "path": str(target)}
    write_json_atomic(output_dir / "metadata.json", manifest)
    return manifest


def smoke_test(client: TushareClient, end_date: str) -> dict[str, Any]:
    start = (datetime.strptime(end_date, "%Y%m%d").date() - timedelta(days=14)).strftime(
        "%Y%m%d"
    )
    result: dict[str, Any] = {}
    try:
        instruments = fetch_etf_instruments(client)
        result["etf_instruments"] = {
            "success": True,
            "rows": len(instruments),
            "fallback_compatible": True,
        }
    except Exception as exc:
        result["etf_instruments"] = {"success": False, "error": str(exc)[:300]}

    calls = (
        ("trade_cal", {"exchange": "SSE", "start_date": start, "end_date": end_date}),
        ("fund_daily", {"ts_code": "510300.SH", "start_date": start, "end_date": end_date}),
        ("fund_adj", {"ts_code": "510300.SH", "start_date": start, "end_date": end_date}),
        ("fut_basic", {"exchange": "CFFEX"}),
        ("fut_daily", {"trade_date": end_date}),
        ("fut_mapping", {"ts_code": "IF.CFX", "start_date": start, "end_date": end_date}),
    )
    for api_name, params in calls:
        try:
            rows = client.query(api_name, params)
            result[api_name] = {
                "success": True,
                "rows": len(rows),
                "fields": sorted(rows[0].keys()) if rows else [],
            }
        except Exception as exc:
            result[api_name] = {"success": False, "error": str(exc)[:300]}
    result["success"] = all(item.get("success") for item in result.values())
    return result


def validate_coverage(total: int, failures: int, minimum: float, label: str) -> float:
    ratio = (total - failures) / total if total else 1.0
    if ratio < minimum:
        raise RuntimeError(f"{label} API coverage too low: {ratio:.2%} < {minimum:.2%}")
    return ratio


def run(args: argparse.Namespace) -> dict[str, Any]:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not configured")
    start_date, end_date = normalize_date(args.start_date), normalize_date(args.end_date)
    if start_date > end_date:
        raise ValueError("--start-date must not be after --end-date")
    if not 0 < args.minimum_success_ratio <= 1:
        raise ValueError("--minimum-success-ratio must be in (0, 1]")
    if args.apply and (args.max_etfs or args.max_contracts):
        raise ValueError("--max-etfs/--max-contracts are dry-run validation options")

    client = TushareClient(token=token, interval=max(args.request_interval, 0.0))
    if args.smoke_test:
        return smoke_test(client, end_date)

    connection = database_connection() if args.apply else None
    result: dict[str, Any] = {
        "success": True,
        "source": "tushare",
        "apply": bool(args.apply),
        "requested_start_date": start_date,
        "end_date": end_date,
    }
    try:
        if connection:
            ensure_schema(connection)

        if args.asset in ("all", "etf"):
            etf_start = start_date
            if connection:
                etf_start = effective_start(start_date, table_watermark(connection, "etf_daily"))
            instruments = fetch_etf_instruments(client)
            if args.max_etfs:
                instruments = instruments[: args.max_etfs]
            etf_rows: list[dict[str, Any]] = []
            etf_failures: list[dict[str, str]] = []
            etf_attempts = 0
            if etf_start <= end_date:
                etf_rows, etf_failures, etf_attempts = fetch_etf_daily(
                    client, instruments, etf_start, end_date
                )
            ratio = validate_coverage(
                etf_attempts, len(etf_failures), args.minimum_success_ratio, "ETF"
            )
            etf_written = 0
            if connection:
                upsert_values(
                    connection,
                    "etf_instruments",
                    [
                        "symbol",
                        "ts_code",
                        "name",
                        "management",
                        "custodian",
                        "benchmark",
                        "fund_type",
                        "list_date",
                        "delist_date",
                        "source",
                    ],
                    etf_instrument_records(instruments),
                    ("symbol",),
                )
                etf_written = upsert_values(
                    connection,
                    "etf_daily",
                    [
                        "trade_date",
                        "symbol",
                        "ts_code",
                        "open",
                        "high",
                        "low",
                        "close",
                        "pre_close",
                        "change",
                        "pct_chg",
                        "volume",
                        "amount",
                        "adj_factor",
                        "source",
                    ],
                    etf_daily_records(etf_rows),
                    ("trade_date", "symbol"),
                )
            result["etf"] = {
                "start_date": etf_start,
                "instruments": len(instruments),
                "rows": len(etf_rows),
                "requests": etf_attempts,
                "written": etf_written,
                "success_ratio": round(ratio, 6),
                "failure_samples": etf_failures[:20],
            }

        if args.asset in ("all", "futures"):
            daily_start, mapping_start = start_date, start_date
            if connection:
                daily_start = effective_start(
                    start_date, table_watermark(connection, "futures_daily")
                )
                mapping_start = effective_start(
                    start_date, table_watermark(connection, "futures_mapping")
                )
            contracts = fetch_futures_contracts(client)
            if args.max_contracts:
                contracts = contracts[: args.max_contracts]
            future_rows: list[dict[str, Any]] = []
            future_failures: list[dict[str, str]] = []
            future_attempts = 0
            mapping_rows: list[dict[str, Any]] = []
            mapping_failures: list[dict[str, str]] = []
            if daily_start <= end_date:
                future_rows, future_failures, future_attempts = fetch_futures_daily(
                    client, contracts, daily_start, end_date
                )
            if mapping_start <= end_date:
                mapping_rows, mapping_failures = fetch_futures_mapping(
                    client, mapping_start, end_date
                )
            ratio = validate_coverage(
                future_attempts,
                len(future_failures),
                args.minimum_success_ratio,
                "futures",
            )
            mapping_total = len(PRODUCTS) * max(len(yearly_ranges(mapping_start, end_date)), 1)
            mapping_ratio = validate_coverage(
                mapping_total,
                len(mapping_failures),
                args.minimum_success_ratio,
                "futures mapping",
            )
            daily_written = mapping_written = 0
            if connection:
                upsert_values(
                    connection,
                    "futures_contracts",
                    [
                        "contract_code",
                        "product_code",
                        "exchange",
                        "name",
                        "multiplier",
                        "trade_unit",
                        "per_unit",
                        "quote_unit",
                        "quote_unit_desc",
                        "delivery_month",
                        "list_date",
                        "delist_date",
                        "delivery_mode_desc",
                        "trade_time_desc",
                        "source",
                    ],
                    futures_contract_records(contracts),
                    ("contract_code",),
                )
                daily_written = upsert_values(
                    connection,
                    "futures_daily",
                    [
                        "trade_date",
                        "contract_code",
                        "product_code",
                        "exchange",
                        "open",
                        "high",
                        "low",
                        "close",
                        "pre_close",
                        "settle",
                        "pre_settle",
                        "change1",
                        "change2",
                        "volume",
                        "amount",
                        "open_interest",
                        "oi_change",
                        "source",
                    ],
                    futures_daily_records(future_rows),
                    ("trade_date", "contract_code"),
                )
                mapping_written = upsert_values(
                    connection,
                    "futures_mapping",
                    [
                        "trade_date",
                        "continuous_code",
                        "product_code",
                        "mapping_contract_code",
                        "source",
                    ],
                    futures_mapping_records(mapping_rows),
                    ("trade_date", "continuous_code"),
                )
            result["futures"] = {
                "daily_start_date": daily_start,
                "mapping_start_date": mapping_start,
                "contracts": len(contracts),
                "daily_rows": len(future_rows),
                "daily_requests": future_attempts,
                "mapping_rows": len(mapping_rows),
                "daily_written": daily_written,
                "mapping_written": mapping_written,
                "success_ratio": round(ratio, 6),
                "mapping_success_ratio": round(mapping_ratio, 6),
                "failure_samples": (future_failures + mapping_failures)[:20],
            }

        if connection and not args.skip_parquet:
            result["parquet"] = export_parquet(connection, Path(args.output_dir))
        return result
    finally:
        if connection:
            connection.close()


def main() -> int:
    args = parse_args()
    load_dotenv(project_root() / ".env", override=False)
    lock_path = Path(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"success": True, "skipped": True, "reason": "already_running"}))
            return 0
        try:
            result = run(args)
            print(json.dumps(result, ensure_ascii=False, default=str))
            return 0 if result.get("success") else 1
        except Exception as exc:
            print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
