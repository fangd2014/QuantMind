#!/usr/bin/env python3
"""Update, audit and safely repair daily data before the rotation report.

The gate never deletes rows or guesses corrected values.  It first runs the
existing Baostock incremental updater, audits PostgreSQL against the Qlib
trading calendar, and only re-fetches a bounded recent window when the audit
finds blocking defects.  A failed repair blocks stock recommendations and can
send a Feishu alert through the already configured webhook.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg2


LOGGER = logging.getLogger("concept_rotation_preflight")
DEFAULT_QLIB_DIR = "/app/db/qlib_data"
DEFAULT_STATUS_FILE = "/data/cache/concept-rotation/preflight.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update and validate stock_daily_latest before daily reporting"
    )
    parser.add_argument(
        "--qlib-dir", default=os.getenv("QLIB_DATA_DIR", DEFAULT_QLIB_DIR)
    )
    parser.add_argument("--status-file", default=DEFAULT_STATUS_FILE)
    parser.add_argument("--minimum-rows", type=int, default=1000)
    parser.add_argument("--minimum-snapshot-ratio", type=float, default=0.85)
    parser.add_argument("--minimum-history-days", type=int, default=26)
    parser.add_argument("--repair-days", type=int, default=35)
    parser.add_argument("--skip-update", action="store_true")
    parser.add_argument("--skip-repair", action="store_true")
    parser.add_argument("--send-feishu-alerts", action="store_true")
    return parser.parse_args()


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


def expected_trade_date(qlib_dir: Path) -> str:
    calendar_path = qlib_dir / "calendars" / "day.txt"
    if not calendar_path.exists():
        raise FileNotFoundError(f"Qlib calendar not found: {calendar_path}")
    dates = [
        line.strip()
        for line in calendar_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not dates:
        raise RuntimeError("Qlib trading calendar is empty")
    return dates[-1]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def evaluate_quality(
    measurements: dict[str, Any],
    expected_date: str,
    minimum_rows: int = 1000,
    minimum_snapshot_ratio: float = 0.85,
    minimum_history_days: int = 26,
) -> tuple[list[str], list[str]]:
    """Turn database measurements into blocking issues and soft warnings."""
    issues: list[str] = []
    warnings: list[str] = []
    latest_date = str(measurements.get("latest_date") or "")
    latest_rows = int(measurements.get("latest_rows") or 0)
    previous_rows = int(measurements.get("previous_rows") or 0)
    history_days = int(measurements.get("history_days") or 0)
    snapshot_ratio = latest_rows / max(previous_rows, 1)
    if latest_date != expected_date:
        issues.append(
            f"最新数据日{latest_date or '缺失'}不等于应有交易日{expected_date}"
        )
    if latest_rows < minimum_rows:
        issues.append(f"最新截面仅{latest_rows}行，低于{minimum_rows}行")
    if previous_rows and snapshot_ratio < minimum_snapshot_ratio:
        issues.append(
            f"最新/前一交易日截面比{snapshot_ratio:.1%}低于{minimum_snapshot_ratio:.1%}"
        )
    if history_days < minimum_history_days:
        issues.append(f"仅有{history_days}个交易日，少于{minimum_history_days}日")
    blocking_counts = {
        "latest_duplicate_rows": "重复主键",
        "latest_invalid_symbol_rows": "证券代码格式异常",
        "latest_null_critical_rows": "关键行情字段缺失",
        "latest_invalid_ohlc_rows": "OHLC价格关系异常",
        "latest_negative_flow_rows": "成交量或成交额为负",
    }
    for key, label in blocking_counts.items():
        count = int(measurements.get(key) or 0)
        if count:
            issues.append(f"{label}{count}行")
    recent_invalid = int(measurements.get("recent_invalid_rows") or 0)
    latest_invalid = sum(int(measurements.get(key) or 0) for key in blocking_counts)
    if recent_invalid > latest_invalid:
        issues.append(
            f"最近{history_days}个交易日另有异常行情{recent_invalid - latest_invalid}行"
        )
    q99 = _finite(measurements.get("pct_change_q99"))
    if q99 > 0:
        normalized = q99 / 100 if q99 > 1 else q99
        if normalized > 0.12:
            warnings.append(f"涨跌幅99分位达到{normalized:.1%}，需关注单位或极端波动")
    amount_ratio = _finite(measurements.get("latest_market_amount_ratio"), 1.0)
    if amount_ratio and not 0.20 <= amount_ratio <= 5.0:
        warnings.append(f"最新全市场成交额为前日{amount_ratio:.2f}倍")
    suspended_rows = int(measurements.get("latest_suspended_rows") or 0)
    if suspended_rows:
        warnings.append(f"停牌或无成交{suspended_rows}行，已从选股计算中隔离")
    extreme_rows = int(measurements.get("recent_extreme_return_rows") or 0)
    if extreme_rows:
        warnings.append(
            f"最近{history_days}个交易日有{extreme_rows}行涨跌幅绝对值超过25%，"
            "可能为上市/退市等不设涨跌停事件，保留原值并由选股规则隔离"
        )
    return issues, warnings


def audit_database(
    expected_date: str,
    minimum_rows: int,
    minimum_snapshot_ratio: float,
    minimum_history_days: int,
) -> dict[str, Any]:
    connection = database_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT trade_date, COUNT(*)::bigint, COALESCE(SUM(amount), 0)
                FROM stock_daily_latest
                GROUP BY trade_date
                ORDER BY trade_date DESC
                LIMIT 31
                """
            )
            dates = cursor.fetchall()
            if not dates:
                raise RuntimeError("stock_daily_latest contains no rows")
            latest_date, latest_rows, latest_amount = dates[0]
            previous_rows = dates[1][1] if len(dates) > 1 else 0
            previous_amount = dates[1][2] if len(dates) > 1 else 0
            cursor.execute(
                """
                WITH latest AS (
                    SELECT *
                    FROM stock_daily_latest
                    WHERE trade_date = %s
                ), stats AS (
                    SELECT percentile_cont(0.99) WITHIN GROUP (
                        ORDER BY ABS(pct_change)
                    ) AS pct_change_q99
                    FROM latest
                )
                SELECT
                    COUNT(*) - COUNT(DISTINCT symbol) AS duplicate_rows,
                    COUNT(*) FILTER (
                        WHERE symbol !~ '^(SH|SZ|BJ)[0-9]{6}$'
                    ) AS invalid_symbol_rows,
                    COUNT(*) FILTER (
                        WHERE open IS NULL OR high IS NULL OR low IS NULL
                           OR close IS NULL
                           OR (
                                (amount IS NULL OR volume IS NULL)
                                AND NOT (open = high AND high = low AND low = close)
                           )
                    ) AS null_critical_rows,
                    COUNT(*) FILTER (
                        WHERE (amount IS NULL OR volume IS NULL)
                          AND open = high AND high = low AND low = close
                    ) AS suspended_rows,
                    COUNT(*) FILTER (
                        WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
                           OR high < GREATEST(open, close, low)
                           OR low > LEAST(open, close, high)
                    ) AS invalid_ohlc_rows,
                    COUNT(*) FILTER (
                        WHERE volume < 0 OR amount < 0
                    ) AS negative_flow_rows,
                    COUNT(*) FILTER (
                        WHERE ABS(pct_change) > CASE
                            WHEN stats.pct_change_q99 > 1 THEN 25.0 ELSE 0.25
                        END
                    ) AS extreme_return_rows,
                    stats.pct_change_q99
                FROM latest CROSS JOIN stats
                GROUP BY stats.pct_change_q99
                """,
                (latest_date,),
            )
            latest_quality = cursor.fetchone()
            recent_dates = [row[0] for row in dates[:minimum_history_days]]
            pct_change_threshold = 25.0 if _finite(latest_quality[7]) > 1 else 0.25
            cursor.execute(
                """
                SELECT
                COUNT(*) FILTER (
                    WHERE symbol !~ '^(SH|SZ|BJ)[0-9]{6}$'
                       OR open IS NULL OR high IS NULL OR low IS NULL
                       OR close IS NULL
                       OR (
                            (amount IS NULL OR volume IS NULL)
                            AND NOT (open = high AND high = low AND low = close)
                       )
                       OR open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
                       OR high < GREATEST(open, close, low)
                       OR low > LEAST(open, close, high)
                       OR volume < 0 OR amount < 0
                ),
                COUNT(*) FILTER (WHERE ABS(pct_change) > %s)
                FROM stock_daily_latest
                WHERE trade_date = ANY(%s)
                """,
                (pct_change_threshold, recent_dates),
            )
            recent_quality = cursor.fetchone()
            recent_invalid = int(recent_quality[0] or 0)
            recent_extreme = int(recent_quality[1] or 0)
    finally:
        connection.close()
    measurements = {
        "expected_date": expected_date,
        "latest_date": latest_date.isoformat(),
        "latest_rows": int(latest_rows),
        "previous_rows": int(previous_rows),
        "snapshot_ratio": int(latest_rows) / max(int(previous_rows), 1),
        "history_days": min(len(dates), minimum_history_days),
        "latest_duplicate_rows": int(latest_quality[0] or 0),
        "latest_invalid_symbol_rows": int(latest_quality[1] or 0),
        "latest_null_critical_rows": int(latest_quality[2] or 0),
        "latest_suspended_rows": int(latest_quality[3] or 0),
        "latest_invalid_ohlc_rows": int(latest_quality[4] or 0),
        "latest_negative_flow_rows": int(latest_quality[5] or 0),
        "latest_extreme_return_rows": int(latest_quality[6] or 0),
        "pct_change_q99": _finite(latest_quality[7]),
        "latest_market_amount_ratio": (
            _finite(latest_amount) / _finite(previous_amount, 1.0)
            if _finite(previous_amount) > 0
            else 1.0
        ),
        "recent_invalid_rows": recent_invalid,
        "recent_extreme_return_rows": recent_extreme,
    }
    issues, warnings = evaluate_quality(
        measurements,
        expected_date,
        minimum_rows=minimum_rows,
        minimum_snapshot_ratio=minimum_snapshot_ratio,
        minimum_history_days=minimum_history_days,
    )
    measurements["issues"] = issues
    measurements["warnings"] = warnings
    measurements["passed"] = not issues
    return measurements


def run_updater(refresh_days: int = 0) -> dict[str, Any]:
    updater = Path(__file__).with_name("sync_daily_from_baostock.py")
    command = [sys.executable, str(updater), "--apply"]
    if refresh_days > 0:
        command.extend(["--refresh-days", str(refresh_days), "--skip-qlib"])
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    output_lines = [line for line in result.stdout.splitlines() if line.strip()]
    payload: dict[str, Any] = {}
    if output_lines:
        try:
            payload = json.loads(output_lines[-1])
        except json.JSONDecodeError:
            payload = {"output_tail": output_lines[-1][-500:]}
    if result.returncode != 0:
        detail = payload.get("error") or result.stderr.strip()[-500:] or "unknown error"
        raise RuntimeError(f"Baostock update failed: {detail}")
    return payload


def send_feishu_alert(webhook: str, title: str, message: str) -> None:
    payload = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": title,
                    "content": [
                        [{"tag": "text", "text": message}],
                        [
                            {
                                "tag": "text",
                                "text": "已阻断当日选股推荐，请检查数据源与任务日志。",
                            }
                        ],
                    ],
                }
            }
        },
    }
    request = urllib.request.Request(
        webhook,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
    code = result.get("code", result.get("StatusCode"))
    if code not in (0, "0"):
        raise RuntimeError(f"Feishu alert rejected: {result}")


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temp_path.replace(path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    status: dict[str, Any] = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "updated": False,
        "repair_attempted": False,
        "repaired": False,
    }
    try:
        if not args.skip_update:
            status["update"] = run_updater()
            status["updated"] = not bool(status["update"].get("skipped"))
        expected = expected_trade_date(Path(args.qlib_dir))
        status["expected_date"] = expected
        audit = audit_database(
            expected,
            args.minimum_rows,
            args.minimum_snapshot_ratio,
            args.minimum_history_days,
        )
        status["initial_audit"] = audit
        if audit["issues"] and not args.skip_repair:
            status["repair_attempted"] = True
            status["repair"] = run_updater(refresh_days=args.repair_days)
            audit = audit_database(
                expected,
                args.minimum_rows,
                args.minimum_snapshot_ratio,
                args.minimum_history_days,
            )
            status["repaired"] = not bool(audit["issues"])
        status["final_audit"] = audit
        status["passed"] = not bool(audit["issues"])
        status["completed_at"] = (
            datetime.now().astimezone().isoformat(timespec="seconds")
        )
        write_status(Path(args.status_file), status)
        if not status["passed"]:
            message = "；".join(audit["issues"])
            if args.send_feishu_alerts and os.getenv("WEB_HOOK", "").strip():
                send_feishu_alert(
                    os.environ["WEB_HOOK"].strip(),
                    f"QuantMind 数据质量异常 {expected}",
                    message,
                )
            print(json.dumps(status, ensure_ascii=False, default=str))
            return 1
        print(json.dumps(status, ensure_ascii=False, default=str))
        return 0
    except Exception as exc:
        status["passed"] = False
        status["error"] = str(exc)
        status["completed_at"] = (
            datetime.now().astimezone().isoformat(timespec="seconds")
        )
        write_status(Path(args.status_file), status)
        LOGGER.exception("Daily data preflight failed")
        if args.send_feishu_alerts and os.getenv("WEB_HOOK", "").strip():
            try:
                send_feishu_alert(
                    os.environ["WEB_HOOK"].strip(),
                    "QuantMind 数据更新/质量门禁失败",
                    str(exc),
                )
            except (urllib.error.URLError, TimeoutError, OSError, RuntimeError):
                LOGGER.exception("Feishu preflight alert failed")
        print(json.dumps(status, ensure_ascii=False, default=str))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
