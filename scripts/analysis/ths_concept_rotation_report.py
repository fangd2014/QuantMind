#!/usr/bin/env python3
"""Generate the daily Tonghuashun concept-board rotation report.

The scoring, RRG, leader confirmation, control proxy, washout/breakout filters,
data preflight gate and Feishu format are shared with the Shenwan report.  Only
the board universe changes to current A-share Tonghuashun concept constituents.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.analysis.concept_rotation_report import (
    TushareQuery,
    build_feishu_payload,
    build_report,
    load_preflight_status,
    load_stock_history,
    query_tushare,
    send_feishu,
    write_outputs,
)


LOGGER = logging.getLogger("ths_concept_rotation_report")
THS_CONCEPT_VERSION = "THS_A_CONCEPT_N_V1"
DEFAULT_OUTPUT_DIR = "/data/uploads/reports/ths-concept-rotation"
DEFAULT_CACHE_DIR = "/data/cache/ths-concept-rotation"
DEFAULT_PUBLIC_BASE_URL = "http://192.168.5.10:18000"
THS_REPORT_PROFILE = {
    "universe": "同花顺A股概念板块",
    "board_label": "同花顺概念",
    "board_kind": "概念板块",
    "quadrant_title": "同花顺概念板块 RRG 四象限",
    "report_title": "同花顺概念板块轮动日报",
    "report_heading": "QuantMind 同花顺概念板块轮动与次日候选",
    "membership_note": (
        "Tushare同花顺A股概念板块及当前成分，使用7日缓存并做覆盖率校验"
    ),
    "membership_risk": (
        "概念口径为Tushare同花顺A股概念板块，当前成分应用于历史窗口，"
        "概念重叠及成分调整可能带来幸存者偏差。"
    ),
    "output_prefix": "ths_concept_rotation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Tonghuashun concept rotation reports and notify Feishu"
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("THS_CONCEPT_REPORT_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("THS_CONCEPT_REPORT_CACHE_DIR", DEFAULT_CACHE_DIR),
    )
    parser.add_argument(
        "--public-base-url",
        default=os.getenv(
            "THS_CONCEPT_REPORT_PUBLIC_BASE_URL",
            os.getenv("CONCEPT_REPORT_PUBLIC_BASE_URL", DEFAULT_PUBLIC_BASE_URL),
        ),
    )
    parser.add_argument("--send-feishu", action="store_true")
    parser.add_argument("--max-buy", type=int, default=5)
    parser.add_argument("--max-watch", type=int, default=10)
    parser.add_argument("--max-control-picks", type=int, default=10)
    parser.add_argument("--member-cache-days", type=float, default=7.0)
    return parser.parse_args()


def _load_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        LOGGER.exception("Ignoring invalid Tonghuashun concept cache")
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("version") != THS_CONCEPT_VERSION
        or not isinstance(payload.get("industries"), list)
        or not isinstance(payload.get("memberships"), dict)
    ):
        return {}
    return payload


def load_ths_concept_universe(
    cache_dir: Path,
    cache_days: float = 7.0,
    query: TushareQuery | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Load current A-share Tonghuashun concept boards and constituents."""
    cache_path = cache_dir / "tushare-ths-a-concept-universe.json.gz"
    stale_payload = _load_cache(cache_path)
    if stale_payload:
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days <= cache_days and len(stale_payload["industries"]) >= 50:
            LOGGER.info("Using %.1f-day-old Tonghuashun concept cache", age_days)
            return stale_payload["industries"], stale_payload["memberships"]

    query_api = query or query_tushare
    try:
        classification = query_api(
            "ths_index",
            {"exchange": "A", "type": "N"},
            ("ts_code", "name", "count", "exchange", "list_date", "type"),
        )
        industries = sorted(
            [
                {
                    "code": str(item["ts_code"]),
                    "name": str(item["name"]),
                    "level": "concept",
                    "source": "THS",
                    "reported_count": int(item.get("count") or 0),
                }
                for item in classification
                if item.get("ts_code")
                and item.get("name")
                and str(item.get("type") or "N").upper() == "N"
                and str(item.get("exchange") or "A").upper() == "A"
                and int(item.get("count") or 0) >= 8
            ],
            key=lambda item: item["code"],
        )
        if len(industries) < 50:
            raise ValueError(
                "Expected at least 50 usable Tonghuashun concepts, "
                f"received {len(industries)}"
            )

        memberships: dict[str, list[dict[str, Any]]] = {}
        for index, concept in enumerate(industries, 1):
            code = concept["code"]
            rows = query_api(
                "ths_member",
                {"ts_code": code},
                (
                    "ts_code",
                    "con_code",
                    "con_name",
                    "weight",
                    "in_date",
                    "out_date",
                    "is_new",
                ),
            )
            current = [
                {
                    "symbol": str(item["con_code"]),
                    "name": str(item.get("con_name") or item["con_code"]),
                    "in_date": item.get("in_date"),
                    "out_date": item.get("out_date"),
                }
                for item in rows
                if item.get("con_code")
                and str(item.get("is_new") or "Y").upper() == "Y"
            ]
            unique = {item["symbol"]: item for item in current}
            memberships[code] = sorted(
                unique.values(), key=lambda item: item["symbol"]
            )
            concept["reported_count"] = len(memberships[code])
            if query is None:
                # ths_member is limited to 200 requests/minute. query_tushare
                # already waits 0.12s; this additional delay keeps the first
                # full refresh safely below that ceiling.
                time.sleep(0.22)
            if index % 25 == 0:
                LOGGER.info(
                    "Downloaded Tonghuashun memberships %s/%s",
                    index,
                    len(industries),
                )

        successful = sum(len(rows) >= 5 for rows in memberships.values())
        minimum_successful = max(40, int(len(industries) * 0.80))
        if successful < minimum_successful:
            raise RuntimeError(
                f"Only {successful}/{len(industries)} Tonghuashun memberships "
                "are usable"
            )
        industries = [
            item for item in industries if len(memberships.get(item["code"], [])) >= 5
        ]
        memberships = {
            item["code"]: memberships[item["code"]] for item in industries
        }
        payload = {
            "version": THS_CONCEPT_VERSION,
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
            LOGGER.exception(
                "Tonghuashun concept fetch failed; using stale membership cache"
            )
            return stale_payload["industries"], stale_payload["memberships"]
        raise


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
    LOGGER.info("Loading Tonghuashun A-share concepts and membership")
    industries, memberships = load_ths_concept_universe(
        cache_dir, cache_days=args.member_cache_days
    )
    LOGGER.info("Analyzing %s Tonghuashun concepts", len(industries))
    report, boards = build_report(
        stock,
        industries,
        memberships,
        max_buy=args.max_buy,
        max_watch=args.max_watch,
        max_control_picks=args.max_control_picks,
        report_profile=THS_REPORT_PROFILE,
    )
    report["data_preflight"] = load_preflight_status(
        str(report["market"]["latest_date"])
    )
    pdf_path, _latest_pdf_path, html_path, _latest_html_path = write_outputs(
        report, boards, output_dir
    )
    base_url = args.public_base_url.rstrip("/")
    uploads_root = output_dir.parents[1]
    pdf_url = f"{base_url}/uploads/{pdf_path.relative_to(uploads_root).as_posix()}"
    html_url = f"{base_url}/uploads/{html_path.relative_to(uploads_root).as_posix()}"
    LOGGER.info("Reports generated: PDF=%s HTML=%s", pdf_path, html_path)
    if args.send_feishu:
        webhook = os.getenv("THS_CONCEPT_WEB_HOOK", "").strip()
        webhook = webhook or os.getenv("WEB_HOOK", "").strip()
        if not webhook:
            raise RuntimeError(
                "THS_CONCEPT_WEB_HOOK and WEB_HOOK are empty; reports were generated"
            )
        send_feishu(webhook, build_feishu_payload(report, pdf_url, html_url))
        LOGGER.info("Feishu notification sent with PDF=%s HTML=%s", pdf_url, html_url)
    print(
        json.dumps(
            {
                "data_date": report["market"]["latest_date"],
                "concept_count": len(boards),
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
        LOGGER.exception("Daily Tonghuashun concept rotation report failed: %s", exc)
        raise SystemExit(1) from exc
