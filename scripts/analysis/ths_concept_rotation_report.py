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
import re
import time
from collections.abc import Callable
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any

import requests

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
THS_CONCEPT_VERSION = "THS_A_CONCEPT_N_V2"
DEFAULT_OUTPUT_DIR = "/data/uploads/reports/ths-concept-rotation"
DEFAULT_CACHE_DIR = "/data/cache/ths-concept-rotation"
DEFAULT_PUBLIC_BASE_URL = "http://192.168.5.10:18000"
THS_PUBLIC_INDEX_URL = "https://q.10jqka.com.cn/gn/"
THS_PUBLIC_DETAIL_URL = (
    "https://q.10jqka.com.cn/gn/detail/page/{page}/code/{code}/"
)
PublicHttpGet = Callable[[str], bytes]
THS_REPORT_PROFILE = {
    "universe": "同花顺A股概念板块",
    "board_label": "同花顺概念",
    "board_kind": "概念板块",
    "quadrant_title": "同花顺概念板块 RRG 四象限",
    "report_title": "同花顺概念板块轮动日报",
    "report_heading": "QuantMind 同花顺概念板块轮动与次日候选",
    "membership_note": (
        "同花顺A股概念板块及当前成分（Tushare优先、同花顺公开页兜底），"
        "使用7日缓存并做覆盖率校验"
    ),
    "membership_risk": (
        "概念口径为同花顺A股概念板块，当前成分应用于历史窗口，"
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


def _decode_ths_page(content: bytes) -> str:
    """Decode Tonghuashun quote pages, which currently declare GBK."""
    charset_match = re.search(
        br"charset\s*=\s*[\"']?([a-zA-Z0-9_-]+)", content[:4096], re.IGNORECASE
    )
    declared = (
        charset_match.group(1).decode("ascii", errors="ignore").lower()
        if charset_match
        else ""
    )
    encodings = ["gb18030", "utf-8"]
    if declared in {"utf-8", "utf8"}:
        encodings = ["utf-8", "gb18030"]
    for encoding in encodings:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("gb18030", errors="replace")


def _plain_text(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", "", fragment)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def parse_ths_public_concepts(content: bytes) -> list[dict[str, Any]]:
    """Parse the public Tonghuashun concept directory without live-network tests."""
    page = _decode_ths_page(content)
    pattern = re.compile(
        r"<a\b[^>]*href=[\"'][^\"']*/gn/detail/code/(\d+)/?[^\"']*[\"']"
        r"[^>]*>(.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )
    concepts: dict[str, dict[str, Any]] = {}
    for route_code, raw_name in pattern.findall(page):
        name = _plain_text(raw_name)
        if not name:
            continue
        code = f"THS{route_code}"
        concepts[code] = {
            "code": code,
            "route_code": route_code,
            "name": name,
            "level": "concept",
            "source": "THS_PUBLIC",
            "reported_count": 0,
        }
    return sorted(concepts.values(), key=lambda item: item["code"])


def parse_ths_public_members(content: bytes) -> tuple[list[dict[str, Any]], int]:
    """Parse one public concept-member page and its total page count."""
    page = _decode_ths_page(content)
    main_match = re.search(
        r"<div\b[^>]*\bid=[\"']maincont[\"'][^>]*>(.*?)(?:</div>\s*</div>|$)",
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )
    main = main_match.group(1) if main_match else page
    body_match = re.search(
        r"<tbody\b[^>]*>(.*?)</tbody>",
        main,
        flags=re.IGNORECASE | re.DOTALL,
    )
    table_body = body_match.group(1) if body_match else ""
    members: dict[str, dict[str, Any]] = {}
    for raw_row in re.findall(
        r"<tr\b[^>]*>(.*?)</tr>", table_body, flags=re.IGNORECASE | re.DOTALL
    ):
        cells = re.findall(
            r"<td\b[^>]*>(.*?)</td>",
            raw_row,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if len(cells) < 3:
            continue
        symbol_match = re.search(r"\b(\d{6})\b", _plain_text(cells[1]))
        if not symbol_match:
            continue
        symbol = symbol_match.group(1)
        name = _plain_text(cells[2]) or symbol
        members[symbol] = {"symbol": symbol, "name": name}

    page_info = re.search(r"class=[\"']page_info[\"'][^>]*>\s*\d+\s*/\s*(\d+)", main)
    if page_info:
        page_count = max(1, int(page_info.group(1)))
    else:
        page_numbers = [
            int(value)
            for value in re.findall(r"\bpage=[\"'](\d+)[\"']", main)
        ]
        page_count = max(page_numbers, default=1)
    return sorted(members.values(), key=lambda item: item["symbol"]), page_count


def _default_ths_public_get() -> tuple[PublicHttpGet, Callable[[], None]]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "Chrome/126.0 Safari/537.36 QuantMind/1.0"
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": THS_PUBLIC_INDEX_URL,
        }
    )

    def get(url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = session.get(url, timeout=(10, 45))
                response.raise_for_status()
                if len(response.content) < 500:
                    raise RuntimeError(
                        "Tonghuashun returned an undersized response: "
                        f"{len(response.content)}"
                    )
                return response.content
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Tonghuashun request failed for {url}: {last_error}")

    return get, session.close


def _load_ths_public_universe(
    http_get: PublicHttpGet | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    close_http: Callable[[], None] = lambda: None
    if http_get is None:
        http_get, close_http = _default_ths_public_get()
        throttle_seconds = 0.12
    else:
        throttle_seconds = 0.0

    try:
        concepts = parse_ths_public_concepts(http_get(THS_PUBLIC_INDEX_URL))
        if len(concepts) < 50:
            raise ValueError(
                "Expected at least 50 Tonghuashun public concepts, "
                f"received {len(concepts)}"
            )

        memberships: dict[str, list[dict[str, Any]]] = {}
        usable_concepts: list[dict[str, Any]] = []
        for index, concept in enumerate(concepts, 1):
            route_code = str(concept["route_code"])
            first_url = THS_PUBLIC_DETAIL_URL.format(page=1, code=route_code)
            first_members, page_count = parse_ths_public_members(http_get(first_url))
            unique = {item["symbol"]: item for item in first_members}
            for page_number in range(2, page_count + 1):
                page_url = THS_PUBLIC_DETAIL_URL.format(
                    page=page_number, code=route_code
                )
                page_members, _ = parse_ths_public_members(http_get(page_url))
                unique.update({item["symbol"]: item for item in page_members})
                if throttle_seconds:
                    time.sleep(throttle_seconds)
            rows = sorted(unique.values(), key=lambda item: item["symbol"])
            if len(rows) >= 5:
                concept["reported_count"] = len(rows)
                concept.pop("route_code", None)
                usable_concepts.append(concept)
                memberships[concept["code"]] = rows
            if index % 25 == 0:
                LOGGER.info(
                    "Downloaded Tonghuashun public memberships %s/%s",
                    index,
                    len(concepts),
                )
            if throttle_seconds:
                time.sleep(throttle_seconds)

        minimum_successful = max(40, int(len(concepts) * 0.80))
        if len(usable_concepts) < minimum_successful:
            raise RuntimeError(
                f"Only {len(usable_concepts)}/{len(concepts)} Tonghuashun public "
                "memberships are usable"
            )
        return usable_concepts, memberships
    finally:
        close_http()


def _load_ths_tushare_universe(
    query_api: TushareQuery,
    add_rate_limit_delay: bool,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
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
                "source": "THS_TUSHARE",
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
        memberships[code] = sorted(unique.values(), key=lambda item: item["symbol"])
        concept["reported_count"] = len(memberships[code])
        if add_rate_limit_delay:
            time.sleep(0.22)
        if index % 25 == 0:
            LOGGER.info(
                "Downloaded Tonghuashun Tushare memberships %s/%s",
                index,
                len(industries),
            )

    successful = sum(len(rows) >= 5 for rows in memberships.values())
    minimum_successful = max(40, int(len(industries) * 0.80))
    if successful < minimum_successful:
        raise RuntimeError(
            f"Only {successful}/{len(industries)} Tonghuashun memberships are usable"
        )
    industries = [
        item for item in industries if len(memberships.get(item["code"], [])) >= 5
    ]
    memberships = {item["code"]: memberships[item["code"]] for item in industries}
    return industries, memberships


def load_ths_concept_universe(
    cache_dir: Path,
    cache_days: float = 7.0,
    query: TushareQuery | None = None,
    public_http_get: PublicHttpGet | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Load current A-share Tonghuashun concept boards and constituents."""
    cache_path = cache_dir / "ths-a-concept-universe.json.gz"
    stale_payload = _load_cache(cache_path)
    if stale_payload:
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days <= cache_days and len(stale_payload["industries"]) >= 50:
            LOGGER.info("Using %.1f-day-old Tonghuashun concept cache", age_days)
            return stale_payload["industries"], stale_payload["memberships"]

    try:
        query_api = query or query_tushare
        try:
            industries, memberships = _load_ths_tushare_universe(
                query_api, add_rate_limit_delay=query is None
            )
        except Exception:
            if query is not None and public_http_get is None:
                raise
            LOGGER.exception(
                "Tushare Tonghuashun interfaces failed; using public Tonghuashun pages"
            )
            industries, memberships = _load_ths_public_universe(public_http_get)
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
