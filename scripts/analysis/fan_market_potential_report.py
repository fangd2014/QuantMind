#!/usr/bin/env python3
"""Build the 10:00 fan-market potential watchlist from completed daily bars.

The job deliberately does not claim intraday confirmation.  It uses the last
complete trading day, identifies SW2021 level-one industries that either just
started with broad participation or completed a controlled pullback, and then
selects at most two liquid setups per industry for the next session's watchlist.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.analysis.concept_rotation_report import (
    load_stock_history,
    load_sw_industry_universe,
    normalize_symbol,
    prepare_stock_history,
    send_feishu,
)


LOGGER = logging.getLogger("fan_market_potential_report")
DEFAULT_OUTPUT_DIR = "/data/uploads/reports/fan-market"
DEFAULT_CACHE_DIR = "/data/cache/concept-rotation"
DEFAULT_PREFLIGHT_FILE = "/data/cache/fan-market/preflight.json"
DEFAULT_LAST_SENT_FILE = "/data/cache/fan-market/last_sent.json"
DEFAULT_PUBLIC_BASE_URL = "http://192.168.5.10:18000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the 10:00 SW-industry fan-market potential watchlist"
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--preflight-file", default=DEFAULT_PREFLIGHT_FILE)
    parser.add_argument("--last-sent-file", default=DEFAULT_LAST_SENT_FILE)
    parser.add_argument(
        "--public-base-url",
        default=os.getenv("FAN_MARKET_PUBLIC_BASE_URL", DEFAULT_PUBLIC_BASE_URL),
    )
    parser.add_argument("--max-boards", type=int, default=5)
    parser.add_argument("--max-stocks", type=int, default=10)
    parser.add_argument("--send-feishu", action="store_true")
    parser.add_argument("--force-send", action="store_true")
    return parser.parse_args()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _prepare_fan_features(stock: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_stock_history(stock)
    grouped = prepared.groupby("symbol", group_keys=False)
    prepared["ret2_calc"] = grouped["close"].pct_change(2, fill_method=None)
    prepared["amount_ma20_calc"] = grouped["amount"].transform(
        lambda values: values.rolling(20, min_periods=15).mean()
    )
    return prepared


def _startup_at(daily: pd.DataFrame, position: int) -> bool:
    if position < 1:
        return False
    previous = daily.iloc[position - 1]
    current = daily.iloc[position]
    cumulative = (1 + _number(previous["board_ret"])) * (
        1 + _number(current["board_ret"])
    ) - 1
    return bool(
        _number(previous["board_ret"]) > 0
        and _number(current["board_ret"]) > 0
        and _number(previous["amount_ratio20"]) >= 1.10
        and _number(current["amount_ratio20"]) >= 1.10
        and 0.02 <= cumulative <= 0.08
        and _number(current["breadth_up"]) >= 0.55
        and _number(current["breadth_ma20"]) >= 0.50
        and (
            _number(current["gain5_count"]) >= 3
            or _number(current["breadth_up"]) >= 0.65
        )
        and _number(current["leader_ret2"]) >= 0.08
        and _number(current["ex_leader_ret"]) > -0.005
    )


def evaluate_board_signal(daily: pd.DataFrame) -> dict[str, Any]:
    """Classify one industry from its daily aggregate metrics.

    This pure function is intentionally public so the hard gates remain easy to
    regression-test without a database or Tushare connection.
    """
    if len(daily) < 22:
        return {"stage": "数据不足", "score": 0.0, "reason": "有效历史不足22日"}
    frame = daily.reset_index(drop=True)
    latest_pos = len(frame) - 1
    latest = frame.iloc[latest_pos]
    hard_risks: list[str] = []
    if (
        _number(latest["board_ret"]) < -0.01
        and _number(latest["amount_ratio20"]) >= 1.20
    ):
        hard_risks.append("放量下跌")
    if not bool(latest.get("leader_above_ma5", False)):
        hard_risks.append("龙头跌破MA5")
    if _number(latest["drop5_ratio"]) > 0.20:
        hard_risks.append("板块跌超5%个股比例过高")
    if _number(latest["board_ret"]) > 0.05 and _number(latest["amount_ratio20"]) > 2.0:
        hard_risks.append("板块过热")
    if _number(latest["ex_leader_ret"]) <= -0.01:
        hard_risks.append("龙头独涨且扩散不足")
    if hard_risks:
        return {
            "stage": "淘汰",
            "score": 0.0,
            "reason": "；".join(hard_risks),
            "risk_flags": hard_risks,
        }

    startup_positions = [
        position
        for position in range(max(1, latest_pos - 5), latest_pos + 1)
        if _startup_at(frame, position)
    ]
    if not startup_positions:
        return {
            "stage": "未触发",
            "score": 0.0,
            "reason": "近6日未出现连续两日放量、上涨扩散和龙头共振",
            "risk_flags": [],
        }

    startup_pos = startup_positions[-1]
    startup = frame.iloc[startup_pos]
    if startup_pos == latest_pos:
        score = min(
            88.0,
            55
            + min(12, 8 * _number(latest["amount_ratio20"]))
            + min(10, 12 * _number(latest["breadth_up"]))
            + min(8, 40 * _number(latest["leader_ret2"]))
            + max(0, 5 * _number(latest["direction_proxy"])),
        )
        return {
            "stage": "启动观察",
            "score": round(score, 2),
            "startup_position": startup_pos,
            "reason": "连续两日放量上涨、上涨扩散与龙头强度共振；首轮启动不追高",
            "risk_flags": [],
        }

    peak_start = max(0, startup_pos - 1)
    peak_end = min(len(frame), startup_pos + 2)
    startup_peak_amount = frame.iloc[peak_start:peak_end]["amount"].max()
    pullback_ratio = _number(latest["amount"]) / _number(startup_peak_amount, 1.0)
    breadth_loss = _number(startup["breadth_ma20"]) - _number(latest["breadth_ma20"])
    direction_positive_days = int(
        (frame.tail(3)["direction_proxy"].fillna(0) > 0).sum()
    )
    pullback_ok = bool(
        -0.025 <= _number(latest["board_ret"]) <= 0.005
        and _number(latest["amount_ratio20"]) <= 0.90
        and pullback_ratio <= 0.75
        and bool(latest.get("leader_above_ma5", False))
        and _number(latest["drop5_ratio"]) <= 0.10
        and breadth_loss <= 0.15
        and -0.08 <= _number(latest["drawdown10"]) <= -0.01
        and direction_positive_days >= 1
    )
    if not pullback_ok:
        return {
            "stage": "继续观察",
            "score": 60.0,
            "startup_position": startup_pos,
            "reason": "启动已确认，但缩量回调、龙头守位或扩散稳定尚未全部满足",
            "risk_flags": [],
        }

    score = (
        50
        + min(15, max(0, (0.95 - _number(latest["amount_ratio20"])) * 30))
        + min(10, max(0, (0.80 - pullback_ratio) * 25))
        + min(10, max(0, (0.18 - breadth_loss) * 35))
        + min(10, max(0, _number(latest["relative_ret5"]) * 200 + 5))
        + min(5, direction_positive_days * 2)
    )
    return {
        "stage": "待二次启动",
        "score": round(min(100.0, score), 2),
        "startup_position": startup_pos,
        "pullback_amount_ratio": round(pullback_ratio, 4),
        "reason": "启动后缩量回调，龙头守住MA5，板块扩散未明显坍塌",
        "risk_flags": [],
    }


def _membership_maps(
    raw_memberships: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, set[str]], dict[str, str]]:
    memberships: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    for board_code, members in raw_memberships.items():
        normalized: set[str] = set()
        for member in members:
            symbol = normalize_symbol(str(member.get("symbol") or ""))
            if not symbol:
                continue
            normalized.add(symbol)
            name = str(member.get("name") or "").strip()
            if name:
                names[symbol] = name
        memberships[board_code] = normalized
    return memberships, names


def build_board_candidates(
    stock: pd.DataFrame,
    industries: list[dict[str, Any]],
    raw_memberships: dict[str, list[dict[str, Any]]],
    max_boards: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, set[str]], pd.DataFrame]:
    prepared = _prepare_fan_features(stock)
    memberships, member_names = _membership_maps(raw_memberships)
    missing_names = prepared["stock_name"].isna() | (
        prepared["stock_name"].astype(str).str.strip() == ""
    )
    prepared.loc[missing_names, "stock_name"] = prepared.loc[
        missing_names, "symbol"
    ].map(member_names)
    market_daily = prepared.groupby("trade_date")["pct_return"].mean().sort_index()
    market_ret5 = (1 + market_daily).rolling(5).apply(np.prod, raw=True) - 1
    board_results: list[dict[str, Any]] = []
    latest_date = prepared["trade_date"].max()

    for industry in industries:
        code = str(industry.get("code") or "")
        members = memberships.get(code, set())
        if len(members) < 5:
            continue
        frame = prepared[prepared["symbol"].isin(members)].copy()
        frame = frame[
            frame["is_st"].fillna(0).eq(0)
            & ~frame["stock_name"].fillna("").str.upper().str.contains("ST")
        ]
        latest_members = frame[frame["trade_date"] == latest_date]
        if len(latest_members) < 5:
            continue
        daily_rows: list[dict[str, Any]] = []
        for trade_date, day in frame.groupby("trade_date", sort=True):
            if len(day) < 5:
                continue
            leader_rank = (
                day["ret5_calc"].fillna(0) * 0.5
                + day["ret20_calc"].fillna(0) * 0.3
                + day["pct_return"].fillna(0) * 0.2
            )
            leader_index = leader_rank.idxmax()
            leader = day.loc[leader_index]
            non_leader = day.drop(index=leader_index)
            total_amount = _number(day["amount"].sum(min_count=1))
            signed_amount = (
                np.sign(day["pct_return"].fillna(0)) * day["amount"].fillna(0)
            ).sum()
            daily_rows.append(
                {
                    "trade_date": trade_date,
                    "board_ret": _number(day["pct_return"].mean()),
                    "amount": total_amount,
                    "breadth_up": _number((day["pct_return"] > 0).mean()),
                    "breadth_ma20": _number((day["close"] > day["ma20_calc"]).mean()),
                    "breadth_strong": _number(
                        (
                            (day["close"] > day["ma20_calc"]) & (day["ret5_calc"] > 0)
                        ).mean()
                    ),
                    "gain5_count": int((day["pct_return"] >= 0.05).sum()),
                    "drop5_ratio": _number((day["pct_return"] <= -0.05).mean()),
                    "limit_up_count": int((day["pct_return"] >= 0.095).sum()),
                    "direction_proxy": signed_amount / total_amount
                    if total_amount > 0
                    else 0.0,
                    "leader_symbol": str(leader["symbol"]),
                    "leader_name": str(leader.get("stock_name") or leader["symbol"]),
                    "leader_ret2": _number(leader["ret2_calc"]),
                    "leader_ret5": _number(leader["ret5_calc"]),
                    "leader_above_ma5": bool(
                        _number(leader["close"]) >= _number(leader["ma5_calc"])
                    ),
                    "ex_leader_ret": _number(non_leader["pct_return"].mean()),
                }
            )
        daily = (
            pd.DataFrame(daily_rows).sort_values("trade_date").reset_index(drop=True)
        )
        if len(daily) < 22:
            continue
        prior_amount = daily["amount"].shift(1).rolling(20, min_periods=15).mean()
        daily["amount_ratio20"] = (daily["amount"] / prior_amount).replace(
            [np.inf, -np.inf], np.nan
        )
        daily["board_index"] = (1 + daily["board_ret"].fillna(0)).cumprod()
        daily["board_ret5"] = daily["board_index"].pct_change(5)
        daily["drawdown10"] = (
            daily["board_index"] / daily["board_index"].rolling(10, min_periods=5).max()
            - 1
        )
        daily["relative_ret5"] = daily["board_ret5"] - daily["trade_date"].map(
            market_ret5
        )
        signal = evaluate_board_signal(daily)
        if signal["stage"] not in {"启动观察", "待二次启动"}:
            continue
        latest = daily.iloc[-1]
        board_results.append(
            {
                "board_code": code,
                "board_name": str(industry.get("name") or code),
                "board_source": "Tushare 申万2021一级行业",
                "stage": signal["stage"],
                "potential_score": signal["score"],
                "member_count": len(members),
                "mapped_count": len(latest_members),
                "coverage": round(len(latest_members) / len(members), 4),
                "board_ret_1d": round(_number(latest["board_ret"]), 6),
                "board_ret_5d": round(_number(latest["board_ret5"]), 6),
                "relative_ret_5d": round(_number(latest["relative_ret5"]), 6),
                "amount_ratio20": round(_number(latest["amount_ratio20"]), 4),
                "breadth_up": round(_number(latest["breadth_up"]), 4),
                "breadth_ma20": round(_number(latest["breadth_ma20"]), 4),
                "direction_proxy": round(_number(latest["direction_proxy"]), 4),
                "leader_symbol": str(latest["leader_symbol"]),
                "leader_name": str(latest["leader_name"]),
                "reason": signal["reason"],
                "trigger": "10:00后板块量价转强、上涨家数扩散，个股不过度高开",
                "invalidation": "龙头跌破昨日低点或MA5、板块放量长阴、跌超5%个股明显扩散",
            }
        )
    order = {"待二次启动": 0, "启动观察": 1}
    board_results.sort(
        key=lambda item: (order[item["stage"]], -_number(item["potential_score"]))
    )
    return board_results[: max(0, max_boards)], memberships, prepared


def select_stock_candidates(
    boards: list[dict[str, Any]],
    prepared: pd.DataFrame,
    memberships: dict[str, set[str]],
    max_stocks: int = 10,
) -> list[dict[str, Any]]:
    if not boards or prepared.empty:
        return []
    latest_date = prepared["trade_date"].max()
    latest = prepared[prepared["trade_date"] == latest_date].copy()
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for board in boards:
        if board["stage"] != "待二次启动":
            continue
        members = latest[
            latest["symbol"].isin(memberships.get(board["board_code"], set()))
        ]
        candidates: list[dict[str, Any]] = []
        for _, row in members.iterrows():
            symbol = str(row["symbol"])
            name = str(row.get("stock_name") or symbol)
            pct_return = _number(row["pct_return"])
            amount_ratio5 = _number(row["amount"] / row["amount_ma5_calc"])
            amount_ma20 = _number(row["amount_ma20_calc"])
            turnover = _number(row["turnover_rate"])
            ret5 = _number(row["ret5_calc"])
            ret20 = _number(row["ret20_calc"])
            drawdown = _number(row["drawdown20"])
            close = _number(row["close"])
            ma5 = _number(row["ma5_calc"])
            ma20 = _number(row["ma20_calc"])
            if (
                symbol in seen
                or _number(row.get("is_st")) != 0
                or "ST" in name.upper()
                or abs(pct_return) >= 0.095
                or amount_ma20 < 100_000_000
                or not 1.0 <= turnover <= 25.0
                or close <= ma20
                or not 0 <= ret20 <= 0.35
            ):
                continue
            washout = bool(
                -0.06 <= ret5 <= 0.03
                and -0.12 <= drawdown <= -0.02
                and 0.55 <= amount_ratio5 <= 0.95
                and close >= ma20
                and _number(row["close_pos"], 0.5) >= 0.50
                and _number(row["obv_balance5"]) >= -0.05
            )
            restart = bool(
                close > ma5 > ma20
                and 0.02 <= ret5 <= 0.15
                and -0.03 <= drawdown <= 0
                and 1.05 <= amount_ratio5 <= 2.0
                and 0 <= pct_return <= 0.07
                and _number(row["close_pos"], 0.5) >= 0.65
                and _number(row["obv_balance5"]) > 0.05
            )
            if not (washout or restart):
                continue
            setup = "洗盘低吸观察" if washout else "二次启动观察"
            score = (
                _number(board["potential_score"]) * 0.30
                + min(20, max(0, ret20 * 100))
                + min(15, max(0, (_number(row["obv_balance5"]) + 0.1) * 50))
                + (15 if washout or restart else 0)
                + min(10, amount_ma20 / 100_000_000)
                + min(10, max(0, _number(row["close_pos"]) * 10))
            )
            candidates.append(
                {
                    "symbol": symbol,
                    "stock_name": name,
                    "board_code": board["board_code"],
                    "board_name": board["board_name"],
                    "setup_type": setup,
                    "stock_score": round(min(100.0, score), 2),
                    "board_score": board["potential_score"],
                    "close": round(close, 3),
                    "return_5d": round(ret5, 6),
                    "return_20d": round(ret20, 6),
                    "drawdown_20d": round(drawdown, 6),
                    "amount_ratio5": round(amount_ratio5, 4),
                    "turnover_rate": round(turnover, 3),
                    "main_flow_state": "上涨/下跌成交额方向代理",
                    "reason": (
                        f"{board['board_name']}处于待二次启动；"
                        f"{setup}，20日均成交额{amount_ma20 / 100_000_000:.1f}亿元，"
                        f"近5日{ret5:.1%}、量比{amount_ratio5:.2f}"
                    ),
                    "trigger": "10:00后放量转强且高开不超过5%，不在快速拉升后追入",
                    "invalidation": "跌破昨日低点或MA20，或出现放量长阴",
                    "action_label": "关注",
                }
            )
        candidates.sort(key=lambda item: -_number(item["stock_score"]))
        for candidate in candidates[:2]:
            if len(selected) >= max_stocks:
                break
            seen.add(candidate["symbol"])
            selected.append(candidate)
        if len(selected) >= max_stocks:
            break
    return selected


def build_report(
    stock: pd.DataFrame,
    industries: list[dict[str, Any]],
    memberships: dict[str, list[dict[str, Any]]],
    max_boards: int = 5,
    max_stocks: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    boards, normalized_memberships, prepared = build_board_candidates(
        stock, industries, memberships, max_boards=max_boards
    )
    stocks = select_stock_candidates(
        boards, prepared, normalized_memberships, max_stocks=max_stocks
    )
    latest_date = pd.Timestamp(prepared["trade_date"].max()).strftime("%Y-%m-%d")
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_cutoff": latest_date,
        "title": "10:00 电风扇行情潜力关注池（基于上一交易日）",
        "universe": "Tushare 申万2021版一级行业",
        "intraday_confirmed": False,
        "boards": boards,
        "stocks": stocks,
        "limits": {
            "max_boards": max_boards,
            "max_stocks": max_stocks,
            "max_per_board": 2,
        },
        "data_notes": [
            "10:00任务仅使用上一完整交易日日线，候选必须等待盘中触发。",
            "真实主力净流入、市值容量和催化字段当前未覆盖，不编造；资金连续性使用上涨/下跌成交额方向代理。",
            "第一轮启动只列板块观察，不直接推荐追涨股票。",
        ],
        "disclaimer": "仅供量化研究和盘中观察，不构成投资建议。",
    }
    return report, boards, stocks


def build_feishu_payload(report: dict[str, Any], html_url: str) -> dict[str, Any]:
    board_lines = []
    for index, board in enumerate(report["boards"], 1):
        board_lines.append(
            f"{index}. {board['board_name']}｜{board['stage']}｜{board['potential_score']:.1f}分\n"
            f"龙头：{board['leader_symbol']} {board['leader_name']}\n"
            f"理由：{board['reason']}\n触发：{board['trigger']}\n失效：{board['invalidation']}"
        )
    stock_lines = []
    for index, item in enumerate(report["stocks"], 1):
        stock_lines.append(
            f"{index}. {item['symbol']} {item['stock_name']}｜{item['board_name']}｜{item['setup_type']}\n"
            f"理由：{item['reason']}\n触发：{item['trigger']}\n失效：{item['invalidation']}"
        )
    content: list[list[dict[str, str]]] = [
        [
            {
                "tag": "text",
                "text": f"数据截止：{report['data_cutoff']}｜口径：申万2021一级行业",
            }
        ],
        [
            {
                "tag": "text",
                "text": "潜力板块\n"
                + (
                    "\n\n".join(board_lines)
                    or "本次无符合硬门槛的板块，不降低标准凑数。"
                ),
            }
        ],
        [
            {
                "tag": "text",
                "text": "股票关注池\n"
                + (
                    "\n\n".join(stock_lines)
                    or "本次无完成缩量回调且满足流动性/形态门槛的股票。"
                ),
            }
        ],
        [{"tag": "a", "text": "查看完整 HTML 报告", "href": html_url}],
        [
            {
                "tag": "text",
                "text": "资金为成交额方向代理；10:00后仍需盘中确认，禁止无条件追高。仅供研究，不构成投资建议。",
            }
        ],
    ]
    return {
        "msg_type": "post",
        "content": {"post": {"zh_cn": {"title": report["title"], "content": content}}},
    }


def render_html(report: dict[str, Any], target: Path) -> None:
    board_rows = (
        "".join(
            "<tr>"
            f"<td>{escape(str(item['board_name']))}</td><td>{escape(str(item['stage']))}</td>"
            f"<td>{_number(item['potential_score']):.1f}</td><td>{_number(item['board_ret_1d']):.2%}</td>"
            f"<td>{_number(item['amount_ratio20']):.2f}</td><td>{_number(item['breadth_up']):.1%}</td>"
            f"<td>{escape(str(item['leader_symbol']))} {escape(str(item['leader_name']))}</td>"
            f"<td>{escape(str(item['reason']))}</td></tr>"
            for item in report["boards"]
        )
        or '<tr><td colspan="8">无符合硬门槛的板块</td></tr>'
    )
    stock_rows = (
        "".join(
            "<tr>"
            f"<td>{escape(str(item['symbol']))}</td><td>{escape(str(item['stock_name']))}</td>"
            f"<td>{escape(str(item['board_name']))}</td><td>{escape(str(item['setup_type']))}</td>"
            f"<td>{_number(item['stock_score']):.1f}</td><td>{escape(str(item['reason']))}</td>"
            f"<td>{escape(str(item['trigger']))}</td><td>{escape(str(item['invalidation']))}</td></tr>"
            for item in report["stocks"]
        )
        or '<tr><td colspan="8">无完成二次启动条件的股票</td></tr>'
    )
    target.write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(report["title"])}</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:#f4f6f9;color:#152235}}main{{max-width:1280px;margin:auto;padding:28px}}.hero{{background:linear-gradient(135deg,#102a43,#0e7490);color:white;padding:28px;border-radius:16px}}.note{{background:#fff7ed;border-left:4px solid #f59e0b;padding:14px;margin:18px 0}}section{{background:white;margin-top:18px;padding:20px;border-radius:14px;box-shadow:0 5px 20px #0f172a12}}input{{padding:10px;width:min(420px,90%);border:1px solid #ccd5df;border-radius:8px}}table{{width:100%;border-collapse:collapse;margin-top:12px;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #e5eaf0;text-align:left;vertical-align:top}}th{{background:#eef5f8;position:sticky;top:0}}.scroll{{overflow:auto}}small{{color:#64748b}}</style></head><body><main>
<div class="hero"><h1>{escape(report["title"])}</h1><p>数据截止 {escape(report["data_cutoff"])}｜{escape(report["universe"])}</p></div>
<div class="note">10:00 任务基于上一交易日完整日线，不代表盘中信号已确认；第一根大阳线只观察，不追高。</div>
<section><h2>潜力板块</h2><input id="q" placeholder="搜索板块、龙头或理由" oninput="filterRows()"><div class="scroll"><table id="boards"><thead><tr><th>板块</th><th>阶段</th><th>分数</th><th>日涨跌</th><th>20日量比</th><th>上涨宽度</th><th>龙头</th><th>理由</th></tr></thead><tbody>{board_rows}</tbody></table></div></section>
<section><h2>股票关注池（最多10只，每板块最多2只）</h2><div class="scroll"><table id="stocks"><thead><tr><th>代码</th><th>名称</th><th>板块</th><th>形态</th><th>分数</th><th>理由</th><th>盘中触发</th><th>失效</th></tr></thead><tbody>{stock_rows}</tbody></table></div></section>
<section><h2>口径限制</h2><p>{escape("；".join(report["data_notes"]))}</p><small>{escape(report["disclaimer"])}</small></section>
<script>function filterRows(){{const q=document.getElementById('q').value.toLowerCase();document.querySelectorAll('#boards tbody tr').forEach(r=>r.style.display=r.innerText.toLowerCase().includes(q)?'':'none')}}</script>
</main></body></html>""",
        encoding="utf-8",
    )


def write_outputs(
    report: dict[str, Any],
    boards: list[dict[str, Any]],
    stocks: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    token = report["data_cutoff"].replace("-", "")
    json_path = output_dir / f"fan_market_{token}.json"
    board_csv = output_dir / f"fan_market_boards_{token}.csv"
    stock_csv = output_dir / f"fan_market_stocks_{token}.csv"
    html_path = output_dir / f"fan_market_{token}.html"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(boards).to_csv(board_csv, index=False)
    pd.DataFrame(stocks).to_csv(stock_csv, index=False)
    render_html(report, html_path)
    shutil.copyfile(json_path, output_dir / "latest.json")
    shutil.copyfile(html_path, output_dir / "latest.html")
    return json_path, board_csv, stock_csv, html_path


def require_preflight(path: Path, data_cutoff: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    audit = payload.get("final_audit") or {}
    if not payload.get("passed") or str(audit.get("latest_date")) != data_cutoff:
        raise RuntimeError("数据门禁未通过或门禁日期与报告日期不一致")
    return payload


def should_send(last_sent_file: Path, data_cutoff: str, force: bool = False) -> bool:
    if force or not last_sent_file.exists():
        return True
    try:
        return (
            json.loads(last_sent_file.read_text(encoding="utf-8")).get("data_cutoff")
            != data_cutoff
        )
    except (OSError, json.JSONDecodeError):
        return True


def mark_sent(last_sent_file: Path, data_cutoff: str) -> None:
    last_sent_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = last_sent_file.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "data_cutoff": data_cutoff,
                "sent_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(last_sent_file)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stock = load_stock_history()
    industries, memberships = load_sw_industry_universe(Path(args.cache_dir))
    report, boards, stocks = build_report(
        stock, industries, memberships, args.max_boards, min(10, args.max_stocks)
    )
    report["data_preflight"] = require_preflight(
        Path(args.preflight_file), report["data_cutoff"]
    )
    json_path, board_csv, stock_csv, html_path = write_outputs(
        report, boards, stocks, Path(args.output_dir)
    )
    uploads_root = Path(args.output_dir).parents[1]
    html_url = (
        f"{args.public_base_url.rstrip('/')}/uploads/"
        f"{html_path.relative_to(uploads_root).as_posix()}"
    )
    sent = False
    if args.send_feishu and should_send(
        Path(args.last_sent_file), report["data_cutoff"], args.force_send
    ):
        webhook = (
            os.getenv("FAN_MARKET_WEB_HOOK", "").strip()
            or os.getenv("WEB_HOOK", "").strip()
        )
        if not webhook:
            raise RuntimeError("FAN_MARKET_WEB_HOOK and WEB_HOOK are empty")
        send_feishu(webhook, build_feishu_payload(report, html_url))
        mark_sent(Path(args.last_sent_file), report["data_cutoff"])
        sent = True
    print(
        json.dumps(
            {
                "data_cutoff": report["data_cutoff"],
                "boards": len(boards),
                "stocks": len(stocks),
                "json": str(json_path),
                "board_csv": str(board_csv),
                "stock_csv": str(stock_csv),
                "html": str(html_path),
                "html_url": html_url,
                "feishu_sent": sent,
                "duplicate_suppressed": bool(args.send_feishu and not sent),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOGGER.exception("Fan-market potential report failed: %s", exc)
        raise SystemExit(1) from exc
