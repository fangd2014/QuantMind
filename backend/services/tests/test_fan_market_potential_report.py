from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analysis.fan_market_potential_report import (
    build_feishu_payload,
    evaluate_board_signal,
    select_stock_candidates,
    should_send,
)


def _daily_metrics() -> pd.DataFrame:
    rows = []
    for index, trade_date in enumerate(
        pd.date_range("2026-06-01", periods=24, freq="B")
    ):
        rows.append(
            {
                "trade_date": trade_date,
                "board_ret": 0.001,
                "amount": 100.0,
                "amount_ratio20": 1.0,
                "breadth_up": 0.58,
                "breadth_ma20": 0.62,
                "gain5_count": 2,
                "drop5_ratio": 0.0,
                "direction_proxy": 0.05,
                "leader_ret2": 0.04,
                "leader_above_ma5": True,
                "ex_leader_ret": 0.002,
                "drawdown10": -0.005,
                "relative_ret5": 0.01,
                "marker": index,
            }
        )
    return pd.DataFrame(rows)


def _make_startup(frame: pd.DataFrame, end_position: int) -> None:
    frame.loc[end_position - 1, ["board_ret", "amount", "amount_ratio20"]] = [
        0.015,
        180.0,
        1.20,
    ]
    frame.loc[
        end_position,
        [
            "board_ret",
            "amount",
            "amount_ratio20",
            "breadth_up",
            "breadth_ma20",
            "gain5_count",
            "leader_ret2",
            "ex_leader_ret",
        ],
    ] = [0.02, 200.0, 1.25, 0.70, 0.65, 4, 0.11, 0.012]


def test_two_day_broad_volume_expansion_enters_startup_observation() -> None:
    daily = _daily_metrics()
    _make_startup(daily, len(daily) - 1)

    result = evaluate_board_signal(daily)

    assert result["stage"] == "启动观察"
    assert result["score"] >= 70
    assert "首轮启动不追高" in result["reason"]


def test_contracted_pullback_with_leader_above_ma5_enters_restart_watch() -> None:
    daily = _daily_metrics()
    _make_startup(daily, 20)
    daily.loc[23, ["board_ret", "amount", "amount_ratio20"]] = [-0.01, 100.0, 0.75]
    daily.loc[23, ["breadth_ma20", "drop5_ratio", "drawdown10"]] = [0.58, 0.05, -0.04]

    result = evaluate_board_signal(daily)

    assert result["stage"] == "待二次启动"
    assert result["pullback_amount_ratio"] <= 0.75
    assert result["score"] >= 70


def test_volume_expanding_decline_is_rejected() -> None:
    daily = _daily_metrics()
    _make_startup(daily, 20)
    daily.loc[23, ["board_ret", "amount_ratio20"]] = [-0.02, 1.30]

    result = evaluate_board_signal(daily)

    assert result["stage"] == "淘汰"
    assert "放量下跌" in result["risk_flags"]


def test_leader_only_move_without_breadth_is_rejected() -> None:
    daily = _daily_metrics()
    _make_startup(daily, 23)
    daily.loc[23, "ex_leader_ret"] = -0.02

    result = evaluate_board_signal(daily)

    assert result["stage"] == "淘汰"
    assert "龙头独涨且扩散不足" in result["risk_flags"]


def _stock_row(
    symbol: str, name: str, *, amount_ratio: float = 0.8
) -> dict[str, object]:
    return {
        "trade_date": pd.Timestamp("2026-07-30"),
        "symbol": symbol,
        "stock_name": name,
        "is_st": 0,
        "pct_return": -0.005,
        "amount": 120_000_000 * amount_ratio,
        "amount_ma5_calc": 120_000_000,
        "amount_ma20_calc": 150_000_000,
        "turnover_rate": 5.0,
        "ret5_calc": -0.02,
        "ret20_calc": 0.12,
        "drawdown20": -0.06,
        "close": 10.2,
        "ma5_calc": 10.3,
        "ma20_calc": 9.8,
        "close_pos": 0.7,
        "obv_balance5": 0.05,
    }


def test_stock_selection_deduplicates_caps_and_filters_st_extremes() -> None:
    boards = [
        {
            "board_code": "801010.SI",
            "board_name": "农林牧渔",
            "stage": "待二次启动",
            "potential_score": 82.0,
        },
        {
            "board_code": "801080.SI",
            "board_name": "电子",
            "stage": "待二次启动",
            "potential_score": 80.0,
        },
    ]
    rows = [
        _stock_row("SH600001", "样本一"),
        _stock_row("SH600002", "样本二"),
        _stock_row("SH600003", "样本三"),
        _stock_row("SH600004", "ST风险"),
        _stock_row("SH600005", "极端股"),
    ]
    rows[-1]["pct_return"] = 0.10
    memberships = {
        "801010.SI": {"SH600001", "SH600002", "SH600003", "SH600004", "SH600005"},
        "801080.SI": {"SH600001", "SH600002", "SH600003"},
    }

    selected = select_stock_candidates(
        boards, pd.DataFrame(rows), memberships, max_stocks=10
    )

    assert len(selected) == 3
    board_counts = {
        code: sum(item["board_code"] == code for item in selected)
        for code in memberships
    }
    assert max(board_counts.values()) <= 2
    assert len(selected) <= 10
    assert len({item["symbol"] for item in selected}) == len(selected)
    assert all("ST" not in item["stock_name"] for item in selected)
    assert all(
        item["main_flow_state"] == "上涨/下跌成交额方向代理" for item in selected
    )


def test_feishu_content_states_cutoff_reasons_triggers_and_invalidations() -> None:
    report = {
        "title": "10:00 电风扇行情潜力关注池（基于上一交易日）",
        "data_cutoff": "2026-07-30",
        "boards": [
            {
                "board_name": "电子",
                "stage": "待二次启动",
                "potential_score": 82.0,
                "leader_symbol": "SH600001",
                "leader_name": "样本一",
                "reason": "缩量回调",
                "trigger": "盘中转强",
                "invalidation": "跌破MA5",
            }
        ],
        "stocks": [
            {
                "symbol": "SH600001",
                "stock_name": "样本一",
                "board_name": "电子",
                "setup_type": "洗盘低吸观察",
                "reason": "缩量守位",
                "trigger": "不过度高开",
                "invalidation": "跌破昨日低点",
            }
        ],
    }

    serialized = str(build_feishu_payload(report, "http://example/report.html"))

    assert "2026-07-30" in serialized
    assert "申万2021一级行业" in serialized
    assert "理由" in serialized
    assert "触发" in serialized
    assert "失效" in serialized
    assert "成交额方向代理" in serialized


def test_duplicate_data_date_is_suppressed_unless_forced(tmp_path: Path) -> None:
    marker = tmp_path / "last_sent.json"
    marker.write_text('{"data_cutoff":"2026-07-30"}', encoding="utf-8")

    assert should_send(marker, "2026-07-30") is False
    assert should_send(marker, "2026-07-30", force=True) is True
    assert should_send(marker, "2026-07-31") is True
