from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analysis.concept_rotation_report import (
    build_feishu_payload,
    build_leading_control_picks,
    build_stock_recommendations,
    classify_quadrant,
    load_preflight_status,
    load_sw_industry_universe,
    normalize_symbol,
    prepare_stock_history,
    render_interactive_html,
    render_report_pdf,
    write_outputs,
)


def _control_candidate(
    symbol: str,
    *,
    stage: str = "洗盘",
    stock_name: str | None = None,
) -> dict[str, object]:
    if stage == "开始拉升":
        close, high20, ret5, amount_ratio, close_pos = 11.0, 11.2, 0.06, 1.35, 0.82
        pct_return = 0.035
    else:
        close, high20, ret5, amount_ratio, close_pos = 10.2, 11.0, -0.02, 0.78, 0.62
        pct_return = -0.005
    return {
        "symbol": symbol,
        "stock_name": stock_name or f"样本{symbol[-2:]}",
        "is_st": 0,
        "close": close,
        "low": close - 0.3,
        "high": close + 0.2,
        "ma5_calc": 10.5 if stage == "开始拉升" else 10.1,
        "ma20_calc": 10.0,
        "ret5_calc": ret5,
        "ret20_calc": 0.16,
        "pct_return": pct_return,
        "amount": amount_ratio * 1_000_000,
        "amount_ma5_calc": 1_000_000,
        "amount_max20_calc": 1_600_000 if stage == "开始拉升" else 2_500_000,
        "ema12_calc": 10.7 if stage == "开始拉升" else 10.25,
        "ema36_calc": 9.9,
        "obv_balance5": 0.24 if stage == "开始拉升" else 0.08,
        "turnover_rate": 5.0,
        "limit_up_today": 0,
        "limit_down_today": 0,
        "high20_calc": high20,
        "low20_calc": 8.8,
        "up_down_amount_ratio20": 1.32,
        "trend_efficiency20": 0.28,
        "volatility20": 0.025,
        "range_pos20": (close - 8.8) / (high20 - 8.8),
        "drawdown20": close / high20 - 1,
        "close_pos": close_pos,
    }


def test_load_sw_industry_universe_uses_sw2021_l1_and_cache(
    tmp_path: Path,
) -> None:
    industries = [
        {
            "index_code": f"801{index:03d}.SI",
            "industry_name": f"申万行业{index}",
            "level": "L1",
            "src": "SW2021",
        }
        for index in range(1, 32)
    ]
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_query(
        api_name: str,
        params: dict[str, str],
        _fields: tuple[str, ...],
    ) -> list[dict[str, str]]:
        calls.append((api_name, params))
        if api_name == "index_classify":
            return industries
        return [
            {
                "l1_code": params["l1_code"],
                "ts_code": f"600{member:03d}.SH",
                "name": f"成分股{member}",
                "is_new": "Y",
            }
            for member in range(1, 9)
        ]

    boards, memberships = load_sw_industry_universe(
        tmp_path, cache_days=7, query=fake_query
    )

    assert len(boards) == 31
    assert boards[0]["code"] == "801001.SI"
    assert boards[0]["name"] == "申万行业1"
    assert len(memberships["801001.SI"]) == 8
    assert calls[0] == (
        "index_classify",
        {"level": "L1", "src": "SW2021"},
    )
    assert sum(name == "index_member_all" for name, _params in calls) == 31

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("fresh cache should avoid Tushare requests")

    cached_boards, cached_memberships = load_sw_industry_universe(
        tmp_path, cache_days=7, query=fail_if_called
    )

    assert cached_boards == boards
    assert cached_memberships == memberships


def test_normalize_symbol_uses_quantmind_prefix_format() -> None:
    assert normalize_symbol("sh600000") == "SH600000"
    assert normalize_symbol("600000.SH") == "SH600000"
    assert normalize_symbol("sz000001") == "SZ000001"
    assert normalize_symbol("430047") == "BJ430047"
    assert normalize_symbol("not-a-stock") is None


def test_load_preflight_status_requires_matching_passed_date(
    tmp_path: Path, monkeypatch
) -> None:
    status_path = tmp_path / "preflight.json"
    status_path.write_text(
        '{"passed":true,"final_audit":{"latest_date":"2026-07-29"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("CONCEPT_REPORT_PREFLIGHT_STATUS_FILE", str(status_path))

    status = load_preflight_status("2026-07-29")

    assert status is not None
    assert status["passed"] is True


def test_classify_quadrant() -> None:
    assert classify_quadrant(101, 102) == "领先区"
    assert classify_quadrant(99, 102) == "改善区"
    assert classify_quadrant(101, 99) == "转弱区"
    assert classify_quadrant(99, 99) == "落后区"


def test_stock_recommendations_filter_and_deduplicate() -> None:
    boards = pd.DataFrame(
        [
            {
                "code": "801080.SI",
                "name": "电子",
                "score": 91.0,
                "quadrant": "领先区",
                "breadth_delta5": 0.12,
                "eligible": True,
            },
            {
                "code": "801890.SI",
                "name": "机械设备",
                "score": 86.0,
                "quadrant": "改善区",
                "breadth_delta5": 0.08,
                "eligible": True,
            },
        ]
    )
    latest = pd.DataFrame(
        [
            {
                "symbol": "SH600001",
                "stock_name": "核心科技",
                "is_st": 0,
                "close": 12.0,
                "low": 10.0,
                "high": 12.2,
                "ma5_calc": 11.0,
                "ma20_calc": 10.5,
                "ret5_calc": 0.18,
                "ret20_calc": 0.18,
                "pct_return": 0.035,
                "amount": 2_000_000,
                "amount_ma5_calc": 1_400_000,
                "turnover_rate": 8.0,
                "limit_up_today": 0,
                "limit_down_today": 0,
            },
            {
                "symbol": "SZ000002",
                "stock_name": "趋势股份",
                "is_st": 0,
                "close": 10.4,
                "low": 10.0,
                "high": 10.8,
                "ma5_calc": 10.1,
                "ma20_calc": 9.9,
                "ret5_calc": 0.03,
                "ret20_calc": 0.09,
                "pct_return": 0.02,
                "amount": 900_000,
                "amount_ma5_calc": 1_000_000,
                "turnover_rate": 4.0,
                "limit_up_today": 0,
                "limit_down_today": 0,
            },
            {
                "symbol": "SZ000003",
                "stock_name": "ST风险",
                "is_st": 1,
                "close": 5.0,
                "low": 4.5,
                "high": 5.0,
                "ma5_calc": 4.7,
                "ma20_calc": 4.5,
                "ret5_calc": 0.02,
                "ret20_calc": 0.2,
                "pct_return": 0.05,
                "amount": 2_000_000,
                "amount_ma5_calc": 1_000_000,
                "turnover_rate": 10.0,
                "limit_up_today": 0,
                "limit_down_today": 0,
            },
        ]
    )
    memberships = {
        "801080.SI": {"SH600001", "SZ000002", "SZ000003"},
        "801890.SI": {"SH600001", "SZ000002"},
    }

    buy, watch = build_stock_recommendations(
        boards, latest, memberships, max_buy=5, max_watch=10
    )

    assert [item["symbol"] for item in buy] == ["SH600001"]
    assert buy[0]["industry_name"] == "电子"
    assert buy[0]["turnover_rate"] == 8.0
    assert [item["symbol"] for item in watch] == ["SZ000002"]
    assert "量能未达到1.05倍" in watch[0]["unmet_conditions"]
    assert all(item["symbol"] != "SZ000003" for item in buy + watch)


def test_prepare_stock_history_normalizes_baostock_units() -> None:
    frame = pd.DataFrame(
        [
            {
                "trade_date": date,
                "symbol": "600001.SH",
                "stock_name": "核心科技",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.0 + index * 0.1,
                "amount": 1_000_000,
                "pct_change": 0.02,
                "turnover_rate": 0.08,
                "float_mv": 0.0,
                "total_mv": 2_000_000_000.0,
            }
            for index, date in enumerate(pd.date_range("2026-06-01", periods=25))
        ]
    )

    prepared = prepare_stock_history(frame)

    assert prepared.iloc[-1]["symbol"] == "SH600001"
    assert prepared.iloc[-1]["pct_return"] == 0.02
    assert prepared.iloc[-1]["turnover_rate"] == 8.0
    assert prepared.iloc[-1]["market_weight"] == 2_000_000_000.0


def test_stock_recommendations_honor_empty_strict_eligibility() -> None:
    boards = pd.DataFrame(
        [
            {
                "code": "801080.SI",
                "name": "电子",
                "score": 91.0,
                "quadrant": "领先区",
                "breadth_delta5": 0.12,
                "eligible": False,
            }
        ]
    )
    latest = pd.DataFrame(
        [
            {
                "symbol": "SH600001",
                "stock_name": "核心科技",
                "is_st": 0,
                "close": 12.0,
                "low": 10.0,
                "high": 12.2,
                "ma5_calc": 11.0,
                "ma20_calc": 10.5,
                "ret5_calc": 0.18,
                "ret20_calc": 0.18,
                "pct_return": 0.035,
                "amount": 2_000_000,
                "amount_ma5_calc": 1_400_000,
                "turnover_rate": 8.0,
                "limit_up_today": 0,
                "limit_down_today": 0,
            }
        ]
    )

    buy, watch = build_stock_recommendations(
        boards, latest, {"801080.SI": {"SH600001"}}
    )

    assert buy == []
    assert watch == []


def test_leading_control_picks_only_include_controlled_wash_or_breakout() -> None:
    boards = pd.DataFrame(
        [
            {
                "code": "801080.SI",
                "name": "电子",
                "score": 91.0,
                "quadrant": "领先区",
                "rs_ratio": 108.0,
            },
            {
                "code": "801890.SI",
                "name": "机械设备",
                "score": 86.0,
                "quadrant": "改善区",
                "rs_ratio": 98.0,
            },
        ]
    )
    wash = _control_candidate("SH600001", stock_name="洗盘科技")
    breakout = _control_candidate("SZ000002", stage="开始拉升", stock_name="启动股份")
    improving = _control_candidate("SZ000003", stock_name="改善样本")
    broken_ma20 = _control_candidate("SH600004", stock_name="破位样本")
    broken_ma20["close"] = 9.8
    st_stock = _control_candidate("SH600005", stock_name="ST风险")
    st_stock["is_st"] = 1
    limit_stock = _control_candidate("SH600006", stock_name="涨停样本")
    limit_stock["limit_up_today"] = 1

    picks = build_leading_control_picks(
        boards,
        pd.DataFrame([wash, breakout, improving, broken_ma20, st_stock, limit_stock]),
        {
            "801080.SI": {
                "SH600001",
                "SZ000002",
                "SH600004",
                "SH600005",
                "SH600006",
            },
            "801890.SI": {"SZ000003"},
        },
    )

    assert {item["symbol"] for item in picks} == {"SH600001", "SZ000002"}
    assert {item["stage"] for item in picks} == {"洗盘", "开始拉升"}
    assert all(item["quadrant"] == "领先区" for item in picks)
    for item in picks:
        assert item["industry_name"] in item["reason"]
        assert "上涨/下跌成交额比" in item["reason"]
        assert item["stage"] in item["reason"]
        assert item["main_force_intent"] in {"洗盘吸筹", "主动拉升"}
        assert item["intent_confidence"] in {"中", "高"}
        assert "OBV方向代理" in item["intent_evidence"]
        assert "MA20" in item["invalidation"]


def test_leading_control_picks_exclude_distribution_risk() -> None:
    boards = pd.DataFrame(
        [
            {
                "code": "801080.SI",
                "name": "电子",
                "score": 91.0,
                "quadrant": "领先区",
                "rs_ratio": 108.0,
            }
        ]
    )
    candidate = _control_candidate(
        "SH600001", stage="开始拉升", stock_name="放量滞涨样本"
    )
    candidate.update(
        {
            "amount": 2_000_000,
            "amount_ma5_calc": 1_000_000,
            "pct_return": 0.01,
            "obv_balance5": -0.25,
            "range_pos20": 0.88,
        }
    )

    picks = build_leading_control_picks(
        boards,
        pd.DataFrame([candidate]),
        {"801080.SI": {"SH600001"}},
    )

    assert picks == []


def test_leading_control_picks_are_capped_at_ten_and_two_per_industry() -> None:
    board_rows = []
    memberships: dict[str, set[str]] = {}
    stock_rows = []
    for industry_index in range(6):
        code = f"801{industry_index:03d}.SI"
        board_rows.append(
            {
                "code": code,
                "name": f"领先行业{industry_index}",
                "score": 90.0 - industry_index,
                "quadrant": "领先区",
                "rs_ratio": 105.0 - industry_index / 10,
            }
        )
        symbols = set()
        for stock_index in range(3):
            symbol = f"SH60{industry_index:02d}{stock_index:02d}"
            symbols.add(symbol)
            stock_rows.append(
                _control_candidate(
                    symbol,
                    stage="开始拉升" if stock_index % 2 else "洗盘",
                )
            )
        memberships[code] = symbols

    picks = build_leading_control_picks(
        pd.DataFrame(board_rows),
        pd.DataFrame(stock_rows),
        memberships,
        limit=99,
    )

    assert len(picks) == 10
    counts = pd.Series([item["industry_code"] for item in picks]).value_counts()
    assert counts.max() <= 2


def test_feishu_payload_contains_report_links_and_candidates() -> None:
    report = {
        "market": {"latest_date": "2026-07-24", "regime": "活跃"},
        "top": [{"name": "电子", "quadrant": "领先区"}],
        "buy_candidates": [
            {
                "symbol": "SH600001",
                "stock_name": "核心科技",
                "industry_name": "电子",
            }
        ],
        "leading_control_picks": [
            {
                "symbol": "SZ000002",
                "stock_name": "启动股份",
                "industry_name": "电子",
                "stage": "开始拉升",
                "main_force_intent": "主动拉升",
                "intent_confidence": "高",
                "intent_evidence": "量比1.35、5日OBV方向代理+0.24",
                "reason": "电子位于领先区；控盘量价代理成立；开始拉升。",
                "invalidation": "跌破MA20时失效",
                "action_label": "次日关注",
            }
        ],
        "watchlist": [],
    }
    payload = build_feishu_payload(
        report,
        "http://example/report.pdf",
        "http://example/report.html",
    )

    assert payload["msg_type"] == "post"
    serialized = str(payload)
    assert "SH600001" in serialized
    assert "SZ000002" in serialized
    assert "开始拉升" in serialized
    assert "主力意图：主动拉升（高置信）" in serialized
    assert "意图依据" in serialized
    assert "控盘量价代理" in serialized
    assert "http://example/report.pdf" in serialized
    assert "http://example/report.html" in serialized
    assert "查看交互四象限" in serialized
    assert "不构成投资建议" in serialized


def test_render_interactive_html_contains_filters_and_board_data(
    tmp_path: Path,
) -> None:
    report = {
        "market": {"latest_date": "2026-07-24", "regime": "活跃"},
        "plot": [
            {
                "code": "801080.SI",
                "name": "电子",
                "score": 91.0,
                "quadrant": "领先区",
                "rs_ratio": 108.0,
                "rs_momentum": 105.0,
                "breadth20": 0.72,
                "breadth_delta5": 0.12,
                "member_count": 42,
                "mapped_count": 39,
                "coverage": 0.93,
                "leader_symbol": "SH600001",
                "leader_name": "核心科技",
                "leader_confirmed": True,
                "eligible": True,
            },
            {
                "code": "801890.SI",
                "name": "机械设备",
                "score": 80.0,
                "quadrant": "改善区",
                "rs_ratio": 96.0,
                "rs_momentum": 104.0,
                "breadth20": 0.64,
                "breadth_delta5": 0.08,
                "member_count": 33,
                "mapped_count": 30,
                "coverage": 0.91,
                "leader_symbol": "SZ000002",
                "leader_name": "趋势股份",
                "leader_confirmed": False,
                "eligible": False,
            },
        ],
        "focus_industries": [
            {
                "code": "801080.SI",
                "name": "电子",
                "quadrant": "领先区",
                "score": 91.0,
                "breadth20": 0.72,
                "breadth_delta5": 0.12,
                "coverage": 0.93,
                "leader_symbol": "SH600001",
                "leader_name": "核心科技",
                "leader_confirmed": True,
            },
            {
                "code": "801890.SI",
                "name": "机械设备",
                "quadrant": "改善区",
                "score": 80.0,
                "breadth20": 0.64,
                "breadth_delta5": 0.08,
                "coverage": 0.91,
                "leader_symbol": "SZ000002",
                "leader_name": "趋势股份",
                "leader_confirmed": False,
            },
        ],
        "leading_control_picks": [
            {
                "symbol": "SZ000002",
                "stock_name": "启动股份",
                "industry_name": "电子",
                "stage": "开始拉升",
                "control_score": 88.2,
                "reason": "电子位于领先区；20日上涨/下跌成交额比1.32，趋势效率0.28；开始拉升。",
                "invalidation": "跌破MA20时失效",
                "action_label": "次日关注",
            }
        ],
    }
    target = tmp_path / "report.html"

    render_interactive_html(report, target)

    html = target.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "申万一级行业 RRG 四象限" in html
    assert "电子" in html
    assert "核心科技" in html
    assert 'id="industry-list"' in html
    assert 'id="leading-control-picks"' in html
    assert "领先区控盘阶段关注" in html
    assert "20日上涨/下跌成交额比1.32" in html
    assert "领先区 / 改善区行业清单" in html
    assert 'id="board-search"' in html
    assert 'id="quadrant-filter"' in html
    assert 'id="board-detail"' in html
    assert "plotly_click" in html
    assert "Plotly.newPlot" in html
    assert 'src="https://cdn.plot.ly' not in html


def test_write_outputs_publishes_dated_and_latest_html(
    tmp_path: Path, monkeypatch
) -> None:
    report = {
        "market": {"latest_date": "2026-07-24"},
        "plot": [],
        "top": [],
        "focus_industries": [],
        "leading_control_picks": [],
        "buy_candidates": [],
        "watchlist": [],
    }
    boards = pd.DataFrame([{"code": "801080.SI", "score": 91.0}])

    def fake_pdf(_report, target: Path) -> None:
        target.write_bytes(b"%PDF-test")

    def fake_html(_report, target: Path) -> None:
        target.write_text("<!doctype html><title>test</title>", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.analysis.concept_rotation_report.render_report_pdf", fake_pdf
    )
    monkeypatch.setattr(
        "scripts.analysis.concept_rotation_report.render_interactive_html", fake_html
    )

    pdf_path, latest_pdf, html_path, latest_html = write_outputs(
        report, boards, tmp_path
    )

    assert pdf_path.name == "concept_rotation_20260724.pdf"
    assert html_path.name == "concept_rotation_20260724.html"
    assert latest_pdf.read_bytes() == pdf_path.read_bytes()
    assert latest_html.read_text(encoding="utf-8") == html_path.read_text(
        encoding="utf-8"
    )


def test_render_report_pdf_smoke(tmp_path: Path) -> None:
    report = {
        "market": {
            "latest_date": "2026-07-24",
            "stock_count": 3,
            "win_rate": 0.66,
            "median_pct": 0.012,
            "limit_up": 1,
            "limit_down": 0,
            "benchmark_5d": 0.03,
            "benchmark_20d": 0.06,
            "industry_count": 2,
            "regime": "活跃",
        },
        "top": [
            {
                "name": "电子",
                "score": 91.0,
                "quadrant": "领先区",
                "rs_ratio": 108.0,
                "rs_momentum": 105.0,
                "breadth20": 0.72,
                "breadth_delta5": 0.12,
                "leader_name": "核心科技",
                "leader_confirmed": True,
            }
        ],
        "plot": [
            {
                "name": "电子",
                "score": 91.0,
                "quadrant": "领先区",
                "rs_ratio": 108.0,
                "rs_momentum": 105.0,
            },
            {
                "name": "机械设备",
                "score": 80.0,
                "quadrant": "改善区",
                "rs_ratio": 96.0,
                "rs_momentum": 104.0,
            },
        ],
        "focus_industries": [
            {
                "code": "801080.SI",
                "name": "电子",
                "score": 91.0,
                "quadrant": "领先区",
                "breadth20": 0.72,
                "breadth_delta5": 0.12,
                "coverage": 0.93,
                "leader_symbol": "SH600001",
                "leader_name": "核心科技",
                "leader_confirmed": True,
            }
        ],
        "buy_candidates": [
            {
                "symbol": "SH600001",
                "stock_name": "核心科技",
                "industry_name": "电子",
                "stock_score": 92.0,
                "ret5": 0.12,
                "ret20": 0.18,
                "amount_ratio": 1.43,
                "turnover_rate": 8.0,
                "close_pos": 0.91,
                "trigger": "开盘涨幅不超过3%，且不跌破前一日低点或MA5",
                "invalidation": "高开超过5%或跌破前一日低点时取消买入",
            }
        ],
        "leading_control_picks": [
            {
                "symbol": "SZ000002",
                "stock_name": "启动股份",
                "industry_name": "电子",
                "industry_score": 91.0,
                "rs_ratio": 108.0,
                "stage": "开始拉升",
                "control_score": 88.2,
                "up_down_amount_ratio20": 1.32,
                "trend_efficiency20": 0.28,
                "amount_ratio": 1.35,
                "drawdown20": -0.018,
                "ret5": 0.06,
                "reason": "电子位于领先区；控盘量价代理成立；开始拉升。",
                "invalidation": "跌破MA20时失效",
                "action_label": "次日关注",
            }
        ],
        "watchlist": [],
        "method": {},
    }
    target = tmp_path / "report.pdf"

    render_report_pdf(report, target)

    assert target.exists()
    assert target.stat().st_size > 4_000
    assert target.read_bytes().startswith(b"%PDF")
