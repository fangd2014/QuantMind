from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analysis.concept_rotation_report import (
    build_feishu_payload,
    build_stock_recommendations,
    classify_quadrant,
    normalize_symbol,
    prepare_stock_history,
    render_interactive_html,
    render_report_pdf,
    write_outputs,
)


def test_normalize_symbol_uses_quantmind_prefix_format() -> None:
    assert normalize_symbol("sh600000") == "SH600000"
    assert normalize_symbol("600000.SH") == "SH600000"
    assert normalize_symbol("sz000001") == "SZ000001"
    assert normalize_symbol("430047") == "BJ430047"
    assert normalize_symbol("not-a-stock") is None


def test_classify_quadrant() -> None:
    assert classify_quadrant(101, 102) == "领先区"
    assert classify_quadrant(99, 102) == "改善区"
    assert classify_quadrant(101, 99) == "转弱区"
    assert classify_quadrant(99, 99) == "落后区"


def test_stock_recommendations_filter_and_deduplicate() -> None:
    boards = pd.DataFrame(
        [
            {
                "code": "gn_ai",
                "name": "人工智能",
                "score": 91.0,
                "quadrant": "领先区",
                "breadth_delta5": 0.12,
                "eligible": True,
            },
            {
                "code": "gn_robot",
                "name": "机器人",
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
        "gn_ai": {"SH600001", "SZ000002", "SZ000003"},
        "gn_robot": {"SH600001", "SZ000002"},
    }

    buy, watch = build_stock_recommendations(
        boards, latest, memberships, max_buy=5, max_watch=10
    )

    assert [item["symbol"] for item in buy] == ["SH600001"]
    assert buy[0]["concept_name"] == "人工智能"
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
                "code": "gn_ai",
                "name": "人工智能",
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

    buy, watch = build_stock_recommendations(boards, latest, {"gn_ai": {"SH600001"}})

    assert buy == []
    assert watch == []


def test_feishu_payload_contains_report_links_and_candidates() -> None:
    report = {
        "market": {"latest_date": "2026-07-24", "regime": "活跃"},
        "top": [{"name": "人工智能", "quadrant": "领先区"}],
        "buy_candidates": [
            {
                "symbol": "SH600001",
                "stock_name": "核心科技",
                "concept_name": "人工智能",
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
                "code": "gn_ai",
                "name": "人工智能",
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
                "code": "gn_robot",
                "name": "机器人",
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
    }
    target = tmp_path / "report.html"

    render_interactive_html(report, target)

    html = target.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "人工智能" in html
    assert "核心科技" in html
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
        "buy_candidates": [],
        "watchlist": [],
    }
    boards = pd.DataFrame([{"code": "gn_ai", "score": 91.0}])

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
            "concept_count": 2,
            "regime": "活跃",
        },
        "top": [
            {
                "name": "人工智能",
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
                "name": "人工智能",
                "score": 91.0,
                "quadrant": "领先区",
                "rs_ratio": 108.0,
                "rs_momentum": 105.0,
            },
            {
                "name": "机器人",
                "score": 80.0,
                "quadrant": "改善区",
                "rs_ratio": 96.0,
                "rs_momentum": 104.0,
            },
        ],
        "buy_candidates": [
            {
                "symbol": "SH600001",
                "stock_name": "核心科技",
                "concept_name": "人工智能",
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
        "watchlist": [],
        "method": {},
    }
    target = tmp_path / "report.pdf"

    render_report_pdf(report, target)

    assert target.exists()
    assert target.stat().st_size > 4_000
    assert target.read_bytes().startswith(b"%PDF")
