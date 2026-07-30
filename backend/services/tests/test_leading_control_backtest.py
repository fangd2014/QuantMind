from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.leading_control_backtest import (
    _build_periods,
    calculate_performance_metrics,
    load_database_tail,
    point_in_time_memberships,
    run_monthly_backtest,
    validate_market_data,
    write_outputs,
)


def _market_rows() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2021-01-28",
            "2021-01-29",
            "2021-02-01",
            "2021-02-25",
            "2021-02-26",
            "2021-03-01",
            "2021-03-30",
            "2021-03-31",
            "2021-04-01",
        ]
    )
    rows: list[dict[str, object]] = []
    for offset, trade_date in enumerate(dates):
        for symbol, base in (("SH600001", 10.0), ("SZ000002", 20.0)):
            price = base + offset
            rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "stock_name": symbol,
                    "open": price,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price + 0.25,
                    "amount": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_point_in_time_memberships_respects_entry_and_exit_dates() -> None:
    memberships = {
        "801080.SI": [
            {
                "symbol": "600001.SH",
                "name": "老成员",
                "in_date": "20180101",
                "out_date": "20210215",
            },
            {
                "symbol": "000002.SZ",
                "name": "新成员",
                "in_date": "20210216",
                "out_date": None,
            },
            {
                "symbol": "300003.SZ",
                "name": "未来成员",
                "in_date": "20220101",
                "out_date": None,
            },
        ]
    }

    january = point_in_time_memberships(memberships, pd.Timestamp("2021-01-29"))
    march = point_in_time_memberships(memberships, pd.Timestamp("2021-03-31"))
    exit_day = point_in_time_memberships(memberships, pd.Timestamp("2021-02-15"))

    assert [item["symbol"] for item in january["801080.SI"]] == ["SH600001"]
    assert [item["symbol"] for item in march["801080.SI"]] == ["SZ000002"]
    assert "SH600001" not in {item["symbol"] for item in exit_day["801080.SI"]}


def test_validate_market_data_repairs_duplicates_and_invalid_rows() -> None:
    data = _market_rows()
    duplicate = data.iloc[[0]].copy()
    duplicate["close"] = 99.0
    invalid = data.iloc[[1]].copy()
    invalid["symbol"] = "SH600099"
    invalid["open"] = -1.0
    dirty = pd.concat([data, duplicate, invalid], ignore_index=True)

    clean, audit = validate_market_data(dirty, min_daily_count=1)

    assert not clean.duplicated(["trade_date", "symbol"]).any()
    assert "SH600099" not in set(clean["symbol"])
    repaired = clean[
        (clean["trade_date"] == data.iloc[0]["trade_date"])
        & (clean["symbol"] == data.iloc[0]["symbol"])
    ]
    assert repaired.iloc[0]["close"] == 99.0
    assert audit["duplicate_rows_removed"] == 1
    assert audit["invalid_price_rows_removed"] == 1


def test_monthly_backtest_uses_month_end_signal_and_never_future_data() -> None:
    market = _market_rows()
    market["adj_open"] = market["open"]
    market.loc[
        (market["trade_date"] == pd.Timestamp("2021-03-01"))
        & (market["symbol"] == "SH600001"),
        "adj_open",
    ] = 30.0
    benchmark = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2021-02-01", "2021-03-01", "2021-04-01"]),
            "open": [100.0, 110.0, 121.0],
        }
    )
    observed_history_max: list[pd.Timestamp] = []

    def signal_builder(history, _industries, _memberships, _max_stocks):
        observed_history_max.append(history["trade_date"].max())
        return {
            "picks": [
                {
                    "symbol": "SH600001",
                    "stock_name": "样本一",
                    "industry_code": "801080.SI",
                    "industry_name": "电子",
                    "stage": "洗盘",
                    "reason": "测试信号",
                }
            ],
            "market": {"regime": "分化"},
            "boards": [],
        }

    result = run_monthly_backtest(
        market,
        industries=[{"code": "801080.SI", "name": "电子"}],
        memberships={"801080.SI": [{"symbol": "SH600001"}]},
        start_date="2021-01-01",
        end_date="2021-04-01",
        transaction_cost_bps=15,
        signal_builder=signal_builder,
        benchmark=benchmark,
    )

    assert observed_history_max == [
        pd.Timestamp("2021-01-29"),
        pd.Timestamp("2021-02-26"),
    ]
    first = result["monthly"][0]
    assert first["signal_date"] == "2021-01-29"
    assert first["buy_date"] == "2021-02-01"
    assert first["sell_date"] == "2021-03-01"
    expected_gross = (30.0 / 12.0) - 1
    expected_net = (1 + expected_gross) * (1 - 0.0015) ** 2 - 1
    assert first["gross_return"] == pytest.approx(expected_gross)
    assert first["net_return"] == pytest.approx(expected_net)
    assert first["holding_count"] == 1
    assert first["benchmark_return"] == pytest.approx(0.10)


@pytest.mark.parametrize("regime", ["退潮", "过热"])
def test_monthly_backtest_holds_cash_in_prohibited_regimes(regime: str) -> None:
    def signal_builder(_history, _industries, _memberships, _max_stocks):
        return {
            "picks": [{"symbol": "SH600001", "stage": "洗盘"}],
            "market": {"regime": regime},
            "boards": [],
        }

    result = run_monthly_backtest(
        _market_rows(),
        industries=[],
        memberships={},
        start_date="2021-01-01",
        end_date="2021-03-01",
        signal_builder=signal_builder,
    )

    assert result["monthly"][0]["holding_count"] == 0
    assert result["monthly"][0]["net_return"] == 0
    assert result["monthly"][0]["cash_reason"] == regime


def test_monthly_backtest_enforces_portfolio_and_industry_caps() -> None:
    symbols = [f"SH600{index:03d}" for index in range(12)]
    market = _market_rows()
    template = market[market["symbol"] == "SH600001"].copy()
    expanded = []
    for index, symbol in enumerate(symbols):
        part = template.copy()
        part["symbol"] = symbol
        part[["open", "high", "low", "close"]] += index
        expanded.append(part)
    market = pd.concat(expanded, ignore_index=True)

    def signal_builder(_history, _industries, _memberships, _max_stocks):
        return {
            "picks": [
                {
                    "symbol": symbol,
                    "industry_code": f"I{index // 3}",
                    "industry_name": f"行业{index // 3}",
                    "stage": "开始拉升",
                    "selection_score": 100 - index,
                }
                for index, symbol in enumerate(symbols)
            ],
            "market": {"regime": "活跃"},
            "boards": [],
        }

    result = run_monthly_backtest(
        market,
        industries=[],
        memberships={},
        start_date="2021-01-01",
        end_date="2021-03-01",
        max_stocks=20,
        signal_builder=signal_builder,
    )

    holdings = result["holdings"]
    assert len(holdings) == 8
    assert (
        max(
            sum(row["industry_code"] == code for row in holdings)
            for code in {row["industry_code"] for row in holdings}
        )
        == 2
    )


def test_performance_metrics_include_drawdown_and_risk_statistics() -> None:
    monthly = pd.DataFrame(
        {
            "net_return": [0.10, -0.20, 0.05],
            "benchmark_return": [0.02, -0.03, 0.01],
            "holding_count": [3, 3, 0],
        }
    )

    metrics, curve = calculate_performance_metrics(monthly)

    assert metrics["total_return"] == pytest.approx(1.1 * 0.8 * 1.05 - 1)
    assert metrics["max_drawdown"] == pytest.approx(-0.20)
    assert metrics["invested_months"] == 2
    assert "sortino_ratio" in metrics
    assert "information_ratio" in metrics
    assert list(curve.columns) == [
        "strategy_equity",
        "benchmark_equity",
        "drawdown",
    ]


def test_drawdown_includes_loss_from_initial_capital() -> None:
    metrics, _curve = calculate_performance_metrics(
        pd.DataFrame(
            {
                "net_return": [-0.20],
                "benchmark_return": [0.0],
                "holding_count": [1],
            }
        )
    )

    assert metrics["max_drawdown"] == pytest.approx(-0.20)


def test_period_builder_produces_exactly_sixty_open_to_open_months() -> None:
    dates = pd.date_range("2021-06-01", "2026-07-01", freq="MS")
    stock = pd.DataFrame({"trade_date": dates, "symbol": "SH600001"})

    periods = _build_periods(stock, "2021-06-01", "2026-07-01")

    assert len(periods) == 60
    assert periods[0][1:] == (
        pd.Timestamp("2021-07-01"),
        pd.Timestamp("2021-08-01"),
        "2021-07",
    )
    assert periods[-1][1:] == (
        pd.Timestamp("2026-06-01"),
        pd.Timestamp("2026-07-01"),
        "2026-06",
    )


def test_database_tail_normalizes_postgres_dates_before_snapshot_merge() -> None:
    columns = [
        "trade_date",
        "symbol",
        "stock_name",
        "is_st",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pct_change",
        "factor",
        "turnover_rate",
        "float_mv",
        "total_mv",
        "limit_up_today",
        "limit_down_today",
    ]

    class Cursor:
        description = [(column,) for column in columns]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, *_args):
            return None

        def fetchall(self):
            return [
                (
                    date(2026, 7, 1),
                    "SH600001",
                    "样本",
                    0,
                    10.0,
                    10.5,
                    9.8,
                    10.2,
                    1000.0,
                    10000.0,
                    2.0,
                    1.2,
                    0.01,
                    1e9,
                    2e9,
                    0,
                    0,
                )
            ]

    class Connection:
        def cursor(self):
            return Cursor()

        def close(self):
            return None

    tail = load_database_tail("2026-06-30", "2026-07-01", connection_factory=Connection)

    assert tail.loc[0, "trade_date"] == pd.Timestamp("2026-07-01")
    assert tail.loc[0, "adj_open"] == pytest.approx(12.0)


def test_write_outputs_creates_complete_self_contained_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {
        "meta": {"title": "领先区控盘阶段五年回测", "generated_at": "2026-07-30"},
        "parameters": {"transaction_cost_bps_per_side": 15},
        "data_quality": {"status": "ok", "issues": []},
        "metrics": {"total_return": 0.2, "max_drawdown": -0.1},
        "equity_curve": [
            {
                "month": "2021-02",
                "strategy_equity": 1.02,
                "benchmark_equity": 1.01,
                "drawdown": 0.0,
            }
        ],
        "annual": [{"year": 2021, "return": 0.2}],
        "monthly_matrix": [{"year": 2021, "02": 0.02}],
        "industry_stage_attribution": [],
        "monthly": [
            {
                "holding_month": "2021-02",
                "net_return": 0.02,
                "benchmark_return": 0.01,
                "holding_count": 1,
            }
        ],
        "holdings": [
            {
                "holding_month": "2021-02",
                "symbol": "SH600001",
                "stock_name": "样本一",
                "industry_name": "电子",
                "stage": "洗盘",
                "net_return": 0.02,
            }
        ],
        "limitations": ["控盘为量价代理，不代表真实机构持仓。"],
    }

    def fake_pdf(_report, target):
        target.write_bytes(b"%PDF-1.4 test")

    monkeypatch.setattr(
        "scripts.analysis.leading_control_backtest.render_report_pdf", fake_pdf
    )
    paths = write_outputs(report, tmp_path)

    assert {"json", "monthly_csv", "holdings_csv", "html", "pdf"} <= set(paths)
    assert all(path.exists() for path in paths.values())
    assert "<script src=" not in paths["html"].read_text(encoding="utf-8")
    assert "领先区控盘阶段五年回测" in paths["html"].read_text(encoding="utf-8")
    serialized = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert serialized["parameters"]["transaction_cost_bps_per_side"] == 15
