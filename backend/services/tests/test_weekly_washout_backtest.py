from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.weekly_washout_backtest import (
    BASELINE_CONFIG,
    CONFIGS,
    build_report,
    build_weekly_periods,
    filter_washout_picks,
    prepare_weekly_observations,
    simulate_weekly_strategy,
    walk_forward_optimize,
    write_outputs,
)


def _pick(
    symbol: str,
    *,
    stage: str = "洗盘",
    industry: str = "801080.SI",
    score: float = 90.0,
    amount_ratio: float = 0.8,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "stock_name": symbol,
        "industry_code": industry,
        "industry_name": "电子",
        "quadrant": "领先区",
        "stage": stage,
        "selection_score": score,
        "amount_ratio": amount_ratio,
        "drawdown20": -0.05,
        "ret5": -0.02,
        "up_down_amount_ratio20": 1.2,
        "trend_efficiency20": 0.2,
        "reason": "领先区、缩量洗盘并守住MA20",
        "invalidation": "跌破MA20失效",
    }


def _observation(
    index: int,
    picks: list[dict[str, object]],
    returns: dict[str, float],
    *,
    regime: str = "活跃",
) -> dict[str, object]:
    prices = {
        symbol: {"buy_open": 10.0, "sell_open": 10.0 * (1 + value)}
        for symbol, value in returns.items()
    }
    buy = pd.Timestamp("2025-01-06") + pd.Timedelta(weeks=index)
    return {
        "holding_week": buy.date().isoformat(),
        "signal_date": (buy - pd.Timedelta(days=3)).date().isoformat(),
        "buy_date": buy.date().isoformat(),
        "sell_date": (buy + pd.Timedelta(weeks=1)).date().isoformat(),
        "regime": regime,
        "market": {"regime": regime},
        "raw_picks": picks,
        "prices": prices,
        "benchmark_return": 0.001,
    }


def test_weekly_periods_use_last_close_then_two_future_first_opens() -> None:
    dates = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-10",
            "2025-01-13",
            "2025-01-17",
            "2025-01-20",
        ]
    )
    stock = pd.DataFrame({"trade_date": dates, "symbol": "SH600001"})

    periods = build_weekly_periods(stock, "2025-01-01", "2025-01-20")

    assert periods[0][:3] == (
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-06"),
        pd.Timestamp("2025-01-13"),
    )
    assert periods[1][:3] == (
        pd.Timestamp("2025-01-10"),
        pd.Timestamp("2025-01-13"),
        pd.Timestamp("2025-01-20"),
    )


def test_prepare_observations_never_passes_future_rows_to_signal() -> None:
    dates = pd.to_datetime(
        [
            "2025-01-03",
            "2025-01-06",
            "2025-01-10",
            "2025-01-13",
            "2025-01-17",
            "2025-01-20",
        ]
    )
    stock = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": "SH600001",
            "open": [10, 11, 12, 13, 14, 15],
            "high": [11, 12, 13, 14, 15, 16],
            "low": [9, 10, 11, 12, 13, 14],
            "close": [10, 11, 12, 13, 14, 15],
            "adj_open": [10, 11, 12, 13, 14, 15],
        }
    )
    benchmark = pd.DataFrame(
        {"trade_date": dates, "open": [100, 101, 102, 103, 104, 105]}
    )
    seen: list[pd.Timestamp] = []

    def builder(history, _industries, _memberships, _limit):
        seen.append(history["trade_date"].max())
        return {"picks": [_pick("SH600001")], "market": {"regime": "活跃"}}

    observations, errors = prepare_weekly_observations(
        stock,
        [],
        {},
        start_date="2025-01-01",
        end_date="2025-01-20",
        weeks=2,
        signal_builder=builder,
        benchmark=benchmark,
    )

    assert not errors
    assert seen == [pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-10")]
    assert observations[0]["buy_date"] == "2025-01-06"
    assert observations[0]["sell_date"] == "2025-01-13"


def test_filter_keeps_only_washout_and_enforces_industry_cap() -> None:
    picks = [
        _pick("SH600001", score=100),
        _pick("SH600002", score=99),
        _pick("SH600003", score=98),
        _pick("SZ000004", stage="开始拉升", industry="801750.SI", score=97),
        _pick("SZ000005", industry="801750.SI", score=96),
    ]

    selected = filter_washout_picks(picks, BASELINE_CONFIG, 10)

    assert [row["symbol"] for row in selected] == [
        "SH600001",
        "SH600002",
        "SZ000005",
    ]
    assert all(row["stage"] == "洗盘" for row in selected)


@pytest.mark.parametrize("regime", ["退潮", "过热", "数据异常"])
def test_forbidden_regime_holds_cash(regime: str) -> None:
    observation = _observation(
        0, [_pick("SH600001")], {"SH600001": 0.10}, regime=regime
    )

    weekly, holdings = simulate_weekly_strategy([observation])

    assert weekly.loc[0, "holding_count"] == 0
    assert weekly.loc[0, "net_return"] == 0
    assert weekly.loc[0, "cash_reason"] == regime
    assert holdings.empty


def test_adjusted_open_return_and_two_sided_cost_are_used() -> None:
    observation = _observation(0, [_pick("SH600001")], {"SH600001": 0.20})

    weekly, holdings = simulate_weekly_strategy([observation], transaction_cost_bps=15)

    expected = 1.2 * (1 - 0.0015) ** 2 - 1
    assert weekly.loc[0, "gross_return"] == pytest.approx(0.20)
    assert weekly.loc[0, "net_return"] == pytest.approx(expected)
    assert holdings.loc[0, "net_return"] == pytest.approx(expected)


def test_missing_exit_quote_is_conservatively_a_total_loss() -> None:
    observation = _observation(0, [_pick("SH600001")], {"SH600001": 0.20})
    observation["prices"]["SH600001"]["sell_open"] = None

    weekly, holdings = simulate_weekly_strategy([observation])

    assert weekly.loc[0, "net_return"] == -1
    assert holdings.loc[0, "exit_status"] == "退出日无报价，保守按全部损失"


@pytest.mark.parametrize("limit", [3, 5, 10])
def test_portfolio_size_limits_are_respected(limit: int) -> None:
    picks = [
        _pick(
            f"SH600{index:03d}",
            industry=f"I{index // 2}",
            score=100 - index,
        )
        for index in range(12)
    ]
    returns = {str(row["symbol"]): 0.01 for row in picks}

    weekly, holdings = simulate_weekly_strategy(
        [_observation(0, picks, returns)], max_stocks=limit
    )

    assert weekly.loc[0, "holding_count"] == limit
    assert len(holdings) == limit


def test_walk_forward_first_choice_is_unchanged_by_future_returns() -> None:
    defensive = _pick("SH600001", amount_ratio=0.8)
    baseline_only = _pick("SH600002", amount_ratio=1.0, score=80)
    observations = []
    for index in range(8):
        observations.append(
            _observation(
                index,
                [defensive, baseline_only],
                {"SH600001": 0.01, "SH600002": -0.08 if index < 4 else 0.02},
            )
        )

    original = walk_forward_optimize(
        observations, train_weeks=4, apply_weeks=2, candidate_configs=CONFIGS
    )
    changed = [dict(row) for row in observations]
    changed[-1] = {
        **changed[-1],
        "prices": {
            **changed[-1]["prices"],
            "SH600002": {"buy_open": 10.0, "sell_open": 1000.0},
        },
    }
    mutated = walk_forward_optimize(
        changed, train_weeks=4, apply_weeks=2, candidate_configs=CONFIGS
    )

    assert (
        original["selections"][0]["chosen_config"]
        == mutated["selections"][0]["chosen_config"]
    )
    assert original["selections"][0]["application_end"] < changed[-1]["holding_week"]


def test_report_outputs_are_self_contained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observations = [
        _observation(index, [_pick("SH600001")], {"SH600001": 0.01})
        for index in range(6)
    ]
    report = build_report(
        observations,
        data_quality={"status": "ok", "issues": []},
        signal_errors=[],
        train_weeks=2,
        apply_weeks=2,
    )

    def fake_pdf(_report, target):
        target.write_bytes(b"%PDF-1.4 test")

    monkeypatch.setattr("scripts.analysis.weekly_washout_backtest.render_pdf", fake_pdf)
    paths = write_outputs(report, tmp_path)

    assert all(path.exists() for path in paths.values())
    assert "Walk-forward 样本外对照" in paths["html"].read_text(encoding="utf-8")
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["parameters"]["frequency"] == "weekly"
    assert payload["metrics"]["weeks"] == 6


def test_period_builder_can_gate_exactly_fifty_two_complete_weeks() -> None:
    dates = pd.date_range("2025-01-03", periods=55, freq="W-FRI")
    stock = pd.DataFrame({"trade_date": dates, "symbol": "SH600001"})

    periods = build_weekly_periods(stock, dates.min(), dates.max(), weeks=52)

    assert len(periods) == 52
