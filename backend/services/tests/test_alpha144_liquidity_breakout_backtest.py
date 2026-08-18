from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.alpha144_liquidity_breakout_backtest import (
    BacktestConfig,
    _load_csi500_benchmark,
    calculate_daily_metrics,
    load_pit_csi500_memberships,
    prepare_alpha144_features,
    run_backtest,
)


def _market_frame(
    dates: pd.DatetimeIndex,
    *,
    limit_up_on_entry: bool = False,
    limit_down_indices: set[int] | None = None,
    breakout_indices: set[int] | None = None,
):
    breakout_indices = {0} if breakout_indices is None else breakout_indices
    limit_down_indices = set() if limit_down_indices is None else limit_down_indices
    rows = []
    for index, trade_date in enumerate(dates):
        if limit_up_on_entry and index == 1:
            open_price = 110.0
        elif index in limit_down_indices:
            open_price = 90.0
        else:
            open_price = 100.0
        rows.append(
            {
                "trade_date": trade_date,
                "symbol": "SH600000",
                "open": open_price,
                "high": open_price,
                "low": open_price,
                "close": open_price,
                "volume": 1_000_000.0,
                "factor": 1.0,
                "liq_amount": 100_000_000.0,
                "adj_open": open_price,
                "adj_high": open_price,
                "adj_low": open_price,
                "adj_close": open_price,
                "previous_adj_close": 100.0,
                "alpha144": 1.0,
                "prior_breakout_close": 99.0,
                "breakout": index in breakout_indices,
            }
        )
    return pd.DataFrame(rows)


def _benchmark_frame(dates: pd.DatetimeIndex, closes: np.ndarray | None = None):
    values = np.full(len(dates), 100.0) if closes is None else closes
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": values,
            "close": values,
            "factor": 1.0,
            "adj_open": values,
            "adj_close": values,
        }
    )


class Alpha144LiquidityBreakoutBacktestTests(unittest.TestCase):
    def test_alpha144_averages_down_days_and_breakout_excludes_today(self):
        dates = pd.date_range("2021-01-04", periods=5, freq="B")
        closes = [10.0, 9.0, 9.9, 8.91, 10.0]
        market = pd.DataFrame(
            {
                "trade_date": dates,
                "symbol": "SH600000",
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": 1_000.0,
                "factor": 1.0,
                "liq_amount": 100.0,
            }
        )

        prepared = prepare_alpha144_features(market, factor_window=3, breakout_window=2)

        self.assertAlmostEqual(float(prepared.loc[3, "alpha144"]), 0.001)
        self.assertAlmostEqual(float(prepared.loc[4, "prior_breakout_close"]), 9.9)
        self.assertTrue(bool(prepared.loc[4, "breakout"]))

    def test_membership_loader_queries_only_missing_snapshots(self):
        dates = [pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-18")]
        calls = []

        def query(trade_date: str):
            calls.append(trade_date)
            return trade_date, [
                {"symbol": f"sh.{600000 + index:06d}", "name": str(index)}
                for index in range(500)
            ]

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "members.json"
            first, audit = load_pit_csi500_memberships(dates, cache_path, query=query)
            second, second_audit = load_pit_csi500_memberships(
                dates, cache_path, query=query
            )

        self.assertEqual(len(first[dates[0]]), 500)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 2)
        self.assertEqual(audit["queried_snapshot_count"], 2)
        self.assertEqual(second_audit["queried_snapshot_count"], 0)

    def test_membership_loader_rejects_future_effective_date(self):
        dates = [pd.Timestamp("2021-01-04")]

        def query(_trade_date: str):
            return "2021-01-05", [
                {"symbol": f"sh.{600000 + index:06d}", "name": str(index)}
                for index in range(500)
            ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "after signal date"):
                load_pit_csi500_memberships(
                    dates, Path(temp_dir) / "members.json", query=query
                )

    def test_membership_loader_requires_exactly_five_hundred_members(self):
        dates = [pd.Timestamp("2021-01-04")]

        def query(trade_date: str):
            return trade_date, [
                {"symbol": f"sh.{600000 + index:06d}", "name": str(index)}
                for index in range(499)
            ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "499 unique members"):
                load_pit_csi500_memberships(
                    dates, Path(temp_dir) / "members.json", query=query
                )

    def test_benchmark_cache_uses_unadjusted_index_prices(self):
        rows = pd.DataFrame(
            {
                "trade_date": pd.date_range("2020-12-01", "2021-01-08", freq="B"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "preclose": 100.0,
                "pct_chg": 0.0,
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            frame, audit = _load_csi500_benchmark(
                Path(temp_dir) / "benchmark.csv",
                "2021-01-04",
                "2021-01-08",
                query=lambda _start, _end: rows,
            )

        self.assertTrue((frame["adj_close"] == frame["close"]).all())
        self.assertTrue(audit["benchmark_cache_refreshed"])

    def test_metrics_exclude_benchmark_return_before_strategy_start(self):
        curve = pd.DataFrame(
            {
                "equity": [100.0, 100.0],
                "strategy_return": [0.0, 0.0],
                "benchmark_return": [0.10, 0.0],
                "holding_count": [0, 0],
            }
        )

        metrics = calculate_daily_metrics(curve)

        self.assertAlmostEqual(metrics["benchmark_total_return"], 0.0)

    def test_orders_execute_next_open_and_exit_after_twenty_sessions(self):
        dates = pd.date_range("2021-01-04", periods=35, freq="B")
        config = BacktestConfig(
            start_date=str(dates[0].date()),
            end_date=str(dates[-1].date()),
            refresh_days=10,
            max_positions=1,
            holding_days=20,
            buy_cost_bps=0.0,
            sell_cost_bps=0.0,
        )
        memberships = {date: {"SH600000"} for date in dates[::10]}

        result = run_backtest(
            _market_frame(dates), _benchmark_frame(dates), memberships, config
        )
        trades = result["trades"]

        buy = trades[trades["side"] == "buy"].iloc[0]
        sell = trades[trades["side"] == "sell"].iloc[0]
        self.assertEqual(pd.Timestamp(buy["trade_date"]), dates[1])
        self.assertEqual(pd.Timestamp(sell["trade_date"]), dates[21])
        self.assertEqual(sell["holding_days"], 20)

    def test_limit_up_at_next_open_cancels_buy(self):
        dates = pd.date_range("2021-01-04", periods=25, freq="B")
        config = BacktestConfig(
            start_date=str(dates[0].date()),
            end_date=str(dates[-1].date()),
            refresh_days=10,
            max_positions=1,
        )
        memberships = {date: {"SH600000"} for date in dates[::10]}

        result = run_backtest(
            _market_frame(dates, limit_up_on_entry=True),
            _benchmark_frame(dates),
            memberships,
            config,
        )

        self.assertEqual(result["metrics"]["buy_count"], 0)
        self.assertEqual(result["metrics"]["limit_up_rejections"], 1)

    def test_factor_pool_refreshes_every_ten_days_but_scans_breakouts_daily(self):
        dates = pd.date_range("2021-01-04", periods=25, freq="B")
        config = BacktestConfig(
            start_date=str(dates[0].date()),
            end_date=str(dates[-1].date()),
            refresh_days=10,
            max_positions=1,
            buy_cost_bps=0.0,
            sell_cost_bps=0.0,
        )
        memberships = {date: {"SH600000"} for date in dates[::10]}

        result = run_backtest(
            _market_frame(dates, breakout_indices={3}),
            _benchmark_frame(dates),
            memberships,
            config,
        )

        buy = result["trades"][result["trades"]["side"] == "buy"].iloc[0]
        self.assertEqual(pd.Timestamp(buy["trade_date"]), dates[4])
        signal = (
            result["selections"]
            .loc[result["selections"]["signal_date"] == dates[3]]
            .iloc[0]
        )
        self.assertFalse(bool(signal["pool_refresh"]))
        self.assertEqual(signal["selected_symbols"], "SH600000")

    def test_market_filter_blocks_refresh_entries(self):
        all_dates = pd.date_range("2020-12-07", periods=45, freq="B")
        start = all_dates[20]
        backtest_dates = all_dates[20:]
        benchmark_closes = np.linspace(100.0, 90.0, len(all_dates))
        benchmark = _benchmark_frame(all_dates, benchmark_closes)
        config = BacktestConfig(
            start_date=str(start.date()),
            end_date=str(all_dates[-1].date()),
            refresh_days=10,
            max_positions=1,
        )
        memberships = {date: {"SH600000"} for date in backtest_dates[::10]}

        result = run_backtest(
            _market_frame(backtest_dates), benchmark, memberships, config
        )

        first_selection = result["selections"].iloc[0]
        self.assertFalse(bool(first_selection["market_open"]))
        self.assertEqual(first_selection["selected_count"], 0)

    def test_market_filter_liquidates_existing_position_next_open(self):
        all_dates = pd.date_range("2020-12-07", periods=45, freq="B")
        start = all_dates[20]
        backtest_dates = all_dates[20:]
        benchmark_closes = np.full(len(all_dates), 100.0)
        benchmark_closes[23:] = 90.0
        config = BacktestConfig(
            start_date=str(start.date()),
            end_date=str(all_dates[-1].date()),
            refresh_days=10,
            max_positions=1,
            buy_cost_bps=0.0,
            sell_cost_bps=0.0,
        )
        memberships = {date: {"SH600000"} for date in backtest_dates[::10]}

        result = run_backtest(
            _market_frame(backtest_dates),
            _benchmark_frame(all_dates, benchmark_closes),
            memberships,
            config,
        )

        sell = result["trades"][result["trades"]["side"] == "sell"].iloc[0]
        self.assertEqual(sell["reason"], "csi500_20d_below_threshold")
        self.assertEqual(pd.Timestamp(sell["trade_date"]), backtest_dates[4])

    def test_limit_down_delays_scheduled_exit_until_next_tradable_open(self):
        dates = pd.date_range("2021-01-04", periods=12, freq="B")
        config = BacktestConfig(
            start_date=str(dates[0].date()),
            end_date=str(dates[-1].date()),
            refresh_days=10,
            max_positions=1,
            holding_days=2,
            buy_cost_bps=0.0,
            sell_cost_bps=0.0,
        )
        memberships = {date: {"SH600000"} for date in dates[::10]}

        result = run_backtest(
            _market_frame(dates, limit_down_indices={3}),
            _benchmark_frame(dates),
            memberships,
            config,
        )

        sell = result["trades"][result["trades"]["side"] == "sell"].iloc[0]
        self.assertEqual(pd.Timestamp(sell["trade_date"]), dates[4])
        self.assertEqual(result["metrics"]["limit_down_retries"], 1)


if __name__ == "__main__":
    unittest.main()
