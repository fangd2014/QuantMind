from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.data.maintenance.sync_daily_from_baostock import (
    DailyRow,
    build_tushare_rows,
    extend_bin_file,
    fetch_tushare_daily_rows,
    normalize_symbol,
    qlib_value,
    run,
    resolve_requested_dates,
)


def _row() -> DailyRow:
    return DailyRow(
        trade_date="2026-07-24",
        symbol="SH600000",
        open=10.0,
        high=11.0,
        low=9.5,
        close=10.5,
        preclose=10.0,
        volume=1_000_000.0,
        amount=10_250_000.0,
        pct_change=5.0,
        turnover_rate=0.012,
        pe_ttm=8.0,
        pb=0.9,
        is_st=0,
    )


class SyncDailyFromBaostockTests(unittest.TestCase):
    def test_normalize_symbol_uses_prefix_format(self) -> None:
        self.assertEqual(normalize_symbol("sh.600000"), "SH600000")
        self.assertEqual(normalize_symbol("sz.000001"), "SZ000001")
        self.assertIsNone(normalize_symbol("600000.SH"))

    def test_qlib_value_preserves_raw_price_and_adjusted_close(self) -> None:
        row = _row()
        self.assertEqual(qlib_value(row, "close", 2.0), 10.5)
        self.assertEqual(qlib_value(row, "adjclose", 2.0), 21.0)
        self.assertEqual(qlib_value(row, "change", 2.0), 0.05)
        self.assertEqual(qlib_value(row, "volume", 2.0), 5_000.0)

    def test_extend_bin_file_keeps_start_index_and_appends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "close.day.bin"
            np.array([1.0, 8.0, 9.0], dtype="<f4").tofile(path)
            calendar = ["2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24"]

            changed = extend_bin_file(
                path,
                calendar,
                {"2026-07-24": _row()},
                factor=2.0,
                apply=True,
            )

            self.assertTrue(changed)
            actual = np.fromfile(path, dtype="<f4")
            np.testing.assert_allclose(
                actual,
                np.array([1.0, 8.0, 9.0, 10.5], dtype="<f4"),
            )

    def test_resolve_requested_dates_can_refresh_existing_database_window(
        self,
    ) -> None:
        class Result:
            error_code = "0"
            error_msg = ""

            def __init__(self) -> None:
                self.rows = iter(
                    [
                        ["2026-07-27", "0"],
                        ["2026-07-28", "1"],
                        ["2026-07-29", "1"],
                    ]
                )
                self.current = None

            def next(self) -> bool:
                self.current = next(self.rows, None)
                return self.current is not None

            def get_row_data(self):
                return self.current

        class Baostock:
            @staticmethod
            def query_trade_dates(**_kwargs):
                return Result()

        dates = resolve_requested_dates(
            Baostock(),
            ["2026-07-28", "2026-07-29"],
            "2026-07-29",
            refresh_days=3,
        )

        self.assertEqual(dates, ["2026-07-28", "2026-07-29"])

    def test_build_tushare_rows_normalizes_codes_units_and_st_names(self) -> None:
        rows = build_tushare_rows(
            daily_rows=[
                {
                    "ts_code": "600000.SH",
                    "trade_date": "20260731",
                    "open": 10,
                    "high": 11,
                    "low": 9,
                    "close": 10.5,
                    "pre_close": 10,
                    "pct_chg": 5,
                    "vol": 12_345,
                    "amount": 67_890,
                }
            ],
            daily_basic_rows=[
                {
                    "ts_code": "600000.SH",
                    "trade_date": "20260731",
                    "turnover_rate": 1.25,
                    "pe_ttm": 8.5,
                    "pb": 0.9,
                }
            ],
            stock_basic_rows=[{"ts_code": "600000.SH", "name": "*ST浦发"}],
            requested_date="2026-07-31",
            instruments={"SH600000"},
        )

        row = rows["SH600000"]["2026-07-31"]
        self.assertEqual(row.symbol, "SH600000")
        self.assertEqual(row.volume, 1_234_500)
        self.assertEqual(row.amount, 67_890_000)
        self.assertEqual(row.turnover_rate, 0.0125)
        self.assertEqual(row.is_st, 1)

    def test_run_uses_tushare_when_open_date_is_missing_from_baostock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            qlib_dir = self._write_qlib_fixture(Path(tmp))
            bs = self._fake_baostock(is_open=True, has_daily_bar=False)
            fallback_rows = {
                "SH600000": {
                    "2026-07-31": DailyRow(
                        trade_date="2026-07-31",
                        symbol="SH600000",
                        open=10,
                        high=11,
                        low=9,
                        close=10.5,
                        preclose=10,
                        volume=1_000_000,
                        amount=10_000_000,
                        pct_change=5,
                        turnover_rate=0.01,
                        pe_ttm=8,
                        pb=0.9,
                        is_st=0,
                    )
                }
            }
            captured: dict[str, object] = {}

            def fake_upsert(rows, _factors, apply):
                captured["rows"] = rows
                captured["apply"] = apply
                return 1

            with (
                patch.dict("sys.modules", {"baostock": bs}),
                patch.dict("os.environ", {"TUSHARE_TOKEN": "test-token"}),
                patch(
                    "scripts.data.maintenance.sync_daily_from_baostock."
                    "fetch_tushare_daily_rows",
                    return_value=fallback_rows,
                ) as fetch_fallback,
                patch(
                    "scripts.data.maintenance.sync_daily_from_baostock."
                    "upsert_database",
                    side_effect=fake_upsert,
                ),
            ):
                result = run(self._args(qlib_dir, apply=True, skip_qlib=True))

            self.assertEqual(result["source"], "tushare")
            self.assertEqual(result["date_end"], "2026-07-31")
            self.assertEqual(result["database_rows"], 1)
            self.assertIn("2026-07-31", captured["rows"]["SH600000"])
            self.assertTrue(captured["apply"])
            fetch_fallback.assert_called_once()

    def test_run_fails_when_open_date_is_missing_and_token_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            qlib_dir = self._write_qlib_fixture(Path(tmp))
            bs = self._fake_baostock(is_open=True, has_daily_bar=False)
            with (
                patch.dict("sys.modules", {"baostock": bs}),
                patch.dict("os.environ", {}, clear=True),
            ):
                with self.assertRaisesRegex(RuntimeError, "TUSHARE_TOKEN"):
                    run(self._args(qlib_dir))

    def test_run_falls_back_when_baostock_benchmark_exists_but_coverage_is_low(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            qlib_dir = self._write_qlib_fixture(Path(tmp))
            bs = self._fake_baostock(is_open=True, has_daily_bar=True)
            fallback_row = replace(_row(), trade_date="2026-07-31")
            with (
                patch.dict("sys.modules", {"baostock": bs}),
                patch.dict("os.environ", {"TUSHARE_TOKEN": "test-token"}),
                patch(
                    "scripts.data.maintenance.sync_daily_from_baostock."
                    "fetch_tushare_daily_rows",
                    return_value={"SH600000": {"2026-07-31": fallback_row}},
                ) as fetch_fallback,
            ):
                result = run(self._args(qlib_dir))

            self.assertEqual(result["source"], "baostock+tushare")
            self.assertEqual(result["success_ratio"], 1.0)
            fetch_fallback.assert_called_once()

    def test_run_can_skip_a_non_trading_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            qlib_dir = self._write_qlib_fixture(Path(tmp))
            bs = self._fake_baostock(is_open=False, has_daily_bar=False)
            with (
                patch.dict("sys.modules", {"baostock": bs}),
                patch.dict("os.environ", {}, clear=True),
            ):
                result = run(self._args(qlib_dir))

            self.assertTrue(result["skipped"])
            self.assertEqual(result["calendar_last_date"], "2026-07-30")

    def test_tushare_empty_daily_result_fails(self) -> None:
        def fake_query(_token, api_name, **_kwargs):
            if api_name == "stock_basic":
                return [{"ts_code": "600000.SH", "name": "浦发银行"}]
            return []

        with patch(
            "scripts.data.maintenance.sync_daily_from_baostock.query_tushare",
            side_effect=fake_query,
        ):
            with self.assertRaisesRegex(RuntimeError, "2026-07-31"):
                fetch_tushare_daily_rows(
                    "test-token", ["2026-07-31"], {"SH600000"}
                )

    @staticmethod
    def _write_qlib_fixture(root: Path) -> Path:
        qlib_dir = root / "qlib"
        (qlib_dir / "calendars").mkdir(parents=True)
        (qlib_dir / "instruments").mkdir()
        (qlib_dir / "features" / "sh600000").mkdir(parents=True)
        (qlib_dir / "calendars" / "day.txt").write_text(
            "2026-07-30\n", encoding="utf-8"
        )
        (qlib_dir / "instruments" / "all.txt").write_text(
            "SH600000\t2000-01-01\t2026-07-30\n", encoding="utf-8"
        )
        return qlib_dir

    @staticmethod
    def _args(qlib_dir: Path, **overrides) -> Namespace:
        values = {
            "target_date": "2026-07-31",
            "qlib_dir": str(qlib_dir),
            "max_symbols": 0,
            "apply": False,
            "skip_database": False,
            "skip_qlib": True,
            "refresh_days": 0,
            "minimum_success_ratio": 0.9,
        }
        values.update(overrides)
        return Namespace(**values)

    @staticmethod
    def _fake_baostock(is_open: bool, has_daily_bar: bool):
        class Result:
            error_code = "0"
            error_msg = ""

            def __init__(self, rows):
                self.rows = iter(rows)
                self.current = None

            def next(self):
                self.current = next(self.rows, None)
                return self.current is not None

            def get_row_data(self):
                return self.current

        class LoginResult:
            error_code = "0"
            error_msg = ""

        class Baostock:
            @staticmethod
            def login():
                return LoginResult()

            @staticmethod
            def logout():
                return LoginResult()

            @staticmethod
            def query_trade_dates(**_kwargs):
                rows = [["2026-07-31", "1"]] if is_open else []
                return Result(rows)

            @staticmethod
            def query_history_k_data_plus(*_args, **_kwargs):
                rows = [["2026-07-31", "10.5"]] if has_daily_bar else []
                return Result(rows)

        return Baostock


if __name__ == "__main__":
    unittest.main()
