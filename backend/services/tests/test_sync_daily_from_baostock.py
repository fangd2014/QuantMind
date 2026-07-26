from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.data.maintenance.sync_daily_from_baostock import (
    DailyRow,
    extend_bin_file,
    normalize_symbol,
    qlib_value,
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


if __name__ == "__main__":
    unittest.main()
