from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "data"
    / "maintenance"
    / "build_model_qlib_features_incremental.py"
)
SPEC = importlib.util.spec_from_file_location("model_qlib_incremental", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _panel(days: int = 130) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=days)
    rows = []
    for symbol, offset in (("SH600000", 0.0), ("SZ000001", 3.0)):
        for index, trade_date in enumerate(dates):
            close = 10.0 + offset + index * 0.05
            rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "raw_open": close - 0.02,
                    "raw_high": close + 0.10,
                    "raw_low": close - 0.10,
                    "raw_close": close,
                    "raw_volume": 1_000_000 + index * 100,
                    "raw_amount": close * (1_000_000 + index * 100),
                    "factor": 1.0,
                    "turnover_ratio": 0.01,
                    "bp": 0.8,
                    "ep_ttm": 0.1,
                    "float_mv": 10_000_000_000.0,
                    "flow_net_amount": 100_000.0,
                    "lrg_trd_tolbuynum": 120.0,
                    "lrg_trd_tolsellnum": 80.0,
                    "micro_effective_spread": 0.001,
                    "micro_imbalance_volume": 0.2,
                    "micro_jump_flag": 0.0,
                    "ind_code_l1": "801010",
                }
            )
    return pd.DataFrame(rows)


def test_recomputed_features_do_not_use_future_rows():
    panel = _panel()
    computed = MODULE.compute_reliable_features(panel)
    target_date = panel["trade_date"].sort_values().unique()[-2]
    before = computed.loc[
        (computed["symbol"] == "SH600000") & (computed["trade_date"] == target_date),
        ["mom_ret_5d", "mom_ret_120d", "vol_std_20"],
    ].iloc[0]

    changed = panel.copy()
    future_mask = (changed["symbol"] == "SH600000") & (
        changed["trade_date"] > target_date
    )
    changed.loc[future_mask, "raw_close"] = 1_000_000.0
    after_frame = MODULE.compute_reliable_features(changed)
    after = after_frame.loc[
        (after_frame["symbol"] == "SH600000")
        & (after_frame["trade_date"] == target_date),
        ["mom_ret_5d", "mom_ret_120d", "vol_std_20"],
    ].iloc[0]
    pd.testing.assert_series_equal(before, after)


def test_materialize_uses_carry_before_model_fill():
    history = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2026-06-24"),
                "symbol": "SH600000",
                "f1": 1.0,
                "f2": 2.0,
            }
        ]
    )
    fresh = pd.DataFrame(
        [
            {"trade_date": pd.Timestamp("2026-06-25"), "symbol": "SH600000"},
            {"trade_date": pd.Timestamp("2026-06-25"), "symbol": "SZ000001"},
        ]
    )
    computed = fresh.copy()
    computed["f1"] = [10.0, 20.0]
    computed["f2"] = np.nan
    contract = MODULE.ModelContract(
        feature_columns=["f1", "f2"], fill_values={"f1": 0.0, "f2": -1.0}
    )

    output, audits = MODULE.materialize_days(
        computed_panel=computed,
        history=history,
        fresh=fresh,
        contract=contract,
        baseline_rows=2,
        min_symbols=1,
        min_row_ratio=0.5,
        min_source_coverage=0.7,
        min_recomputed_coverage=0.4,
    )

    rows = output.set_index("symbol")
    assert rows.loc["SH600000", "f2"] == 2.0
    assert rows.loc["SZ000001", "f2"] == -1.0
    assert audits[0]["recomputed_cells"] == 2
    assert audits[0]["carried_cells"] == 1
    assert audits[0]["model_fill_cells"] == 1
    assert audits[0]["passed"] is True
