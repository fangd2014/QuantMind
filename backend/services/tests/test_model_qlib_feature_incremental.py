from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


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


class _ClosableConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FixedDate(date):
    @classmethod
    def today(cls) -> date:
        return cls(2026, 7, 31)


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


def test_run_skips_when_snapshot_matches_latest_available_source_date(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "model"
    snapshot_dir = tmp_path / "snapshots"
    audit_dir = tmp_path / "audit"
    model_dir.mkdir()
    snapshot_dir.mkdir()
    (model_dir / "metadata.json").write_text(
        json.dumps({"feature_columns": ["f1"], "fill_values": {"f1": 0.0}}),
        encoding="utf-8",
    )
    yearly_path = snapshot_dir / "model_features_2026.parquet"
    yearly_path.touch()
    connection = _ClosableConnection()

    monkeypatch.setattr(MODULE, "date", _FixedDate)
    monkeypatch.setattr(MODULE, "database_connection", lambda: connection)

    def source_date(_connection, *, not_after: date) -> pd.Timestamp:
        assert not_after == date(2026, 7, 31)
        return pd.Timestamp("2026-07-30")

    monkeypatch.setattr(
        MODULE,
        "latest_stock_date",
        source_date,
    )
    monkeypatch.setattr(
        MODULE,
        "parquet_date_range",
        lambda _path: (
            pd.Timestamp("2026-01-02"),
            pd.Timestamp("2026-07-30"),
            123_456,
        ),
    )

    def fail_if_loaded(*_args, **_kwargs):
        raise AssertionError("no nonexistent calendar date should be queried")

    monkeypatch.setattr(MODULE, "load_stock_rows", fail_if_loaded)
    args = SimpleNamespace(
        start_date=None,
        end_date=None,
        model_dir=model_dir,
        snapshot_dir=snapshot_dir,
        audit_dir=audit_dir,
        apply=True,
        backup=False,
        min_symbols=1000,
        min_row_ratio=0.85,
        min_source_coverage=0.75,
        min_recomputed_coverage=0.35,
    )

    result = MODULE.run(args)

    assert result == {
        "success": True,
        "skipped": True,
        "reason": "feature_parquet_already_current",
        "parquet_last_date": "2026-07-30",
        "source_last_date": "2026-07-30",
    }
    assert connection.closed is True


def test_explicit_current_end_date_skips_without_database_access(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "model"
    snapshot_dir = tmp_path / "snapshots"
    model_dir.mkdir()
    snapshot_dir.mkdir()
    (model_dir / "metadata.json").write_text(
        json.dumps({"feature_columns": ["f1"], "fill_values": {"f1": 0.0}}),
        encoding="utf-8",
    )
    (snapshot_dir / "model_features_2026.parquet").touch()
    monkeypatch.setattr(
        MODULE,
        "parquet_date_range",
        lambda _path: (
            pd.Timestamp("2026-01-02"),
            pd.Timestamp("2026-07-30"),
            123_456,
        ),
    )

    def fail_if_connected():
        raise AssertionError("an already-current explicit range needs no database")

    monkeypatch.setattr(MODULE, "database_connection", fail_if_connected)
    args = SimpleNamespace(
        start_date=None,
        end_date="2026-07-30",
        model_dir=model_dir,
        snapshot_dir=snapshot_dir,
        audit_dir=tmp_path / "audit",
        apply=True,
        backup=False,
        min_symbols=1000,
        min_row_ratio=0.85,
        min_source_coverage=0.75,
        min_recomputed_coverage=0.35,
    )

    result = MODULE.run(args)

    assert result == {
        "success": True,
        "skipped": True,
        "reason": "feature_parquet_already_current",
        "parquet_last_date": "2026-07-30",
    }


def test_explicit_unavailable_end_date_stays_strict_and_closes_connection(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "model"
    snapshot_dir = tmp_path / "snapshots"
    model_dir.mkdir()
    snapshot_dir.mkdir()
    (model_dir / "metadata.json").write_text(
        json.dumps({"feature_columns": ["f1"], "fill_values": {"f1": 0.0}}),
        encoding="utf-8",
    )
    (snapshot_dir / "model_features_2026.parquet").touch()
    connection = _ClosableConnection()
    monkeypatch.setattr(MODULE, "database_connection", lambda: connection)
    monkeypatch.setattr(
        MODULE,
        "parquet_date_range",
        lambda _path: (
            pd.Timestamp("2026-01-02"),
            pd.Timestamp("2026-07-30"),
            123_456,
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "load_stock_rows",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )

    def fail_if_source_date_is_resolved(*_args, **_kwargs):
        raise AssertionError("explicit end dates must not be clamped")

    monkeypatch.setattr(MODULE, "latest_stock_date", fail_if_source_date_is_resolved)
    args = SimpleNamespace(
        start_date=None,
        end_date="2026-07-31",
        model_dir=model_dir,
        snapshot_dir=snapshot_dir,
        audit_dir=tmp_path / "audit",
        apply=True,
        backup=False,
        min_symbols=1000,
        min_row_ratio=0.85,
        min_source_coverage=0.75,
        min_recomputed_coverage=0.35,
    )

    with pytest.raises(
        RuntimeError,
        match=r"stock_daily_latest has no rows in \[2026-07-31, 2026-07-31\]",
    ):
        MODULE.run(args)

    assert connection.closed is True
