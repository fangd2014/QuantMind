from __future__ import annotations

import builtins
import gzip
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from backend.services.engine.qlib_app.services.sector_momentum_leader_core import (
    DEFAULT_PARAMS,
    StrictDataError,
    _board_daily_rows,
    _prepare_features,
    anti_fall,
    build_sector_signals,
    capped_cap_weights,
    chain_board_index,
    classify_board_phase,
    close_position,
    healthy_volume,
    load_sector_universe,
    normalize_stock_code,
    point_in_time_members,
    select_role_candidate,
    size_fit,
    trend_stability,
    upper_shadow_ratio,
    validate_membership_universe,
    validate_stock_daily,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("600000.SH", "SH600000"),
        ("sz000001", "SZ000001"),
        ("430047", "BJ430047"),
        ("SH.688001", "SH688001"),
        ("bad", None),
    ],
)
def test_normalize_stock_code_uses_prefix_format(
    raw: str, expected: str | None
) -> None:
    assert normalize_stock_code(raw) == expected


def test_membership_intervals_are_strict_and_half_open() -> None:
    universe = validate_membership_universe(
        [{"code": "B1", "name": "板块"}],
        {
            "B1": [
                {
                    "symbol": "600000.SH",
                    "in_date": "20240102",
                    "out_date": "20240105",
                }
            ]
        },
    )

    assert point_in_time_members(universe.memberships, "2024-01-01")["B1"] == []
    assert [
        row["symbol"]
        for row in point_in_time_members(universe.memberships, "2024-01-02")["B1"]
    ] == ["SH600000"]
    assert point_in_time_members(universe.memberships, "2024-01-05")["B1"] == []


def test_membership_strict_failures_reject_missing_dates_and_overlap() -> None:
    boards = [{"code": "B1", "name": "板块"}]
    with pytest.raises(StrictDataError, match="missing in_date"):
        validate_membership_universe(
            boards, {"B1": [{"symbol": "SH600000", "in_date": None}]}
        )
    with pytest.raises(StrictDataError, match="overlapping"):
        validate_membership_universe(
            boards,
            {
                "B1": [
                    {
                        "symbol": "SH600000",
                        "in_date": "20240101",
                        "out_date": "20240201",
                    },
                    {
                        "symbol": "600000.SH",
                        "in_date": "20240115",
                        "out_date": None,
                    },
                ]
            },
        )


def test_sw_loader_reads_historical_cache_without_scripts_package(
    tmp_path, monkeypatch
) -> None:
    cache_path = tmp_path / "tushare-sw2021-l1-membership-history.json.gz"
    with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "version": "SW2021",
                "historical": True,
                "industries": [{"code": "801010.SI", "name": "农林牧渔"}],
                "memberships": {
                    "801010.SI": [
                        {
                            "symbol": "600000.SH",
                            "in_date": "20200101",
                            "out_date": None,
                        }
                    ]
                },
            },
            handle,
        )

    original_import = builtins.__import__

    def reject_scripts_import(name, *args, **kwargs):
        if name == "scripts" or name.startswith("scripts."):
            raise AssertionError(f"unexpected scripts import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_scripts_import)

    def reject_query(*_args, **_kwargs):
        raise AssertionError("fresh cache must not be refreshed")

    universe = load_sector_universe("sw_l1", tmp_path, query=reject_query)

    assert universe.boards[0]["code"] == "801010.SI"
    assert universe.memberships["801010.SI"][0]["symbol"] == "SH600000"


def _sw_query_fixture(api_name, params, _fields):
    if api_name == "index_classify":
        return [
            {
                "index_code": f"80{index:04d}.SI",
                "industry_name": f"行业{index}",
                "level": "L1",
                "src": "SW2021",
            }
            for index in range(28)
        ]
    index = int(str(params["l1_code"])[2:6])
    if params["is_new"] == "N":
        return []
    return [
        {
            "ts_code": f"{600000 + index}.SH",
            "name": f"股票{index}",
            "in_date": "20200101",
            "out_date": None,
        }
    ]


def test_sw_loader_refreshes_expired_cache(tmp_path) -> None:
    cache_path = tmp_path / "tushare-sw2021-l1-membership-history.json.gz"
    with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "version": "SW2021",
                "historical": True,
                "industries": [{"code": "OLD.SI", "name": "旧行业"}],
                "memberships": {
                    "OLD.SI": [
                        {
                            "symbol": "600000.SH",
                            "in_date": "20200101",
                            "out_date": None,
                        }
                    ]
                },
            },
            handle,
        )
    os.utime(cache_path, (0, 0))

    universe = load_sector_universe("sw_l1", tmp_path, query=_sw_query_fixture)

    assert len(universe.boards) == 28
    assert universe.boards[0]["code"] != "OLD.SI"


def test_sw_loader_falls_back_to_validated_stale_cache(tmp_path) -> None:
    cache_path = tmp_path / "tushare-sw2021-l1-membership-history.json.gz"
    with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "version": "SW2021",
                "historical": True,
                "industries": [{"code": "OLD.SI", "name": "旧行业"}],
                "memberships": {
                    "OLD.SI": [
                        {
                            "symbol": "600000.SH",
                            "in_date": "20200101",
                            "out_date": None,
                        }
                    ]
                },
            },
            handle,
        )
    os.utime(cache_path, (0, 0))

    def failing_query(*_args, **_kwargs):
        raise RuntimeError("upstream unavailable")

    universe = load_sector_universe("sw_l1", tmp_path, query=failing_query)

    assert [board["code"] for board in universe.boards] == ["OLD.SI"]


def test_sw_loader_rejects_incomplete_refresh_without_overwriting_cache(
    tmp_path,
) -> None:
    cache_path = tmp_path / "tushare-sw2021-l1-membership-history.json.gz"

    def incomplete_query(api_name, params, fields):
        rows = _sw_query_fixture(api_name, params, fields)
        if api_name == "index_member_all" and params["l1_code"] != "800000.SI":
            return []
        return rows

    with pytest.raises(StrictDataError, match="only 1/28"):
        load_sector_universe("sw_l1", tmp_path, query=incomplete_query)

    assert not cache_path.exists()


def test_ths_loader_keeps_new_and_old_members(tmp_path) -> None:
    def query(api_name, _params, _fields):
        if api_name == "ths_index":
            return [
                {
                    "ts_code": "885001.TI",
                    "name": "测试概念",
                    "exchange": "A",
                    "type": "N",
                }
            ]
        return [
            {
                "con_code": "600000.SH",
                "con_name": "旧成员",
                "in_date": "20200101",
                "out_date": "20230101",
                "is_new": "N",
            },
            {
                "con_code": "000001.SZ",
                "con_name": "新成员",
                "in_date": "20230101",
                "out_date": None,
                "is_new": "Y",
            },
        ]

    universe = load_sector_universe("ths_concept", tmp_path, query=query)

    assert [row["symbol"] for row in universe.memberships["885001.TI"]] == [
        "SH600000",
        "SZ000001",
    ]
    assert "幸存者偏差" in universe.metadata["survivorship_warning"]


def test_ths_strict_loader_rejects_public_fallback(tmp_path) -> None:
    with pytest.raises(StrictDataError, match="public-page fallback"):
        load_sector_universe("ths_concept", tmp_path)


def test_ths_loader_uses_default_tushare_query_when_token_exists(
    tmp_path, monkeypatch
) -> None:
    calls = []

    def fake_query(api_name, params, fields):
        calls.append((api_name, params, fields))
        if api_name == "ths_index":
            return [
                {
                    "ts_code": "885001.TI",
                    "name": "测试概念",
                    "exchange": "A",
                    "type": "N",
                }
            ]
        return [
            {
                "con_code": "600000.SH",
                "con_name": "历史成员",
                "in_date": "20200101",
                "out_date": None,
                "is_new": "Y",
            }
        ]

    monkeypatch.setenv("TUSHARE_TOKEN", "fixture-token")
    monkeypatch.setattr(
        "scripts.analysis.concept_rotation_report.query_tushare", fake_query
    )

    universe = load_sector_universe("ths_concept", tmp_path)

    assert universe.memberships["885001.TI"][0]["symbol"] == "SH600000"
    assert [call[0] for call in calls] == ["ths_index", "ths_member"]


@pytest.mark.parametrize(
    "market_values",
    [
        np.ones(10),
        np.arange(1, 12),
        np.array([10_000_000, *([1.0] * 10)]),
    ],
)
def test_capped_cap_water_filling(market_values: np.ndarray) -> None:
    weights = capped_cap_weights(market_values)

    assert weights.sum() == pytest.approx(1.0)
    assert weights.max() <= 0.10 + 1e-12
    positive_lambda = (
        weights[weights < 0.10 - 1e-10] / market_values[weights < 0.10 - 1e-10]
    )
    if len(positive_lambda):
        assert positive_lambda.max() == pytest.approx(positive_lambda.min())


def test_capped_cap_rejects_infeasible_and_missing_values() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        capped_cap_weights(np.ones(9))
    with pytest.raises(ValueError, match="finite positive"):
        capped_cap_weights([1] * 9 + [math.nan])


def test_chain_board_index_uses_one_base_and_compounds_subsequent_returns() -> None:
    result = chain_board_index([math.nan, 0.10, -0.10, 0.05])
    assert np.isnan(result[0])
    assert result[1:].tolist() == pytest.approx([100.0, 90.0, 94.5])


def _stock_rows(
    days: int = 30, boards: int = 1
) -> tuple[pd.DataFrame, list[dict], dict]:
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows: list[dict[str, object]] = []
    board_rows: list[dict[str, str]] = []
    memberships: dict[str, list[dict[str, object]]] = {}
    for board_index in range(boards):
        code = f"B{board_index:02d}"
        board_rows.append({"code": code, "name": code})
        memberships[code] = []
        for member_index in range(10):
            digits = 600000 + board_index * 100 + member_index
            symbol = f"SH{digits:06d}"
            memberships[code].append(
                {"symbol": symbol, "in_date": "20200101", "out_date": None}
            )
            for day_index, day in enumerate(dates):
                close = 10 * 1.01**day_index
                rows.append(
                    {
                        "trade_date": day,
                        "symbol": symbol,
                        "open": close * 0.995,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "close": close,
                        "volume": 10_000_000,
                        "amount": 500_000_000,
                        "adj_factor": 1.0,
                        "is_st": 0,
                        "turnover_rate": 10.0,
                        "float_mv": 10_000_000_000 + member_index,
                        "limit_up_today": 0,
                        "limit_down_today": 0,
                    }
                )
    return pd.DataFrame(rows), board_rows, memberships


def test_stock_validation_derives_adjusted_prices_and_converts_turnover() -> None:
    raw, _, _ = _stock_rows(days=2)
    clean, audit = validate_stock_daily(raw)

    assert clean["adj_close"].equals(clean["close"] * clean["adj_factor"])
    assert clean["turnover_rate"].iloc[0] == pytest.approx(0.10)
    assert audit["turnover_unit_conversion"] == "percent_to_decimal"


def test_stock_validation_handles_suspension_and_unknown_st_conservatively() -> None:
    raw, _, _ = _stock_rows(days=2)
    suspension_index = raw.index[0]
    st_unknown_index = raw.index[1]
    close = raw.loc[suspension_index, "close"]
    raw.loc[suspension_index, ["open", "high", "low", "close"]] = close
    raw.loc[suspension_index, ["volume", "amount"]] = np.nan
    raw.loc[st_unknown_index, "is_st"] = np.nan

    clean, audit = validate_stock_daily(raw)

    suspension = clean.loc[
        (clean["trade_date"] == raw.loc[suspension_index, "trade_date"])
        & (clean["symbol"] == raw.loc[suspension_index, "symbol"])
    ].iloc[0]
    unknown_st = clean.loc[
        (clean["trade_date"] == raw.loc[st_unknown_index, "trade_date"])
        & (clean["symbol"] == raw.loc[st_unknown_index, "symbol"])
    ].iloc[0]
    assert suspension["volume"] == 0
    assert suspension["amount"] == 0
    assert bool(suspension["is_suspended"]) is True
    assert unknown_st["is_st"] == 1
    assert audit["suspension_liquidity_filled"] == 1
    assert audit["missing_is_st_excluded"] == 1


def test_stock_validation_rejects_partial_missing_liquidity() -> None:
    raw, _, _ = _stock_rows(days=2)
    raw.loc[raw.index[0], "volume"] = np.nan

    with pytest.raises(StrictDataError, match="missing only together"):
        validate_stock_daily(raw)


def test_stock_validation_rejects_nonflat_missing_liquidity() -> None:
    raw, _, _ = _stock_rows(days=2)
    raw.loc[raw.index[0], ["volume", "amount"]] = np.nan

    with pytest.raises(StrictDataError, match="flat-price"):
        validate_stock_daily(raw)


def test_stock_validation_rejects_bad_adjustment() -> None:
    raw, _, _ = _stock_rows(days=2)
    raw["adj_close"] = raw["close"] * 2
    with pytest.raises(StrictDataError, match="inconsistent"):
        validate_stock_daily(raw)


def _board_rows_with_twelve_members() -> tuple[pd.DataFrame, list[dict], dict]:
    raw, boards, memberships = _stock_rows(days=6)
    dates = sorted(raw["trade_date"].unique())
    extra_rows: list[dict[str, object]] = []
    for member_index in (10, 11):
        symbol = f"SH{600000 + member_index:06d}"
        memberships["B00"].append(
            {"symbol": symbol, "in_date": "20200101", "out_date": None}
        )
        for day_index, day in enumerate(dates):
            close = 10 * 1.01**day_index
            extra_rows.append(
                {
                    "trade_date": day,
                    "symbol": symbol,
                    "open": close * 0.995,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 10_000_000,
                    "amount": 500_000_000,
                    "adj_factor": 1.0,
                    "is_st": 0,
                    "turnover_rate": 10.0,
                    "float_mv": 10_000_000_000 + member_index,
                    "limit_up_today": 0,
                    "limit_down_today": 0,
                }
            )
    combined = pd.concat([raw, pd.DataFrame(extra_rows)], ignore_index=True)
    return combined, boards, memberships


def test_board_metrics_recalculate_coverage_for_partial_limit_flags() -> None:
    raw, boards, memberships = _board_rows_with_twelve_members()
    latest = raw["trade_date"].max()
    raw.loc[
        (raw["trade_date"] == latest)
        & (raw["symbol"] == "SH600011"),
        ["limit_up_today", "limit_down_today"],
    ] = np.nan
    raw.loc[
        (raw["trade_date"] == latest) & (raw["symbol"] == "SH600010"),
        ["limit_up_today", "limit_down_today"],
    ] = [1, 1]
    raw.loc[
        (raw["trade_date"] == latest) & (raw["symbol"] == "SH600000"),
        ["limit_up_today", "limit_down_today"],
    ] = [1, np.nan]
    clean, _ = validate_stock_daily(raw)
    diagnostics: list[dict] = []

    daily = _board_daily_rows(
        _prepare_features(clean), boards, memberships, DEFAULT_PARAMS, diagnostics
    )
    latest_board = daily.loc[daily["trade_date"] == latest].iloc[0]

    assert bool(latest_board["evaluable"])
    assert latest_board["available_count"] == 10
    assert latest_board["coverage"] == pytest.approx(10 / 12)
    assert latest_board["limit_up_count"] == 1
    assert latest_board["limit_up_ratio"] == pytest.approx(0.1)


def test_board_with_all_limit_flags_missing_is_soft_excluded() -> None:
    raw, boards, memberships = _board_rows_with_twelve_members()
    latest = raw["trade_date"].max()
    raw.loc[
        raw["trade_date"] == latest,
        ["limit_up_today", "limit_down_today"],
    ] = np.nan
    clean, _ = validate_stock_daily(raw)
    diagnostics: list[dict] = []

    daily = _board_daily_rows(
        _prepare_features(clean), boards, memberships, DEFAULT_PARAMS, diagnostics
    )
    latest_board = daily.loc[daily["trade_date"] == latest].iloc[0]

    assert not bool(latest_board["evaluable"])
    assert latest_board["reason"] == "insufficient_enhanced_coverage"
    assert latest_board["available_count"] == 0
    assert latest_board["coverage"] == 0
    assert any(
        item["date"] == latest
        and item["reason"] == "insufficient_enhanced_coverage"
        and item["soft"]
        for item in diagnostics
    )


def test_factor_formulas_have_the_unique_prd_behavior() -> None:
    assert close_position(9, 8, 10) == pytest.approx(0.5)
    assert close_position(10, 10, 10) == 0.5
    assert upper_shadow_ratio(9, 10, 8, 9.5) == pytest.approx(0.25)
    assert healthy_volume(1.5) == pytest.approx(1.0)
    assert math.isnan(healthy_volume(0))
    assert anti_fall([0.02, 0.03], [-0.01, 0.01]) == pytest.approx(0.03)
    assert anti_fall([0.02], [0.01]) == 0.0
    assert size_fit(math.sqrt(5e9 * 3e10), 5e9, 3e10) == pytest.approx(1.0)
    assert trend_stability(11, 10, 9) == pytest.approx(1.0)
    assert math.isnan(trend_stability(11, 10, 0))


def _phase_metrics() -> dict[str, float | bool]:
    return {
        "board_close": 105.0,
        "ma5": 103.0,
        "ma20": 100.0,
        "breadth_ma20": 0.60,
        "breadth_change_3d": 0.08,
        "relative_return_3d": 0.03,
        "relative_return_5d": 0.05,
        "limit_up_count": 2,
        "limit_up_ratio": 0.02,
        "amount_share_vs_median20": 1.20,
        "amount_share_quantile": 0.80,
        "top3_return_median": 0.01,
        "top3_amount_ratio_median": 1.20,
        "crowding_quantile": 0.50,
        "crossed_ma20_within_3d": True,
    }


def test_phase_priority_and_boundaries() -> None:
    metrics = _phase_metrics()
    assert classify_board_phase(metrics)[0] == "launch"
    diffusion = {
        **metrics,
        "breadth_ma20": 0.65,
        "breadth_change_3d": 0.01,
        "limit_up_count": 3,
    }
    assert classify_board_phase(diffusion)[0] == "diffusion"
    overheated = {**diffusion, "breadth_ma20": 0.80, "relative_return_5d": 0.13}
    assert classify_board_phase(overheated)[0] == "overheated"
    retreat = {
        **overheated,
        "board_close": 90.0,
        "breadth_ma20": 0.39,
        "breadth_change_3d": -0.10,
        "relative_return_3d": -0.03,
    }
    assert classify_board_phase(retreat)[0] == "retreat"


def _role_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "SH600001",
                "open": 10.4,
                "high": 11.1,
                "low": 10.2,
                "close": 11.0,
                "adj_close": 11.0,
                "is_st": 0,
                "is_suspended": False,
                "limit_up_today": False,
                "limit_down_today": False,
                "limit_up_last5": True,
                "float_mv": 2e10,
                "amount_ma5": 1.2e9,
                "turnover_rate": 0.10,
                "ma5": 10.5,
                "ma20": 10.0,
                "ma20_lag5": 9.5,
                "rps20": 0.95,
                "amount_ratio": 1.5,
                "return_3d": 0.12,
                "return_5d": 0.15,
                "volatility_20d": 0.30,
                "max_drawdown_20d": -0.04,
                "anti_fall": 0.03,
            },
            {
                "symbol": "SH600002",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "adj_close": 10.2,
                "is_st": 0,
                "is_suspended": False,
                "limit_up_today": False,
                "limit_down_today": False,
                "limit_up_last5": True,
                "float_mv": 1.2e10,
                "amount_ma5": 1.05e9,
                "turnover_rate": 0.08,
                "ma5": 10.1,
                "ma20": 10.0,
                "ma20_lag5": 9.8,
                "rps20": 0.88,
                "amount_ratio": 1.3,
                "return_3d": 0.06,
                "return_5d": 0.08,
                "volatility_20d": 0.35,
                "max_drawdown_20d": -0.08,
                "anti_fall": 0.01,
            },
        ]
    )


def test_phase_selects_only_the_matching_deterministic_role() -> None:
    rows = _role_rows()
    leader = select_role_candidate(rows.sample(frac=1, random_state=1), "launch")
    core = select_role_candidate(rows.sample(frac=1, random_state=2), "diffusion")

    assert leader is not None
    assert leader["symbol"] == "SH600001"
    assert leader["role"] == "leader"
    assert core is not None
    assert core["symbol"] == "SH600001"
    assert core["role"] == "core"
    assert select_role_candidate(rows, "overheated") is None


def test_leader_rejects_high_volume_long_upper_shadow() -> None:
    rows = _role_rows().iloc[[0]].copy()
    rows["amount_ratio"] = 2.1
    rows["open"] = 10.0
    rows["close"] = 10.1
    rows["high"] = 11.0
    rows["low"] = 10.0

    assert select_role_candidate(rows, "launch") is None


def test_chain_index_is_not_rebased_and_future_rows_do_not_change_history() -> None:
    stock, boards, memberships = _stock_rows(days=30)
    end = stock["trade_date"].sort_values().unique()[24]
    params = {"topk_sectors": 1, "topk_stocks": 1}
    original = build_sector_signals(
        stock, boards, memberships, "2024-01-02", end, params=params
    )
    future = stock["trade_date"] > end
    mutated = stock.copy()
    mutated.loc[future, ["close", "high", "amount", "float_mv"]] *= 100
    changed = build_sector_signals(
        mutated, boards, memberships, "2024-01-02", end, params=params
    )

    original_states = original.board_states
    changed_states = changed.board_states
    assert original_states.keys() == changed_states.keys()
    for day in original_states:
        left = original_states[day]["B00"]
        right = changed_states[day]["B00"]
        assert left["board_close"] == pytest.approx(right["board_close"], nan_ok=True)
        assert left["board_score"] == pytest.approx(right["board_score"], nan_ok=True)
    second_valid = sorted(original_states)[2]
    assert original_states[second_valid]["B00"]["board_close"] == pytest.approx(101.0)
    assert original.metadata["signal_lag_days"] == 0


def test_top_ten_board_ranking_is_stable_on_ties() -> None:
    stock, boards, memberships = _stock_rows(days=25, boards=12)
    result = build_sector_signals(
        stock,
        boards,
        memberships,
        "2024-01-02",
        "2024-02-05",
        params={"topk_sectors": 10},
    )

    latest = result.board_states[max(result.board_states)]
    ranked = sorted(
        (state["rank"], code)
        for code, state in latest.items()
        if state["rank"] is not None and state["rank"] <= 10
    )
    assert len(ranked) == 10
    assert [code for _, code in ranked] == [f"B{index:02d}" for index in range(10)]


def test_missing_one_previous_market_cap_degrades_whole_board_to_equal_weight() -> None:
    stock, boards, memberships = _stock_rows(days=8)
    dates = sorted(stock["trade_date"].unique())
    stock.loc[
        (stock["trade_date"] == dates[4]) & (stock["symbol"] == "SH600000"),
        "float_mv",
    ] = np.nan

    result = build_sector_signals(
        stock,
        boards,
        memberships,
        dates[0],
        dates[-1],
        params={"topk_sectors": 1},
    )

    state = result.board_states[pd.Timestamp(dates[5])]["B00"]
    assert state["weighting"] == "equal_missing_previous_float_mv"
    assert any(
        diagnostic["date"] == pd.Timestamp(dates[5])
        and diagnostic["reason"] == "equal_missing_previous_float_mv"
        for diagnostic in result.metadata["diagnostics"]
    )


def test_strict_mode_fails_when_warmed_board_count_cannot_fill_topk() -> None:
    stock, boards, memberships = _stock_rows(days=25)
    with pytest.raises(StrictDataError, match="evaluable boards after warm-up"):
        build_sector_signals(
            stock,
            boards,
            memberships,
            "2024-01-02",
            "2024-02-05",
            params={"topk_sectors": 2},
        )


def test_soft_exclusion_is_kept_in_full_board_state() -> None:
    stock, boards, memberships = _stock_rows(days=5)
    memberships["B00"] = memberships["B00"][:9]

    result = build_sector_signals(
        stock,
        boards,
        memberships,
        "2024-01-02",
        "2024-01-08",
        params={"topk_sectors": 1},
    )

    first = result.board_states[min(result.board_states)]["B00"]
    assert first["evaluable"] is False
    assert first["reason"] == "insufficient_members"
    assert first["rank"] is None
    assert first["phase"] is None
