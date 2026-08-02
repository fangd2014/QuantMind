from contextlib import asynccontextmanager
from types import SimpleNamespace

import pandas as pd
import pytest

import backend.shared.database_manager_v2 as database_manager
from backend.services.engine.qlib_app.schemas.backtest import (
    QlibBacktestRequest,
    SectorMomentumLeaderCoreParams,
)
from backend.services.engine.qlib_app.services import (
    backtest_service_runtime as runtime_module,
)
from backend.services.engine.qlib_app.services import (
    sector_momentum_leader_core as core_module,
)
from backend.services.engine.qlib_app.services.backtest_service import (
    QlibBacktestService,
)
from backend.services.engine.qlib_app.services.backtest_service_runtime import (
    QlibBacktestServiceRuntimeMixin,
)
from backend.services.engine.qlib_app.services.strategy_builder import (
    SectorMomentumLeaderCoreBuilder,
    SectorSignalPayload,
    StrategyFactory,
)
from backend.services.engine.qlib_app.services.strategy_templates import (
    get_template_by_id,
    invalidate_templates_cache,
)
from backend.services.engine.qlib_app.utils import (
    extended_strategies as extended_module,
)
from backend.services.engine.qlib_app.utils.extended_strategies import (
    OrderDir,
    RedisSectorMomentumLeaderCoreStrategy,
)


def _request(**overrides):
    payload = {
        "strategy_type": "sector_momentum_leader_core",
        "strategy_params": {},
        "start_date": "2026-01-05",
        "end_date": "2026-01-09",
        "deal_price": "close",
        "signal_lag_days": 0,
    }
    payload.update(overrides)
    return QlibBacktestRequest(**payload)


def test_request_uses_dedicated_schema_and_forces_safe_execution():
    request = _request()

    assert isinstance(request.strategy_params, SectorMomentumLeaderCoreParams)
    assert request.strategy_params.board_universe == "sw_l1"
    assert request.strategy_params.max_holding_days == 10
    assert request.deal_price == "open"
    assert request.signal_lag_days == 1


def test_runtime_signal_logging_supports_dedicated_schema():
    request = _request()

    assert QlibBacktestServiceRuntimeMixin._strategy_signal_for_log(request) is None


def test_factory_returns_dedicated_builder_and_unknown_id_fails():
    builder, is_fallback, normalized = StrategyFactory.resolve_builder(
        "sector_momentum_leader_core"
    )

    assert isinstance(builder, SectorMomentumLeaderCoreBuilder)
    assert is_fallback is False
    assert normalized == "sector_momentum_leader_core"
    with pytest.raises(ValueError, match="Unknown strategy_type"):
        QlibBacktestService._resolve_strategy_builder(
            _request(strategy_type="sector_momentum_leader_cor_typo")
        )


def test_template_exposes_complete_parameter_metadata():
    invalidate_templates_cache()
    template = get_template_by_id("sector_momentum_leader_core")

    assert template is not None
    assert template.name == "板块动量轮动+龙头中军选股"
    params = {item.name: item for item in template.params}
    assert set(SectorMomentumLeaderCoreParams.model_fields) == set(params)
    assert params["board_universe"].options == ["sw_l1", "ths_concept"]
    assert params["board_universe"].type == "select"
    assert params["leader_gap_down"].model_dump()["step"] == 0.01
    assert params["max_holding_days"].max == 10
    assert "MODEL_SPEC_V1" not in params


@pytest.mark.asyncio
async def test_runtime_reads_local_daily_data_and_aligns_all_inputs_once(
    monkeypatch, tmp_path
):
    captured = {}

    class FakeMappings:
        def all(self):
            return [{"trade_date": "2026-01-05", "symbol": "SH600000"}]

    class FakeResult:
        def mappings(self):
            return FakeMappings()

    class FakeSession:
        async def execute(self, statement, parameters):
            captured["sql"] = str(statement)
            captured["parameters"] = parameters
            return FakeResult()

    @asynccontextmanager
    async def fake_get_session(*, read_only=False):
        captured["read_only"] = read_only
        yield FakeSession()

    signals = pd.DataFrame(
        [
            {
                "score": 88.0,
                "board_code": "801010.SI",
                "board_name": "农林牧渔",
                "phase": "launch",
                "role": "leader",
                "board_score": 80.0,
                "signal_date": "2026-01-05",
            }
        ],
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-01-05"), "SH600000")],
            names=["datetime", "instrument"],
        ),
    )
    board_states = {
        pd.Timestamp("2026-01-05"): {
            "801010.SI": {
                "board_code": "801010.SI",
                "rank": 1,
                "board_score": 80.0,
                "phase": "launch",
                "signal_date": "2026-01-05",
                "evaluable": True,
            }
        }
    }

    def fake_load_sector_universe(board_universe, cache_dir, strict):
        captured["universe"] = (board_universe, cache_dir, strict)
        return SimpleNamespace(boards=[{"code": "801010.SI"}], memberships={})

    def fake_build_sector_signals(stock_daily, boards, memberships, *args, **kwargs):
        captured["stock_daily"] = stock_daily
        captured["build_args"] = (boards, memberships, args, kwargs)
        return SimpleNamespace(
            signals=signals,
            board_states=board_states,
            metadata={"diagnostics": [{"kind": "fixture"}]},
        )

    monkeypatch.setattr(database_manager, "get_session", fake_get_session)
    monkeypatch.setattr(core_module, "load_sector_universe", fake_load_sector_universe)
    monkeypatch.setattr(core_module, "build_sector_signals", fake_build_sector_signals)
    monkeypatch.delenv("SECTOR_MEMBERSHIP_CACHE_DIR", raising=False)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        runtime_module.D,
        "calendar",
        lambda freq="day": pd.DatetimeIndex(["2026-01-05", "2026-01-06"]),
        raising=False,
    )
    monkeypatch.setattr(
        QlibBacktestServiceRuntimeMixin,
        "_sector_market_breadth_by_execution_date",
        staticmethod(lambda _daily, _dates: {"2026-01-06": 0.55}),
    )

    runtime = QlibBacktestServiceRuntimeMixin()
    payload, metadata = await runtime._build_signal_data(_request())

    assert isinstance(payload, SectorSignalPayload)
    assert captured["read_only"] is True
    assert "FROM stock_daily_latest" in captured["sql"]
    assert captured["stock_daily"].iloc[0]["symbol"] == "SH600000"
    assert captured["universe"] == (
        "sw_l1",
        tmp_path / "sector-memberships",
        True,
    )
    assert payload.score.index.get_level_values("datetime").tolist() == [
        pd.Timestamp("2026-01-06")
    ]
    detail = payload.candidate_details["2026-01-06"]["SH600000"]
    assert detail["signal_date"] == "2026-01-05"
    state = payload.board_states["2026-01-06"]["801010.SI"]
    assert state["signal_date"] == "2026-01-05"
    assert state["board_score"] == 80.0
    assert metadata["source"] == "sector_momentum_leader_core"
    assert metadata["signal_lag_days"] == 1
    assert metadata["diagnostic_count"] == 1
    assert metadata["diagnostic_reason_counts"] == {"fixture": 1}
    assert metadata["diagnostics_sample"] == [{"kind": "fixture"}]
    assert "candidate_details" not in metadata
    assert "board_states" not in metadata
    assert payload.candidate_details["2026-01-06"]["SH600000"] == detail
    assert payload.board_states["2026-01-06"]["801010.SI"] == state
    assert metadata["market_breadth_by_date"] == {"2026-01-06": 0.55}


def test_board_only_signal_date_is_aligned_for_next_day_exit(monkeypatch):
    signals = pd.DataFrame(
        {"score": [88.0]},
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-01-05"), "SH600000")],
            names=["datetime", "instrument"],
        ),
    )
    board_states = {
        pd.Timestamp("2026-01-05"): {
            "B1": {"phase": "launch", "evaluable": True}
        },
        pd.Timestamp("2026-01-06"): {
            "B1": {"phase": "retreat", "evaluable": True}
        },
    }
    monkeypatch.setattr(
        runtime_module.D,
        "calendar",
        lambda freq="day": pd.DatetimeIndex(
            ["2026-01-05", "2026-01-06", "2026-01-07"]
        ),
        raising=False,
    )

    execution_dates = QlibBacktestServiceRuntimeMixin._sector_execution_date_map(
        signals, board_states
    )
    aligned_states = QlibBacktestServiceRuntimeMixin._align_sector_board_states(
        board_states, execution_dates
    )

    assert execution_dates[pd.Timestamp("2026-01-06")] == pd.Timestamp("2026-01-07")
    assert aligned_states["2026-01-07"]["B1"]["phase"] == "retreat"

    strategy = RedisSectorMomentumLeaderCoreStrategy.__new__(
        RedisSectorMomentumLeaderCoreStrategy
    )
    strategy._position_context = {
        "SH600000": {"board_code": "B1", "role": "core"}
    }
    strategy.board_states = aligned_states
    strategy._board_rank_breach_days = {"SH600000": 0}
    strategy.board_exit_rank = 20
    strategy.board_exit_days = 2
    strategy.board_score_drop_exit = 20.0

    assert (
        strategy._board_exit_reason("SH600000", "2026-01-07")
        == "board_phase_retreat"
    )


def test_builder_passes_full_params_details_and_board_states():
    request = _request(strategy_params={"board_universe": "ths_concept"})
    score = pd.DataFrame(
        {"score": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-01-06"), "SH600000")],
            names=["datetime", "instrument"],
        ),
    )
    payload = SectorSignalPayload(
        score=score,
        candidate_details={"2026-01-06": {"SH600000": {"role": "leader"}}},
        board_states={"2026-01-06": {"881001.TI": {"rank": 1}}},
        metadata={"source": "sector_momentum_leader_core"},
    )

    config = SectorMomentumLeaderCoreBuilder().build(request, {}, payload, "bt-1")
    kwargs = config["kwargs"]

    assert config["class"] == "RedisSectorMomentumLeaderCoreStrategy"
    assert kwargs["signal"] is score
    assert kwargs["board_universe"] == "ths_concept"
    assert kwargs["candidate_details"] == payload.candidate_details
    assert kwargs["board_states"] == payload.board_states
    assert kwargs["signal_metadata"] is payload.metadata
    assert kwargs["max_holding_days"] == 10


def test_holding_age_is_fill_driven_and_repeated_buys_do_not_reset_it(monkeypatch):
    strategy = RedisSectorMomentumLeaderCoreStrategy.__new__(
        RedisSectorMomentumLeaderCoreStrategy
    )
    strategy.candidate_details = {
        "2026-01-06": {
            "SH600000": {
                "board_code": "801010.SI",
                "board_score": 80.0,
                "role": "leader",
                "signal_date": "2026-01-05",
            }
        }
    }
    strategy._entry_trade_step = {}
    strategy._position_context = {}
    strategy._board_rank_breach_days = {}
    strategy._forced_exit_blocked = {}
    strategy._trade_step_by_date = {"2026-01-06": 5}
    strategy.execution_diagnostics = []
    position_amount = {"SH600000": 100.0}
    fake_position = SimpleNamespace(
        get_stock_amount=lambda stock: position_amount.get(stock, 0.0),
        get_stock_list=lambda: [
            stock for stock, amount in position_amount.items() if amount > 0
        ],
    )
    monkeypatch.setattr(
        RedisSectorMomentumLeaderCoreStrategy,
        "trade_position",
        property(lambda _self: fake_position),
    )
    buy = SimpleNamespace(
        stock_id="SH600000",
        direction=OrderDir.BUY,
        deal_amount=100.0,
        start_time=pd.Timestamp("2026-01-06"),
    )

    strategy._record_fills([(buy, 0, 0, 10.0)], trade_step=6)
    strategy._record_fills([(buy, 0, 0, 10.0)], trade_step=8)

    assert strategy._entry_trade_step["SH600000"] == 5
    strategy.max_holding_days = 10
    strategy.board_states = {}
    assert strategy._forced_exit_reasons(14, "2026-01-19") == {}
    assert strategy._forced_exit_reasons(15, "2026-01-20") == {
        "SH600000": "max_holding_days"
    }

    sell = SimpleNamespace(
        stock_id="SH600000",
        direction=OrderDir.SELL,
        deal_amount=50.0,
        start_time=pd.Timestamp("2026-01-20"),
    )
    position_amount["SH600000"] = 50.0
    strategy._record_fills([(sell, 0, 0, 10.0)], trade_step=15)
    assert strategy._entry_trade_step["SH600000"] == 5
    position_amount["SH600000"] = 0.0
    strategy._record_fills([(sell, 0, 0, 10.0)], trade_step=16)
    assert "SH600000" not in strategy._entry_trade_step


def test_sell_fill_keeps_tracking_when_position_amount_cannot_be_read(monkeypatch):
    strategy = RedisSectorMomentumLeaderCoreStrategy.__new__(
        RedisSectorMomentumLeaderCoreStrategy
    )
    strategy.execution_diagnostics = []
    strategy._entry_trade_step = {"SH600000": 5}
    strategy._position_context = {"SH600000": {"board_code": "B1"}}
    strategy._board_rank_breach_days = {"SH600000": 1}
    strategy._forced_exit_blocked = {
        "SH600000": {"reason": "max_holding_days", "retries": 2}
    }
    strategy._trade_step_by_date = {}
    fake_position = SimpleNamespace(
        get_stock_amount=lambda _stock: (_ for _ in ()).throw(
            RuntimeError("position unavailable")
        )
    )
    monkeypatch.setattr(
        RedisSectorMomentumLeaderCoreStrategy,
        "trade_position",
        property(lambda _self: fake_position),
    )
    sell = SimpleNamespace(
        stock_id="SH600000",
        direction=OrderDir.SELL,
        deal_amount=100.0,
        start_time=pd.Timestamp("2026-01-20"),
    )

    strategy._record_fills([(sell, 0, 0, 10.0)], trade_step=15)

    assert strategy._entry_trade_step == {"SH600000": 5}
    assert strategy._position_context == {"SH600000": {"board_code": "B1"}}
    assert strategy._board_rank_breach_days == {"SH600000": 1}
    assert strategy._forced_exit_blocked == {
        "SH600000": {"reason": "max_holding_days", "retries": 2}
    }
    assert strategy.execution_diagnostics == [
        {
            "event": "position_amount_read_failed",
            "date": "2026-01-20",
            "stock": "SH600000",
            "error_type": "RuntimeError",
        }
    ]


def test_position_equity_failure_remains_fail_closed_and_is_diagnosable(monkeypatch):
    strategy = RedisSectorMomentumLeaderCoreStrategy.__new__(
        RedisSectorMomentumLeaderCoreStrategy
    )
    strategy.execution_diagnostics = []
    fake_position = SimpleNamespace(
        calculate_value=lambda: (_ for _ in ()).throw(
            RuntimeError("position unavailable")
        )
    )
    monkeypatch.setattr(
        RedisSectorMomentumLeaderCoreStrategy,
        "trade_position",
        property(lambda _self: fake_position),
    )

    assert strategy._position_equity() == 0.0
    assert strategy.execution_diagnostics == [
        {
            "event": "position_equity_read_failed",
            "error_type": "RuntimeError",
        }
    ]


def test_forced_exit_diagnostics_cover_block_retry_recovery_and_completion(
    monkeypatch,
):
    strategy = RedisSectorMomentumLeaderCoreStrategy.__new__(
        RedisSectorMomentumLeaderCoreStrategy
    )
    strategy.candidate_details = {}
    strategy.board_states = {}
    strategy.signal_metadata = {"execution_diagnostics": []}
    strategy.execution_diagnostics = strategy.signal_metadata["execution_diagnostics"]
    strategy._entry_trade_step = {"SH600000": 0}
    strategy._position_context = {"SH600000": {}}
    strategy._board_rank_breach_days = {"SH600000": 0}
    strategy._forced_exit_blocked = {}
    strategy._trade_step_by_date = {}
    strategy.max_holding_days = 1
    strategy.rebalance_days = 1
    strategy.stop_loss = -0.03
    strategy.leader_trailing_stop = 0.06
    strategy.core_trailing_stop = 0.05
    position_amount = {"SH600000": 100.0}
    fake_position = SimpleNamespace(
        get_stock_amount=lambda stock: position_amount.get(stock, 0.0),
        get_stock_list=lambda: [
            stock for stock, amount in position_amount.items() if amount > 0
        ],
    )
    tradable = {"value": False}
    fake_exchange = SimpleNamespace(
        is_stock_tradable=lambda **_kwargs: tradable["value"]
    )
    monkeypatch.setattr(
        RedisSectorMomentumLeaderCoreStrategy,
        "trade_position",
        property(lambda _self: fake_position),
    )
    monkeypatch.setattr(
        RedisSectorMomentumLeaderCoreStrategy,
        "trade_exchange",
        property(lambda _self: fake_exchange),
    )
    monkeypatch.setattr(
        extended_module,
        "TradeDecisionWO",
        lambda orders, owner: SimpleNamespace(
            orders=orders,
            strategy=owner,
            get_decision=lambda: orders,
        ),
    )
    monkeypatch.setattr(
        extended_module,
        "Order",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    contexts = iter(
        [
            (2, pd.Timestamp("2026-01-07"), pd.Timestamp("2026-01-07"), "2026-01-07"),
            (3, pd.Timestamp("2026-01-08"), pd.Timestamp("2026-01-08"), "2026-01-08"),
            (4, pd.Timestamp("2026-01-09"), pd.Timestamp("2026-01-09"), "2026-01-09"),
        ]
    )
    strategy._current_trade_context = lambda: next(contexts)
    strategy._load_prior_close_snapshot = lambda *_args: {}
    strategy._generate_execution_date_decision = lambda *_args: SimpleNamespace(
        get_decision=lambda: []
    )

    strategy.generate_trade_decision()
    assert strategy._forced_exit_blocked["SH600000"]["retries"] == 0

    strategy.generate_trade_decision()
    assert strategy._forced_exit_blocked["SH600000"]["retries"] == 1

    tradable["value"] = True
    recovered = strategy.generate_trade_decision()
    assert len(recovered.orders) == 1
    assert "SH600000" in strategy._forced_exit_blocked

    sell = recovered.orders[0]
    sell.deal_amount = sell.amount
    position_amount["SH600000"] = 0.0
    strategy._record_fills([(sell, 0, 0, 10.0)], trade_step=4)

    assert "SH600000" not in strategy._forced_exit_blocked
    assert strategy.signal_metadata["execution_diagnostics"] == [
        {
            "event": "forced_exit_blocked",
            "date": "2026-01-07",
            "stock": "SH600000",
            "reason": "max_holding_days",
            "retries": 0,
        },
        {
            "event": "forced_exit_retry",
            "date": "2026-01-08",
            "stock": "SH600000",
            "reason": "max_holding_days",
            "retries": 1,
        },
        {
            "event": "forced_exit_recovered",
            "date": "2026-01-09",
            "stock": "SH600000",
            "reason": "max_holding_days",
            "retries": 1,
        },
        {
            "event": "forced_exit_completed",
            "date": "2026-01-09",
            "stock": "SH600000",
            "reason": "max_holding_days",
            "retries": 1,
        },
    ]


def test_execution_gap_uses_open_and_previous_close_and_fails_closed(monkeypatch):
    strategy = RedisSectorMomentumLeaderCoreStrategy.__new__(
        RedisSectorMomentumLeaderCoreStrategy
    )
    strategy.candidate_details = {"2026-01-06": {"SH600000": {"role": "leader"}}}
    strategy.leader_gap_down = -0.03
    strategy.leader_gap_up = 0.05
    strategy.core_gap_down = -0.02
    strategy.core_gap_up = 0.03
    prices = {"SH600000": 106.0}
    fake_exchange = SimpleNamespace(
        get_deal_price=lambda stock_id, **_kwargs: prices[stock_id]
    )
    monkeypatch.setattr(
        RedisSectorMomentumLeaderCoreStrategy,
        "trade_exchange",
        property(lambda _self: fake_exchange),
    )
    order = SimpleNamespace(stock_id="SH600000")

    assert not strategy._passes_gap_filter(
        order,
        "2026-01-06",
        pd.Timestamp("2026-01-06"),
        pd.Timestamp("2026-01-06"),
        {"SH600000": {"close": 100.0}},
    )
    prices["SH600000"] = 104.0
    assert strategy._passes_gap_filter(
        order,
        "2026-01-06",
        pd.Timestamp("2026-01-06"),
        pd.Timestamp("2026-01-06"),
        {"SH600000": {"close": 100.0}},
    )
    assert not strategy._passes_gap_filter(
        order,
        "2026-01-06",
        pd.Timestamp("2026-01-06"),
        pd.Timestamp("2026-01-06"),
        {},
    )


def test_stop_trailing_and_trend_exit_state_machine():
    strategy = RedisSectorMomentumLeaderCoreStrategy.__new__(
        RedisSectorMomentumLeaderCoreStrategy
    )
    strategy.stop_loss = -0.03
    strategy.leader_trailing_stop = 0.06
    strategy.core_trailing_stop = 0.05
    strategy._position_context = {
        "STOP": {"entry_price": 100.0, "highest_close": 100.0, "role": "leader"},
        "TRAIL": {"entry_price": 90.0, "highest_close": 110.0, "role": "leader"},
        "MA20": {"entry_price": 90.0, "highest_close": 100.0, "role": "core"},
        "MA5": {
            "entry_price": 90.0,
            "highest_close": 100.0,
            "role": "core",
            "below_ma5_days": 0,
        },
    }

    assert (
        strategy._stock_exit_reason(
            "STOP", {"STOP": {"close": 96.0, "ma5": 95.0, "ma20": 94.0}}
        )
        == "stop_loss"
    )
    assert (
        strategy._stock_exit_reason(
            "TRAIL", {"TRAIL": {"close": 103.0, "ma5": 100.0, "ma20": 99.0}}
        )
        == "trailing_stop"
    )
    assert (
        strategy._stock_exit_reason(
            "MA20", {"MA20": {"close": 99.0, "ma5": 100.0, "ma20": 100.0}}
        )
        == "trend_below_ma20"
    )
    ma5_snapshot = {"MA5": {"close": 99.0, "ma5": 100.0, "ma20": 90.0}}
    assert strategy._stock_exit_reason("MA5", ma5_snapshot) is None
    assert strategy._stock_exit_reason("MA5", ma5_snapshot) == "trend_below_ma5_twice"


def test_board_score_drop_exit_uses_real_board_state_field() -> None:
    strategy = RedisSectorMomentumLeaderCoreStrategy.__new__(
        RedisSectorMomentumLeaderCoreStrategy
    )
    strategy._position_context = {
        "SH600000": {
            "board_code": "B1",
            "entry_board_score": 80.0,
            "role": "core",
        }
    }
    strategy.board_states = {
        "2026-01-06": {
            "B1": {
                "evaluable": True,
                "phase": "watch",
                "rank": 1,
                "board_score": 55.0,
            }
        }
    }
    strategy.board_exit_rank = 20
    strategy.board_exit_days = 2
    strategy.board_score_drop_exit = 20.0
    strategy._board_rank_breach_days = {"SH600000": 0}

    assert (
        strategy._board_exit_reason("SH600000", "2026-01-06")
        == "board_score_drop"
    )


def test_buy_order_is_capped_by_stock_board_and_market_position(monkeypatch):
    strategy = RedisSectorMomentumLeaderCoreStrategy.__new__(
        RedisSectorMomentumLeaderCoreStrategy
    )
    strategy.candidate_details = {
        "2026-01-06": {"SH600000": {"board_code": "B1", "role": "leader"}}
    }
    strategy.signal_metadata = {"market_breadth_by_date": {"2026-01-06": 0.35}}
    strategy.market_breadth_pause = 0.30
    strategy.market_breadth_reduce = 0.40
    strategy.weak_market_position = 0.50
    strategy.max_stock_weight = 0.10
    strategy.max_board_weight = 0.20
    fake_position = SimpleNamespace(
        calculate_value=lambda: 1_000_000.0,
        get_stock_amount=lambda _stock: 0.0,
    )
    monkeypatch.setattr(
        RedisSectorMomentumLeaderCoreStrategy,
        "trade_position",
        property(lambda _self: fake_position),
    )
    fake_exchange = SimpleNamespace(
        get_deal_price=lambda **_kwargs: 10.0,
        get_factor=lambda **_kwargs: 1.0,
        round_amount_by_trade_unit=lambda amount, _factor: int(amount // 100) * 100,
    )
    monkeypatch.setattr(
        RedisSectorMomentumLeaderCoreStrategy,
        "trade_exchange",
        property(lambda _self: fake_exchange),
    )
    order = SimpleNamespace(stock_id="SH600000", amount=20_000.0)

    capped = strategy._cap_buy_order(
        order,
        "2026-01-06",
        pd.Timestamp("2026-01-06"),
        pd.Timestamp("2026-01-06"),
        {"B1": 190_000.0},
        total_stock_value=450_000.0,
    )
    assert capped is order
    assert capped.amount == 1_000

    strategy.signal_metadata["market_breadth_by_date"]["2026-01-06"] = 0.20
    assert (
        strategy._cap_buy_order(
            SimpleNamespace(stock_id="SH600000", amount=20_000.0),
            "2026-01-06",
            pd.Timestamp("2026-01-06"),
            pd.Timestamp("2026-01-06"),
            {},
            total_stock_value=0.0,
        )
        is None
    )


def test_runtime_market_breadth_uses_only_signal_date_data():
    dates = pd.bdate_range("2026-01-01", periods=21)
    rows = []
    for symbol, slope in (("SH600000", 1.0), ("SZ000001", -1.0)):
        for index, day in enumerate(dates):
            rows.append(
                {
                    "trade_date": day,
                    "symbol": symbol,
                    "close": 100 + slope * index,
                    "adj_factor": 1.0,
                }
            )
    stock_daily = pd.DataFrame(rows)
    signal_date = pd.Timestamp(dates[-1])
    execution_date = signal_date + pd.offsets.BDay(1)

    breadth = QlibBacktestServiceRuntimeMixin._sector_market_breadth_by_execution_date(
        stock_daily, {signal_date: pd.Timestamp(execution_date)}
    )
    future = stock_daily.copy()
    future.loc[len(future)] = {
        "trade_date": pd.Timestamp(execution_date),
        "symbol": "SH600000",
        "close": 10_000.0,
        "adj_factor": 1.0,
    }
    mutated = QlibBacktestServiceRuntimeMixin._sector_market_breadth_by_execution_date(
        future, {signal_date: pd.Timestamp(execution_date)}
    )

    assert breadth == {str(pd.Timestamp(execution_date).date()): 0.5}
    assert mutated == breadth


def test_strategy_reads_already_lagged_score_on_execution_date(monkeypatch):
    strategy = RedisSectorMomentumLeaderCoreStrategy.__new__(
        RedisSectorMomentumLeaderCoreStrategy
    )
    calls = []
    fake_signal = SimpleNamespace(
        get_signal=lambda start_time, end_time: (
            calls.append((start_time, end_time)) or pd.Series(dtype=float)
        )
    )
    strategy.signal = fake_signal
    monkeypatch.setattr(
        extended_module,
        "TradeDecisionWO",
        lambda orders, owner: SimpleNamespace(orders=orders, strategy=owner),
    )
    execution_date = pd.Timestamp("2026-01-06")

    strategy._generate_execution_date_decision(execution_date, execution_date)

    assert calls == [(execution_date, execution_date)]
