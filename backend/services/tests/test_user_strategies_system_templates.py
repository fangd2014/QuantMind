import sys
import types
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

# services/__init__.py eagerly imports the full Qlib backtest runtime.  These API
# tests only need the leaf strategy_templates module, so expose the package path
# without executing that unrelated heavyweight initializer.
_SERVICES_PACKAGE = "backend.services.engine.qlib_app.services"
if _SERVICES_PACKAGE not in sys.modules:
    services_package = types.ModuleType(_SERVICES_PACKAGE)
    services_package.__path__ = [
        str(Path(__file__).resolve().parents[1] / "engine/qlib_app/services")
    ]
    sys.modules[_SERVICES_PACKAGE] = services_package

from backend.services.engine.qlib_app.api import user_strategies
from backend.services.engine.qlib_app.services.strategy_templates import get_all_templates
from backend.shared import strategy_storage
from backend.shared.strategy_storage import StrategyStorageService


class _StorageStub:
    def __init__(self, items):
        self._items = items

    def list(self, **_kwargs):
        return self._items


class _WritableStorageStub:
    def __init__(self, current=None):
        self.current = current
        self.saved = None

    async def save(self, **kwargs):
        self.saved = kwargs
        if self.current is not None:
            self.current = {
                **self.current,
                "id": kwargs.get("strategy_id", self.current.get("id")),
                "name": kwargs["name"],
                "code": kwargs["code"],
                **kwargs["metadata"],
            }
        return {"id": kwargs.get("strategy_id") or "303", "cos_url": None}

    async def get(self, _strategy_id, user_id=None):
        return self.current


def _request() -> Request:
    request = Request({"type": "http", "headers": []})
    request.state.user = {"user_id": "test-user", "tenant_id": "default"}
    return request


def test_strategy_update_route_is_registered_once():
    matching = [
        route
        for route in user_strategies.router.routes
        if route.path == "/{strategy_id}" and "PUT" in (route.methods or set())
    ]

    assert len(matching) == 1
    assert matching[0].endpoint is user_strategies.update_user_strategy


def test_system_template_items_include_all_file_templates_and_defaults():
    templates = get_all_templates()

    items = user_strategies._build_system_template_items(templates)

    assert len(items) == 11
    assert all(item["id"].startswith("sys_") for item in items)
    assert all(item["is_system"] is True for item in items)

    sector_item = next(
        item for item in items if item["id"] == "sys_sector_momentum_leader_core"
    )
    sector_template = next(
        template for template in templates if template.id == "sector_momentum_leader_core"
    )
    assert sector_item["parameters"]["strategy_type"] == sector_template.id
    assert sector_item["parameters"]["board_universe"] == "sw_l1"
    assert sector_item["parameters"]["max_holding_days"] == 10
    assert sector_item["execution_defaults"] == sector_item["execution_config"]
    assert sector_item["live_defaults"] == sector_item["live_trade_config"]
    assert set(sector_item["parameters"]) == {
        "strategy_type",
        *(parameter.name for parameter in sector_template.params),
    }


@pytest.mark.asyncio
async def test_list_merges_system_templates_and_filters_synced_copies(monkeypatch):
    storage = _StorageStub(
        [
            {
                "id": "101",
                "name": "板块动量轮动+龙头中军选股",
                "description": "旧同步副本",
                "status": "ACTIVE",
                "tags": ["advanced", "SystemSync"],
                "is_verified": True,
                "parameters": {"strategy_type": "sector_momentum_leader_core"},
                "created_at": None,
                "updated_at": None,
            },
            {
                "id": "202",
                "name": "我的个人策略",
                "description": "用户创建",
                "status": "DRAFT",
                "tags": ["personal"],
                "is_verified": False,
                "parameters": {"topk": 3},
                "created_at": None,
                "updated_at": None,
            },
        ]
    )
    monkeypatch.setattr(user_strategies, "get_strategy_storage_service", lambda: storage)
    monkeypatch.setattr(
        user_strategies, "_fetch_latest_backtest_summaries", AsyncMock(return_value={})
    )
    monkeypatch.setattr(
        user_strategies, "_fetch_real_trading_status", AsyncMock(return_value=None)
    )

    response = await user_strategies.list_user_strategies(
        _request(), category=None, search=None, tags=None
    )

    system_items = [item for item in response.strategies if item.is_system]
    assert response.total == 12
    assert len(system_items) == 11
    assert sum(item.name == "板块动量轮动+龙头中军选股" for item in response.strategies) == 1
    assert any(item.id == "202" and not item.is_system for item in response.strategies)


@pytest.mark.asyncio
async def test_system_template_detail_returns_complete_defaults():
    storage = object.__new__(StrategyStorageService)

    detail = await storage.get("sys_sector_momentum_leader_core", user_id="test-user")

    template = next(
        item for item in get_all_templates() if item.id == "sector_momentum_leader_core"
    )
    assert detail is not None
    assert detail["is_system"] is True
    assert detail["parameters"]["board_universe"] == "sw_l1"
    assert detail["parameters"]["max_holding_days"] == 10
    assert detail["execution_defaults"] == detail["execution_config"]
    assert detail["live_defaults"] == detail["live_trade_config"]
    assert set(detail["parameters"]) == {
        "strategy_type",
        *(parameter.name for parameter in template.params),
    }


def test_storage_list_preserves_database_parameters(monkeypatch):
    now = datetime.now(timezone.utc)
    row = (
        202,
        "我的个人策略",
        "用户创建",
        "DRAFT",
        None,
        None,
        "hash",
        ["personal"],
        False,
        {},
        {"topk": 3},
        now,
        now,
    )

    class _Result:
        def fetchall(self):
            return [row]

    class _Session:
        def execute(self, *_args, **_kwargs):
            return _Result()

    @contextmanager
    def _get_db():
        yield _Session()

    monkeypatch.setattr(strategy_storage, "get_db", _get_db)
    storage = object.__new__(StrategyStorageService)
    storage._has_cos_key_col = False

    items = storage.list(user_id="test-user")

    assert items[0]["parameters"] == {"topk": 3}


@pytest.mark.asyncio
async def test_create_strategy_persists_expert_mode_payload(monkeypatch):
    storage = _WritableStorageStub()
    monkeypatch.setattr(
        user_strategies, "get_strategy_storage_service", lambda: storage
    )

    response = await user_strategies.create_user_strategy(
        user_strategies.SaveStrategyRequest(
            name=" 专家策略_0802_1030 ",
            code='STRATEGY_CONFIG = {"class": "RedisTopkStrategy", "kwargs": {}}',
            description="由专家模式生成",
            category="manual_created",
            tags=["ExpertMode"],
            parameters={"topk": 50},
        ),
        _request(),
    )

    assert response["success"] is True
    assert response["strategy_id"] == "303"
    assert storage.saved["name"] == "专家策略_0802_1030"
    assert storage.saved["metadata"]["strategy_type"] == "CUSTOM"
    assert storage.saved["metadata"]["config"]["category"] == "manual_created"
    assert storage.saved["metadata"]["parameters"] == {"topk": 50}
    assert storage.saved["metadata"]["is_verified"] is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "CUSTOM"),
        ("manual_created", "CUSTOM"),
        ("quantitative", "QUANTITATIVE"),
        (" TECHNICAL ", "TECHNICAL"),
    ],
)
def test_strategy_type_is_normalized_to_database_enum(value, expected):
    assert strategy_storage.normalize_strategy_type(value) == expected


def test_storage_upsert_normalizes_strategy_type_before_sql(monkeypatch):
    captured_params = {}

    class _Result:
        def scalar(self):
            return 303

    class _Session:
        def execute(self, _sql, params=None):
            if params:
                captured_params.update(params)
            return _Result()

    @contextmanager
    def _get_db():
        yield _Session()

    monkeypatch.setattr(strategy_storage, "get_db", _get_db)
    storage = object.__new__(StrategyStorageService)
    storage._has_cos_key_col = False

    strategy_id = storage._db_upsert(
        user_id="1",
        strategy_id=None,
        name="专家策略",
        code="pass",
        cos_key="unused",
        cos_url=None,
        file_size=4,
        hash_val="hash",
        metadata={"strategy_type": "manual_created"},
    )

    assert strategy_id == "303"
    assert captured_params["stype"] == "CUSTOM"


@pytest.mark.asyncio
async def test_update_strategy_code_returns_to_unverified_draft(monkeypatch):
    storage = _WritableStorageStub(
        {
            "id": "202",
            "name": "旧策略",
            "description": "旧描述",
            "code": "OLD_CODE = True",
            "strategy_type": "CUSTOM",
            "status": "ACTIVE",
            "config": {},
            "parameters": {"topk": 3},
            "execution_config": {},
            "tags": ["personal"],
            "is_public": False,
            "is_verified": True,
        }
    )
    monkeypatch.setattr(
        user_strategies, "get_strategy_storage_service", lambda: storage
    )

    updated = await user_strategies.update_user_strategy(
        "202",
        user_strategies.UpdateStrategyRequest(code="NEW_CODE = True"),
        _request(),
    )

    assert storage.saved["metadata"]["status"] == "DRAFT"
    assert storage.saved["metadata"]["is_verified"] is False
    assert storage.saved["metadata"]["parameters"] == {"topk": 3}
    assert updated["code"] == "NEW_CODE = True"


@pytest.mark.asyncio
async def test_update_rejects_system_templates():
    with pytest.raises(HTTPException, match="系统内置策略不可修改") as exc_info:
        await user_strategies.update_user_strategy(
            "sys_sector_momentum_leader_core",
            user_strategies.UpdateStrategyRequest(code="pass"),
            _request(),
        )

    assert exc_info.value.status_code == 400
