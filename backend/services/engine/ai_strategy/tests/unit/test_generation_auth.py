import os
import sys
import types
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_URL", "sqlite:///tmp-test.db")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")

fake_redis_client = types.ModuleType("backend.shared.redis_sentinel_client")
fake_redis_client.get_redis_sentinel_client = lambda: SimpleNamespace(
    setex=lambda *args, **kwargs: None,
    get=lambda *args, **kwargs: None,
)
sys.modules.setdefault("backend.shared.redis_sentinel_client", fake_redis_client)

fake_selection = types.ModuleType("backend.services.engine.ai_strategy.services.selection")
fake_selection.get_intent_parser = lambda: None
fake_selection.get_sql_generator = lambda: None
sys.modules.setdefault("backend.services.engine.ai_strategy.services.selection", fake_selection)

from backend.services.engine.ai_strategy.api.v1 import generation, routes


@pytest.fixture
def auth_client():
    app = FastAPI()

    @app.middleware("http")
    async def inject_user(request: Request, call_next):
        user_id = request.headers.get("X-User-Id")
        tenant_id = request.headers.get("X-Tenant-Id", "default")
        if user_id:
            request.state.user = {"user_id": user_id, "tenant_id": tenant_id}
        return await call_next(request)

    app.include_router(routes.router, prefix="/api/v1")
    app.include_router(generation.router, prefix="/api/v1")
    return TestClient(app)


def test_strategy_generate_uses_profile_key_with_request_scoped_deepseek(monkeypatch, auth_client):
    captured: dict[str, object] = {}

    async def _fake_load_profile_api_key(user_id: str) -> str | None:
        captured["lookup_user_id"] = user_id
        return "profile-real-key"

    class _DummyProvider:
        def __init__(self, api_key: str | None = None):
            captured["provider_api_key"] = api_key

        async def generate(self, req):
            captured["provider_user_id"] = req.user_id
            return SimpleNamespace(
                strategy_name="S1",
                rationale="R1",
                artifacts=[],
                metadata={},
                provider="deepseek",
                generated_at="2026-08-20T00:00:00",
            )

    monkeypatch.setenv("DEEPSEEK_API_KEY", "mock-api-key")
    monkeypatch.setattr(routes, "_load_profile_api_key", _fake_load_profile_api_key)
    monkeypatch.setattr(routes, "DeepseekProvider", _DummyProvider)
    monkeypatch.setattr(routes, "get_provider", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("global provider should not be used")))

    resp = auth_client.post(
        "/api/v1/strategy/generate",
        headers={"X-User-Id": "user-1"},
        json={"description": "test strategy", "user_id": "user-1"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert captured["lookup_user_id"] == "user-1"
    assert captured["provider_api_key"] == "profile-real-key"
    assert captured["provider_user_id"] == "user-1"


def test_generate_qlib_rejects_cross_user_request(auth_client):
    resp = auth_client.post(
        "/api/v1/generate-qlib",
        headers={"X-User-Id": "user-1"},
        json={"user_id": "user-2", "conditions": {}},
    )

    assert resp.status_code == 403
    assert "user_id" in resp.json()["detail"]


def test_remote_import_rejects_foreign_cos_prefix(auth_client):
    resp = auth_client.post(
        "/api/v1/remote/import",
        headers={"X-User-Id": "user-1"},
        json={
            "user_id": "user-1",
            "files": ["user_strategies/user-2/2026/08/strategy.py"],
        },
    )

    assert resp.status_code == 403
    assert "COS 文件不属于当前用户" in resp.json()["detail"]
