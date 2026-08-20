import pytest
from starlette.requests import Request

from backend.services.engine.routers.ai_ide import config as llm_config


class _Response:
    status_code = 200
    text = ""


class _Client:
    def __init__(self) -> None:
        self.payload = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def put(self, _url, *, headers, json):
        assert headers["X-User-Id"] == "user-1"
        self.payload = json
        return _Response()


@pytest.mark.asyncio
async def test_empty_deepseek_key_clears_profile(monkeypatch):
    client = _Client()
    monkeypatch.setattr(llm_config.httpx, "AsyncClient", lambda **_kwargs: client)

    request = Request({"type": "http", "headers": []})
    request.state.user = {"user_id": "user-1", "tenant_id": "default"}

    result = await llm_config.save_llm_config(
        request,
        llm_config.LLMConfig(deepseek_api_key=""),
    )

    assert client.payload == {"api_key": ""}
    assert result == {"success": True, "message": "API Key 已清除"}
