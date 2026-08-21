from __future__ import annotations

from backend.shared.runtime_secrets import get_quantdb_api_key


def test_quantdb_key_is_loaded_from_runtime_env(monkeypatch, tmp_path):
    runtime_env = tmp_path / "runtime.env"
    runtime_env.write_text("QUANTDB_API_KEY=test-runtime-key\n", encoding="utf-8")
    monkeypatch.setenv("QM_RUNTIME_ENV_FILE", str(runtime_env))
    monkeypatch.delenv("QUANTDB_API_KEY", raising=False)

    assert get_quantdb_api_key() == "test-runtime-key"


def test_nonempty_process_env_has_priority(monkeypatch, tmp_path):
    runtime_env = tmp_path / "runtime.env"
    runtime_env.write_text("QUANTDB_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setenv("QM_RUNTIME_ENV_FILE", str(runtime_env))
    monkeypatch.setenv("QUANTDB_API_KEY", "process-key")

    assert get_quantdb_api_key() == "process-key"
