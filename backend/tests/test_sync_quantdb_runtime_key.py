"""standalone QuantDB 同步脚本应读取后台管理页保存的 runtime.env。"""

import importlib
import sys

import pytest


def _load_sync_module(monkeypatch, tmp_path):
    pytest.importorskip("quantdb_sdk")
    monkeypatch.setenv("QM_QUANTDB_DATA_DIR", str(tmp_path / "quantdb"))
    runtime_env = tmp_path / "runtime.env"
    runtime_env.write_text("QUANTDB_API_KEY=runtime-key-for-test\n", encoding="utf-8")
    monkeypatch.setenv("QM_RUNTIME_ENV_FILE", str(runtime_env))
    monkeypatch.delenv("QUANTDB_API_KEY", raising=False)
    sys.modules.pop("sync_quantdb", None)
    return importlib.import_module("sync_quantdb")


def test_standalone_sync_uses_runtime_env_key(monkeypatch, tmp_path):
    module = _load_sync_module(monkeypatch, tmp_path)
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "QuantDBClient", FakeClient)
    module.make_client()

    assert captured["api_key"] == "runtime-key-for-test"


def test_daily_sync_creates_all_required_root_directories(monkeypatch, tmp_path):
    pytest.importorskip("pandas")
    sys.modules.pop("backend.scripts.quantdb_daily_sync", None)
    module = importlib.import_module("backend.scripts.quantdb_daily_sync")
    data_dir = tmp_path / "quantdb"
    monkeypatch.setattr(module, "QUANTDB_DATA_DIR", data_dir)

    module.ensure_data_layout()

    assert {
        "1_kline_data",
        "2_base_sector",
        "3_financial_data",
        "5_technical_derived",
        "6_ml_datasets",
    } <= {p.name for p in data_dir.iterdir()}
