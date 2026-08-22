from __future__ import annotations

from backend.services.engine.rd_agent.market_adapters.a_share import AShareAdapter


def test_a_share_uses_shared_qlib_path_resolver(monkeypatch):
    from backend.shared import qlib_paths

    monkeypatch.setattr(
        qlib_paths,
        "resolve_qlib_provider_uri",
        lambda market="CN": "/mounted/qlib_data",
    )

    assert AShareAdapter().get_qlib_provider_uri() == "/mounted/qlib_data"


def test_a_share_ready_requires_complete_qlib_layout(monkeypatch, tmp_path):
    adapter = AShareAdapter()
    monkeypatch.setattr(adapter, "get_qlib_provider_uri", lambda: str(tmp_path))

    assert adapter.is_data_ready() is False

    (tmp_path / "calendars").mkdir()
    (tmp_path / "calendars" / "day.txt").write_text(
        "2026-08-21\n", encoding="utf-8"
    )
    (tmp_path / "instruments").mkdir()
    (tmp_path / "instruments" / "all.txt").write_text(
        "SH600000\t2020-01-01\t2026-08-21\n", encoding="utf-8"
    )
    (tmp_path / "features").mkdir()

    assert adapter.is_data_ready() is True
