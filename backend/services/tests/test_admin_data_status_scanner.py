from pathlib import Path

from backend.services.api.routers.admin import data_status_scanner


def test_a_share_qlib_dir_uses_shared_provider_resolution(monkeypatch) -> None:
    monkeypatch.setattr(
        data_status_scanner,
        "resolve_qlib_provider_uri",
        lambda market: "/app/db/qlib_data" if market == "CN" else "",
    )

    assert data_status_scanner._resolve_qlib_dir("a_share") == Path(
        "/app/db/qlib_data"
    )


def test_non_a_share_qlib_dir_keeps_market_specific_mapping(monkeypatch) -> None:
    expected = Path("/tmp/quantmind-hk-cache")
    monkeypatch.setitem(data_status_scanner._MARKET_QLIB_DIRS, "hong_kong", expected)

    assert data_status_scanner._resolve_qlib_dir("hong_kong") == expected
