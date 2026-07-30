from backend.shared.feature_catalog_persistence import (
    DEFAULT_CATALOG_PATH,
    _SCHEMA_STATEMENTS,
    _catalog_payload,
)


def test_feature_catalog_schema_covers_all_registry_tables():
    ddl = "\n".join(_SCHEMA_STATEMENTS)
    for table in (
        "qm_feature_category",
        "qm_feature_definition",
        "qm_feature_set_version",
        "qm_feature_set_item",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in ddl


def test_bundled_feature_catalog_is_seedable():
    payload = _catalog_payload(DEFAULT_CATALOG_PATH)
    assert payload["version_id"]
    assert payload["categories"]
    feature_count = sum(
        len(category.get("features") or []) for category in payload["categories"]
    )
    assert feature_count >= 52
