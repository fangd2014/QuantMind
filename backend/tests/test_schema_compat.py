from __future__ import annotations

from backend.shared.schema_compat import ensure_data_quality_alerts_table


class _Cursor:
    def __init__(self):
        self.sql = ""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, sql: str):
        self.sql = sql


class _Connection:
    def __init__(self):
        self.autocommit = False
        self.closed = False
        self.cursor_instance = _Cursor()

    def cursor(self):
        return self.cursor_instance

    def close(self):
        self.closed = True


def test_data_quality_alerts_compat_migration_is_complete():
    conn = _Connection()

    ensure_data_quality_alerts_table(connect=lambda **_kwargs: conn)

    assert conn.autocommit is True
    assert conn.closed is True
    assert "CREATE TABLE IF NOT EXISTS data_quality_alerts" in conn.cursor_instance.sql
    assert "idx_dqa_created_at" in conn.cursor_instance.sql
    assert "idx_dqa_unack_severity" in conn.cursor_instance.sql
    assert "idx_dqa_market_field" in conn.cursor_instance.sql
