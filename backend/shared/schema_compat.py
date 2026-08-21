"""独立于全量初始化脚本的幂等兼容迁移。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

_DATA_QUALITY_ALERTS_MIGRATION = (
    Path(__file__).resolve().parent
    / "migrations"
    / "20260821_ensure_data_quality_alerts.sql"
)


def ensure_data_quality_alerts_table(
    connect: Callable[..., Any] | None = None,
) -> None:
    """确保数据质量告警表与索引存在。

    该迁移必须独立执行：旧库的整份 ``db_init.sql`` 可能因前序对象冲突而
    中断，不能让后面的告警表建表语句因此被跳过。``connect`` 参数仅用于
    单元测试注入，生产默认使用 psycopg2。
    """
    if connect is None:
        import psycopg2

        connect = psycopg2.connect

    sql = _DATA_QUALITY_ALERTS_MIGRATION.read_text(encoding="utf-8")
    conn = connect(
        host=os.getenv("DB_HOST", os.getenv("POSTGRES_HOST", "db")),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", os.getenv("POSTGRES_DB", "quantmind")),
        user=os.getenv("DB_USER", os.getenv("POSTGRES_USER", "quantmind")),
        password=os.getenv(
            "DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", "quantmind2026")
        ),
    )
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)
    finally:
        conn.close()
