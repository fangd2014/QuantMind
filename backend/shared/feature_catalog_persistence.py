from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import text

from backend.shared.database_manager_v2 import get_session


logger = logging.getLogger(__name__)
DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "features"
    / "model_training_feature_catalog_v1.json"
)


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS qm_feature_category (
      category_id TEXT PRIMARY KEY,
      category_name TEXT NOT NULL,
      sort_order INTEGER NOT NULL DEFAULT 0,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS qm_feature_definition (
      feature_id TEXT PRIMARY KEY,
      feature_key TEXT NOT NULL UNIQUE,
      feature_name TEXT NOT NULL,
      formula TEXT,
      source_table_fields TEXT,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS qm_feature_set_version (
      version_id TEXT PRIMARY KEY,
      version_name TEXT NOT NULL,
      description TEXT,
      feature_count INTEGER NOT NULL DEFAULT 0,
      status TEXT NOT NULL DEFAULT 'inactive',
      effective_at TIMESTAMPTZ,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS qm_feature_set_item (
      version_id TEXT NOT NULL REFERENCES qm_feature_set_version(version_id) ON DELETE CASCADE,
      feature_key TEXT NOT NULL REFERENCES qm_feature_definition(feature_key) ON DELETE CASCADE,
      category_id TEXT NOT NULL REFERENCES qm_feature_category(category_id),
      order_no INTEGER NOT NULL DEFAULT 0,
      enabled BOOLEAN NOT NULL DEFAULT TRUE,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      PRIMARY KEY (version_id, feature_key)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_qm_feature_set_version_status
      ON qm_feature_set_version (status, effective_at DESC, created_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_qm_feature_set_item_order
      ON qm_feature_set_item (version_id, category_id, order_no)
    """,
)


def _catalog_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("categories"), list):
        raise ValueError(f"invalid feature catalog: {path}")
    return payload


async def ensure_feature_catalog(
    catalog_path: Path = DEFAULT_CATALOG_PATH,
) -> dict[str, Any]:
    """Create and idempotently seed the feature registry used by model APIs."""
    payload = _catalog_payload(catalog_path)
    version_id = str(payload.get("version_id") or "qm_feature_set_v1").strip()
    if not version_id:
        raise ValueError("feature catalog version_id is required")

    categories = [item for item in payload.get("categories") or [] if isinstance(item, dict)]
    feature_count = 0
    async with get_session() as session:
        for statement in _SCHEMA_STATEMENTS:
            await session.execute(text(statement))

        await session.execute(
            text(
                """
                INSERT INTO qm_feature_set_version (
                  version_id, version_name, description, feature_count,
                  status, effective_at, created_at, updated_at
                ) VALUES (
                  :version_id, :version_name, :description, :feature_count,
                  'active', NOW(), NOW(), NOW()
                )
                ON CONFLICT (version_id) DO UPDATE SET
                  version_name = EXCLUDED.version_name,
                  description = EXCLUDED.description,
                  feature_count = EXCLUDED.feature_count,
                  status = 'active',
                  effective_at = COALESCE(qm_feature_set_version.effective_at, NOW()),
                  updated_at = NOW()
                """
            ),
            {
                "version_id": version_id,
                "version_name": str(payload.get("name") or version_id),
                "description": str(payload.get("description") or ""),
                "feature_count": int(payload.get("feature_count") or 0),
            },
        )
        await session.execute(
            text(
                """
                UPDATE qm_feature_set_version
                SET status = 'inactive', updated_at = NOW()
                WHERE version_id <> :version_id AND status = 'active'
                """
            ),
            {"version_id": version_id},
        )

        for category_index, category in enumerate(categories, start=1):
            category_id = str(category.get("id") or "").strip()
            if not category_id:
                continue
            await session.execute(
                text(
                    """
                    INSERT INTO qm_feature_category (
                      category_id, category_name, sort_order, created_at, updated_at
                    ) VALUES (
                      :category_id, :category_name, :sort_order, NOW(), NOW()
                    )
                    ON CONFLICT (category_id) DO UPDATE SET
                      category_name = EXCLUDED.category_name,
                      sort_order = EXCLUDED.sort_order,
                      updated_at = NOW()
                    """
                ),
                {
                    "category_id": category_id,
                    "category_name": str(category.get("name") or category_id),
                    "sort_order": int(category.get("order") or category_index),
                },
            )
            features = [item for item in category.get("features") or [] if isinstance(item, dict)]
            for feature_index, feature in enumerate(features, start=1):
                feature_key = str(feature.get("key") or "").strip()
                if not feature_key:
                    continue
                feature_id = str(feature.get("feature_id") or f"feat_{feature_key}").strip()
                feature_count += 1
                await session.execute(
                    text(
                        """
                        INSERT INTO qm_feature_definition (
                          feature_id, feature_key, feature_name, formula,
                          source_table_fields, created_at, updated_at
                        ) VALUES (
                          :feature_id, :feature_key, :feature_name, :formula,
                          :source_table_fields, NOW(), NOW()
                        )
                        ON CONFLICT (feature_key) DO UPDATE SET
                          feature_name = EXCLUDED.feature_name,
                          formula = EXCLUDED.formula,
                          source_table_fields = EXCLUDED.source_table_fields,
                          updated_at = NOW()
                        """
                    ),
                    {
                        "feature_id": feature_id,
                        "feature_key": feature_key,
                        "feature_name": str(
                            feature.get("description")
                            or feature.get("feature_name")
                            or feature_key
                        ),
                        "formula": str(feature.get("formula") or ""),
                        "source_table_fields": str(
                            feature.get("source") or feature.get("source_table_fields") or ""
                        ),
                    },
                )
                await session.execute(
                    text(
                        """
                        INSERT INTO qm_feature_set_item (
                          version_id, feature_key, category_id, order_no,
                          enabled, created_at, updated_at
                        ) VALUES (
                          :version_id, :feature_key, :category_id, :order_no,
                          :enabled, NOW(), NOW()
                        )
                        ON CONFLICT (version_id, feature_key) DO UPDATE SET
                          category_id = EXCLUDED.category_id,
                          order_no = EXCLUDED.order_no,
                          enabled = EXCLUDED.enabled,
                          updated_at = NOW()
                        """
                    ),
                    {
                        "version_id": version_id,
                        "feature_key": feature_key,
                        "category_id": category_id,
                        "order_no": int(feature.get("order_no") or feature_index),
                        "enabled": bool(feature.get("enabled", True)),
                    },
                )

        await session.execute(
            text(
                """
                UPDATE qm_feature_set_version
                SET feature_count = :feature_count, updated_at = NOW()
                WHERE version_id = :version_id
                """
            ),
            {"version_id": version_id, "feature_count": feature_count},
        )
    logger.info(
        "Feature catalog ready: version=%s features=%s", version_id, feature_count
    )
    return {"version_id": version_id, "feature_count": feature_count}
