from __future__ import annotations

import asyncio

from sqlalchemy import String
from sqlalchemy.dialects.postgresql.asyncpg import dialect as asyncpg_dialect

from backend.services.trade.portfolio.models import Portfolio
from backend.services.trade.routers.real_trading_utils import (
    _fetch_active_portfolio_snapshot,
)


class _EmptyScalarResult:
    def scalars(self):
        return self

    def first(self):
        return None


class _CapturingSession:
    def __init__(self) -> None:
        self.statement = None

    async def execute(self, statement):
        self.statement = statement
        return _EmptyScalarResult()


def test_portfolio_user_id_matches_production_varchar_schema() -> None:
    assert isinstance(Portfolio.__table__.c.user_id.type, String)


def test_active_portfolio_query_binds_user_id_as_varchar() -> None:
    session = _CapturingSession()

    result = asyncio.run(
        _fetch_active_portfolio_snapshot(
            session,
            tenant_id="default",
            user_id="10000001",
            strategy_id=None,
            mode="REAL",
        )
    )

    assert result is None
    compiled = session.statement.compile(dialect=asyncpg_dialect())
    user_id_bind = next(
        bind
        for key, bind in compiled.binds.items()
        if key.startswith("user_id") and not key.startswith("%(")
    )
    assert isinstance(user_id_bind.type, String)
    assert user_id_bind.value == "10000001"
