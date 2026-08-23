from contextlib import asynccontextmanager
import logging
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from backend.services.api.user_app.schemas.user import UserLogin
from backend.services.api.user_app.services import auth_service as auth_service_module
from backend.services.api.user_app.services.auth_service import AuthService


class _LoginAttemptManager:
    def is_locked(self, _tenant_id: str, _identifier: str) -> bool:
        return False

    def record_failed_attempt(self, _tenant_id: str, _identifier: str) -> None:
        pass


class _UserResult:
    def __init__(self, user) -> None:
        self.user = user

    def scalar_one_or_none(self):
        return self.user


class _UserSession:
    def __init__(self, user) -> None:
        self.user = user

    async def execute(self, _statement):
        return _UserResult(self.user)


class AuthLoggingSecurityTest(unittest.IsolatedAsyncioTestCase):
    async def test_successful_login_never_logs_plaintext_password(self) -> None:
        secret_password = "DoNotLog-Secret-2026"
        user = SimpleNamespace(
            user_id="10000001",
            tenant_id="default",
            username="admin",
            password_hash="stored-hash",
            is_active=True,
            is_locked=False,
        )

        @asynccontextmanager
        async def _get_session(*, read_only: bool):
            self.assertTrue(read_only)
            yield _UserSession(user)

        messages: list[str] = []

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                messages.append(record.getMessage())

        service = AuthService.__new__(AuthService)
        service.login_attempt_manager = _LoginAttemptManager()
        service._verify_password = (
            lambda password, _hash: password == secret_password
        )
        service._finalize_login = AsyncMock(return_value=object())

        handler = _Handler()
        logger = auth_service_module.logger
        original_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            with patch.object(auth_service_module, "get_session", _get_session):
                await service.login(
                    UserLogin(
                        tenant_id="default",
                        username="admin",
                        password=secret_password,
                    )
                )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)

        self.assertNotIn(secret_password, "\n".join(messages))
