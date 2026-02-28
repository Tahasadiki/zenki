"""Tests for the ZenkiOrchestrator (zenki.sdk.orchestrator).

These tests mock the SDK's ``query`` function so that no real LLM calls are
made, while still exercising session management, model routing, and error
handling end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from zenki.config.settings import ZenkiSettings
from zenki.db.database import ZenkiDatabase
from zenki.sdk.orchestrator import ZenkiOrchestrator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> ZenkiDatabase:
    """Return an initialised ZenkiDatabase backed by a temp file."""
    database = ZenkiDatabase(tmp_path / "test_orchestrator.db")
    database.initialize()
    return database


@pytest.fixture()
def settings() -> ZenkiSettings:
    """Return default settings."""
    return ZenkiSettings()


@pytest.fixture()
def orchestrator(settings: ZenkiSettings, db: ZenkiDatabase) -> ZenkiOrchestrator:
    """Return a ZenkiOrchestrator with mocked SDK internals."""
    with patch("zenki.sdk.orchestrator.get_agents", return_value={}), \
         patch("zenki.sdk.orchestrator.create_zenki_tools", return_value={}), \
         patch("zenki.sdk.orchestrator.create_zenki_hooks", return_value={}):
        orch = ZenkiOrchestrator(settings=settings, db=db)
    # Patch run_query so it never calls the real SDK
    orch.run_query = AsyncMock(return_value="Mock response")
    return orch


# ===========================================================================
# Basic message processing
# ===========================================================================


class TestProcessMessage:
    """Tests for ZenkiOrchestrator.process_message."""

    @pytest.mark.asyncio
    async def test_returns_response(self, orchestrator: ZenkiOrchestrator) -> None:
        result = await orchestrator.process_message("Hello!")
        assert result == "Mock response"

    @pytest.mark.asyncio
    async def test_stores_user_message_in_db(
        self, orchestrator: ZenkiOrchestrator, db: ZenkiDatabase
    ) -> None:
        await orchestrator.process_message("Test message", user_id="u1")

        sessions = db.list_sessions(user_id="u1")
        assert len(sessions) >= 1
        messages = db.get_messages_for_session(sessions[0].id)
        user_messages = [m for m in messages if m.role == "user"]
        assert len(user_messages) == 1
        assert user_messages[0].content == "Test message"

    @pytest.mark.asyncio
    async def test_stores_assistant_response_in_db(
        self, orchestrator: ZenkiOrchestrator, db: ZenkiDatabase
    ) -> None:
        await orchestrator.process_message("Ping", user_id="u2")

        sessions = db.list_sessions(user_id="u2")
        messages = db.get_messages_for_session(sessions[0].id)
        assistant_messages = [m for m in messages if m.role == "assistant"]
        assert len(assistant_messages) == 1
        assert assistant_messages[0].content == "Mock response"


# ===========================================================================
# Session management
# ===========================================================================


class TestSessionHandling:
    """Tests for session creation and reuse within the orchestrator."""

    @pytest.mark.asyncio
    async def test_creates_session_when_none_provided(
        self, orchestrator: ZenkiOrchestrator, db: ZenkiDatabase
    ) -> None:
        await orchestrator.process_message("First message", user_id="auto-user")
        sessions = db.list_sessions(user_id="auto-user")
        assert len(sessions) == 1

    @pytest.mark.asyncio
    async def test_reuses_existing_session(
        self, orchestrator: ZenkiOrchestrator, db: ZenkiDatabase
    ) -> None:
        await orchestrator.process_message(
            "Message 1", user_id="reuse-user", channel_type="cli"
        )
        await orchestrator.process_message(
            "Message 2", user_id="reuse-user", channel_type="cli"
        )

        sessions = db.list_sessions(user_id="reuse-user")
        assert len(sessions) == 1
        messages = db.get_messages_for_session(sessions[0].id)
        # 2 user messages + 2 assistant messages = 4 total.
        assert len(messages) == 4

    @pytest.mark.asyncio
    async def test_uses_provided_session_id(
        self, orchestrator: ZenkiOrchestrator, db: ZenkiDatabase
    ) -> None:
        session = orchestrator.session_manager.create_session(
            user_id="sid-user", channel_type="cli"
        )
        await orchestrator.process_message("Hi", session_id=session.id, user_id="sid-user")

        messages = db.get_messages_for_session(session.id)
        assert len(messages) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_creates_new_session_if_provided_id_invalid(
        self, orchestrator: ZenkiOrchestrator, db: ZenkiDatabase
    ) -> None:
        result = await orchestrator.process_message(
            "Hello",
            session_id="invalid-session-id",
            user_id="fallback-user",
        )
        assert result == "Mock response"
        sessions = db.list_sessions(user_id="fallback-user")
        assert len(sessions) == 1


# ===========================================================================
# Model routing
# ===========================================================================


class TestModelRouting:
    """Tests verifying that model routing is applied via the router."""

    @pytest.mark.asyncio
    async def test_simple_message_routes_to_haiku(
        self, orchestrator: ZenkiOrchestrator
    ) -> None:
        await orchestrator.process_message("hi")
        orchestrator.run_query.assert_called_once()
        _, kwargs = orchestrator.run_query.call_args
        assert kwargs.get("model_override") == "haiku"

    @pytest.mark.asyncio
    async def test_complex_message_routes_to_opus(
        self, orchestrator: ZenkiOrchestrator
    ) -> None:
        await orchestrator.process_message("Write code to implement a REST API service")
        _, kwargs = orchestrator.run_query.call_args
        assert kwargs.get("model_override") == "opus"

    @pytest.mark.asyncio
    async def test_default_message_routes_to_sonnet(
        self, orchestrator: ZenkiOrchestrator
    ) -> None:
        await orchestrator.process_message("What is the weather like today?")
        _, kwargs = orchestrator.run_query.call_args
        assert kwargs.get("model_override") == "sonnet"


# ===========================================================================
# Error handling
# ===========================================================================


class TestErrorHandling:
    """Tests that SDK errors are wrapped by SmartErrorHandler."""

    @pytest.mark.asyncio
    async def test_error_returns_user_message(
        self, orchestrator: ZenkiOrchestrator
    ) -> None:
        orchestrator.run_query = AsyncMock(side_effect=RuntimeError("LLM exploded"))
        result = await orchestrator.process_message("trigger error")
        assert "error" in result.lower() or "Error" in result

    @pytest.mark.asyncio
    async def test_error_stores_message_in_db(
        self, orchestrator: ZenkiOrchestrator, db: ZenkiDatabase
    ) -> None:
        orchestrator.run_query = AsyncMock(side_effect=RuntimeError("Connection failed"))
        await orchestrator.process_message("trigger error", user_id="err-user")

        sessions = db.list_sessions(user_id="err-user")
        assert len(sessions) >= 1
        messages = db.get_messages_for_session(sessions[0].id)
        # Should have user message + error assistant message.
        assert len(messages) == 2
        assert messages[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_rate_limit_error_message(
        self, orchestrator: ZenkiOrchestrator
    ) -> None:
        orchestrator.run_query = AsyncMock(side_effect=RuntimeError("rate limit exceeded"))
        result = await orchestrator.process_message("test rate limit")
        assert isinstance(result, str)
        assert len(result) > 0


# ===========================================================================
# Close session
# ===========================================================================


class TestCloseSession:
    """Tests for ZenkiOrchestrator.close_session."""

    @pytest.mark.asyncio
    async def test_close_session(
        self, orchestrator: ZenkiOrchestrator, db: ZenkiDatabase
    ) -> None:
        session = orchestrator.session_manager.create_session(
            user_id="close-user", channel_type="cli"
        )
        await orchestrator.close_session(session.id)
        closed = db.get_session(session.id)
        assert closed is not None
        assert closed.ended_at is not None
