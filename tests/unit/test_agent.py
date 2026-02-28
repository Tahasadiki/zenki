"""Tests for the ZenkiAgent orchestrator (zenki.core.agent)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from zenki.config.settings import ZenkiSettings
from zenki.core.agent import ZenkiAgent
from zenki.db.database import ZenkiDatabase
from zenki.llm.base import BaseLLMProvider, LLMMessage, LLMResponse

# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------


class MockProvider(BaseLLMProvider):
    """A deterministic LLM provider for testing."""

    def __init__(self, response_text: str = "Mock response") -> None:
        self.response_text = response_text
        self.last_messages: list[LLMMessage] | None = None
        self.last_system: str | None = None
        self.last_model: str | None = None
        self.call_count: int = 0
        self.should_raise: Exception | None = None

    async def send(
        self,
        messages: list[LLMMessage],
        system: str = "",
        model: str | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        self.last_messages = messages
        self.last_system = system
        self.last_model = model

        if self.should_raise is not None:
            raise self.should_raise

        return LLMResponse(
            content=self.response_text,
            model=model or "mock-model",
            tokens_in=10,
            tokens_out=20,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        system: str = "",
        model: str | None = None,
    ) -> AsyncIterator[str]:
        yield self.response_text  # pragma: no cover

    def get_available_models(self) -> list[str]:
        return ["mock-model"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> ZenkiDatabase:
    """Return an initialised ZenkiDatabase backed by a temp file."""
    database = ZenkiDatabase(tmp_path / "test_agent.db")
    database.initialize()
    return database


@pytest.fixture()
def settings() -> ZenkiSettings:
    """Return default settings."""
    return ZenkiSettings()


@pytest.fixture()
def provider() -> MockProvider:
    """Return a fresh MockProvider."""
    return MockProvider()


@pytest.fixture()
def agent(settings: ZenkiSettings, db: ZenkiDatabase, provider: MockProvider) -> ZenkiAgent:
    """Return a ZenkiAgent wired with mock dependencies."""
    return ZenkiAgent(settings=settings, db=db, llm_provider=provider)


# ===========================================================================
# Basic message processing
# ===========================================================================


class TestProcessMessage:
    """Tests for ZenkiAgent.process_message."""

    async def test_returns_llm_response(self, agent: ZenkiAgent) -> None:
        result = await agent.process_message("Hello!")
        assert result == "Mock response"

    async def test_stores_user_message_in_db(self, agent: ZenkiAgent, db: ZenkiDatabase) -> None:
        await agent.process_message("Test message", user_id="u1")

        # Find the session that was created.
        sessions = db.list_sessions(user_id="u1")
        assert len(sessions) >= 1
        messages = db.get_messages_for_session(sessions[0].id)
        user_messages = [m for m in messages if m.role == "user"]
        assert len(user_messages) == 1
        assert user_messages[0].content == "Test message"

    async def test_stores_assistant_response_in_db(
        self, agent: ZenkiAgent, db: ZenkiDatabase
    ) -> None:
        await agent.process_message("Ping", user_id="u2")

        sessions = db.list_sessions(user_id="u2")
        messages = db.get_messages_for_session(sessions[0].id)
        assistant_messages = [m for m in messages if m.role == "assistant"]
        assert len(assistant_messages) == 1
        assert assistant_messages[0].content == "Mock response"

    async def test_assistant_message_has_model_and_tokens(
        self, agent: ZenkiAgent, db: ZenkiDatabase
    ) -> None:
        await agent.process_message("Tell me something", user_id="u3")

        sessions = db.list_sessions(user_id="u3")
        messages = db.get_messages_for_session(sessions[0].id)
        assistant_msg = [m for m in messages if m.role == "assistant"][0]
        assert assistant_msg.model_used is not None
        assert assistant_msg.tokens_in == 10
        assert assistant_msg.tokens_out == 20


# ===========================================================================
# Session management
# ===========================================================================


class TestSessionHandling:
    """Tests for session creation and reuse within the agent."""

    async def test_creates_session_when_none_provided(
        self, agent: ZenkiAgent, db: ZenkiDatabase
    ) -> None:
        await agent.process_message("First message", user_id="auto-user")
        sessions = db.list_sessions(user_id="auto-user")
        assert len(sessions) == 1

    async def test_reuses_existing_session(
        self, agent: ZenkiAgent, db: ZenkiDatabase
    ) -> None:
        await agent.process_message(
            "Message 1", user_id="reuse-user", channel_type="cli"
        )
        await agent.process_message(
            "Message 2", user_id="reuse-user", channel_type="cli"
        )

        sessions = db.list_sessions(user_id="reuse-user")
        assert len(sessions) == 1
        messages = db.get_messages_for_session(sessions[0].id)
        # 2 user messages + 2 assistant messages = 4 total.
        assert len(messages) == 4

    async def test_uses_provided_session_id(
        self, agent: ZenkiAgent, db: ZenkiDatabase
    ) -> None:
        # Create a session manually.
        session = agent.session_manager.create_session(
            user_id="sid-user", channel_type="cli"
        )
        await agent.process_message("Hi", session_id=session.id, user_id="sid-user")

        messages = db.get_messages_for_session(session.id)
        assert len(messages) == 2  # user + assistant

    async def test_creates_new_session_if_provided_id_invalid(
        self, agent: ZenkiAgent, db: ZenkiDatabase
    ) -> None:
        result = await agent.process_message(
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
    """Tests verifying that model routing is applied."""

    async def test_simple_message_routes_to_haiku(
        self, agent: ZenkiAgent, provider: MockProvider
    ) -> None:
        await agent.process_message("hi")
        # The router classifies "hi" as haiku, which resolves to its full model id.
        assert provider.last_model is not None
        assert "haiku" in provider.last_model

    async def test_complex_message_routes_to_opus(
        self, agent: ZenkiAgent, provider: MockProvider
    ) -> None:
        await agent.process_message("Write code to implement a REST API service")
        assert provider.last_model is not None
        assert "opus" in provider.last_model

    async def test_default_message_routes_to_sonnet(
        self, agent: ZenkiAgent, provider: MockProvider
    ) -> None:
        await agent.process_message("What is the weather like today?")
        assert provider.last_model is not None
        assert "sonnet" in provider.last_model


# ===========================================================================
# Error handling
# ===========================================================================


class TestErrorHandling:
    """Tests that LLM errors are wrapped by SmartErrorHandler."""

    async def test_llm_error_returns_user_message(
        self, agent: ZenkiAgent, provider: MockProvider
    ) -> None:
        provider.should_raise = RuntimeError("LLM exploded")
        result = await agent.process_message("trigger error")
        # SmartErrorHandler should produce a user-friendly message.
        assert "error" in result.lower() or "Error" in result

    async def test_llm_error_stores_error_in_db(
        self, agent: ZenkiAgent, provider: MockProvider, db: ZenkiDatabase
    ) -> None:
        provider.should_raise = RuntimeError("Connection failed")
        await agent.process_message("trigger error", user_id="err-user")

        sessions = db.list_sessions(user_id="err-user")
        assert len(sessions) >= 1
        messages = db.get_messages_for_session(sessions[0].id)
        # Should have user message + error assistant message.
        assert len(messages) == 2
        assert messages[1].role == "assistant"

    async def test_rate_limit_error_message(
        self, agent: ZenkiAgent, provider: MockProvider
    ) -> None:
        provider.should_raise = RuntimeError("rate limit exceeded")
        result = await agent.process_message("test rate limit")
        # The error handler recognizes rate limit patterns.
        assert isinstance(result, str)
        assert len(result) > 0


# ===========================================================================
# System prompt
# ===========================================================================


class TestSystemPrompt:
    """Tests that the system prompt is correctly built and sent."""

    async def test_system_prompt_sent_to_provider(
        self, agent: ZenkiAgent, provider: MockProvider
    ) -> None:
        await agent.process_message("Hello")
        assert provider.last_system is not None
        assert len(provider.last_system) > 0
        # Should contain personality-related content.
        assert "Zenki" in provider.last_system

    async def test_conversation_history_sent_to_provider(
        self, agent: ZenkiAgent, provider: MockProvider
    ) -> None:
        await agent.process_message("First", user_id="hist-user", channel_type="cli")
        await agent.process_message("Second", user_id="hist-user", channel_type="cli")

        # After the second call, the provider should receive conversation history.
        assert provider.last_messages is not None
        # History should include: user "First", assistant "Mock response",
        # user "Second" = 3 messages.
        assert len(provider.last_messages) == 3


# ===========================================================================
# Close session
# ===========================================================================


class TestCloseSession:
    """Tests for ZenkiAgent.close_session."""

    async def test_close_session(self, agent: ZenkiAgent, db: ZenkiDatabase) -> None:
        session = agent.session_manager.create_session(
            user_id="close-user", channel_type="cli"
        )
        await agent.close_session(session.id)
        closed = db.get_session(session.id)
        assert closed is not None
        assert closed.ended_at is not None
