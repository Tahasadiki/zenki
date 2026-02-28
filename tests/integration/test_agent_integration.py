"""Integration tests for the full Zenki pipeline.

Tests exercise real database interactions, memory management,
consolidation, channels, and the scheduler.  The SDK orchestrator is
tested with ``run_query`` mocked so no real LLM calls are made.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from zenki.channels.cli_channel.adapter import CLIChannel
from zenki.channels.registry import ChannelRegistry
from zenki.config.settings import ZenkiSettings
from zenki.db.database import ZenkiDatabase
from zenki.db.models import User
from zenki.memory.consolidation import MemoryConsolidator
from zenki.memory.manager import MemoryManager
from zenki.scheduler.scheduler import ZenkiScheduler
from zenki.sdk.orchestrator import ZenkiOrchestrator


@pytest.fixture
def setup_env(tmp_path):
    """Set up a full Zenki environment for integration testing."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    settings = ZenkiSettings()

    db_path = config_dir / "db" / "zenki.db"
    db_path.parent.mkdir(parents=True)
    db = ZenkiDatabase(db_path)
    db.initialize()

    db.create_user(User(id="test-user", display_name="Test User"))

    memory_dir = config_dir / "memory"
    memory_dir.mkdir()

    return {
        "settings": settings,
        "db": db,
        "config_dir": config_dir,
        "memory_dir": memory_dir,
    }


def _make_orchestrator(setup_env) -> ZenkiOrchestrator:
    """Create a ZenkiOrchestrator with mocked SDK internals."""
    with patch("zenki.sdk.orchestrator.get_agents", return_value={}), \
         patch("zenki.sdk.orchestrator.create_zenki_tools", return_value={}), \
         patch("zenki.sdk.orchestrator.create_zenki_hooks", return_value={}):
        orch = ZenkiOrchestrator(
            settings=setup_env["settings"],
            db=setup_env["db"],
        )
    orch.run_query = AsyncMock(return_value="Mock response")
    return orch


class TestFullPipeline:
    """Test the full agent pipeline end-to-end."""

    @pytest.mark.asyncio
    async def test_conversation_flow(self, setup_env):
        """Test a complete conversation flow."""
        orchestrator = _make_orchestrator(setup_env)

        response1 = await orchestrator.process_message(
            message="Hello, I'm working on a Python project",
            user_id="test-user",
            channel_type="cli",
        )
        assert response1

        response2 = await orchestrator.process_message(
            message="Can you help me with testing?",
            user_id="test-user",
            channel_type="cli",
        )
        assert response2

        sessions = setup_env["db"].list_sessions()
        assert len(sessions) >= 1

    @pytest.mark.asyncio
    async def test_session_persistence(self, setup_env):
        """Test that sessions persist across messages."""
        orchestrator = _make_orchestrator(setup_env)

        await orchestrator.process_message(
            message="First message",
            user_id="test-user",
            channel_type="cli",
        )

        sessions = setup_env["db"].list_sessions()
        session_id = sessions[0].id

        response2 = await orchestrator.process_message(
            message="Second message",
            session_id=session_id,
            user_id="test-user",
            channel_type="cli",
        )
        assert response2

        messages = setup_env["db"].get_messages_for_session(session_id)
        assert len(messages) >= 4  # 2 user + 2 assistant

    @pytest.mark.asyncio
    async def test_memory_manager_integration(self, setup_env):
        """Test memory manager works with the database."""
        from zenki.db.models import Session
        db = setup_env["db"]

        manager = MemoryManager(
            settings=setup_env["settings"],
            db=db,
            config_dir=setup_env["config_dir"],
        )

        manager.store_fact("User prefers Python", "preference", ["python", "language"])

        session = db.create_session(Session(user_id="test-user", channel_type="cli"))

        manager.store_conversation_summary(
            session_id=session.id,
            summary="Discussed Python testing strategies",
            topics=["python", "testing"],
            entities=["pytest"],
        )

        context = manager.build_context("python testing")
        assert "core" in context
        assert "episodic" in context
        assert "semantic" in context

    @pytest.mark.asyncio
    async def test_consolidation_integration(self, setup_env):
        """Test memory consolidation after conversations."""
        db = setup_env["db"]

        from zenki.db.models import Message, Session

        session = Session(
            user_id="test-user",
            channel_type="cli",
        )
        session = db.create_session(session)

        db.create_message(Message(
            session_id=session.id, role="user",
            content="How do I use Docker for Python?",
        ))
        db.create_message(Message(
            session_id=session.id, role="assistant",
            content="You can create a Dockerfile with a Python base image.",
        ))

        session.ended_at = datetime.now(UTC)
        db.update_session(session)

        consolidator = MemoryConsolidator(
            db=db,
            memory_dir=setup_env["memory_dir"],
        )
        result = await consolidator.run_consolidation(days_back=7)
        assert result.sessions_reviewed >= 1

    def test_channel_registry(self):
        """Test channel registry with CLI channel."""
        registry = ChannelRegistry()
        cli = CLIChannel()
        registry.register(cli)

        assert registry.get("cli") is cli
        assert registry.default_channel_type == "cli"

    @pytest.mark.asyncio
    async def test_scheduler_with_database(self, setup_env):
        """Test scheduler creates and retrieves tasks."""
        scheduler = ZenkiScheduler(setup_env["db"])

        scheduler.add_task(
            user_id="test-user",
            description="Daily standup reminder",
            cron_expression="0 9 * * 1-5",
            task_type="reminder",
            task_config={"message": "Time for standup!"},
        )

        tasks = scheduler.list_tasks()
        assert any(t.description == "Daily standup reminder" for t in tasks)


class TestModelRouting:
    """Test smart model routing via the orchestrator."""

    @pytest.mark.asyncio
    async def test_simple_greeting_routes_to_haiku(self, setup_env):
        """Verify simple greetings route to the cheapest model."""
        orchestrator = _make_orchestrator(setup_env)

        await orchestrator.process_message(
            message="Hi there!",
            user_id="test-user",
            channel_type="cli",
        )

        _, kwargs = orchestrator.run_query.call_args
        assert kwargs.get("model_override") == "haiku"

    @pytest.mark.asyncio
    async def test_code_request_routes_to_opus(self, setup_env):
        """Verify code generation requests route to opus."""
        orchestrator = _make_orchestrator(setup_env)

        await orchestrator.process_message(
            message="Write a Python function to sort a binary tree",
            user_id="test-user",
            channel_type="cli",
        )

        _, kwargs = orchestrator.run_query.call_args
        assert kwargs.get("model_override") == "opus"
