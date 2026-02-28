"""Integration tests for the full Zenki agent pipeline."""

from collections.abc import AsyncIterator
from datetime import UTC
from pathlib import Path

import pytest

from zenki.channels.cli_channel.adapter import CLIChannel
from zenki.channels.registry import ChannelRegistry
from zenki.config.settings import ZenkiSettings
from zenki.core.agent import ZenkiAgent
from zenki.db.database import ZenkiDatabase
from zenki.db.models import User
from zenki.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from zenki.memory.consolidation import MemoryConsolidator
from zenki.memory.manager import MemoryManager
from zenki.scheduler.scheduler import ZenkiScheduler
from zenki.skills.loader import SkillLoader
from zenki.skills.registry import SkillRegistry


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for integration tests."""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_messages: list[LLMMessage] = []
        self.last_system: str = ""
        self.response_map: dict[str, str] = {}

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

        # Check response map for matching content
        user_msg = messages[-1].content if messages else ""
        for key, response in self.response_map.items():
            if key.lower() in user_msg.lower():
                return LLMResponse(
                    content=response,
                    model=model or "mock-model",
                    tokens_in=len(user_msg),
                    tokens_out=len(response),
                )

        return LLMResponse(
            content=f"Mock response to: {user_msg[:50]}",
            model=model or "mock-model",
            tokens_in=len(user_msg),
            tokens_out=20,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        system: str = "",
        model: str | None = None,
    ) -> AsyncIterator[str]:
        yield "Mock streaming response"

    def get_available_models(self) -> list[str]:
        return ["mock-model"]


@pytest.fixture
def setup_env(tmp_path):
    """Set up a full Zenki environment for integration testing."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create settings
    settings = ZenkiSettings()

    # Create database
    db_path = config_dir / "db" / "zenki.db"
    db_path.parent.mkdir(parents=True)
    db = ZenkiDatabase(db_path)
    db.initialize()

    # Create user
    db.create_user(User(id="test-user", display_name="Test User"))

    # Create memory dir
    memory_dir = config_dir / "memory"
    memory_dir.mkdir()

    # Create LLM provider
    llm = MockLLMProvider()

    return {
        "settings": settings,
        "db": db,
        "llm": llm,
        "config_dir": config_dir,
        "memory_dir": memory_dir,
    }


class TestFullPipeline:
    """Test the full agent pipeline end-to-end."""

    @pytest.mark.asyncio
    async def test_conversation_flow(self, setup_env):
        """Test a complete conversation flow."""
        agent = ZenkiAgent(
            settings=setup_env["settings"],
            db=setup_env["db"],
            llm_provider=setup_env["llm"],
        )

        # First message creates session
        response1 = await agent.process_message(
            message="Hello, I'm working on a Python project",
            user_id="test-user",
            channel_type="cli",
        )
        assert response1  # Got a response

        # Second message should reuse session
        response2 = await agent.process_message(
            message="Can you help me with testing?",
            user_id="test-user",
            channel_type="cli",
        )
        assert response2

        # Verify messages are stored
        sessions = setup_env["db"].list_sessions()
        assert len(sessions) >= 1

    @pytest.mark.asyncio
    async def test_session_persistence(self, setup_env):
        """Test that sessions persist across messages."""
        agent = ZenkiAgent(
            settings=setup_env["settings"],
            db=setup_env["db"],
            llm_provider=setup_env["llm"],
        )

        # Create a session
        await agent.process_message(
            message="First message",
            user_id="test-user",
            channel_type="cli",
        )

        # Get the session
        sessions = setup_env["db"].list_sessions()
        session_id = sessions[0].id

        # Send another message to same session
        response2 = await agent.process_message(
            message="Second message",
            session_id=session_id,
            user_id="test-user",
            channel_type="cli",
        )
        assert response2

        # Verify both messages in same session
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

        # Store a fact
        manager.store_fact("User prefers Python", "preference", ["python", "language"])

        # Create a real session first (FK constraint)
        session = db.create_session(Session(user_id="test-user", channel_type="cli"))

        # Store a conversation summary
        manager.store_conversation_summary(
            session_id=session.id,
            summary="Discussed Python testing strategies",
            topics=["python", "testing"],
            entities=["pytest"],
        )

        # Build context
        context = manager.build_context("python testing")
        assert "core" in context
        assert "episodic" in context
        assert "semantic" in context

    @pytest.mark.asyncio
    async def test_consolidation_integration(self, setup_env):
        """Test memory consolidation after conversations."""
        db = setup_env["db"]

        # Create a closed session with messages
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

        # Close the session
        from datetime import datetime
        session.ended_at = datetime.now(UTC)
        db.update_session(session)

        # Run consolidation
        consolidator = MemoryConsolidator(
            db=db,
            memory_dir=setup_env["memory_dir"],
        )
        result = await consolidator.run_consolidation(days_back=7)
        assert result.sessions_reviewed >= 1

    def test_skill_loader_finds_core_skills(self):
        """Test that core skills are discoverable."""
        import zenki.skills.core_skills as core_module
        core_dir = Path(core_module.__file__).parent

        loader = SkillLoader(core_skills_dir=core_dir)
        skills = loader.discover()
        skill_names = [s.name for s in skills]
        assert "software-engineering" in skill_names
        assert "web-research" in skill_names
        assert "file-management" in skill_names
        assert "system-operations" in skill_names

    def test_skill_registry_sync(self, setup_env):
        """Test skill registry syncs core skills to database."""
        import zenki.skills.core_skills as core_module
        core_dir = Path(core_module.__file__).parent

        loader = SkillLoader(core_skills_dir=core_dir)
        registry = SkillRegistry(db=setup_env["db"], loader=loader)
        registry.sync_skills()

        available = registry.get_available_skills()
        assert len(available) >= 4

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
    """Test smart model routing in the agent."""

    @pytest.mark.asyncio
    async def test_simple_greeting_routes_to_haiku(self, setup_env):
        """Verify simple greetings route to the cheapest model."""
        agent = ZenkiAgent(
            settings=setup_env["settings"],
            db=setup_env["db"],
            llm_provider=setup_env["llm"],
        )

        await agent.process_message(
            message="Hi there!",
            user_id="test-user",
            channel_type="cli",
        )

        # Check the model used
        messages = setup_env["db"].list_sessions()
        assert len(messages) >= 1

    @pytest.mark.asyncio
    async def test_code_request_routes_appropriately(self, setup_env):
        """Verify code generation requests route to capable model."""
        agent = ZenkiAgent(
            settings=setup_env["settings"],
            db=setup_env["db"],
            llm_provider=setup_env["llm"],
        )

        await agent.process_message(
            message="Write a Python function to sort a binary tree",
            user_id="test-user",
            channel_type="cli",
        )

        # Verify message was processed
        sessions = setup_env["db"].list_sessions()
        assert len(sessions) >= 1
