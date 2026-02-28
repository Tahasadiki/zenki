"""Comprehensive integration tests for Zenki.

These tests exercise the full Zenki stack end-to-end, simulating real usage
scenarios without requiring external API calls. They test:

1. Full initialization → chat → memory pipeline
2. Session management (create, resume, parallel)
3. Memory system (all 4 tiers working together)
4. Model routing based on message complexity
5. Error handling and recovery
6. Consolidation cycle
7. Edge cases and stress scenarios
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zenki.config.settings import ZenkiSettings
from zenki.core.errors import ErrorSeverity, RecoveryStrategy, SmartErrorHandler
from zenki.core.prompts import build_system_prompt
from zenki.core.session import SessionManager
from zenki.db.database import ZenkiDatabase
from zenki.db.models import (
    EpisodicMemory,
    Message,
    ScheduledTask,
    SemanticMemory,
    Session,
    Skill,
    User,
)
from zenki.memory.consolidation import MemoryConsolidator, calculate_importance
from zenki.memory.core_memory import CoreMemory
from zenki.memory.embeddings import DummyEmbeddingProvider, get_embedding_provider
from zenki.memory.episodic import EpisodicMemoryStore
from zenki.memory.manager import MemoryManager
from zenki.memory.retrieval import cosine_similarity, rank_memories
from zenki.memory.semantic import SemanticMemoryStore
from zenki.memory.working import WorkingMemory
from zenki.sdk import _context
from zenki.sdk.router import ModelRouter


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for each test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def db(tmp_dir):
    """Create a fresh database for each test."""
    db_path = tmp_dir / "test_zenki.db"
    database = ZenkiDatabase(db_path)
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def settings(tmp_dir):
    """Create test settings that write to the temp directory."""
    config_path = tmp_dir / "config.json"
    s = ZenkiSettings()
    # Override config dir to use temp
    return s


@pytest.fixture
def session_manager(db):
    """Create a session manager backed by the test database."""
    return SessionManager(db)


@pytest.fixture
def core_memory(tmp_dir):
    """Create a core memory instance in the temp directory."""
    memory_dir = tmp_dir / "memory"
    cm = CoreMemory(memory_dir=memory_dir)
    cm.ensure_files()
    return cm


@pytest.fixture
def embedding_provider():
    """Provide a dummy embedding provider."""
    return DummyEmbeddingProvider(dimension=384)


@pytest.fixture
def episodic_store(db, embedding_provider):
    """Create an episodic memory store."""
    return EpisodicMemoryStore(db=db, embedding_provider=embedding_provider)


@pytest.fixture
def semantic_store(db, embedding_provider):
    """Create a semantic memory store."""
    return SemanticMemoryStore(db=db, embedding_provider=embedding_provider)


@pytest.fixture
def memory_manager(tmp_dir, db):
    """Create a full memory manager with all tiers."""
    settings = ZenkiSettings()
    # Patch settings to use dummy embeddings so we don't need sentence-transformers
    settings.memory.embeddings.provider = "dummy"
    config_dir = tmp_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return MemoryManager(settings=settings, db=db, config_dir=config_dir)


@pytest.fixture(autouse=True)
def cleanup_context():
    """Clean up the global context after each test."""
    yield
    _context.clear_all()


# ============================================================================
# 1. Full Pipeline: Init → Chat → Memory
# ============================================================================


class TestFullPipeline:
    """Test the complete flow from initialization to memory storage."""

    def test_initialization_creates_all_components(self, db, tmp_dir, memory_manager):
        """Verify that all core components can be initialized together."""
        settings = ZenkiSettings()
        session_mgr = SessionManager(db)

        # All components should be non-None
        assert db is not None
        assert session_mgr is not None
        assert memory_manager is not None
        assert memory_manager.working is not None
        assert memory_manager.core is not None

    def test_full_conversation_flow(self, db, session_manager, memory_manager):
        """Simulate a full conversation: create session → add messages → store memory."""
        # 1. Create a session
        session = session_manager.create_session(
            user_id="test-user",
            channel_type="cli",
        )
        assert session.id is not None
        assert session.channel_type == "cli"

        # 2. Add user message
        user_msg = session_manager.add_message(
            session_id=session.id,
            role="user",
            content="Tell me about Python decorators",
        )
        assert user_msg.role == "user"

        # 3. Add assistant response
        assistant_msg = session_manager.add_message(
            session_id=session.id,
            role="assistant",
            content="Python decorators are a way to modify functions. They use the @syntax...",
        )
        assert assistant_msg.role == "assistant"

        # 4. Store to working memory
        memory_manager.working.add_message("user", "Tell me about Python decorators")
        memory_manager.working.add_message("assistant", "Python decorators are a way...")
        assert memory_manager.working.message_count == 2

        # 5. Store an episodic summary
        memory_manager.store_conversation_summary(
            session_id=session.id,
            summary="User asked about Python decorators. Discussed @syntax and use cases.",
            topics=["python", "decorators", "programming"],
            entities=["Python"],
        )

        # 6. Store a semantic fact
        memory_manager.store_fact(
            content="User is interested in Python programming patterns",
            category="preference",
            tags=["python", "learning"],
        )

        # 7. Verify memory context can be built
        context = memory_manager.build_context("python decorators")
        assert "working" in context
        assert "core" in context
        assert "episodic" in context
        assert "semantic" in context
        assert len(context["working"]) == 2

        # 8. Verify episodic retrieval
        assert len(context["episodic"]) >= 1
        found_decorator_summary = any(
            "decorator" in ep.get("summary", "").lower()
            for ep in context["episodic"]
        )
        assert found_decorator_summary

        # 9. Verify semantic retrieval
        assert len(context["semantic"]) >= 1

    def test_conversation_with_session_resume(self, db, session_manager):
        """Verify session resume preserves conversation history."""
        # Create initial session with messages
        session = session_manager.create_session(
            user_id="alice",
            channel_type="cli",
        )
        session_manager.add_message(session.id, "user", "Hello Zenki")
        session_manager.add_message(session.id, "assistant", "Hello Alice!")
        session_manager.add_message(session.id, "user", "What's my name?")

        # "Resume" by fetching session and messages
        resumed = session_manager.get_session(session.id)
        assert resumed is not None
        assert resumed.id == session.id

        messages = session_manager.get_messages(session.id)
        assert len(messages) == 3
        assert messages[0].content == "Hello Zenki"
        assert messages[1].content == "Hello Alice!"
        assert messages[2].content == "What's my name?"


# ============================================================================
# 2. Session Management
# ============================================================================


class TestSessionManagement:
    """Test session creation, retrieval, parallel sessions, and lifecycle."""

    def test_create_multiple_sessions_for_same_user(self, session_manager):
        """User can have multiple active sessions across channels."""
        cli_session = session_manager.create_session(
            user_id="user1",
            channel_type="cli",
        )
        slack_session = session_manager.create_session(
            user_id="user1",
            channel_type="slack",
            channel_id="C12345",
        )

        assert cli_session.id != slack_session.id
        assert cli_session.channel_type == "cli"
        assert slack_session.channel_type == "slack"

        # Both should appear in active sessions
        active = session_manager.list_active_sessions(user_id="user1")
        assert len(active) == 2

    def test_parallel_sessions_isolation(self, session_manager):
        """Messages in parallel sessions should not interfere."""
        session_a = session_manager.create_session(
            user_id="user1",
            channel_type="cli",
            thread_id="thread-a",
        )
        session_b = session_manager.create_session(
            user_id="user1",
            channel_type="cli",
            thread_id="thread-b",
        )

        # Add different messages to each session
        session_manager.add_message(session_a.id, "user", "Message for session A")
        session_manager.add_message(session_b.id, "user", "Message for session B")
        session_manager.add_message(session_a.id, "assistant", "Reply to A")
        session_manager.add_message(session_b.id, "assistant", "Reply to B")

        # Verify isolation
        msgs_a = session_manager.get_messages(session_a.id)
        msgs_b = session_manager.get_messages(session_b.id)

        assert len(msgs_a) == 2
        assert len(msgs_b) == 2
        assert msgs_a[0].content == "Message for session A"
        assert msgs_b[0].content == "Message for session B"
        assert msgs_a[1].content == "Reply to A"
        assert msgs_b[1].content == "Reply to B"

    def test_get_or_create_reuses_active_session(self, session_manager):
        """get_or_create should return existing active session for same channel/thread."""
        session1 = session_manager.get_or_create_session(
            user_id="user1",
            channel_type="slack",
            channel_id="C12345",
            thread_id="thread-1",
        )
        session2 = session_manager.get_or_create_session(
            user_id="user1",
            channel_type="slack",
            channel_id="C12345",
            thread_id="thread-1",
        )

        assert session1.id == session2.id  # Same session reused

    def test_get_or_create_separates_threads(self, session_manager):
        """Different threads in the same channel should get different sessions."""
        session1 = session_manager.get_or_create_session(
            user_id="user1",
            channel_type="slack",
            channel_id="C12345",
            thread_id="thread-1",
        )
        session2 = session_manager.get_or_create_session(
            user_id="user1",
            channel_type="slack",
            channel_id="C12345",
            thread_id="thread-2",
        )

        assert session1.id != session2.id  # Different sessions

    def test_close_and_reopen_creates_new_session(self, session_manager):
        """Closing a session and calling get_or_create should create a new one."""
        session1 = session_manager.get_or_create_session(
            user_id="user1",
            channel_type="cli",
        )
        session_manager.close_session(session1.id, summary="Session ended")

        session2 = session_manager.get_or_create_session(
            user_id="user1",
            channel_type="cli",
        )

        assert session1.id != session2.id
        # Old session should have summary
        old = session_manager.get_session(session1.id)
        assert old.summary == "Session ended"
        assert old.ended_at is not None

    def test_session_last_active_updates_on_message(self, session_manager):
        """Adding a message should update the session's last_active timestamp."""
        session = session_manager.create_session(
            user_id="user1",
            channel_type="cli",
        )

        # Initially last_active might be None
        original = session_manager.get_session(session.id)
        assert original.last_active is None

        session_manager.add_message(session.id, "user", "Hello")

        updated = session_manager.get_session(session.id)
        assert updated.last_active is not None

    def test_many_parallel_sessions(self, session_manager):
        """Stress test: create many parallel sessions for one user."""
        sessions = []
        for i in range(20):
            s = session_manager.create_session(
                user_id="stress-user",
                channel_type="slack",
                channel_id=f"channel-{i}",
                thread_id=f"thread-{i}",
            )
            sessions.append(s)
            session_manager.add_message(s.id, "user", f"Message {i}")

        # All should be active
        active = session_manager.list_active_sessions(user_id="stress-user")
        assert len(active) == 20

        # Each should have exactly one message
        for s in sessions:
            msgs = session_manager.get_messages(s.id)
            assert len(msgs) == 1

    def test_message_ordering_is_chronological(self, session_manager):
        """Messages should always be returned in chronological order."""
        session = session_manager.create_session(
            user_id="user1",
            channel_type="cli",
        )

        contents = ["first", "second", "third", "fourth", "fifth"]
        for content in contents:
            session_manager.add_message(session.id, "user", content)

        messages = session_manager.get_messages(session.id)
        actual_order = [m.content for m in messages]
        assert actual_order == contents

    def test_message_limit_returns_most_recent(self, session_manager):
        """When limiting messages, should get the most recent ones."""
        session = session_manager.create_session(
            user_id="user1",
            channel_type="cli",
        )

        for i in range(10):
            session_manager.add_message(session.id, "user", f"Message {i}")

        messages = session_manager.get_messages(session.id, limit=3)
        assert len(messages) == 3
        assert messages[0].content == "Message 7"
        assert messages[1].content == "Message 8"
        assert messages[2].content == "Message 9"


# ============================================================================
# 3. Memory System
# ============================================================================


class TestWorkingMemory:
    """Test Tier 1: Working memory behavior."""

    def test_token_budget_enforcement(self):
        """Working memory should drop older messages when budget is exceeded."""
        wm = WorkingMemory(max_tokens=100)

        # Add messages that exceed the budget
        wm.add_message("user", "x" * 200)  # ~50 tokens
        wm.add_message("assistant", "y" * 200)  # ~50 tokens
        wm.add_message("user", "z" * 200)  # ~50 tokens

        context = wm.get_context()
        # Should have dropped the oldest message(s) to fit budget
        assert len(context) < 3

    def test_clear_removes_all_messages(self):
        """Clear should empty working memory."""
        wm = WorkingMemory()
        wm.add_message("user", "Hello")
        wm.add_message("assistant", "Hi")
        assert wm.message_count == 2

        wm.clear()
        assert wm.message_count == 0
        assert wm.get_context() == []

    def test_summary_generation(self):
        """Summary should include message roles and truncated content."""
        wm = WorkingMemory()
        wm.add_message("user", "Tell me about machine learning algorithms")
        wm.add_message("assistant", "Machine learning encompasses various algorithms...")

        summary = wm.get_summary()
        assert "user:" in summary
        assert "assistant:" in summary
        assert "2 messages" in summary

    def test_empty_summary(self):
        """Empty working memory should report no conversation."""
        wm = WorkingMemory()
        assert wm.get_summary() == "No conversation yet."

    def test_total_tokens_property(self):
        """total_tokens should estimate correctly."""
        wm = WorkingMemory()
        wm.add_message("user", "a" * 100)  # ~25 tokens
        wm.add_message("assistant", "b" * 200)  # ~50 tokens
        assert wm.total_tokens == 75


class TestCoreMemory:
    """Test Tier 2: Core memory file management."""

    def test_ensure_files_creates_all(self, core_memory, tmp_dir):
        """All canonical memory files should be created."""
        memory_dir = tmp_dir / "memory"
        expected_files = [
            "identity.md", "preferences.md", "projects.md",
            "relationships.md", "patterns.md", "personality.md",
        ]
        for f in expected_files:
            assert (memory_dir / f).exists()

    def test_update_and_get(self, core_memory):
        """Write to a core memory section and read it back."""
        core_memory.update("identity", "Name: Alice\nRole: Software Engineer")
        content = core_memory.get("identity")
        assert "Alice" in content
        assert "Software Engineer" in content

    def test_get_nonexistent_key(self, core_memory):
        """Reading a non-existent key should return empty string."""
        assert core_memory.get("nonexistent") == ""

    def test_context_string_combines_all(self, core_memory):
        """get_context_string should combine all non-empty sections."""
        core_memory.update("identity", "Alice, software engineer")
        core_memory.update("preferences", "Prefers Python and VS Code")

        context = core_memory.get_context_string()
        assert "### Identity" in context
        assert "### Preferences" in context
        assert "Alice" in context
        assert "Python" in context

    def test_context_string_skips_empty_sections(self, core_memory):
        """Empty sections should not appear in context string."""
        core_memory.update("identity", "Alice")
        # All others are empty

        context = core_memory.get_context_string()
        assert "### Identity" in context
        assert "### Preferences" not in context
        assert "### Projects" not in context

    def test_load_all_returns_dict(self, core_memory):
        """load_all should return all sections as a dict."""
        core_memory.update("identity", "Bob")
        data = core_memory.load_all()
        assert isinstance(data, dict)
        assert "identity" in data
        assert data["identity"] == "Bob"
        assert "preferences" in data


class TestEpisodicMemory:
    """Test Tier 3: Episodic memory storage and retrieval."""

    def _create_session(self, db, session_id: str) -> str:
        """Helper to create a user + session so FK constraints are satisfied."""
        user_id = f"user-{session_id}"
        if db.get_user(user_id) is None:
            db.create_user(User(id=user_id))
        session = Session(id=session_id, user_id=user_id, channel_type="cli")
        db.create_session(session)
        return session_id

    def test_store_and_retrieve(self, episodic_store, db):
        """Store a memory and retrieve it by query."""
        sid = self._create_session(db, "sess-ep-1")
        episodic_store.store(
            session_id=sid,
            summary="Discussion about Python web frameworks like Django and Flask",
            key_topics=["python", "web", "django", "flask"],
            key_entities=["Django", "Flask"],
            importance=0.8,
        )

        results = episodic_store.retrieve("Django web framework", top_k=5)
        assert len(results) >= 1
        memory, score = results[0]
        assert "Django" in memory.summary
        assert score > 0

    def test_retrieve_no_results(self, episodic_store):
        """Querying with no matching memories should return empty."""
        results = episodic_store.retrieve("quantum physics", top_k=5)
        assert results == []

    def test_retrieve_scores_multiple_matches(self, episodic_store, db):
        """Higher relevance should produce higher scores."""
        sid1 = self._create_session(db, "sess-ep-2")
        sid2 = self._create_session(db, "sess-ep-3")
        episodic_store.store(
            session_id=sid1,
            summary="Deep discussion about Python decorators and metaclasses",
            key_topics=["python", "decorators", "metaclasses"],
            importance=0.9,
        )
        episodic_store.store(
            session_id=sid2,
            summary="Brief mention of Python in a JavaScript discussion",
            key_topics=["javascript", "python"],
            importance=0.4,
        )

        results = episodic_store.retrieve("python decorators", top_k=5)
        assert len(results) >= 1
        # The first result should be the one about decorators (higher relevance)
        memory, score = results[0]
        assert "decorator" in memory.summary.lower()

    def test_get_recent(self, episodic_store, db):
        """get_recent should return most recent memories."""
        for i in range(5):
            sid = self._create_session(db, f"sess-ep-recent-{i}")
            episodic_store.store(
                session_id=sid,
                summary=f"Session {i} summary",
                key_topics=[f"topic-{i}"],
            )

        recent = episodic_store.get_recent(limit=3)
        assert len(recent) == 3

    def test_importance_affects_ranking(self, episodic_store, db):
        """Higher importance should boost retrieval ranking."""
        sid1 = self._create_session(db, "sess-ep-imp-1")
        sid2 = self._create_session(db, "sess-ep-imp-2")
        episodic_store.store(
            session_id=sid1,
            summary="Important Python discussion",
            key_topics=["python"],
            importance=0.9,
        )
        episodic_store.store(
            session_id=sid2,
            summary="Routine Python check-in",
            key_topics=["python"],
            importance=0.1,
        )

        results = episodic_store.retrieve("python", top_k=5)
        assert len(results) >= 2
        # Higher importance should come first (given similar text match)
        assert results[0][0].importance >= results[1][0].importance


class TestSemanticMemory:
    """Test Tier 4: Semantic memory storage and retrieval."""

    def test_store_and_retrieve_fact(self, semantic_store):
        """Store a fact and retrieve it."""
        semantic_store.store(
            content="User prefers dark mode in all applications",
            category="preference",
            tags=["ui", "dark-mode"],
            importance=0.7,
        )

        results = semantic_store.retrieve("dark mode preference")
        assert len(results) >= 1
        mem, score = results[0]
        assert "dark mode" in mem.content

    def test_category_filter(self, semantic_store):
        """Retrieval should respect category filter."""
        semantic_store.store(
            content="Python is user's favorite language",
            category="preference",
            tags=["programming"],
        )
        semantic_store.store(
            content="Python was created by Guido van Rossum",
            category="fact",
            tags=["programming", "history"],
        )

        # Filter by category
        prefs = semantic_store.retrieve("python", category="preference")
        facts = semantic_store.retrieve("python", category="fact")

        assert len(prefs) >= 1
        assert all(m.category == "preference" for m, _ in prefs)

        assert len(facts) >= 1
        assert all(m.category == "fact" for m, _ in facts)

    def test_get_by_category(self, semantic_store):
        """get_by_category should return all memories in a category."""
        for i in range(3):
            semantic_store.store(
                content=f"Insight {i}",
                category="insight",
                tags=[f"tag-{i}"],
            )
        semantic_store.store(
            content="A fact",
            category="fact",
            tags=["other"],
        )

        insights = semantic_store.get_by_category("insight")
        assert len(insights) == 3
        assert all(m.category == "insight" for m in insights)

    def test_multiple_categories(self, semantic_store):
        """Store memories across multiple categories and retrieve correctly."""
        categories = ["fact", "preference", "knowledge", "insight", "lesson"]
        for cat in categories:
            semantic_store.store(
                content=f"Test {cat} about programming",
                category=cat,
                tags=["programming"],
            )

        for cat in categories:
            results = semantic_store.retrieve("programming", category=cat)
            assert len(results) >= 1
            assert results[0][0].category == cat


class TestMemoryManager:
    """Test cross-tier memory operations."""

    def test_build_context_returns_all_tiers(self, memory_manager):
        """build_context should gather data from all memory tiers."""
        # Seed some data
        memory_manager.working.add_message("user", "Hello")
        memory_manager.update_core_memory("identity", "Test User")

        context = memory_manager.build_context("test query")
        assert "working" in context
        assert "core" in context
        assert "episodic" in context
        assert "semantic" in context

    def test_store_and_retrieve_across_tiers(self, memory_manager, db):
        """Store data in different tiers and verify cross-tier retrieval."""
        # Tier 1: Working
        memory_manager.working.add_message("user", "I like machine learning")

        # Tier 2: Core
        memory_manager.update_core_memory("preferences", "Enjoys machine learning")

        # Tier 3: Episodic (need a session)
        session_mgr = SessionManager(db)
        session = session_mgr.create_session(user_id="test", channel_type="cli")
        memory_manager.store_conversation_summary(
            session_id=session.id,
            summary="Discussed machine learning models",
            topics=["machine learning", "AI"],
            entities=["TensorFlow"],
        )

        # Tier 4: Semantic
        memory_manager.store_fact(
            content="User is a machine learning researcher",
            category="fact",
            tags=["machine learning", "career"],
        )

        # Build context and verify all tiers contributed
        context = memory_manager.build_context("machine learning")
        assert len(context["working"]) >= 1
        assert "machine learning" in context["core"].lower()
        assert len(context["episodic"]) >= 1
        assert len(context["semantic"]) >= 1

    def test_read_core_section_all(self, memory_manager):
        """read_core_section('all') should return combined context."""
        memory_manager.update_core_memory("identity", "Test Identity")
        result = memory_manager.read_core_section("all")
        assert "Test Identity" in result

    def test_update_core_section_append(self, memory_manager):
        """Appending to core memory should preserve existing content."""
        memory_manager.update_core_section("preferences", "Likes Python", mode="replace")
        memory_manager.update_core_section("preferences", "Also likes Rust", mode="append")

        content = memory_manager.read_core_section("preferences")
        assert "Likes Python" in content
        assert "Also likes Rust" in content

    def test_update_core_section_replace(self, memory_manager):
        """Replacing core memory should overwrite existing content."""
        memory_manager.update_core_section("preferences", "Likes Python", mode="replace")
        memory_manager.update_core_section("preferences", "Likes Rust", mode="replace")

        content = memory_manager.read_core_section("preferences")
        assert "Likes Rust" in content
        assert "Likes Python" not in content


# ============================================================================
# 4. Model Routing
# ============================================================================


class TestModelRouting:
    """Test the model router's complexity classification."""

    def test_simple_greetings_route_to_haiku(self):
        """Simple greetings should use the cheapest model."""
        router = ModelRouter()
        assert router.classify_complexity("hello") == "haiku"
        assert router.classify_complexity("Hi!") == "haiku"
        assert router.classify_complexity("thanks") == "haiku"
        assert router.classify_complexity("ok") == "haiku"
        assert router.classify_complexity("yes") == "haiku"
        assert router.classify_complexity("no") == "haiku"

    def test_complex_coding_routes_to_opus(self):
        """Complex coding tasks should use the most capable model."""
        router = ModelRouter()
        assert router.classify_complexity("write code to parse JSON") == "opus"
        assert router.classify_complexity("implement a binary search tree") == "opus"
        assert router.classify_complexity("refactor the authentication module") == "opus"
        assert router.classify_complexity("debug this error in the database layer") == "opus"

    def test_moderate_queries_route_to_sonnet(self):
        """Regular queries should default to sonnet."""
        router = ModelRouter()
        assert router.classify_complexity("What's the weather like?") == "sonnet"
        assert router.classify_complexity("Tell me about Python") == "sonnet"
        assert router.classify_complexity("Summarize this article") == "sonnet"

    def test_edge_cases(self):
        """Edge cases in routing."""
        router = ModelRouter()
        # Empty string
        assert router.classify_complexity("") == "sonnet"
        # Very long message defaults to sonnet
        assert router.classify_complexity("a " * 1000) == "sonnet"
        # Mixed case
        assert router.classify_complexity("HELLO") == "haiku"
        assert router.classify_complexity("Write Code") == "opus"


# ============================================================================
# 5. Error Handling
# ============================================================================


class TestErrorHandling:
    """Test the smart error handler."""

    @pytest.mark.asyncio
    async def test_rate_limit_detection(self):
        """Rate limit errors should be recognized and retried."""
        handler = SmartErrorHandler()
        error = Exception("API rate limit exceeded. Please wait.")
        result = await handler.handle(error)
        assert result.strategy_used == RecoveryStrategy.RETRY or result.strategy_used == RecoveryStrategy.ESCALATE

    @pytest.mark.asyncio
    async def test_network_error_detection(self):
        """Network errors should be recognized."""
        handler = SmartErrorHandler()
        error = Exception("Connection timed out after 30s")
        result = await handler.handle(error)
        assert result.strategy_used in (RecoveryStrategy.RETRY, RecoveryStrategy.ESCALATE)

    @pytest.mark.asyncio
    async def test_auth_error_escalates(self):
        """Auth errors should always escalate to user."""
        handler = SmartErrorHandler()
        error = Exception("401 Unauthorized - Invalid API key")
        result = await handler.handle(error)
        assert result.recovered is False

    @pytest.mark.asyncio
    async def test_novel_error_escalates(self):
        """Unknown errors should escalate to user."""
        handler = SmartErrorHandler()
        error = Exception("Something completely unknown happened")
        result = await handler.handle(error)
        assert result.recovered is False
        assert result.strategy_used == RecoveryStrategy.ESCALATE

    @pytest.mark.asyncio
    async def test_file_not_found_escalates(self):
        """FileNotFoundError should escalate."""
        handler = SmartErrorHandler()
        error = FileNotFoundError("/path/to/missing/file.txt")
        result = await handler.handle(error)
        assert result.recovered is False

    @pytest.mark.asyncio
    async def test_permission_error_escalates(self):
        """PermissionError should escalate with high severity."""
        handler = SmartErrorHandler()
        error = PermissionError("Access denied to /etc/shadow")
        result = await handler.handle(error)
        assert result.recovered is False

    @pytest.mark.asyncio
    async def test_retry_with_success(self):
        """Retry strategy should succeed if retry_fn eventually works."""
        handler = SmartErrorHandler()
        call_count = 0

        async def retry_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Connection timed out")
            # Success on second attempt

        error = Exception("Connection timed out initially")
        result = await handler.handle(error, retry_fn=retry_fn)
        assert result.recovered is True
        assert result.strategy_used == RecoveryStrategy.RETRY

    @pytest.mark.asyncio
    async def test_custom_pattern_registration(self):
        """Custom error patterns can be registered and matched."""
        from zenki.core.errors import ErrorPattern

        handler = SmartErrorHandler()
        handler.register_pattern(ErrorPattern(
            name="custom_error",
            match_fn=lambda e: "CUSTOM" in str(e),
            strategy=RecoveryStrategy.SKIP,
            message_template="Custom error handled: {error}",
        ))

        result = await handler.handle(Exception("CUSTOM error occurred"))
        assert result.recovered is True
        assert result.strategy_used == RecoveryStrategy.SKIP


# ============================================================================
# 6. Consolidation
# ============================================================================


class TestConsolidation:
    """Test memory consolidation and importance decay."""

    def test_importance_decays_over_time(self):
        """Importance should decrease for old memories."""
        old_time = datetime.now(UTC) - timedelta(days=30)
        new_time = datetime.now(UTC)

        old_importance = calculate_importance(
            base_importance=0.5,
            created_at=old_time,
            access_count=0,
        )
        new_importance = calculate_importance(
            base_importance=0.5,
            created_at=new_time,
            access_count=0,
        )

        assert old_importance < new_importance

    def test_access_boosts_importance(self):
        """Frequently accessed memories should have higher importance."""
        created = datetime.now(UTC) - timedelta(days=7)

        low_access = calculate_importance(0.5, created, access_count=0)
        high_access = calculate_importance(0.5, created, access_count=10)

        assert high_access > low_access

    def test_importance_capped_at_one(self):
        """Importance should never exceed 1.0."""
        importance = calculate_importance(
            base_importance=1.0,
            created_at=datetime.now(UTC),
            access_count=100,
        )
        assert importance <= 1.0

    @pytest.mark.asyncio
    async def test_consolidation_reviews_closed_sessions(self, db, tmp_dir):
        """Consolidation should review closed sessions and generate summaries."""
        # Create a user, session, and messages
        user = User(id="test-user", display_name="Test")
        db.create_user(user)

        session = Session(
            user_id="test-user",
            channel_type="cli",
            ended_at=datetime.now(UTC) - timedelta(hours=1),
        )
        db.create_session(session)

        db.create_message(Message(
            session_id=session.id,
            role="user",
            content="How do I implement a REST API?",
        ))
        db.create_message(Message(
            session_id=session.id,
            role="assistant",
            content="You can use Flask or FastAPI to create a REST API...",
        ))

        memory_dir = tmp_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        consolidator = MemoryConsolidator(db=db, memory_dir=memory_dir)
        result = await consolidator.run_consolidation(days_back=7)

        assert result.sessions_reviewed >= 1
        assert result.summaries_generated >= 1

    @pytest.mark.asyncio
    async def test_consolidation_skips_active_sessions(self, db, tmp_dir):
        """Consolidation should skip sessions that haven't been closed."""
        user = User(id="test-user", display_name="Test")
        db.create_user(user)

        session = Session(
            user_id="test-user",
            channel_type="cli",
            # ended_at is None - session still active
        )
        db.create_session(session)

        db.create_message(Message(
            session_id=session.id,
            role="user",
            content="Active session message",
        ))

        memory_dir = tmp_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        consolidator = MemoryConsolidator(db=db, memory_dir=memory_dir)
        result = await consolidator.run_consolidation()

        # Should not review active sessions
        assert result.summaries_generated == 0

    @pytest.mark.asyncio
    async def test_consolidation_decay(self, db, tmp_dir):
        """Consolidation should decay old memory importance."""
        user = User(id="test-user", display_name="Test")
        db.create_user(user)

        # Create an old episodic memory
        old_memory = EpisodicMemory(
            session_id=None,
            summary="Very old conversation about weather",
            key_topics=["weather"],
            importance=0.8,
        )
        # Manually set created_at to be old
        old_memory.created_at = datetime.now(UTC) - timedelta(days=90)
        db.create_episodic_memory(old_memory)

        memory_dir = tmp_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        consolidator = MemoryConsolidator(db=db, memory_dir=memory_dir)
        result = await consolidator.run_consolidation()

        assert result.memories_decayed >= 1


# ============================================================================
# 7. Edge Cases and Stress Tests
# ============================================================================


class TestEdgeCases:
    """Edge cases, boundary conditions, and stress scenarios."""

    def test_empty_message_handling(self, session_manager):
        """System should handle empty messages gracefully."""
        session = session_manager.create_session(
            user_id="user1",
            channel_type="cli",
        )
        msg = session_manager.add_message(session.id, "user", "")
        assert msg.content == ""
        messages = session_manager.get_messages(session.id)
        assert len(messages) == 1

    def test_very_long_message(self, session_manager):
        """System should handle very long messages."""
        session = session_manager.create_session(
            user_id="user1",
            channel_type="cli",
        )
        long_content = "x" * 100_000
        msg = session_manager.add_message(session.id, "user", long_content)
        assert len(msg.content) == 100_000

        retrieved = session_manager.get_messages(session.id)
        assert len(retrieved[0].content) == 100_000

    def test_unicode_in_messages(self, session_manager):
        """System should handle unicode content correctly."""
        session = session_manager.create_session(
            user_id="user1",
            channel_type="cli",
        )
        unicode_content = "Hello 🌍! これはテストです。 Привет! مرحبا"
        session_manager.add_message(session.id, "user", unicode_content)

        messages = session_manager.get_messages(session.id)
        assert messages[0].content == unicode_content

    def test_unicode_in_memory(self, memory_manager):
        """Memory system should handle unicode."""
        memory_manager.working.add_message("user", "Je parle français 🇫🇷")
        memory_manager.update_core_memory("preferences", "Parle français 🇫🇷")

        context = memory_manager.build_context("français")
        assert len(context["working"]) >= 1

    def test_special_characters_in_search(self, semantic_store):
        """Search should handle special characters gracefully."""
        semantic_store.store(
            content="User uses C++ and C# for game development",
            category="fact",
            tags=["programming", "c++", "c#"],
        )

        # These shouldn't crash
        results = semantic_store.retrieve("C++")
        assert len(results) >= 1

        results = semantic_store.retrieve("C#")
        assert len(results) >= 1

    def test_sql_injection_prevention(self, db):
        """Database should be safe against SQL injection attempts."""
        user = User(
            id="test'; DROP TABLE users;--",
            display_name="Bobby Tables",
        )
        db.create_user(user)

        # Should be stored safely
        retrieved = db.get_user("test'; DROP TABLE users;--")
        assert retrieved is not None
        assert retrieved.display_name == "Bobby Tables"

        # Users table should still exist
        count = db.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        assert count >= 1

    def test_concurrent_session_creation(self, db):
        """Multiple sessions created rapidly should all have unique IDs."""
        session_mgr = SessionManager(db)

        sessions = []
        for i in range(50):
            s = session_mgr.create_session(
                user_id=f"user-{i % 5}",
                channel_type="cli",
            )
            sessions.append(s)

        # All IDs should be unique
        ids = [s.id for s in sessions]
        assert len(ids) == len(set(ids))

    def test_working_memory_with_zero_budget(self):
        """Working memory with very small budget should still function."""
        wm = WorkingMemory(max_tokens=1)
        wm.add_message("user", "Hello world this is a test message")
        context = wm.get_context()
        # Should return at most one message (if it fits)
        assert len(context) <= 1

    def test_database_reopen_after_close(self, tmp_dir):
        """Database should work after close and reopen."""
        db_path = tmp_dir / "test.db"
        db = ZenkiDatabase(db_path)
        db.initialize()

        user = User(id="persist-test", display_name="Test")
        db.create_user(user)
        db.close()

        # Reopen
        db2 = ZenkiDatabase(db_path)
        db2.initialize()
        retrieved = db2.get_user("persist-test")
        assert retrieved is not None
        assert retrieved.display_name == "Test"
        db2.close()

    def test_message_with_tool_calls(self, session_manager):
        """Messages with tool_calls metadata should be stored correctly."""
        session = session_manager.create_session(
            user_id="user1",
            channel_type="cli",
        )
        tool_calls = [
            {"name": "memory_search", "args": {"query": "test"}, "result": "found"},
            {"name": "web_search", "args": {"query": "weather"}, "result": "sunny"},
        ]
        msg = session_manager.add_message(
            session.id, "assistant", "Here are the results...",
            tool_calls=tool_calls,
        )
        assert msg.tool_calls == tool_calls

        retrieved = session_manager.get_messages(session.id)
        assert retrieved[0].tool_calls == tool_calls


# ============================================================================
# 8. Prompt Building
# ============================================================================


class TestPromptBuilding:
    """Test system prompt construction."""

    def test_basic_prompt_generation(self):
        """build_system_prompt should produce valid prompt text."""
        prompt = build_system_prompt(
            personality={
                "tone": "friendly",
                "verbosity": "balanced",
                "proactivity": "moderate",
            }
        )
        assert "Zenki" in prompt
        assert "friendly" in prompt
        assert "balanced" in prompt

    def test_prompt_with_core_memory(self):
        """Prompt should include core memory when provided."""
        prompt = build_system_prompt(
            personality={"tone": "professional", "verbosity": "minimal", "proactivity": "low"},
            core_memory={
                "identity": "Alice, a software engineer",
                "preferences": "Prefers concise answers",
                "projects": "",
                "relationships": "",
                "patterns": "",
            },
        )
        assert "Alice" in prompt
        assert "concise answers" in prompt

    def test_prompt_with_retrieved_memories(self):
        """Prompt should include retrieved memories when provided."""
        prompt = build_system_prompt(
            personality={"tone": "casual", "verbosity": "verbose", "proactivity": "high"},
            retrieved_memories={
                "episodic": "Previously discussed Python web frameworks",
                "semantic": "User prefers FastAPI over Flask",
            },
        )
        assert "Python web frameworks" in prompt
        assert "FastAPI" in prompt


# ============================================================================
# 9. Retrieval Utilities
# ============================================================================


class TestRetrievalUtilities:
    """Test vector similarity and ranking utilities."""

    def test_cosine_similarity_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        vec = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_cosine_similarity_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) + 1.0) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        """Zero vector should return 0.0."""
        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0

    def test_cosine_similarity_mismatched_dimensions(self):
        """Mismatched vectors should raise ValueError."""
        with pytest.raises(ValueError):
            cosine_similarity([1, 2], [1, 2, 3])

    def test_rank_memories_empty(self):
        """Empty input should return empty output."""
        assert rank_memories([]) == []

    def test_rank_memories_sorts_by_score(self):
        """Memories should be ranked by combined score."""
        mem1 = MagicMock()
        mem1.created_at = datetime.now(UTC)
        mem2 = MagicMock()
        mem2.created_at = datetime.now(UTC) - timedelta(days=30)

        ranked = rank_memories([(mem1, 0.9), (mem2, 0.5)])
        assert len(ranked) == 2
        # mem1 should rank higher (higher relevance + more recent)
        assert ranked[0][1] > ranked[1][1]


# ============================================================================
# 10. Embedding Providers
# ============================================================================


class TestEmbeddingProviders:
    """Test embedding provider factory and implementations."""

    def test_dummy_provider_returns_vectors(self):
        """DummyEmbeddingProvider should return correct-dimensioned vectors."""
        provider = DummyEmbeddingProvider(dimension=128)
        vectors = provider.embed(["hello", "world"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 128
        assert len(vectors[1]) == 128

    def test_dummy_provider_normalized(self):
        """Dummy vectors should be unit-normalized."""
        import math
        provider = DummyEmbeddingProvider(dimension=64)
        vectors = provider.embed(["test"])
        norm = math.sqrt(sum(x * x for x in vectors[0]))
        assert abs(norm - 1.0) < 0.01

    def test_factory_creates_dummy(self):
        """Factory should create DummyEmbeddingProvider for 'dummy' config."""
        provider = get_embedding_provider({"provider": "dummy", "dimension": 256})
        assert isinstance(provider, DummyEmbeddingProvider)
        assert provider.dimension == 256

    def test_factory_unknown_provider(self):
        """Factory should raise for unknown provider type."""
        with pytest.raises(ValueError, match="Unknown"):
            get_embedding_provider({"provider": "nonexistent"})


# ============================================================================
# 11. SDK Context Management
# ============================================================================


class TestSDKContext:
    """Test the global service context registry."""

    def test_set_and_get_all_services(self):
        """All service types should round-trip correctly."""
        mock_db = MagicMock()
        mock_mem = MagicMock()
        mock_sched = MagicMock()
        mock_chan = MagicMock()

        _context.set_database(mock_db)
        _context.set_memory_manager(mock_mem)
        _context.set_scheduler(mock_sched)
        _context.set_channel_registry(mock_chan)

        assert _context.get_database() is mock_db
        assert _context.get_memory_manager() is mock_mem
        assert _context.get_scheduler() is mock_sched
        assert _context.get_channel_registry() is mock_chan

    def test_clear_removes_all(self):
        """clear_all should remove all registered services."""
        _context.set_database(MagicMock())
        _context.clear_all()

        assert _context.get_database() is None
        assert _context.get_memory_manager() is None

    def test_get_without_set_returns_none(self):
        """Getting an unset service should return None."""
        assert _context.get_database() is None
        assert _context.get_memory_manager() is None
        assert _context.get_scheduler() is None
        assert _context.get_channel_registry() is None


# ============================================================================
# 12. Database CRUD Operations
# ============================================================================


class TestDatabaseCRUD:
    """Test database operations for all entity types."""

    def test_user_create_and_get(self, db):
        """Create and retrieve a user."""
        user = User(id="user-1", display_name="Alice")
        db.create_user(user)

        retrieved = db.get_user("user-1")
        assert retrieved is not None
        assert retrieved.display_name == "Alice"

    def test_user_not_found(self, db):
        """Getting a non-existent user returns None."""
        assert db.get_user("nonexistent") is None

    def test_session_crud(self, db):
        """Full session CRUD cycle."""
        user = User(id="user-1")
        db.create_user(user)

        session = Session(user_id="user-1", channel_type="cli")
        db.create_session(session)

        retrieved = db.get_session(session.id)
        assert retrieved is not None
        assert retrieved.channel_type == "cli"

        # Update
        retrieved.summary = "Updated summary"
        db.update_session(retrieved)

        updated = db.get_session(session.id)
        assert updated.summary == "Updated summary"

    def test_list_sessions_by_user(self, db):
        """List sessions filtered by user."""
        db.create_user(User(id="alice"))
        db.create_user(User(id="bob"))

        db.create_session(Session(user_id="alice", channel_type="cli"))
        db.create_session(Session(user_id="alice", channel_type="slack"))
        db.create_session(Session(user_id="bob", channel_type="cli"))

        alice_sessions = db.list_sessions(user_id="alice")
        assert len(alice_sessions) == 2

        bob_sessions = db.list_sessions(user_id="bob")
        assert len(bob_sessions) == 1

    def test_episodic_memory_crud(self, db):
        """Create and search episodic memories."""
        memory = EpisodicMemory(
            summary="Discussion about Python testing best practices",
            key_topics=["python", "testing", "pytest"],
            importance=0.7,
        )
        db.create_episodic_memory(memory)

        results = db.search_episodic_memories("testing")
        assert len(results) >= 1
        assert "testing" in results[0].summary.lower()

    def test_semantic_memory_crud(self, db):
        """Create and search semantic memories."""
        memory = SemanticMemory(
            content="User's favorite editor is VS Code",
            category="preference",
            tags=["editor", "vscode"],
            importance=0.6,
        )
        db.create_semantic_memory(memory)

        results = db.search_semantic_memories("VS Code")
        assert len(results) >= 1
        assert "VS Code" in results[0].content

    def test_semantic_memory_category_filter(self, db):
        """Search semantic memories with category filter."""
        db.create_semantic_memory(SemanticMemory(
            content="Python is great",
            category="fact",
            tags=["python"],
        ))
        db.create_semantic_memory(SemanticMemory(
            content="User likes Python",
            category="preference",
            tags=["python"],
        ))

        facts = db.search_semantic_memories("Python", category="fact")
        assert len(facts) == 1
        assert facts[0].category == "fact"

    def test_scheduled_task_crud(self, db):
        """Create and retrieve scheduled tasks."""
        db.create_user(User(id="user-1"))

        task = ScheduledTask(
            user_id="user-1",
            description="Daily standup reminder",
            cron_expression="0 9 * * 1-5",
            task_type="reminder",
        )
        db.create_scheduled_task(task)

        tasks = db.get_scheduled_tasks(user_id="user-1")
        assert len(tasks) == 1
        assert tasks[0].description == "Daily standup reminder"

    def test_skill_crud(self, db):
        """Create and retrieve skills."""
        skill = Skill(
            name="code_review",
            skill_type="core",
            path="/skills/code_review.md",
            description="Automated code review",
        )
        db.create_skill(skill)

        retrieved = db.get_skill(skill.id)
        assert retrieved is not None
        assert retrieved.name == "code_review"

        # List skills
        skills = db.list_skills(skill_type="core")
        assert len(skills) == 1

    def test_message_ordering(self, db):
        """Messages should be returned in chronological order."""
        db.create_user(User(id="user-1"))
        session = Session(user_id="user-1", channel_type="cli")
        db.create_session(session)

        for i in range(5):
            db.create_message(Message(
                session_id=session.id,
                role="user",
                content=f"Message {i}",
            ))

        messages = db.get_messages_for_session(session.id)
        assert len(messages) == 5
        for i, msg in enumerate(messages):
            assert msg.content == f"Message {i}"


# ============================================================================
# 13. Settings Validation
# ============================================================================


class TestSettingsValidation:
    """Test configuration validation and loading."""

    def test_default_settings_are_valid(self):
        """Default settings should pass validation."""
        settings = ZenkiSettings()
        assert settings.version == "1.0.0"
        assert settings.llm.default_model == "sonnet"
        assert settings.personality.tone == "professional"

    def test_invalid_model_rejected(self):
        """Invalid model name should be rejected."""
        with pytest.raises(Exception):
            ZenkiSettings(llm={"default_model": "gpt-4"})

    def test_invalid_tone_rejected(self):
        """Invalid personality tone should be rejected."""
        with pytest.raises(Exception):
            ZenkiSettings(personality={"tone": "sarcastic"})

    def test_settings_save_and_load(self, tmp_dir):
        """Settings should round-trip through save/load."""
        settings = ZenkiSettings()
        settings.user.display_name = "Test User"
        settings.personality.tone = "casual"

        path = tmp_dir / "config.json"
        settings.save(path)

        loaded = ZenkiSettings.load(path)
        assert loaded.user.display_name == "Test User"
        assert loaded.personality.tone == "casual"

    def test_context_budget_validation(self):
        """Context budget percentages should be validated."""
        from zenki.config.settings import ContextBudgetConfig

        # Valid budget
        budget = ContextBudgetConfig(
            core_memory_pct=10,
            episodic_pct=15,
            semantic_pct=15,
            working_pct=60,
        )
        assert budget.working_pct == 60

        # Out of range should fail
        with pytest.raises(Exception):
            ContextBudgetConfig(working_pct=200)


# ============================================================================
# 14. SDK Tools (Unit-level with context)
# ============================================================================


class TestSDKToolsWithContext:
    """Test SDK tool operations via direct database/manager calls.

    The @tool decorator wraps functions as SdkMcpTool objects which are not
    directly callable in tests. Instead, we test the underlying operations
    that the tools perform, using the same context and database.
    """

    def test_memory_search_via_db(self, db):
        """Verify the database search that memory_search would use."""
        _context.set_database(db)

        db.create_semantic_memory(SemanticMemory(
            content="User loves hiking in the mountains",
            category="preference",
            tags=["outdoor", "hiking"],
            importance=0.8,
        ))

        # Simulate what memory_search tool does internally
        memories = db.search_semantic_memories(query="hiking", category=None)
        assert len(memories) >= 1
        assert "hiking" in memories[0].content.lower()

    def test_memory_store_via_db(self, db):
        """Verify the database store that memory_store would use."""
        _context.set_database(db)

        memory = SemanticMemory(
            content="User speaks French fluently",
            category="fact",
            tags=["language", "french"],
            importance=0.7,
        )
        db.create_semantic_memory(memory)

        # Verify it's in the database
        memories = db.search_semantic_memories("French")
        assert len(memories) >= 1
        assert memories[0].content == "User speaks French fluently"

    def test_memory_read_core_via_manager(self, memory_manager):
        """Verify the core memory read that memory_read_core would use."""
        _context.set_memory_manager(memory_manager)
        memory_manager.update_core_memory("identity", "Test Identity Content")

        # Simulate what memory_read_core tool does internally
        content = memory_manager.read_core_section("identity")
        assert "Test Identity Content" in content

    def test_memory_update_core_via_manager(self, memory_manager):
        """Verify the core memory update that memory_update_core would use."""
        _context.set_memory_manager(memory_manager)

        # Simulate what memory_update_core tool does internally
        memory_manager.update_core_section("preferences", "Prefers dark mode", mode="replace")

        content = memory_manager.read_core_section("preferences")
        assert "dark mode" in content

    def test_get_session_history_via_db(self, db):
        """Verify the session history fetch that get_session_history would use."""
        _context.set_database(db)

        db.create_user(User(id="user-1"))
        session = Session(user_id="user-1", channel_type="cli")
        db.create_session(session)

        db.create_message(Message(
            session_id=session.id,
            role="user",
            content="Hello there",
        ))
        db.create_message(Message(
            session_id=session.id,
            role="assistant",
            content="Hi! How can I help?",
        ))

        # Simulate what get_session_history tool does internally
        messages = db.get_messages_for_session(session_id=session.id)
        messages = messages[-10:]

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_tool_names_convention(self):
        """Verify the MCP tool naming convention."""
        from zenki.sdk.tools import ZENKI_TOOL_NAMES

        for name in ZENKI_TOOL_NAMES:
            assert name.startswith("mcp__zenki__"), f"Tool {name} missing zenki prefix"

    def test_create_zenki_tools_returns_dict(self):
        """create_zenki_tools should return a dict with zenki server."""
        from zenki.sdk.tools import create_zenki_tools

        tools = create_zenki_tools()
        assert isinstance(tools, dict)
        assert "zenki" in tools


# ============================================================================
# 15. Multi-User Scenarios
# ============================================================================


class TestMultiUser:
    """Test multi-user isolation and concurrent access patterns."""

    def test_users_have_separate_sessions(self, db):
        """Different users should have separate session spaces."""
        session_mgr = SessionManager(db)

        alice_session = session_mgr.create_session(
            user_id="alice",
            channel_type="cli",
        )
        bob_session = session_mgr.create_session(
            user_id="bob",
            channel_type="cli",
        )

        session_mgr.add_message(alice_session.id, "user", "Alice's message")
        session_mgr.add_message(bob_session.id, "user", "Bob's message")

        alice_active = session_mgr.list_active_sessions(user_id="alice")
        bob_active = session_mgr.list_active_sessions(user_id="bob")

        assert len(alice_active) == 1
        assert len(bob_active) == 1

        alice_msgs = session_mgr.get_messages(alice_session.id)
        bob_msgs = session_mgr.get_messages(bob_session.id)

        assert alice_msgs[0].content == "Alice's message"
        assert bob_msgs[0].content == "Bob's message"

    def test_user_auto_creation(self, db):
        """SessionManager should auto-create users on first session."""
        session_mgr = SessionManager(db)

        # No user exists yet
        assert db.get_user("new-user") is None

        # Creating a session should auto-create the user
        session = session_mgr.create_session(
            user_id="new-user",
            channel_type="cli",
        )

        assert db.get_user("new-user") is not None
