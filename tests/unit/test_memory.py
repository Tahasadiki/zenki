"""Comprehensive tests for the Zenki 4-tier memory system."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from zenki.config.settings import ZenkiSettings
from zenki.db.database import ZenkiDatabase
from zenki.db.models import EpisodicMemory, SemanticMemory, Session, User
from zenki.memory.core_memory import CORE_MEMORY_FILES, CoreMemory
from zenki.memory.embeddings import (
    DummyEmbeddingProvider,
    LocalEmbeddingProvider,
    get_embedding_provider,
)
from zenki.memory.episodic import EpisodicMemoryStore
from zenki.memory.manager import MemoryManager
from zenki.memory.retrieval import cosine_similarity, rank_memories
from zenki.memory.semantic import SemanticMemoryStore
from zenki.memory.working import WorkingMemory

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def dummy_provider() -> DummyEmbeddingProvider:
    """Return a DummyEmbeddingProvider with default dimension."""
    return DummyEmbeddingProvider(dimension=384)


@pytest.fixture()
def db(tmp_path: Path) -> ZenkiDatabase:
    """Create an initialised in-tmp database."""
    db_path = tmp_path / "zenki_test.db"
    database = ZenkiDatabase(db_path)
    database.initialize()

    # Seed a user and session so foreign-key constraints are satisfied.
    user = User(id="test-user", display_name="Tester")
    database.create_user(user)
    session = Session(id="test-session", user_id="test-user", channel_type="cli")
    database.create_session(session)

    yield database
    database.close()


@pytest.fixture()
def memory_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for core memory files."""
    return tmp_path / "memory"


@pytest.fixture()
def settings() -> ZenkiSettings:
    """Return default ZenkiSettings."""
    return ZenkiSettings()


# ======================================================================
# WorkingMemory tests
# ======================================================================


class TestWorkingMemory:
    """Tests for Tier 1: WorkingMemory."""

    def test_add_and_get_messages(self) -> None:
        wm = WorkingMemory(max_tokens=60000)
        wm.add_message("user", "Hello!")
        wm.add_message("assistant", "Hi there, how can I help?")

        ctx = wm.get_context()
        assert len(ctx) == 2
        assert ctx[0]["role"] == "user"
        assert ctx[0]["content"] == "Hello!"
        assert ctx[1]["role"] == "assistant"

    def test_get_context_empty(self) -> None:
        wm = WorkingMemory()
        assert wm.get_context() == []

    def test_clear(self) -> None:
        wm = WorkingMemory()
        wm.add_message("user", "Test")
        wm.clear()
        assert wm.get_context() == []
        assert wm.message_count == 0

    def test_estimate_tokens(self) -> None:
        wm = WorkingMemory()
        # 20 chars -> 5 tokens
        assert wm.estimate_tokens("a" * 20) == 5
        # Empty string -> 1 (minimum)
        assert wm.estimate_tokens("") == 1
        # Very long string
        assert wm.estimate_tokens("x" * 4000) == 1000

    def test_budget_enforcement(self) -> None:
        # Set a very small budget
        wm = WorkingMemory(max_tokens=10)
        # Each message ~25 tokens (100 chars / 4)
        wm.add_message("user", "x" * 100)
        wm.add_message("user", "y" * 100)
        wm.add_message("user", "z" * 20)  # ~5 tokens, fits in budget

        ctx = wm.get_context()
        # Only the last message should fit (5 tokens < 10 budget)
        assert len(ctx) == 1
        assert ctx[0]["content"] == "z" * 20

    def test_budget_returns_most_recent_fitting(self) -> None:
        wm = WorkingMemory(max_tokens=50)
        wm.add_message("user", "a" * 40)   # 10 tokens
        wm.add_message("user", "b" * 80)   # 20 tokens
        wm.add_message("user", "c" * 40)   # 10 tokens

        ctx = wm.get_context()
        # Should fit b (20) + c (10) = 30 <= 50, but not a+b+c = 40
        assert len(ctx) == 3  # Actually 10+20+10=40 <= 50, all fit
        # Verify all three fit within 50 tokens
        total = sum(wm.estimate_tokens(m["content"]) for m in ctx)
        assert total <= 50

    def test_get_summary_no_messages(self) -> None:
        wm = WorkingMemory()
        assert wm.get_summary() == "No conversation yet."

    def test_get_summary_with_messages(self) -> None:
        wm = WorkingMemory()
        wm.add_message("user", "Hello")
        wm.add_message("assistant", "Hi!")

        summary = wm.get_summary()
        assert "2 messages" in summary
        assert "user: Hello" in summary
        assert "assistant: Hi!" in summary

    def test_get_summary_truncates_long_content(self) -> None:
        wm = WorkingMemory()
        long_msg = "a" * 200
        wm.add_message("user", long_msg)

        summary = wm.get_summary()
        assert "..." in summary

    def test_message_count(self) -> None:
        wm = WorkingMemory()
        assert wm.message_count == 0
        wm.add_message("user", "one")
        wm.add_message("assistant", "two")
        assert wm.message_count == 2

    def test_total_tokens(self) -> None:
        wm = WorkingMemory()
        wm.add_message("user", "a" * 40)  # 10 tokens
        wm.add_message("assistant", "b" * 80)  # 20 tokens
        assert wm.total_tokens == 30


# ======================================================================
# CoreMemory tests
# ======================================================================


class TestCoreMemory:
    """Tests for Tier 2: CoreMemory."""

    def test_ensure_files_creates_directory_and_files(self, memory_dir: Path) -> None:
        cm = CoreMemory(memory_dir)
        cm.ensure_files()

        assert memory_dir.exists()
        for filename in CORE_MEMORY_FILES:
            assert (memory_dir / filename).exists()

    def test_ensure_files_idempotent(self, memory_dir: Path) -> None:
        cm = CoreMemory(memory_dir)
        cm.ensure_files()
        # Write something
        (memory_dir / "identity.md").write_text("I am a test user")
        # Ensure again should not overwrite
        cm.ensure_files()
        assert (memory_dir / "identity.md").read_text() == "I am a test user"

    def test_load_all(self, memory_dir: Path) -> None:
        cm = CoreMemory(memory_dir)
        cm.ensure_files()
        (memory_dir / "identity.md").write_text("Test user identity")

        data = cm.load_all()
        assert "identity" in data
        assert data["identity"] == "Test user identity"
        assert "preferences" in data
        assert "projects" in data

    def test_get(self, memory_dir: Path) -> None:
        cm = CoreMemory(memory_dir)
        cm.ensure_files()
        (memory_dir / "preferences.md").write_text("Dark mode please")

        assert cm.get("preferences") == "Dark mode please"

    def test_get_nonexistent_returns_empty(self, memory_dir: Path) -> None:
        cm = CoreMemory(memory_dir)
        assert cm.get("nonexistent") == ""

    def test_update(self, memory_dir: Path) -> None:
        cm = CoreMemory(memory_dir)
        cm.update("projects", "# Active Projects\n- Zenki")

        assert (memory_dir / "projects.md").read_text() == "# Active Projects\n- Zenki"

    def test_update_creates_dir_if_needed(self, tmp_path: Path) -> None:
        deep_dir = tmp_path / "a" / "b" / "c"
        cm = CoreMemory(deep_dir)
        cm.update("identity", "New user")
        assert (deep_dir / "identity.md").read_text() == "New user"

    def test_get_context_string_empty(self, memory_dir: Path) -> None:
        cm = CoreMemory(memory_dir)
        cm.ensure_files()
        # All files are empty
        assert cm.get_context_string() == ""

    def test_get_context_string_with_content(self, memory_dir: Path) -> None:
        cm = CoreMemory(memory_dir)
        cm.ensure_files()
        (memory_dir / "identity.md").write_text("Name: Alice")
        (memory_dir / "preferences.md").write_text("Theme: dark")

        ctx = cm.get_context_string()
        assert "### Identity" in ctx
        assert "Name: Alice" in ctx
        assert "### Preferences" in ctx
        assert "Theme: dark" in ctx

    def test_get_context_string_skips_empty_sections(self, memory_dir: Path) -> None:
        cm = CoreMemory(memory_dir)
        cm.ensure_files()
        (memory_dir / "identity.md").write_text("Name: Bob")

        ctx = cm.get_context_string()
        assert "### Identity" in ctx
        # Empty files should not produce headings
        assert "### Preferences" not in ctx
        assert "### Projects" not in ctx


# ======================================================================
# EpisodicMemoryStore tests
# ======================================================================


class TestEpisodicMemoryStore:
    """Tests for Tier 3: EpisodicMemoryStore."""

    def test_store(self, db: ZenkiDatabase, dummy_provider: DummyEmbeddingProvider) -> None:
        store = EpisodicMemoryStore(db, dummy_provider)
        mem = store.store(
            session_id="test-session",
            summary="Discussed Python project setup",
            key_topics=["python", "setup"],
            key_entities=["user"],
            importance=0.7,
        )

        assert isinstance(mem, EpisodicMemory)
        assert mem.summary == "Discussed Python project setup"
        assert mem.key_topics == ["python", "setup"]
        assert mem.importance == 0.7

    def test_retrieve_by_text(
        self, db: ZenkiDatabase, dummy_provider: DummyEmbeddingProvider
    ) -> None:
        store = EpisodicMemoryStore(db, dummy_provider)
        store.store(
            session_id="test-session",
            summary="Discussed Python project setup with pytest",
            key_topics=["python", "pytest"],
            key_entities=["user"],
        )
        store.store(
            session_id="test-session",
            summary="Talked about cooking recipes",
            key_topics=["cooking", "recipes"],
            key_entities=["user"],
        )

        results = store.retrieve("python pytest", min_score=0.0)
        assert len(results) >= 1
        # The Python-related memory should be the top result
        top_mem, top_score = results[0]
        assert "python" in top_mem.summary.lower() or "pytest" in " ".join(top_mem.key_topics)

    def test_retrieve_returns_tuples(
        self, db: ZenkiDatabase, dummy_provider: DummyEmbeddingProvider
    ) -> None:
        store = EpisodicMemoryStore(db, dummy_provider)
        store.store(
            session_id="test-session",
            summary="Quick test",
            key_topics=["test"],
        )
        results = store.retrieve("test", min_score=0.0)
        for mem, score in results:
            assert isinstance(mem, EpisodicMemory)
            assert isinstance(score, float)

    def test_retrieve_respects_top_k(
        self, db: ZenkiDatabase, dummy_provider: DummyEmbeddingProvider
    ) -> None:
        store = EpisodicMemoryStore(db, dummy_provider)
        for i in range(10):
            store.store(
                session_id="test-session",
                summary=f"Meeting about topic {i} with a test keyword",
                key_topics=["test"],
            )

        results = store.retrieve("test", top_k=3, min_score=0.0)
        assert len(results) <= 3

    def test_retrieve_respects_min_score(
        self, db: ZenkiDatabase, dummy_provider: DummyEmbeddingProvider
    ) -> None:
        store = EpisodicMemoryStore(db, dummy_provider)
        store.store(
            session_id="test-session",
            summary="Completely unrelated topic about quantum physics",
            key_topics=["quantum", "physics"],
        )

        # Query something very different
        results = store.retrieve("cooking recipes food", min_score=0.9)
        # With min_score=0.9, unlikely to match
        assert len(results) == 0

    def test_get_recent(
        self, db: ZenkiDatabase, dummy_provider: DummyEmbeddingProvider
    ) -> None:
        store = EpisodicMemoryStore(db, dummy_provider)
        for i in range(5):
            store.store(
                session_id="test-session",
                summary=f"Session summary {i}",
                key_topics=[f"topic{i}"],
            )

        recent = store.get_recent(limit=3)
        assert len(recent) == 3
        assert all(isinstance(m, EpisodicMemory) for m in recent)

    def test_store_with_defaults(
        self, db: ZenkiDatabase, dummy_provider: DummyEmbeddingProvider
    ) -> None:
        store = EpisodicMemoryStore(db, dummy_provider)
        mem = store.store(
            session_id="test-session",
            summary="Simple summary",
        )
        assert mem.key_topics == []
        assert mem.key_entities == []
        assert mem.importance == 0.5


# ======================================================================
# SemanticMemoryStore tests
# ======================================================================


class TestSemanticMemoryStore:
    """Tests for Tier 4: SemanticMemoryStore."""

    def test_store(self, db: ZenkiDatabase, dummy_provider: DummyEmbeddingProvider) -> None:
        store = SemanticMemoryStore(db, dummy_provider)
        mem = store.store(
            content="User prefers dark mode",
            category="preference",
            tags=["ui", "theme"],
            source="session-1",
            importance=0.8,
        )

        assert isinstance(mem, SemanticMemory)
        assert mem.content == "User prefers dark mode"
        assert mem.category == "preference"
        assert mem.tags == ["ui", "theme"]
        assert mem.importance == 0.8

    def test_retrieve_by_text(
        self, db: ZenkiDatabase, dummy_provider: DummyEmbeddingProvider
    ) -> None:
        store = SemanticMemoryStore(db, dummy_provider)
        store.store(content="User prefers dark mode", category="preference", tags=["ui"])
        store.store(content="Python is user's main language", category="fact", tags=["programming"])

        results = store.retrieve("dark mode", min_score=0.0)
        assert len(results) >= 1
        top_mem, score = results[0]
        assert "dark" in top_mem.content.lower() or "mode" in top_mem.content.lower()

    def test_retrieve_with_category_filter(
        self, db: ZenkiDatabase, dummy_provider: DummyEmbeddingProvider
    ) -> None:
        store = SemanticMemoryStore(db, dummy_provider)
        store.store(content="User likes Python", category="preference", tags=["python"])
        store.store(content="Python was created by Guido", category="fact", tags=["python"])

        # Filter by category
        results = store.retrieve("Python", category="fact", min_score=0.0)
        for mem, score in results:
            assert mem.category == "fact"

    def test_get_by_category(
        self, db: ZenkiDatabase, dummy_provider: DummyEmbeddingProvider
    ) -> None:
        store = SemanticMemoryStore(db, dummy_provider)
        store.store(content="Fact one", category="fact")
        store.store(content="Fact two", category="fact")
        store.store(content="A preference", category="preference")

        facts = store.get_by_category("fact")
        assert len(facts) == 2
        assert all(m.category == "fact" for m in facts)

    def test_retrieve_respects_top_k(
        self, db: ZenkiDatabase, dummy_provider: DummyEmbeddingProvider
    ) -> None:
        store = SemanticMemoryStore(db, dummy_provider)
        for i in range(10):
            store.store(
                content=f"Knowledge item {i} about test subject",
                category="knowledge",
                tags=["test"],
            )

        results = store.retrieve("test", top_k=3, min_score=0.0)
        assert len(results) <= 3

    def test_store_with_defaults(
        self, db: ZenkiDatabase, dummy_provider: DummyEmbeddingProvider
    ) -> None:
        store = SemanticMemoryStore(db, dummy_provider)
        mem = store.store(content="A fact", category="fact")
        assert mem.tags == []
        assert mem.source is None
        assert mem.importance == 0.5


# ======================================================================
# MemoryManager tests
# ======================================================================


class TestMemoryManager:
    """Tests for the MemoryManager orchestrator."""

    def test_lazy_init_working(
        self, settings: ZenkiSettings, db: ZenkiDatabase, tmp_path: Path
    ) -> None:
        mgr = MemoryManager(settings, db, tmp_path)
        assert mgr._working is None
        wm = mgr.working
        assert isinstance(wm, WorkingMemory)
        assert mgr._working is not None

    def test_lazy_init_core(
        self, settings: ZenkiSettings, db: ZenkiDatabase, tmp_path: Path
    ) -> None:
        mgr = MemoryManager(settings, db, tmp_path)
        assert mgr._core is None
        cm = mgr.core
        assert isinstance(cm, CoreMemory)
        assert mgr._core is not None

    def test_lazy_init_episodic(
        self, settings: ZenkiSettings, db: ZenkiDatabase, tmp_path: Path
    ) -> None:
        mgr = MemoryManager(settings, db, tmp_path)
        assert mgr._episodic is None
        es = mgr.episodic
        assert isinstance(es, EpisodicMemoryStore)

    def test_lazy_init_semantic(
        self, settings: ZenkiSettings, db: ZenkiDatabase, tmp_path: Path
    ) -> None:
        mgr = MemoryManager(settings, db, tmp_path)
        assert mgr._semantic is None
        ss = mgr.semantic
        assert isinstance(ss, SemanticMemoryStore)

    def test_build_context_integrates_all_tiers(
        self, settings: ZenkiSettings, db: ZenkiDatabase, tmp_path: Path
    ) -> None:
        mgr = MemoryManager(settings, db, tmp_path)

        # Populate working memory
        mgr.working.add_message("user", "Tell me about Python")
        mgr.working.add_message("assistant", "Python is a great language!")

        # Populate core memory
        mgr.core.update("identity", "Name: Test User")

        # Populate episodic memory
        mgr.episodic.store(
            session_id="test-session",
            summary="Discussed Python programming",
            key_topics=["python"],
        )

        # Populate semantic memory
        mgr.semantic.store(
            content="User is a Python developer",
            category="fact",
            tags=["python"],
        )

        context = mgr.build_context("Python programming")

        assert "working" in context
        assert "core" in context
        assert "episodic" in context
        assert "semantic" in context

        # Working memory should have messages
        assert len(context["working"]) == 2

        # Core memory should have content
        assert "Test User" in context["core"]

        # Episodic and semantic are lists of dicts
        assert isinstance(context["episodic"], list)
        assert isinstance(context["semantic"], list)

    def test_store_conversation_summary(
        self, settings: ZenkiSettings, db: ZenkiDatabase, tmp_path: Path
    ) -> None:
        mgr = MemoryManager(settings, db, tmp_path)
        mgr.store_conversation_summary(
            session_id="test-session",
            summary="We discussed project setup",
            topics=["setup", "project"],
            entities=["user"],
        )

        # Verify it was stored
        recent = mgr.episodic.get_recent(limit=1)
        assert len(recent) == 1
        assert recent[0].summary == "We discussed project setup"

    def test_store_fact(
        self, settings: ZenkiSettings, db: ZenkiDatabase, tmp_path: Path
    ) -> None:
        mgr = MemoryManager(settings, db, tmp_path)
        mgr.store_fact(
            content="User lives in San Francisco",
            category="fact",
            tags=["location"],
        )

        results = mgr.semantic.retrieve("San Francisco", min_score=0.0)
        assert len(results) >= 1

    def test_update_core_memory(
        self, settings: ZenkiSettings, db: ZenkiDatabase, tmp_path: Path
    ) -> None:
        mgr = MemoryManager(settings, db, tmp_path)
        mgr.update_core_memory("identity", "Name: Updated User")

        content = mgr.core.get("identity")
        assert content == "Name: Updated User"

    def test_build_context_empty(
        self, settings: ZenkiSettings, db: ZenkiDatabase, tmp_path: Path
    ) -> None:
        """build_context works even when all tiers are empty."""
        mgr = MemoryManager(settings, db, tmp_path)
        context = mgr.build_context("anything")

        assert context["working"] == []
        assert isinstance(context["core"], str)
        assert context["episodic"] == []
        assert context["semantic"] == []


# ======================================================================
# Retrieval utilities tests
# ======================================================================


class TestRetrievalUtilities:
    """Tests for cosine_similarity and rank_memories."""

    def test_cosine_similarity_identical(self) -> None:
        vec = [1.0, 0.0, 0.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_opposite(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_cosine_similarity_general(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        # Manual: dot=32, |a|=sqrt(14), |b|=sqrt(77)
        expected = 32 / (math.sqrt(14) * math.sqrt(77))
        assert cosine_similarity(a, b) == pytest.approx(expected, rel=1e-6)

    def test_cosine_similarity_zero_vector(self) -> None:
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert cosine_similarity(a, b) == 0.0

    def test_cosine_similarity_different_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_rank_memories_empty(self) -> None:
        assert rank_memories([]) == []

    def test_rank_memories_single(self) -> None:
        mem = EpisodicMemory(summary="test", created_at=datetime.now(UTC))
        result = rank_memories([(mem, 0.8)])
        assert len(result) == 1
        # Single item: recency should be 1.0 (age = 0, so 1 - 0/max = 1)
        _m, score = result[0]
        assert score > 0

    def test_rank_memories_ordering(self) -> None:
        now = datetime.now(UTC)
        old_mem = EpisodicMemory(
            summary="old",
            created_at=now - timedelta(days=30),
        )
        new_mem = EpisodicMemory(
            summary="new",
            created_at=now - timedelta(seconds=10),
        )

        memories = [(old_mem, 0.7), (new_mem, 0.7)]
        ranked = rank_memories(memories, recency_weight=0.5)

        # With equal relevance scores, the newer memory should rank higher
        assert ranked[0][0].summary == "new"

    def test_rank_memories_recency_vs_relevance(self) -> None:
        now = datetime.now(UTC)
        old_relevant = EpisodicMemory(
            summary="old but relevant",
            created_at=now - timedelta(days=30),
        )
        new_irrelevant = EpisodicMemory(
            summary="new but less relevant",
            created_at=now - timedelta(seconds=10),
        )

        # High relevance for old, low for new
        memories = [(old_relevant, 0.95), (new_irrelevant, 0.1)]
        ranked = rank_memories(memories, recency_weight=0.2)

        # With recency_weight=0.2, relevance dominates
        assert ranked[0][0].summary == "old but relevant"


# ======================================================================
# Embedding provider tests
# ======================================================================


class TestDummyEmbeddingProvider:
    """Tests for the DummyEmbeddingProvider."""

    def test_returns_correct_count(self) -> None:
        provider = DummyEmbeddingProvider(dimension=128)
        texts = ["hello", "world", "test"]
        embeddings = provider.embed(texts)
        assert len(embeddings) == 3

    def test_returns_correct_dimension(self) -> None:
        provider = DummyEmbeddingProvider(dimension=256)
        embeddings = provider.embed(["hello"])
        assert len(embeddings[0]) == 256

    def test_returns_float_vectors(self) -> None:
        provider = DummyEmbeddingProvider()
        embeddings = provider.embed(["test"])
        assert all(isinstance(v, float) for v in embeddings[0])

    def test_vectors_are_normalised(self) -> None:
        provider = DummyEmbeddingProvider(dimension=384)
        embeddings = provider.embed(["test vector"])
        vec = embeddings[0]
        norm = math.sqrt(sum(x * x for x in vec))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_empty_input(self) -> None:
        provider = DummyEmbeddingProvider()
        embeddings = provider.embed([])
        assert embeddings == []

    def test_default_dimension(self) -> None:
        provider = DummyEmbeddingProvider()
        assert provider.dimension == 384


class TestGetEmbeddingProvider:
    """Tests for the get_embedding_provider factory function."""

    def test_dummy_provider(self) -> None:
        provider = get_embedding_provider({"provider": "dummy"})
        assert isinstance(provider, DummyEmbeddingProvider)

    def test_dummy_provider_custom_dimension(self) -> None:
        provider = get_embedding_provider({"provider": "dummy", "dimension": 128})
        assert isinstance(provider, DummyEmbeddingProvider)
        assert provider.dimension == 128

    def test_local_provider(self) -> None:
        provider = get_embedding_provider({"provider": "local"})
        assert isinstance(provider, LocalEmbeddingProvider)

    def test_local_provider_custom_model(self) -> None:
        provider = get_embedding_provider({"provider": "local", "model": "custom-model"})
        assert isinstance(provider, LocalEmbeddingProvider)
        assert provider.model_name == "custom-model"

    def test_default_is_local(self) -> None:
        provider = get_embedding_provider({})
        assert isinstance(provider, LocalEmbeddingProvider)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_provider({"provider": "openai"})

    def test_local_provider_lazy_loads(self) -> None:
        provider = get_embedding_provider({"provider": "local"})
        assert isinstance(provider, LocalEmbeddingProvider)
        # Model should not be loaded yet
        assert provider._model is None
