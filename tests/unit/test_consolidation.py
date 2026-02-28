"""Tests for memory consolidation engine."""

from datetime import UTC, datetime, timedelta

import pytest

from zenki.db.database import ZenkiDatabase
from zenki.db.models import Message, Session, User
from zenki.memory.consolidation import (
    ConsolidationResult,
    MemoryConsolidator,
    calculate_importance,
)


@pytest.fixture
def db(tmp_path):
    database = ZenkiDatabase(tmp_path / "test.db")
    database.initialize()
    database.create_user(User(id="test-user"))
    return database


@pytest.fixture
def consolidator(db, tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    return MemoryConsolidator(db, memory_dir)


class TestCalculateImportance:
    def test_new_memory_full_importance(self):
        now = datetime.now(UTC)
        result = calculate_importance(1.0, now, 0, now)
        assert abs(result - 1.0) < 0.01

    def test_old_memory_decays(self):
        now = datetime.now(UTC)
        old = now - timedelta(days=100)
        result = calculate_importance(1.0, old, 0, now)
        assert result < 0.5

    def test_access_boosts_importance(self):
        now = datetime.now(UTC)
        old = now - timedelta(days=30)
        no_access = calculate_importance(0.5, old, 0, now)
        with_access = calculate_importance(0.5, old, 10, now)
        assert with_access > no_access

    def test_access_boost_capped(self):
        now = datetime.now(UTC)
        result = calculate_importance(0.5, now, 1000, now)
        # Should be capped
        assert result <= 1.0

    def test_zero_importance_stays_low(self):
        now = datetime.now(UTC)
        result = calculate_importance(0.0, now, 5, now)
        assert result == 0.0

    def test_decay_rate_affects_speed(self):
        now = datetime.now(UTC)
        old = now - timedelta(days=10)
        slow_decay = calculate_importance(1.0, old, 0, now, decay_rate=0.001)
        fast_decay = calculate_importance(1.0, old, 0, now, decay_rate=0.1)
        assert slow_decay > fast_decay

    def test_handles_naive_datetimes(self):
        now = datetime.utcnow()
        result = calculate_importance(0.5, now, 0, now)
        assert 0 <= result <= 1.0


class TestConsolidationResult:
    def test_defaults(self):
        result = ConsolidationResult()
        assert result.sessions_reviewed == 0
        assert result.summaries_generated == 0
        assert result.facts_extracted == 0
        assert result.errors == []

    def test_to_dict(self):
        result = ConsolidationResult()
        result.sessions_reviewed = 5
        result.facts_extracted = 3
        d = result.to_dict()
        assert d["sessions_reviewed"] == 5
        assert d["facts_extracted"] == 3
        assert isinstance(d["errors"], list)

    def test_str_representation(self):
        result = ConsolidationResult()
        result.sessions_reviewed = 2
        result.summaries_generated = 1
        s = str(result)
        assert "Sessions reviewed: 2" in s
        assert "Summaries generated: 1" in s


class TestMemoryConsolidator:
    def test_consolidation_dir_created(self, consolidator, tmp_path):
        assert consolidator.consolidation_dir.exists()

    @pytest.mark.asyncio
    async def test_empty_consolidation(self, consolidator):
        result = await consolidator.run_consolidation()
        assert result.sessions_reviewed == 0
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_reviews_closed_sessions(self, db, consolidator):
        # Create a closed session with messages
        session = Session(
            user_id="test-user",
            channel_type="cli",
            ended_at=datetime.now(UTC),
        )
        session = db.create_session(session)

        msg = Message(
            session_id=session.id,
            role="user",
            content="How do I deploy to production?",
        )
        db.create_message(msg)

        result = await consolidator.run_consolidation(days_back=7)
        assert result.sessions_reviewed >= 1

    @pytest.mark.asyncio
    async def test_skips_already_summarized_sessions(self, db, consolidator):
        session = Session(
            user_id="test-user",
            channel_type="cli",
            ended_at=datetime.now(UTC),
            summary="Already summarized",
        )
        db.create_session(session)

        result = await consolidator.run_consolidation(days_back=7)
        assert result.sessions_reviewed == 0

    @pytest.mark.asyncio
    async def test_skips_active_sessions(self, db, consolidator):
        session = Session(
            user_id="test-user",
            channel_type="cli",
            # No ended_at = still active
        )
        db.create_session(session)

        result = await consolidator.run_consolidation(days_back=7)
        assert result.sessions_reviewed == 0

    @pytest.mark.asyncio
    async def test_generates_summary(self, db, consolidator):
        session = Session(
            user_id="test-user",
            channel_type="cli",
            ended_at=datetime.now(UTC),
        )
        session = db.create_session(session)

        db.create_message(Message(
            session_id=session.id, role="user",
            content="How do I deploy my app to AWS?",
        ))
        db.create_message(Message(
            session_id=session.id, role="assistant",
            content="You can use ECS or Lambda for deployment.",
        ))

        result = await consolidator.run_consolidation(days_back=7)
        assert result.summaries_generated >= 1

    @pytest.mark.asyncio
    async def test_saves_report(self, consolidator):
        await consolidator.run_consolidation()
        report_files = list(consolidator.consolidation_dir.glob("*_consolidation.md"))
        assert len(report_files) >= 1

    @pytest.mark.asyncio
    async def test_report_contains_stats(self, consolidator):
        await consolidator.run_consolidation()
        report_files = list(consolidator.consolidation_dir.glob("*_consolidation.md"))
        content = report_files[0].read_text()
        assert "Consolidation Report" in content
        assert "Sessions reviewed" in content

    def test_generate_simple_summary(self, consolidator):
        class FakeMsg:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        messages = [
            FakeMsg("user", "How do I deploy to production?"),
            FakeMsg("assistant", "Use CI/CD pipeline."),
            FakeMsg("user", "What about staging?"),
        ]
        summary = consolidator._generate_simple_summary(messages)
        assert "deploy" in summary.lower()

    def test_generate_summary_empty_messages(self, consolidator):
        summary = consolidator._generate_simple_summary([])
        assert summary == ""

    def test_extract_topics(self, consolidator):
        class FakeMsg:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        messages = [
            FakeMsg("user", "Tell me about Python and Docker"),
            FakeMsg("assistant", "Sure, Python is..."),
        ]
        topics = consolidator._extract_topics(messages)
        assert isinstance(topics, list)
