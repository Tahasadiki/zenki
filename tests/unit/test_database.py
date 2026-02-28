"""Comprehensive tests for the Zenki database layer."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from zenki.db import (
    EpisodicMemory,
    Message,
    ScheduledTask,
    SemanticMemory,
    Session,
    Skill,
    User,
    ZenkiDatabase,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> ZenkiDatabase:
    """Return an initialised ZenkiDatabase backed by a temp file."""
    database = ZenkiDatabase(tmp_path / "test.db")
    database.initialize()
    return database


@pytest.fixture()
def user(db: ZenkiDatabase) -> User:
    """Create and return a persisted test user."""
    u = User(display_name="Alice")
    return db.create_user(u)


@pytest.fixture()
def session(db: ZenkiDatabase, user: User) -> Session:
    """Create and return a persisted test session."""
    s = Session(user_id=user.id, channel_type="cli")
    return db.create_session(s)


# ===========================================================================
# Database initialisation
# ===========================================================================


class TestDatabaseInitialization:
    """Tests for database setup and table creation."""

    def test_initialize_creates_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "init_test.db"
        database = ZenkiDatabase(db_path)
        database.initialize()
        assert db_path.exists()
        database.close()

    def test_initialize_creates_all_tables(self, db: ZenkiDatabase) -> None:
        rows = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = sorted(row["name"] for row in rows)
        expected = sorted([
            "users",
            "sessions",
            "messages",
            "episodic_memories",
            "semantic_memories",
            "scheduled_tasks",
            "skills",
        ])
        assert table_names == expected

    def test_initialize_is_idempotent(self, db: ZenkiDatabase) -> None:
        """Calling initialize() twice should not raise."""
        db.initialize()
        rows = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        assert len(rows) == 7

    def test_wal_mode_enabled(self, db: ZenkiDatabase) -> None:
        mode = db.conn.execute("PRAGMA journal_mode").fetchone()
        assert mode[0] == "wal"

    def test_foreign_keys_enabled(self, db: ZenkiDatabase) -> None:
        fk = db.conn.execute("PRAGMA foreign_keys").fetchone()
        assert fk[0] == 1

    def test_close_and_reopen(self, tmp_path: Path) -> None:
        db_path = tmp_path / "reopen.db"
        database = ZenkiDatabase(db_path)
        database.initialize()
        database.create_user(User(display_name="Bob"))
        database.close()

        database2 = ZenkiDatabase(db_path)
        database2.initialize()
        users = database2.conn.execute("SELECT * FROM users").fetchall()
        assert len(users) == 1
        database2.close()

    def test_indexes_created(self, db: ZenkiDatabase) -> None:
        rows = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
        index_names = {row["name"] for row in rows}
        expected_indexes = {
            "idx_messages_session",
            "idx_episodic_importance",
            "idx_semantic_category",
            "idx_semantic_importance",
            "idx_tasks_next_run",
            "idx_skills_status",
        }
        assert expected_indexes.issubset(index_names)


# ===========================================================================
# User CRUD
# ===========================================================================


class TestUserCRUD:
    """Tests for User create and read operations."""

    def test_create_user_returns_user(self, db: ZenkiDatabase) -> None:
        u = User(display_name="TestUser")
        result = db.create_user(u)
        assert result.id == u.id
        assert result.display_name == "TestUser"

    def test_get_user_by_id(self, db: ZenkiDatabase, user: User) -> None:
        fetched = db.get_user(user.id)
        assert fetched is not None
        assert fetched.id == user.id
        assert fetched.display_name == user.display_name

    def test_get_user_not_found(self, db: ZenkiDatabase) -> None:
        result = db.get_user("nonexistent-id")
        assert result is None

    def test_user_has_uuid_id(self, db: ZenkiDatabase) -> None:
        u = User(display_name="UUID check")
        db.create_user(u)
        assert len(u.id) == 36  # UUID4 format: 8-4-4-4-12

    def test_user_has_created_at(self, db: ZenkiDatabase) -> None:
        u = User(display_name="Timestamp check")
        db.create_user(u)
        assert u.created_at is not None
        assert isinstance(u.created_at, datetime)

    def test_user_config_json_roundtrip(self, db: ZenkiDatabase) -> None:
        config = {"theme": "dark", "language": "en", "notifications": True}
        u = User(display_name="ConfigUser", config=config)
        db.create_user(u)

        fetched = db.get_user(u.id)
        assert fetched is not None
        assert fetched.config == config
        assert fetched.config["theme"] == "dark"
        assert fetched.config["notifications"] is True

    def test_user_empty_config_roundtrip(self, db: ZenkiDatabase) -> None:
        u = User(display_name="EmptyConfig")
        db.create_user(u)
        fetched = db.get_user(u.id)
        assert fetched is not None
        assert fetched.config == {}

    def test_user_default_display_name_is_none(self) -> None:
        u = User()
        assert u.display_name is None

    def test_duplicate_user_id_raises(self, db: ZenkiDatabase) -> None:
        u = User(display_name="First")
        db.create_user(u)
        duplicate = User(id=u.id, display_name="Second")
        with pytest.raises(sqlite3.IntegrityError):
            db.create_user(duplicate)


# ===========================================================================
# Session CRUD
# ===========================================================================


class TestSessionCRUD:
    """Tests for Session create, read, update, and list operations."""

    def test_create_session(self, db: ZenkiDatabase, user: User) -> None:
        s = Session(user_id=user.id, channel_type="slack")
        result = db.create_session(s)
        assert result.id == s.id
        assert result.channel_type == "slack"

    def test_get_session_by_id(self, db: ZenkiDatabase, session: Session) -> None:
        fetched = db.get_session(session.id)
        assert fetched is not None
        assert fetched.id == session.id
        assert fetched.user_id == session.user_id

    def test_get_session_not_found(self, db: ZenkiDatabase) -> None:
        assert db.get_session("nonexistent") is None

    def test_update_session(self, db: ZenkiDatabase, session: Session) -> None:
        session.summary = "Updated summary"
        session.ended_at = datetime.now(UTC)
        db.update_session(session)

        fetched = db.get_session(session.id)
        assert fetched is not None
        assert fetched.summary == "Updated summary"
        assert fetched.ended_at is not None

    def test_list_sessions_all(self, db: ZenkiDatabase, user: User) -> None:
        db.create_session(Session(user_id=user.id, channel_type="cli"))
        db.create_session(Session(user_id=user.id, channel_type="slack"))
        sessions = db.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_by_user(self, db: ZenkiDatabase) -> None:
        u1 = db.create_user(User(display_name="User1"))
        u2 = db.create_user(User(display_name="User2"))
        db.create_session(Session(user_id=u1.id, channel_type="cli"))
        db.create_session(Session(user_id=u1.id, channel_type="slack"))
        db.create_session(Session(user_id=u2.id, channel_type="cli"))

        u1_sessions = db.list_sessions(user_id=u1.id)
        u2_sessions = db.list_sessions(user_id=u2.id)
        assert len(u1_sessions) == 2
        assert len(u2_sessions) == 1

    def test_session_metadata_json_roundtrip(self, db: ZenkiDatabase, user: User) -> None:
        meta = {"platform": "macos", "version": "1.0", "tags": ["important"]}
        s = Session(user_id=user.id, channel_type="cli", metadata=meta)
        db.create_session(s)
        fetched = db.get_session(s.id)
        assert fetched is not None
        assert fetched.metadata == meta
        assert fetched.metadata["tags"] == ["important"]

    def test_session_optional_fields_null(self, db: ZenkiDatabase, user: User) -> None:
        s = Session(user_id=user.id, channel_type="cli")
        db.create_session(s)
        fetched = db.get_session(s.id)
        assert fetched is not None
        assert fetched.channel_id is None
        assert fetched.thread_id is None
        assert fetched.context_summary is None
        assert fetched.ended_at is None
        assert fetched.summary is None

    def test_session_with_all_fields(self, db: ZenkiDatabase, user: User) -> None:
        now = datetime.now(UTC)
        s = Session(
            user_id=user.id,
            channel_type="slack",
            channel_id="C12345",
            thread_id="T67890",
            context_summary="Discussing project plans",
            started_at=now,
            last_active=now,
            summary="A productive session",
            metadata={"priority": "high"},
        )
        db.create_session(s)
        fetched = db.get_session(s.id)
        assert fetched is not None
        assert fetched.channel_id == "C12345"
        assert fetched.thread_id == "T67890"
        assert fetched.context_summary == "Discussing project plans"
        assert fetched.summary == "A productive session"


# ===========================================================================
# Message CRUD
# ===========================================================================


class TestMessageCRUD:
    """Tests for Message create and retrieval operations."""

    def test_create_message(self, db: ZenkiDatabase, session: Session) -> None:
        msg = Message(session_id=session.id, role="user", content="Hello!")
        result = db.create_message(msg)
        assert result.id == msg.id
        assert result.role == "user"

    def test_get_messages_for_session(self, db: ZenkiDatabase, session: Session) -> None:
        db.create_message(Message(session_id=session.id, role="user", content="Hi"))
        db.create_message(Message(session_id=session.id, role="assistant", content="Hello!"))
        db.create_message(Message(session_id=session.id, role="user", content="How are you?"))

        messages = db.get_messages_for_session(session.id)
        assert len(messages) == 3
        assert messages[0].content == "Hi"
        assert messages[1].role == "assistant"
        assert messages[2].content == "How are you?"

    def test_get_messages_empty_session(self, db: ZenkiDatabase, session: Session) -> None:
        messages = db.get_messages_for_session(session.id)
        assert messages == []

    def test_message_tool_calls_json_roundtrip(self, db: ZenkiDatabase, session: Session) -> None:
        tool_calls = [
            {"name": "search", "args": {"query": "python"}},
            {"name": "read_file", "args": {"path": "/tmp/test"}},
        ]
        msg = Message(
            session_id=session.id,
            role="assistant",
            content="Let me search that.",
            tool_calls=tool_calls,
        )
        db.create_message(msg)

        messages = db.get_messages_for_session(session.id)
        assert len(messages) == 1
        assert messages[0].tool_calls == tool_calls
        assert messages[0].tool_calls[0]["name"] == "search"

    def test_message_tool_calls_none(self, db: ZenkiDatabase, session: Session) -> None:
        msg = Message(session_id=session.id, role="user", content="No tools here")
        db.create_message(msg)
        messages = db.get_messages_for_session(session.id)
        assert messages[0].tool_calls is None

    def test_message_with_model_and_tokens(self, db: ZenkiDatabase, session: Session) -> None:
        msg = Message(
            session_id=session.id,
            role="assistant",
            content="Response",
            model_used="claude-sonnet-4-6",
            tokens_in=150,
            tokens_out=300,
        )
        db.create_message(msg)
        messages = db.get_messages_for_session(session.id)
        assert messages[0].model_used == "claude-sonnet-4-6"
        assert messages[0].tokens_in == 150
        assert messages[0].tokens_out == 300

    def test_message_all_roles(self, db: ZenkiDatabase, session: Session) -> None:
        for role in ("user", "assistant", "system", "tool"):
            msg = Message(session_id=session.id, role=role, content=f"I am {role}")
            db.create_message(msg)

        messages = db.get_messages_for_session(session.id)
        roles = [m.role for m in messages]
        assert "user" in roles
        assert "assistant" in roles
        assert "system" in roles
        assert "tool" in roles

    def test_messages_ordered_by_created_at(self, db: ZenkiDatabase, session: Session) -> None:
        db.create_message(Message(session_id=session.id, role="user", content="first"))
        db.create_message(Message(session_id=session.id, role="assistant", content="second"))
        db.create_message(Message(session_id=session.id, role="user", content="third"))

        messages = db.get_messages_for_session(session.id)
        for i in range(len(messages) - 1):
            assert messages[i].created_at <= messages[i + 1].created_at


# ===========================================================================
# Episodic Memory CRUD
# ===========================================================================


class TestEpisodicMemoryCRUD:
    """Tests for EpisodicMemory create and search operations."""

    def test_create_episodic_memory(self, db: ZenkiDatabase, session: Session) -> None:
        mem = EpisodicMemory(
            session_id=session.id,
            summary="Discussed Python best practices",
            key_topics=["python", "best-practices"],
            key_entities=["user", "assistant"],
            importance=0.8,
        )
        result = db.create_episodic_memory(mem)
        assert result.id == mem.id
        assert result.summary == "Discussed Python best practices"

    def test_search_episodic_by_summary(self, db: ZenkiDatabase, session: Session) -> None:
        db.create_episodic_memory(EpisodicMemory(
            session_id=session.id,
            summary="Discussed Python testing frameworks",
            key_topics=["python", "testing"],
        ))
        db.create_episodic_memory(EpisodicMemory(
            session_id=session.id,
            summary="Talked about database design patterns",
            key_topics=["database", "design"],
        ))

        results = db.search_episodic_memories("Python")
        assert len(results) == 1
        assert "Python" in results[0].summary

    def test_search_episodic_by_topic(self, db: ZenkiDatabase, session: Session) -> None:
        db.create_episodic_memory(EpisodicMemory(
            session_id=session.id,
            summary="General chat",
            key_topics=["machine-learning", "deep-learning"],
        ))

        results = db.search_episodic_memories("machine-learning")
        assert len(results) == 1

    def test_search_episodic_by_entity(self, db: ZenkiDatabase, session: Session) -> None:
        db.create_episodic_memory(EpisodicMemory(
            session_id=session.id,
            summary="Meeting notes",
            key_entities=["ProjectAlpha", "TeamBeta"],
        ))

        results = db.search_episodic_memories("ProjectAlpha")
        assert len(results) == 1

    def test_search_episodic_no_results(self, db: ZenkiDatabase, session: Session) -> None:
        db.create_episodic_memory(EpisodicMemory(
            session_id=session.id,
            summary="Something unrelated",
        ))
        results = db.search_episodic_memories("quantum-physics")
        assert len(results) == 0

    def test_episodic_key_topics_json_roundtrip(self, db: ZenkiDatabase, session: Session) -> None:
        topics = ["topic1", "topic2", "topic3"]
        mem = EpisodicMemory(
            session_id=session.id,
            summary="Test topics",
            key_topics=topics,
        )
        db.create_episodic_memory(mem)

        results = db.search_episodic_memories("Test topics")
        assert results[0].key_topics == topics

    def test_episodic_key_entities_json_roundtrip(
        self, db: ZenkiDatabase, session: Session,
    ) -> None:
        entities = ["EntityA", "EntityB"]
        mem = EpisodicMemory(
            session_id=session.id,
            summary="Test entities",
            key_entities=entities,
        )
        db.create_episodic_memory(mem)

        results = db.search_episodic_memories("Test entities")
        assert results[0].key_entities == entities

    def test_episodic_importance_ordering(self, db: ZenkiDatabase, session: Session) -> None:
        db.create_episodic_memory(EpisodicMemory(
            session_id=session.id,
            summary="Low priority matching item",
            importance=0.2,
        ))
        db.create_episodic_memory(EpisodicMemory(
            session_id=session.id,
            summary="High priority matching item",
            importance=0.9,
        ))

        results = db.search_episodic_memories("priority matching")
        assert len(results) == 2
        assert results[0].importance >= results[1].importance


# ===========================================================================
# Semantic Memory CRUD
# ===========================================================================


class TestSemanticMemoryCRUD:
    """Tests for SemanticMemory create and search operations."""

    def test_create_semantic_memory(self, db: ZenkiDatabase, session: Session) -> None:
        mem = SemanticMemory(
            session_id=session.id,
            content="Python uses indentation for block scoping",
            category="fact",
            source="conversation",
            tags=["python", "syntax"],
            importance=0.7,
        )
        result = db.create_semantic_memory(mem)
        assert result.id == mem.id
        assert result.category == "fact"

    def test_search_semantic_by_content(self, db: ZenkiDatabase, session: Session) -> None:
        db.create_semantic_memory(SemanticMemory(
            session_id=session.id,
            content="User prefers dark mode in all applications",
            category="preference",
        ))
        db.create_semantic_memory(SemanticMemory(
            session_id=session.id,
            content="SQLite supports JSON functions",
            category="fact",
        ))

        results = db.search_semantic_memories("dark mode")
        assert len(results) == 1
        assert "dark mode" in results[0].content

    def test_search_semantic_by_tag(self, db: ZenkiDatabase, session: Session) -> None:
        db.create_semantic_memory(SemanticMemory(
            session_id=session.id,
            content="Some content",
            category="knowledge",
            tags=["rust", "systems-programming"],
        ))

        results = db.search_semantic_memories("rust")
        assert len(results) == 1

    def test_search_semantic_by_category_filter(self, db: ZenkiDatabase, session: Session) -> None:
        db.create_semantic_memory(SemanticMemory(
            session_id=session.id,
            content="Python is great for prototyping",
            category="fact",
            tags=["python"],
        ))
        db.create_semantic_memory(SemanticMemory(
            session_id=session.id,
            content="User likes Python for scripting",
            category="preference",
            tags=["python"],
        ))

        facts = db.search_semantic_memories("Python", category="fact")
        prefs = db.search_semantic_memories("Python", category="preference")
        all_results = db.search_semantic_memories("Python")

        assert len(facts) == 1
        assert facts[0].category == "fact"
        assert len(prefs) == 1
        assert prefs[0].category == "preference"
        assert len(all_results) == 2

    def test_search_semantic_no_results(self, db: ZenkiDatabase, session: Session) -> None:
        db.create_semantic_memory(SemanticMemory(
            session_id=session.id,
            content="Unrelated content here",
            category="fact",
        ))
        results = db.search_semantic_memories("nonexistent-query")
        assert len(results) == 0

    def test_semantic_tags_json_roundtrip(self, db: ZenkiDatabase, session: Session) -> None:
        tags = ["tag1", "tag2", "tag3"]
        mem = SemanticMemory(
            session_id=session.id,
            content="Tagged content",
            category="knowledge",
            tags=tags,
        )
        db.create_semantic_memory(mem)
        results = db.search_semantic_memories("Tagged content")
        assert results[0].tags == tags

    def test_semantic_all_categories(self, db: ZenkiDatabase, session: Session) -> None:
        for cat in ("fact", "preference", "knowledge", "insight", "lesson"):
            db.create_semantic_memory(SemanticMemory(
                session_id=session.id,
                content=f"Content for {cat}",
                category=cat,
            ))

        for cat in ("fact", "preference", "knowledge", "insight", "lesson"):
            results = db.search_semantic_memories(cat, category=cat)
            assert len(results) == 1
            assert results[0].category == cat

    def test_semantic_importance_ordering(self, db: ZenkiDatabase, session: Session) -> None:
        db.create_semantic_memory(SemanticMemory(
            session_id=session.id,
            content="Low importance searchable item",
            category="fact",
            importance=0.1,
        ))
        db.create_semantic_memory(SemanticMemory(
            session_id=session.id,
            content="High importance searchable item",
            category="fact",
            importance=0.95,
        ))

        results = db.search_semantic_memories("importance searchable")
        assert len(results) == 2
        assert results[0].importance >= results[1].importance


# ===========================================================================
# Scheduled Task CRUD
# ===========================================================================


class TestScheduledTaskCRUD:
    """Tests for ScheduledTask create, get, and update operations."""

    def test_create_scheduled_task(self, db: ZenkiDatabase, user: User) -> None:
        task = ScheduledTask(
            user_id=user.id,
            description="Daily standup reminder",
            cron_expression="0 9 * * 1-5",
            task_type="reminder",
        )
        result = db.create_scheduled_task(task)
        assert result.id == task.id
        assert result.task_type == "reminder"

    def test_get_scheduled_tasks_all(self, db: ZenkiDatabase, user: User) -> None:
        db.create_scheduled_task(ScheduledTask(
            user_id=user.id,
            description="Task 1",
            cron_expression="0 9 * * *",
            task_type="reminder",
        ))
        db.create_scheduled_task(ScheduledTask(
            user_id=user.id,
            description="Task 2",
            cron_expression="0 17 * * *",
            task_type="action",
        ))

        tasks = db.get_scheduled_tasks()
        assert len(tasks) == 2

    def test_get_scheduled_tasks_by_user(self, db: ZenkiDatabase) -> None:
        u1 = db.create_user(User(display_name="User1"))
        u2 = db.create_user(User(display_name="User2"))

        db.create_scheduled_task(ScheduledTask(
            user_id=u1.id,
            description="U1 task",
            cron_expression="0 9 * * *",
            task_type="reminder",
        ))
        db.create_scheduled_task(ScheduledTask(
            user_id=u2.id,
            description="U2 task",
            cron_expression="0 10 * * *",
            task_type="action",
        ))

        u1_tasks = db.get_scheduled_tasks(user_id=u1.id)
        u2_tasks = db.get_scheduled_tasks(user_id=u2.id)
        assert len(u1_tasks) == 1
        assert u1_tasks[0].description == "U1 task"
        assert len(u2_tasks) == 1

    def test_get_scheduled_tasks_enabled_only(self, db: ZenkiDatabase, user: User) -> None:
        db.create_scheduled_task(ScheduledTask(
            user_id=user.id,
            description="Enabled task",
            cron_expression="0 9 * * *",
            task_type="reminder",
            enabled=True,
        ))
        db.create_scheduled_task(ScheduledTask(
            user_id=user.id,
            description="Disabled task",
            cron_expression="0 10 * * *",
            task_type="action",
            enabled=False,
        ))

        enabled = db.get_scheduled_tasks(enabled_only=True)
        all_tasks = db.get_scheduled_tasks()
        assert len(enabled) == 1
        assert enabled[0].description == "Enabled task"
        assert len(all_tasks) == 2

    def test_update_scheduled_task(self, db: ZenkiDatabase, user: User) -> None:
        task = ScheduledTask(
            user_id=user.id,
            description="Original description",
            cron_expression="0 9 * * *",
            task_type="reminder",
        )
        db.create_scheduled_task(task)

        task.description = "Updated description"
        task.last_result = "Success"
        task.enabled = False
        db.update_scheduled_task(task)

        tasks = db.get_scheduled_tasks()
        updated = [t for t in tasks if t.id == task.id][0]
        assert updated.description == "Updated description"
        assert updated.last_result == "Success"
        assert updated.enabled is False

    def test_task_config_json_roundtrip(self, db: ZenkiDatabase, user: User) -> None:
        config = {"channel": "#general", "message": "Hello team!", "mention": True}
        task = ScheduledTask(
            user_id=user.id,
            description="Configured task",
            cron_expression="0 9 * * *",
            task_type="action",
            task_config=config,
        )
        db.create_scheduled_task(task)

        tasks = db.get_scheduled_tasks()
        assert tasks[0].task_config == config
        assert tasks[0].task_config["mention"] is True

    def test_task_all_types(self, db: ZenkiDatabase, user: User) -> None:
        for task_type in ("reminder", "action", "self_improve", "monitor"):
            db.create_scheduled_task(ScheduledTask(
                user_id=user.id,
                description=f"Task of type {task_type}",
                cron_expression="0 9 * * *",
                task_type=task_type,
            ))

        tasks = db.get_scheduled_tasks()
        types = {t.task_type for t in tasks}
        assert types == {"reminder", "action", "self_improve", "monitor"}

    def test_task_enabled_bool_roundtrip(self, db: ZenkiDatabase, user: User) -> None:
        """Ensure boolean enabled is stored as int and restored as bool."""
        task_true = ScheduledTask(
            user_id=user.id,
            description="Enabled",
            cron_expression="* * * * *",
            task_type="reminder",
            enabled=True,
        )
        task_false = ScheduledTask(
            user_id=user.id,
            description="Disabled",
            cron_expression="* * * * *",
            task_type="reminder",
            enabled=False,
        )
        db.create_scheduled_task(task_true)
        db.create_scheduled_task(task_false)

        tasks = db.get_scheduled_tasks()
        by_desc = {t.description: t for t in tasks}
        assert by_desc["Enabled"].enabled is True
        assert by_desc["Disabled"].enabled is False


# ===========================================================================
# Skill CRUD
# ===========================================================================


class TestSkillCRUD:
    """Tests for Skill create, read, update, and list operations."""

    def test_create_skill(self, db: ZenkiDatabase) -> None:
        skill = Skill(
            name="web_search",
            skill_type="core",
            status="approved",
            path="/skills/web_search",
            description="Search the web",
        )
        result = db.create_skill(skill)
        assert result.id == skill.id
        assert result.name == "web_search"

    def test_get_skill_by_id(self, db: ZenkiDatabase) -> None:
        skill = Skill(
            name="file_read",
            skill_type="core",
            path="/skills/file_read",
        )
        db.create_skill(skill)

        fetched = db.get_skill(skill.id)
        assert fetched is not None
        assert fetched.name == "file_read"
        assert fetched.skill_type == "core"

    def test_get_skill_not_found(self, db: ZenkiDatabase) -> None:
        assert db.get_skill("nonexistent") is None

    def test_update_skill(self, db: ZenkiDatabase) -> None:
        skill = Skill(
            name="code_gen",
            skill_type="learned",
            path="/skills/code_gen",
            status="pending",
        )
        db.create_skill(skill)

        skill.status = "approved"
        skill.usage_count = 5
        skill.approved_at = datetime.now(UTC)
        db.update_skill(skill)

        fetched = db.get_skill(skill.id)
        assert fetched is not None
        assert fetched.status == "approved"
        assert fetched.usage_count == 5
        assert fetched.approved_at is not None

    def test_list_skills_all(self, db: ZenkiDatabase) -> None:
        db.create_skill(Skill(name="skill_a", skill_type="core", path="/a"))
        db.create_skill(Skill(name="skill_b", skill_type="learned", path="/b"))
        db.create_skill(Skill(name="skill_c", skill_type="core", path="/c"))

        skills = db.list_skills()
        assert len(skills) == 3

    def test_list_skills_by_type(self, db: ZenkiDatabase) -> None:
        db.create_skill(Skill(name="core1", skill_type="core", path="/c1"))
        db.create_skill(Skill(name="core2", skill_type="core", path="/c2"))
        db.create_skill(Skill(name="learned1", skill_type="learned", path="/l1"))

        core = db.list_skills(skill_type="core")
        learned = db.list_skills(skill_type="learned")
        assert len(core) == 2
        assert len(learned) == 1

    def test_list_skills_by_status(self, db: ZenkiDatabase) -> None:
        db.create_skill(Skill(name="approved1", skill_type="core", path="/a", status="approved"))
        db.create_skill(Skill(name="pending1", skill_type="learned", path="/p", status="pending"))
        db.create_skill(Skill(name="rejected1", skill_type="learned", path="/r", status="rejected"))

        approved = db.list_skills(status="approved")
        pending = db.list_skills(status="pending")
        assert len(approved) == 1
        assert len(pending) == 1

    def test_list_skills_by_type_and_status(self, db: ZenkiDatabase) -> None:
        db.create_skill(Skill(name="s1", skill_type="core", path="/1", status="approved"))
        db.create_skill(Skill(name="s2", skill_type="core", path="/2", status="pending"))
        db.create_skill(Skill(name="s3", skill_type="learned", path="/3", status="approved"))

        result = db.list_skills(skill_type="core", status="approved")
        assert len(result) == 1
        assert result[0].name == "s1"

    def test_skill_metadata_json_roundtrip(self, db: ZenkiDatabase) -> None:
        meta = {"version": "1.0", "author": "zenki", "tags": ["utility"]}
        skill = Skill(
            name="meta_skill",
            skill_type="learned",
            path="/skills/meta",
            metadata=meta,
        )
        db.create_skill(skill)

        fetched = db.get_skill(skill.id)
        assert fetched is not None
        assert fetched.metadata == meta
        assert fetched.metadata["tags"] == ["utility"]

    def test_skill_unique_name_constraint(self, db: ZenkiDatabase) -> None:
        db.create_skill(Skill(name="unique_skill", skill_type="core", path="/u1"))
        with pytest.raises(sqlite3.IntegrityError):
            db.create_skill(Skill(name="unique_skill", skill_type="learned", path="/u2"))

    def test_list_skills_ordered_by_name(self, db: ZenkiDatabase) -> None:
        db.create_skill(Skill(name="zebra", skill_type="core", path="/z"))
        db.create_skill(Skill(name="alpha", skill_type="core", path="/a"))
        db.create_skill(Skill(name="middle", skill_type="core", path="/m"))

        skills = db.list_skills()
        names = [s.name for s in skills]
        assert names == ["alpha", "middle", "zebra"]


# ===========================================================================
# Foreign key constraints
# ===========================================================================


class TestForeignKeyConstraints:
    """Tests verifying foreign key relationships are enforced."""

    def test_session_requires_valid_user(self, db: ZenkiDatabase) -> None:
        """Creating a session with a non-existent user_id should fail."""
        s = Session(user_id="nonexistent-user-id", channel_type="cli")
        with pytest.raises(sqlite3.IntegrityError):
            db.create_session(s)

    def test_message_requires_valid_session(self, db: ZenkiDatabase) -> None:
        """Creating a message with a non-existent session_id should fail."""
        msg = Message(session_id="nonexistent-session-id", role="user", content="Hello")
        with pytest.raises(sqlite3.IntegrityError):
            db.create_message(msg)

    def test_scheduled_task_requires_valid_user(self, db: ZenkiDatabase) -> None:
        """Creating a scheduled task with a non-existent user_id should fail."""
        task = ScheduledTask(
            user_id="nonexistent-user-id",
            description="Bad task",
            cron_expression="0 9 * * *",
            task_type="reminder",
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.create_scheduled_task(task)

    def test_episodic_memory_with_valid_session(self, db: ZenkiDatabase, session: Session) -> None:
        """Episodic memory should accept a valid session_id."""
        mem = EpisodicMemory(session_id=session.id, summary="Valid session link")
        result = db.create_episodic_memory(mem)
        assert result.session_id == session.id

    def test_episodic_memory_with_null_session(self, db: ZenkiDatabase) -> None:
        """Episodic memory should accept None as session_id."""
        mem = EpisodicMemory(session_id=None, summary="No session link")
        result = db.create_episodic_memory(mem)
        assert result.session_id is None

    def test_semantic_memory_with_valid_session(self, db: ZenkiDatabase, session: Session) -> None:
        """Semantic memory should accept a valid session_id."""
        mem = SemanticMemory(
            session_id=session.id,
            content="Valid session link",
            category="fact",
        )
        result = db.create_semantic_memory(mem)
        assert result.session_id == session.id


# ===========================================================================
# JSON field serialization edge cases
# ===========================================================================


class TestJsonSerialization:
    """Tests for edge cases in JSON field roundtrips."""

    def test_nested_json_in_user_config(self, db: ZenkiDatabase) -> None:
        config = {
            "preferences": {
                "colors": ["red", "blue"],
                "settings": {"volume": 80, "muted": False},
            },
            "count": 42,
        }
        u = User(display_name="Nested", config=config)
        db.create_user(u)
        fetched = db.get_user(u.id)
        assert fetched is not None
        assert fetched.config == config
        assert fetched.config["preferences"]["colors"][1] == "blue"
        assert fetched.config["preferences"]["settings"]["muted"] is False

    def test_unicode_in_json_fields(self, db: ZenkiDatabase) -> None:
        config = {"name": "Zenki", "lang": "ja"}
        u = User(display_name="Unicode", config=config)
        db.create_user(u)
        fetched = db.get_user(u.id)
        assert fetched is not None
        assert fetched.config["lang"] == "ja"

    def test_empty_list_json_roundtrip(self, db: ZenkiDatabase, session: Session) -> None:
        mem = EpisodicMemory(
            session_id=session.id,
            summary="Empty lists",
            key_topics=[],
            key_entities=[],
        )
        db.create_episodic_memory(mem)
        results = db.search_episodic_memories("Empty lists")
        assert results[0].key_topics == []
        assert results[0].key_entities == []

    def test_large_json_field(self, db: ZenkiDatabase) -> None:
        """Store and retrieve a large config dictionary."""
        config = {f"key_{i}": f"value_{i}" for i in range(100)}
        u = User(display_name="LargeConfig", config=config)
        db.create_user(u)
        fetched = db.get_user(u.id)
        assert fetched is not None
        assert len(fetched.config) == 100
        assert fetched.config["key_50"] == "value_50"

    def test_special_chars_in_json(self, db: ZenkiDatabase) -> None:
        config = {"sql_test": "'; DROP TABLE users; --", "newlines": "line1\nline2"}
        u = User(display_name="SpecialChars", config=config)
        db.create_user(u)
        fetched = db.get_user(u.id)
        assert fetched is not None
        assert fetched.config["sql_test"] == "'; DROP TABLE users; --"
        assert fetched.config["newlines"] == "line1\nline2"


# ===========================================================================
# Model validation (Pydantic)
# ===========================================================================


class TestModelValidation:
    """Tests for Pydantic model validation on the data models."""

    def test_user_from_attributes(self) -> None:
        assert User.model_config.get("from_attributes") is True

    def test_session_from_attributes(self) -> None:
        assert Session.model_config.get("from_attributes") is True

    def test_message_invalid_role_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Message(session_id="s1", role="invalid_role", content="hello")

    def test_semantic_memory_invalid_category_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SemanticMemory(content="test", category="invalid_category")

    def test_skill_invalid_type_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Skill(name="bad", skill_type="invalid", path="/bad")

    def test_skill_invalid_status_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Skill(name="bad", skill_type="core", path="/bad", status="invalid")

    def test_scheduled_task_invalid_type_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ScheduledTask(
                user_id="u1",
                description="bad",
                cron_expression="* * * * *",
                task_type="invalid_type",
            )

    def test_user_default_id_is_uuid(self) -> None:
        u = User()
        # UUID4 format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
        parts = u.id.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8

    def test_user_default_created_at_is_utc(self) -> None:
        u = User()
        assert u.created_at.tzinfo is not None


# ===========================================================================
# Datetime roundtrip
# ===========================================================================


class TestDatetimeRoundtrip:
    """Tests that datetime values survive storage and retrieval correctly."""

    def test_user_created_at_roundtrip(self, db: ZenkiDatabase) -> None:
        u = User(display_name="TimestampTest")
        db.create_user(u)
        fetched = db.get_user(u.id)
        assert fetched is not None
        # Allow for microsecond differences due to ISO format
        assert abs((fetched.created_at - u.created_at).total_seconds()) < 1

    def test_session_datetime_fields_roundtrip(self, db: ZenkiDatabase, user: User) -> None:
        now = datetime.now(UTC)
        s = Session(
            user_id=user.id,
            channel_type="cli",
            started_at=now,
            last_active=now,
        )
        db.create_session(s)
        fetched = db.get_session(s.id)
        assert fetched is not None
        assert fetched.started_at is not None
        assert fetched.last_active is not None

    def test_message_created_at_roundtrip(self, db: ZenkiDatabase, session: Session) -> None:
        msg = Message(session_id=session.id, role="user", content="test")
        db.create_message(msg)
        messages = db.get_messages_for_session(session.id)
        assert len(messages) == 1
        assert messages[0].created_at is not None
        assert isinstance(messages[0].created_at, datetime)
