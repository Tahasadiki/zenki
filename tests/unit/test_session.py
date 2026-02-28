"""Tests for the SessionManager (zenki.core.session)."""

from __future__ import annotations

from pathlib import Path

import pytest

from zenki.core.session import SessionManager
from zenki.db.database import ZenkiDatabase
from zenki.db.models import User

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> ZenkiDatabase:
    """Return an initialised ZenkiDatabase backed by a temp file."""
    database = ZenkiDatabase(tmp_path / "test_session.db")
    database.initialize()
    return database


@pytest.fixture()
def sm(db: ZenkiDatabase) -> SessionManager:
    """Return a SessionManager wired to the test database."""
    return SessionManager(db)


@pytest.fixture()
def user(db: ZenkiDatabase) -> User:
    """Create and return a persisted test user."""
    u = User(display_name="Alice")
    return db.create_user(u)


# ===========================================================================
# Session creation
# ===========================================================================


class TestCreateSession:
    """Tests for SessionManager.create_session."""

    def test_create_session_returns_session(self, sm: SessionManager, user: User) -> None:
        session = sm.create_session(user_id=user.id, channel_type="cli")
        assert session.user_id == user.id
        assert session.channel_type == "cli"
        assert session.ended_at is None

    def test_create_session_with_optional_fields(self, sm: SessionManager, user: User) -> None:
        session = sm.create_session(
            user_id=user.id,
            channel_type="slack",
            channel_id="C123",
            thread_id="T456",
            context_summary="Test context",
        )
        assert session.channel_id == "C123"
        assert session.thread_id == "T456"
        assert session.context_summary == "Test context"

    def test_create_session_auto_creates_user(self, sm: SessionManager, db: ZenkiDatabase) -> None:
        """If the user does not exist, create_session should create them."""
        session = sm.create_session(user_id="brand-new-user", channel_type="cli")
        assert session.user_id == "brand-new-user"
        # Verify user was actually created in the database.
        created_user = db.get_user("brand-new-user")
        assert created_user is not None

    def test_create_session_has_unique_id(self, sm: SessionManager, user: User) -> None:
        s1 = sm.create_session(user_id=user.id, channel_type="cli")
        s2 = sm.create_session(user_id=user.id, channel_type="cli")
        assert s1.id != s2.id


# ===========================================================================
# Session retrieval
# ===========================================================================


class TestGetSession:
    """Tests for SessionManager.get_session."""

    def test_get_existing_session(self, sm: SessionManager, user: User) -> None:
        created = sm.create_session(user_id=user.id, channel_type="cli")
        fetched = sm.get_session(created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.channel_type == "cli"

    def test_get_nonexistent_session(self, sm: SessionManager) -> None:
        result = sm.get_session("does-not-exist")
        assert result is None


# ===========================================================================
# get_or_create_session
# ===========================================================================


class TestGetOrCreateSession:
    """Tests for SessionManager.get_or_create_session."""

    def test_creates_new_when_none_exists(self, sm: SessionManager, user: User) -> None:
        session = sm.get_or_create_session(
            user_id=user.id,
            channel_type="cli",
        )
        assert session is not None
        assert session.user_id == user.id
        assert session.channel_type == "cli"
        assert session.ended_at is None

    def test_finds_existing_active_session(self, sm: SessionManager, user: User) -> None:
        original = sm.create_session(
            user_id=user.id,
            channel_type="slack",
            channel_id="C100",
            thread_id="T200",
        )
        found = sm.get_or_create_session(
            user_id=user.id,
            channel_type="slack",
            channel_id="C100",
            thread_id="T200",
        )
        assert found.id == original.id

    def test_does_not_find_closed_session(self, sm: SessionManager, user: User) -> None:
        old = sm.create_session(
            user_id=user.id,
            channel_type="cli",
        )
        sm.close_session(old.id)

        new = sm.get_or_create_session(
            user_id=user.id,
            channel_type="cli",
        )
        assert new.id != old.id

    def test_different_channel_creates_new(self, sm: SessionManager, user: User) -> None:
        s1 = sm.create_session(user_id=user.id, channel_type="cli")
        s2 = sm.get_or_create_session(
            user_id=user.id,
            channel_type="slack",
            channel_id="C999",
        )
        assert s2.id != s1.id

    def test_different_thread_creates_new(self, sm: SessionManager, user: User) -> None:
        s1 = sm.create_session(
            user_id=user.id,
            channel_type="slack",
            channel_id="C1",
            thread_id="T1",
        )
        s2 = sm.get_or_create_session(
            user_id=user.id,
            channel_type="slack",
            channel_id="C1",
            thread_id="T2",
        )
        assert s2.id != s1.id


# ===========================================================================
# Messages
# ===========================================================================


class TestAddAndGetMessages:
    """Tests for SessionManager.add_message and get_messages."""

    def test_add_message(self, sm: SessionManager, user: User) -> None:
        session = sm.create_session(user_id=user.id, channel_type="cli")
        msg = sm.add_message(session.id, role="user", content="Hello!")
        assert msg.session_id == session.id
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_get_messages_returns_all(self, sm: SessionManager, user: User) -> None:
        session = sm.create_session(user_id=user.id, channel_type="cli")
        sm.add_message(session.id, role="user", content="Hello")
        sm.add_message(session.id, role="assistant", content="Hi there!")
        sm.add_message(session.id, role="user", content="How are you?")

        messages = sm.get_messages(session.id)
        assert len(messages) == 3
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[2].content == "How are you?"

    def test_get_messages_respects_limit(self, sm: SessionManager, user: User) -> None:
        session = sm.create_session(user_id=user.id, channel_type="cli")
        for i in range(10):
            sm.add_message(session.id, role="user", content=f"Message {i}")

        messages = sm.get_messages(session.id, limit=3)
        assert len(messages) == 3
        # Should be the last 3 messages.
        assert messages[0].content == "Message 7"
        assert messages[2].content == "Message 9"

    def test_add_message_with_model_and_tokens(self, sm: SessionManager, user: User) -> None:
        session = sm.create_session(user_id=user.id, channel_type="cli")
        msg = sm.add_message(
            session.id,
            role="assistant",
            content="Response",
            model_used="claude-sonnet-4-6",
            tokens_in=100,
            tokens_out=200,
        )
        assert msg.model_used == "claude-sonnet-4-6"
        assert msg.tokens_in == 100
        assert msg.tokens_out == 200

    def test_add_message_with_tool_calls(self, sm: SessionManager, user: User) -> None:
        session = sm.create_session(user_id=user.id, channel_type="cli")
        tools = [{"name": "search", "args": {"q": "test"}}]
        msg = sm.add_message(
            session.id,
            role="assistant",
            content="Searching...",
            tool_calls=tools,
        )
        assert msg.tool_calls == tools

    def test_add_message_updates_last_active(
        self, sm: SessionManager, user: User, db: ZenkiDatabase
    ) -> None:
        session = sm.create_session(user_id=user.id, channel_type="cli")
        original_last_active = session.last_active
        sm.add_message(session.id, role="user", content="ping")

        updated = db.get_session(session.id)
        assert updated is not None
        assert updated.last_active is not None
        if original_last_active is not None:
            assert updated.last_active >= original_last_active

    def test_get_messages_empty_session(self, sm: SessionManager, user: User) -> None:
        session = sm.create_session(user_id=user.id, channel_type="cli")
        messages = sm.get_messages(session.id)
        assert messages == []


# ===========================================================================
# Close session
# ===========================================================================


class TestCloseSession:
    """Tests for SessionManager.close_session."""

    def test_close_sets_ended_at(self, sm: SessionManager, user: User) -> None:
        session = sm.create_session(user_id=user.id, channel_type="cli")
        assert session.ended_at is None

        closed = sm.close_session(session.id)
        assert closed.ended_at is not None

    def test_close_with_summary(self, sm: SessionManager, user: User) -> None:
        session = sm.create_session(user_id=user.id, channel_type="cli")
        closed = sm.close_session(session.id, summary="Discussed testing")
        assert closed.summary == "Discussed testing"

    def test_close_nonexistent_raises(self, sm: SessionManager) -> None:
        with pytest.raises(ValueError, match="Session not found"):
            sm.close_session("nonexistent-id")


# ===========================================================================
# List active sessions
# ===========================================================================


class TestListActiveSessions:
    """Tests for SessionManager.list_active_sessions."""

    def test_list_active_returns_open_sessions(self, sm: SessionManager, user: User) -> None:
        s1 = sm.create_session(user_id=user.id, channel_type="cli")
        s2 = sm.create_session(user_id=user.id, channel_type="slack")
        sm.close_session(s1.id)

        active = sm.list_active_sessions()
        assert len(active) == 1
        assert active[0].id == s2.id

    def test_list_active_filters_by_user(self, sm: SessionManager, db: ZenkiDatabase) -> None:
        u1 = db.create_user(User(display_name="User1"))
        u2 = db.create_user(User(display_name="User2"))
        sm.create_session(user_id=u1.id, channel_type="cli")
        sm.create_session(user_id=u1.id, channel_type="slack")
        sm.create_session(user_id=u2.id, channel_type="cli")

        u1_active = sm.list_active_sessions(user_id=u1.id)
        u2_active = sm.list_active_sessions(user_id=u2.id)
        assert len(u1_active) == 2
        assert len(u2_active) == 1

    def test_list_active_empty(self, sm: SessionManager, user: User) -> None:
        s = sm.create_session(user_id=user.id, channel_type="cli")
        sm.close_session(s.id)

        active = sm.list_active_sessions()
        assert active == []

    def test_list_active_no_user_filter(self, sm: SessionManager, db: ZenkiDatabase) -> None:
        u1 = db.create_user(User(display_name="A"))
        u2 = db.create_user(User(display_name="B"))
        sm.create_session(user_id=u1.id, channel_type="cli")
        sm.create_session(user_id=u2.id, channel_type="cli")

        all_active = sm.list_active_sessions()
        assert len(all_active) == 2
