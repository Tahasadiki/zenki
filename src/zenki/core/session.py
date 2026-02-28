"""Session lifecycle management for Zenki.

Provides a high-level SessionManager that wraps the database layer
to create, retrieve, and manage conversation sessions and their messages.
"""

from __future__ import annotations

from datetime import UTC, datetime

from zenki.db.database import ZenkiDatabase
from zenki.db.models import Message, Session, User


class SessionManager:
    """Manages the lifecycle of conversation sessions.

    Wraps :class:`~zenki.db.database.ZenkiDatabase` to provide
    higher-level session operations such as get-or-create semantics,
    message recording, and active session listing.
    """

    def __init__(self, db: ZenkiDatabase) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def _ensure_user(self, user_id: str) -> User:
        """Return the user for *user_id*, creating one if it does not exist."""
        user = self._db.get_user(user_id)
        if user is None:
            user = self._db.create_user(User(id=user_id, display_name=user_id))
        return user

    def create_session(
        self,
        user_id: str,
        channel_type: str,
        channel_id: str | None = None,
        thread_id: str | None = None,
        context_summary: str | None = None,
    ) -> Session:
        """Create a new conversation session.

        Ensures the user exists before creating the session.

        Returns the newly created :class:`Session`.
        """
        self._ensure_user(user_id)
        session = Session(
            user_id=user_id,
            channel_type=channel_type,
            channel_id=channel_id,
            thread_id=thread_id,
            context_summary=context_summary,
        )
        return self._db.create_session(session)

    def get_session(self, session_id: str) -> Session | None:
        """Fetch a session by its ID, or return ``None``."""
        return self._db.get_session(session_id)

    def get_or_create_session(
        self,
        user_id: str,
        channel_type: str,
        channel_id: str | None = None,
        thread_id: str | None = None,
    ) -> Session:
        """Find an existing active session for the channel/thread, or create a new one.

        An *active* session is one whose ``ended_at`` field is ``None``.
        The match is performed on ``user_id``, ``channel_type``,
        ``channel_id``, and ``thread_id``.
        """
        # Search existing sessions for a match.
        sessions = self._db.list_sessions(user_id=user_id)
        for session in sessions:
            if session.ended_at is not None:
                continue
            if (
                session.channel_type == channel_type
                and session.channel_id == channel_id
                and session.thread_id == thread_id
            ):
                return session

        # No matching active session found — create a new one.
        return self.create_session(
            user_id=user_id,
            channel_type=channel_type,
            channel_id=channel_id,
            thread_id=thread_id,
        )

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model_used: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        tool_calls: list | None = None,
    ) -> Message:
        """Record a message in the given session.

        Also updates the session's ``last_active`` timestamp.
        """
        message = Message(
            session_id=session_id,
            role=role,
            content=content,
            model_used=model_used,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tool_calls=tool_calls,
        )
        result = self._db.create_message(message)

        # Update last_active on the session.
        session = self._db.get_session(session_id)
        if session is not None:
            session.last_active = datetime.now(UTC)
            self._db.update_session(session)

        return result

    def get_messages(self, session_id: str, limit: int = 50) -> list[Message]:
        """Retrieve messages for a session, most recent up to *limit*.

        Messages are returned in chronological order (oldest first).
        """
        all_messages = self._db.get_messages_for_session(session_id)
        # Return the last *limit* messages, preserving chronological order.
        return all_messages[-limit:]

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def close_session(self, session_id: str, summary: str | None = None) -> Session:
        """Close a session by setting its ``ended_at`` timestamp.

        Optionally attach a *summary* describing the session.

        Returns the updated :class:`Session`.

        Raises :class:`ValueError` if the session does not exist.
        """
        session = self._db.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        session.ended_at = datetime.now(UTC)
        if summary is not None:
            session.summary = summary
        return self._db.update_session(session)

    def list_active_sessions(self, user_id: str | None = None) -> list[Session]:
        """List sessions that have not been closed (``ended_at is None``).

        Optionally filtered by *user_id*.
        """
        sessions = self._db.list_sessions(user_id=user_id)
        return [s for s in sessions if s.ended_at is None]
