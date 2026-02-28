"""Database management for Zenki using SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from zenki.db.models import (
    EpisodicMemory,
    Message,
    ScheduledTask,
    SemanticMemory,
    Session,
    Skill,
    User,
)


def _dt_to_str(dt: datetime | None) -> str | None:
    """Convert a datetime to an ISO 8601 string for storage."""
    if dt is None:
        return None
    return dt.isoformat()


def _str_to_dt(s: str | None) -> datetime | None:
    """Parse an ISO 8601 string back to a datetime."""
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


class ZenkiDatabase:
    """SQLite-backed database for Zenki.

    Uses WAL mode for concurrent reads and parameterized queries
    to prevent SQL injection.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        """Return the active connection, opening one if needed."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create all tables if they do not already exist."""
        c = self.conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                display_name TEXT,
                created_at TEXT NOT NULL,
                config TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id),
                channel_type TEXT NOT NULL,
                channel_id TEXT,
                thread_id TEXT,
                context_summary TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                last_active TEXT,
                summary TEXT,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model_used TEXT,
                tokens_in INTEGER,
                tokens_out INTEGER,
                tool_calls TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, created_at);

            CREATE TABLE IF NOT EXISTS episodic_memories (
                id TEXT PRIMARY KEY,
                session_id TEXT REFERENCES sessions(id),
                summary TEXT NOT NULL,
                key_topics TEXT NOT NULL DEFAULT '[]',
                key_entities TEXT NOT NULL DEFAULT '[]',
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_episodic_importance
                ON episodic_memories(importance DESC);

            CREATE TABLE IF NOT EXISTS semantic_memories (
                id TEXT PRIMARY KEY,
                session_id TEXT REFERENCES sessions(id),
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                source TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_semantic_category
                ON semantic_memories(category);
            CREATE INDEX IF NOT EXISTS idx_semantic_importance
                ON semantic_memories(importance DESC);

            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id),
                description TEXT NOT NULL,
                original_request TEXT,
                cron_expression TEXT NOT NULL,
                next_run_at TEXT,
                last_run_at TEXT,
                last_result TEXT,
                task_type TEXT NOT NULL,
                task_config TEXT NOT NULL DEFAULT '{}',
                notify_channel TEXT,
                enabled INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_next_run
                ON scheduled_tasks(next_run_at) WHERE enabled = 1;

            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                skill_type TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                path TEXT NOT NULL,
                description TEXT,
                generated_from TEXT,
                usage_count INTEGER DEFAULT 0,
                last_used_at TEXT,
                approved_at TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_skills_status
                ON skills(status);
            """
        )
        c.commit()

    # ==================================================================
    # Users
    # ==================================================================

    def create_user(self, user: User) -> User:
        """Insert a new user row and return the model."""
        self.conn.execute(
            "INSERT INTO users (id, display_name, created_at, config) VALUES (?, ?, ?, ?)",
            (
                user.id,
                user.display_name,
                _dt_to_str(user.created_at),
                json.dumps(user.config),
            ),
        )
        self.conn.commit()
        return user

    def get_user(self, user_id: str) -> User | None:
        """Fetch a user by ID, or return None."""
        row = self.conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            return None
        return User(
            id=row["id"],
            display_name=row["display_name"],
            created_at=_str_to_dt(row["created_at"]),  # type: ignore[arg-type]
            config=json.loads(row["config"]),
        )

    # ==================================================================
    # Sessions
    # ==================================================================

    def create_session(self, session: Session) -> Session:
        """Insert a new session row."""
        self.conn.execute(
            """INSERT INTO sessions
               (id, user_id, channel_type, channel_id, thread_id,
                context_summary, started_at, ended_at, last_active, summary, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session.id,
                session.user_id,
                session.channel_type,
                session.channel_id,
                session.thread_id,
                session.context_summary,
                _dt_to_str(session.started_at),
                _dt_to_str(session.ended_at),
                _dt_to_str(session.last_active),
                session.summary,
                json.dumps(session.metadata),
            ),
        )
        self.conn.commit()
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Fetch a session by ID."""
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def update_session(self, session: Session) -> Session:
        """Update an existing session row."""
        self.conn.execute(
            """UPDATE sessions SET
                user_id = ?, channel_type = ?, channel_id = ?, thread_id = ?,
                context_summary = ?, started_at = ?, ended_at = ?,
                last_active = ?, summary = ?, metadata = ?
               WHERE id = ?""",
            (
                session.user_id,
                session.channel_type,
                session.channel_id,
                session.thread_id,
                session.context_summary,
                _dt_to_str(session.started_at),
                _dt_to_str(session.ended_at),
                _dt_to_str(session.last_active),
                session.summary,
                json.dumps(session.metadata),
                session.id,
            ),
        )
        self.conn.commit()
        return session

    def list_sessions(self, user_id: str | None = None) -> list[Session]:
        """List sessions, optionally filtered by user_id."""
        if user_id is not None:
            rows = self.conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY started_at DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC"
            ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"],
            user_id=row["user_id"],
            channel_type=row["channel_type"],
            channel_id=row["channel_id"],
            thread_id=row["thread_id"],
            context_summary=row["context_summary"],
            started_at=_str_to_dt(row["started_at"]),  # type: ignore[arg-type]
            ended_at=_str_to_dt(row["ended_at"]),
            last_active=_str_to_dt(row["last_active"]),
            summary=row["summary"],
            metadata=json.loads(row["metadata"]),
        )

    # ==================================================================
    # Messages
    # ==================================================================

    def create_message(self, message: Message) -> Message:
        """Insert a new message row."""
        self.conn.execute(
            """INSERT INTO messages
               (id, session_id, role, content, model_used, tokens_in, tokens_out,
                tool_calls, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                message.id,
                message.session_id,
                message.role,
                message.content,
                message.model_used,
                message.tokens_in,
                message.tokens_out,
                json.dumps(message.tool_calls) if message.tool_calls is not None else None,
                _dt_to_str(message.created_at),
            ),
        )
        self.conn.commit()
        return message

    def get_messages_for_session(self, session_id: str) -> list[Message]:
        """Get all messages for a session, ordered by creation time."""
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        tool_calls_raw = row["tool_calls"]
        return Message(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            model_used=row["model_used"],
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            tool_calls=json.loads(tool_calls_raw) if tool_calls_raw is not None else None,
            created_at=_str_to_dt(row["created_at"]),  # type: ignore[arg-type]
        )

    # ==================================================================
    # Episodic Memories
    # ==================================================================

    def create_episodic_memory(self, memory: EpisodicMemory) -> EpisodicMemory:
        """Insert a new episodic memory row."""
        self.conn.execute(
            """INSERT INTO episodic_memories
               (id, session_id, summary, key_topics, key_entities,
                importance, access_count, last_accessed, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory.id,
                memory.session_id,
                memory.summary,
                json.dumps(memory.key_topics),
                json.dumps(memory.key_entities),
                memory.importance,
                memory.access_count,
                _dt_to_str(memory.last_accessed),
                _dt_to_str(memory.created_at),
            ),
        )
        self.conn.commit()
        return memory

    def search_episodic_memories(self, query: str) -> list[EpisodicMemory]:
        """Search episodic memories by text (summary, topics, entities).

        This is a simple LIKE-based search. Vector-based retrieval will be added later.
        """
        pattern = f"%{query}%"
        rows = self.conn.execute(
            """SELECT * FROM episodic_memories
               WHERE summary LIKE ?
                  OR key_topics LIKE ?
                  OR key_entities LIKE ?
               ORDER BY importance DESC, created_at DESC""",
            (pattern, pattern, pattern),
        ).fetchall()
        return [self._row_to_episodic(r) for r in rows]

    def _row_to_episodic(self, row: sqlite3.Row) -> EpisodicMemory:
        return EpisodicMemory(
            id=row["id"],
            session_id=row["session_id"],
            summary=row["summary"],
            key_topics=json.loads(row["key_topics"]),
            key_entities=json.loads(row["key_entities"]),
            importance=row["importance"],
            access_count=row["access_count"],
            last_accessed=_str_to_dt(row["last_accessed"]),
            created_at=_str_to_dt(row["created_at"]),  # type: ignore[arg-type]
        )

    # ==================================================================
    # Semantic Memories
    # ==================================================================

    def create_semantic_memory(self, memory: SemanticMemory) -> SemanticMemory:
        """Insert a new semantic memory row."""
        self.conn.execute(
            """INSERT INTO semantic_memories
               (id, session_id, content, category, source, tags,
                importance, access_count, last_accessed, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory.id,
                memory.session_id,
                memory.content,
                memory.category,
                memory.source,
                json.dumps(memory.tags),
                memory.importance,
                memory.access_count,
                _dt_to_str(memory.last_accessed),
                _dt_to_str(memory.created_at),
                _dt_to_str(memory.updated_at),
            ),
        )
        self.conn.commit()
        return memory

    def search_semantic_memories(
        self,
        query: str,
        category: str | None = None,
    ) -> list[SemanticMemory]:
        """Search semantic memories by text, optionally filtered by category.

        This is a simple LIKE-based search. Vector-based retrieval will be added later.
        """
        pattern = f"%{query}%"
        if category is not None:
            rows = self.conn.execute(
                """SELECT * FROM semantic_memories
                   WHERE (content LIKE ? OR tags LIKE ?)
                     AND category = ?
                   ORDER BY importance DESC, created_at DESC""",
                (pattern, pattern, category),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM semantic_memories
                   WHERE content LIKE ? OR tags LIKE ?
                   ORDER BY importance DESC, created_at DESC""",
                (pattern, pattern),
            ).fetchall()
        return [self._row_to_semantic(r) for r in rows]

    def _row_to_semantic(self, row: sqlite3.Row) -> SemanticMemory:
        return SemanticMemory(
            id=row["id"],
            session_id=row["session_id"],
            content=row["content"],
            category=row["category"],
            source=row["source"],
            tags=json.loads(row["tags"]),
            importance=row["importance"],
            access_count=row["access_count"],
            last_accessed=_str_to_dt(row["last_accessed"]),
            created_at=_str_to_dt(row["created_at"]),  # type: ignore[arg-type]
            updated_at=_str_to_dt(row["updated_at"]),
        )

    # ==================================================================
    # Scheduled Tasks
    # ==================================================================

    def create_scheduled_task(self, task: ScheduledTask) -> ScheduledTask:
        """Insert a new scheduled task row."""
        self.conn.execute(
            """INSERT INTO scheduled_tasks
               (id, user_id, description, original_request, cron_expression,
                next_run_at, last_run_at, last_result, task_type, task_config,
                notify_channel, enabled, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task.id,
                task.user_id,
                task.description,
                task.original_request,
                task.cron_expression,
                _dt_to_str(task.next_run_at),
                _dt_to_str(task.last_run_at),
                task.last_result,
                task.task_type,
                json.dumps(task.task_config),
                task.notify_channel,
                1 if task.enabled else 0,
                _dt_to_str(task.created_at),
            ),
        )
        self.conn.commit()
        return task

    def get_scheduled_tasks(
        self, user_id: str | None = None, enabled_only: bool = False
    ) -> list[ScheduledTask]:
        """List scheduled tasks, optionally filtered by user and/or enabled status."""
        sql = "SELECT * FROM scheduled_tasks WHERE 1=1"
        params: list[str | int] = []
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        if enabled_only:
            sql += " AND enabled = 1"
        sql += " ORDER BY next_run_at ASC"
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_task(r) for r in rows]

    def update_scheduled_task(self, task: ScheduledTask) -> ScheduledTask:
        """Update an existing scheduled task row."""
        self.conn.execute(
            """UPDATE scheduled_tasks SET
                user_id = ?, description = ?, original_request = ?,
                cron_expression = ?, next_run_at = ?, last_run_at = ?,
                last_result = ?, task_type = ?, task_config = ?,
                notify_channel = ?, enabled = ?, created_at = ?
               WHERE id = ?""",
            (
                task.user_id,
                task.description,
                task.original_request,
                task.cron_expression,
                _dt_to_str(task.next_run_at),
                _dt_to_str(task.last_run_at),
                task.last_result,
                task.task_type,
                json.dumps(task.task_config),
                task.notify_channel,
                1 if task.enabled else 0,
                _dt_to_str(task.created_at),
                task.id,
            ),
        )
        self.conn.commit()
        return task

    def _row_to_task(self, row: sqlite3.Row) -> ScheduledTask:
        return ScheduledTask(
            id=row["id"],
            user_id=row["user_id"],
            description=row["description"],
            original_request=row["original_request"],
            cron_expression=row["cron_expression"],
            next_run_at=_str_to_dt(row["next_run_at"]),
            last_run_at=_str_to_dt(row["last_run_at"]),
            last_result=row["last_result"],
            task_type=row["task_type"],
            task_config=json.loads(row["task_config"]),
            notify_channel=row["notify_channel"],
            enabled=bool(row["enabled"]),
            created_at=_str_to_dt(row["created_at"]),  # type: ignore[arg-type]
        )

    # ==================================================================
    # Skills
    # ==================================================================

    def create_skill(self, skill: Skill) -> Skill:
        """Insert a new skill row."""
        self.conn.execute(
            """INSERT INTO skills
               (id, name, skill_type, status, path, description, generated_from,
                usage_count, last_used_at, approved_at, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                skill.id,
                skill.name,
                skill.skill_type,
                skill.status,
                skill.path,
                skill.description,
                skill.generated_from,
                skill.usage_count,
                _dt_to_str(skill.last_used_at),
                _dt_to_str(skill.approved_at),
                _dt_to_str(skill.created_at),
                json.dumps(skill.metadata),
            ),
        )
        self.conn.commit()
        return skill

    def get_skill(self, skill_id: str) -> Skill | None:
        """Fetch a skill by ID."""
        row = self.conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_skill(row)

    def update_skill(self, skill: Skill) -> Skill:
        """Update an existing skill row."""
        self.conn.execute(
            """UPDATE skills SET
                name = ?, skill_type = ?, status = ?, path = ?,
                description = ?, generated_from = ?, usage_count = ?,
                last_used_at = ?, approved_at = ?, created_at = ?, metadata = ?
               WHERE id = ?""",
            (
                skill.name,
                skill.skill_type,
                skill.status,
                skill.path,
                skill.description,
                skill.generated_from,
                skill.usage_count,
                _dt_to_str(skill.last_used_at),
                _dt_to_str(skill.approved_at),
                _dt_to_str(skill.created_at),
                json.dumps(skill.metadata),
                skill.id,
            ),
        )
        self.conn.commit()
        return skill

    def list_skills(
        self, skill_type: str | None = None, status: str | None = None
    ) -> list[Skill]:
        """List skills, optionally filtered by type and/or status."""
        sql = "SELECT * FROM skills WHERE 1=1"
        params: list[str] = []
        if skill_type is not None:
            sql += " AND skill_type = ?"
            params.append(skill_type)
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY name ASC"
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_skill(r) for r in rows]

    def _row_to_skill(self, row: sqlite3.Row) -> Skill:
        return Skill(
            id=row["id"],
            name=row["name"],
            skill_type=row["skill_type"],
            status=row["status"],
            path=row["path"],
            description=row["description"],
            generated_from=row["generated_from"],
            usage_count=row["usage_count"],
            last_used_at=_str_to_dt(row["last_used_at"]),
            approved_at=_str_to_dt(row["approved_at"]),
            created_at=_str_to_dt(row["created_at"]),  # type: ignore[arg-type]
            metadata=json.loads(row["metadata"]),
        )
