"""Pydantic data models for all Zenki database entities."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(UTC)


def _new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid4())


class User(BaseModel):
    """User profile (single-user, multi-user ready)."""

    model_config = {"from_attributes": True}

    id: str = Field(default_factory=_new_id)
    display_name: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    config: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """Channel-aware conversation session."""

    model_config = {"from_attributes": True}

    id: str = Field(default_factory=_new_id)
    user_id: str
    channel_type: str
    channel_id: str | None = None
    thread_id: str | None = None
    context_summary: str | None = None
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    last_active: datetime | None = None
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """Full conversation message log entry."""

    model_config = {"from_attributes": True}

    id: str = Field(default_factory=_new_id)
    session_id: str
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    model_used: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tool_calls: list[Any] | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class EpisodicMemory(BaseModel):
    """Tier 3: Episodic memory - conversation summaries."""

    model_config = {"from_attributes": True}

    id: str = Field(default_factory=_new_id)
    session_id: str | None = None
    summary: str
    key_topics: list[str] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: datetime | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class SemanticMemory(BaseModel):
    """Tier 4: Semantic memory - facts, knowledge, insights."""

    model_config = {"from_attributes": True}

    id: str = Field(default_factory=_new_id)
    session_id: str | None = None
    content: str
    category: Literal["fact", "preference", "knowledge", "insight", "lesson"]
    source: str | None = None
    tags: list[str] = Field(default_factory=list)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: datetime | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime | None = None


class ScheduledTask(BaseModel):
    """Scheduled task for reminders, actions, self-improvement, and monitoring."""

    model_config = {"from_attributes": True}

    id: str = Field(default_factory=_new_id)
    user_id: str
    description: str
    original_request: str | None = None
    cron_expression: str
    next_run_at: datetime | None = None
    last_run_at: datetime | None = None
    last_result: str | None = None
    task_type: Literal["reminder", "action", "self_improve", "monitor"]
    task_config: dict[str, Any] = Field(default_factory=dict)
    notify_channel: str | None = None
    enabled: bool = True
    created_at: datetime = Field(default_factory=_utcnow)


class Skill(BaseModel):
    """Skill registry entry for core and learned skills."""

    model_config = {"from_attributes": True}

    id: str = Field(default_factory=_new_id)
    name: str
    skill_type: Literal["core", "learned"]
    status: Literal["pending", "approved", "rejected", "disabled"] = "pending"
    path: str
    description: str | None = None
    generated_from: str | None = None
    usage_count: int = 0
    last_used_at: datetime | None = None
    approved_at: datetime | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
