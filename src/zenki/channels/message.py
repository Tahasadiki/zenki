"""Unified message model for all channels."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class IncomingMessage:
    """A message received from any channel."""

    content: str
    user_id: str
    channel_type: str  # 'slack', 'cli', 'telegram', etc.
    channel_id: str = ""  # channel-specific identifier
    thread_id: str | None = None  # thread within channel
    session_id: str | None = None  # existing session to continue
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_threaded(self) -> bool:
        """Whether this message is part of a thread."""
        return self.thread_id is not None


@dataclass
class OutgoingMessage:
    """A message to send to a channel."""

    content: str
    channel_id: str = ""
    thread_id: str | None = None
    session_id: str | None = None
    format_type: str = "markdown"  # 'markdown', 'plain', 'rich'
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Notification:
    """A proactive notification to send to the user."""

    content: str
    title: str = ""
    priority: str = "normal"  # 'low', 'normal', 'high', 'urgent'
    source: str = ""  # what generated this notification
    metadata: dict[str, Any] = field(default_factory=dict)
