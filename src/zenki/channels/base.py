"""Abstract base class for all communication channels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from zenki.channels.message import IncomingMessage, Notification, OutgoingMessage

# Type alias for message handlers
MessageHandler = Callable[[IncomingMessage], Awaitable[OutgoingMessage | None]]


class BaseChannel(ABC):
    """Abstract base class for all communication channels.

    Each channel adapter translates channel-specific events into a common
    message format, manages sessions according to channel rules, and formats
    outgoing messages for the channel.
    """

    def __init__(self, channel_type: str, config: dict[str, Any] | None = None) -> None:
        self.channel_type = channel_type
        self.config = config or {}
        self._message_handler: MessageHandler | None = None
        self._running = False

    def set_message_handler(self, handler: MessageHandler) -> None:
        """Set the handler that processes incoming messages."""
        self._message_handler = handler

    @property
    def is_running(self) -> bool:
        """Whether the channel is currently active."""
        return self._running

    @abstractmethod
    async def start(self) -> None:
        """Start listening for messages."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel gracefully."""
        ...

    @abstractmethod
    async def send_message(self, message: OutgoingMessage) -> None:
        """Send a message to a specific channel/thread."""
        ...

    @abstractmethod
    async def send_notification(self, notification: Notification, user_id: str) -> None:
        """Send a proactive notification to the user."""
        ...

    @abstractmethod
    def format_content(self, content: str, format_type: str = "markdown") -> str:
        """Format content for this channel (e.g., markdown -> Slack mrkdwn)."""
        ...

    def get_session_rules(self) -> dict[str, Any]:
        """Return channel-specific session management rules.

        Override in subclasses to define how sessions are managed.
        Default: one session per channel_id.
        """
        return {
            "session_per": "channel_id",  # or "thread_id", "user"
            "auto_close_after_minutes": None,  # None = never auto-close
            "include_channel_context": False,
        }
