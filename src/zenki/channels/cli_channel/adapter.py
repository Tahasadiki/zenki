"""CLI channel adapter - treats the terminal as a communication channel."""

from __future__ import annotations

import logging
from typing import Any

from zenki.channels.base import BaseChannel
from zenki.channels.message import IncomingMessage, Notification, OutgoingMessage

logger = logging.getLogger(__name__)


class CLIChannel(BaseChannel):
    """CLI as a communication channel.

    This adapter enables the terminal/CLI to be used as a channel alongside
    Slack, Telegram, etc. It's primarily used for:
    - Direct chat via `zenki chat`
    - Development and testing
    - Admin interactions
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(channel_type="cli", config=config)
        self._output_buffer: list[str] = []

    async def start(self) -> None:
        """Start the CLI channel (no-op, always available)."""
        self._running = True
        logger.info("CLI channel started")

    async def stop(self) -> None:
        """Stop the CLI channel."""
        self._running = False
        logger.info("CLI channel stopped")

    async def send_message(self, message: OutgoingMessage) -> None:
        """Send a message to the CLI (store in buffer for retrieval)."""
        formatted = self.format_content(message.content, message.format_type)
        self._output_buffer.append(formatted)

    async def send_notification(self, notification: Notification, user_id: str) -> None:
        """Send a notification via CLI (store in buffer)."""
        prefix = f"[{notification.priority.upper()}]" if notification.priority != "normal" else ""
        title = f" {notification.title}:" if notification.title else ""
        msg = f"{prefix}{title} {notification.content}"
        self._output_buffer.append(msg.strip())

    def format_content(self, content: str, format_type: str = "markdown") -> str:
        """Format content for CLI output (pass through markdown)."""
        return content

    def get_session_rules(self) -> dict[str, Any]:
        """CLI session rules: one session per chat invocation."""
        return {
            "session_per": "invocation",
            "auto_close_after_minutes": None,
            "include_channel_context": False,
        }

    def get_output(self) -> list[str]:
        """Get and clear the output buffer."""
        output = self._output_buffer.copy()
        self._output_buffer.clear()
        return output

    def create_incoming_message(
        self,
        content: str,
        user_id: str = "default",
        session_id: str | None = None,
    ) -> IncomingMessage:
        """Create an IncomingMessage from CLI input."""
        return IncomingMessage(
            content=content,
            user_id=user_id,
            channel_type="cli",
            channel_id="terminal",
            session_id=session_id,
        )
