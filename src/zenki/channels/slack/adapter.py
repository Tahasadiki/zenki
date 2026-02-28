"""Slack channel adapter using slack-bolt."""

from __future__ import annotations

import logging
from typing import Any

from zenki.channels.base import BaseChannel
from zenki.channels.message import IncomingMessage, Notification, OutgoingMessage

logger = logging.getLogger(__name__)


class SlackChannel(BaseChannel):
    """Slack communication channel using the Bolt framework.

    Handles Slack events via webhooks. Each Slack thread maps to a session.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(channel_type="slack", config=config)
        self._app = None
        self._bolt_app = None

    def _ensure_bolt(self) -> None:
        """Lazy-initialize the Slack Bolt app."""
        if self._bolt_app is not None:
            return

        try:
            from slack_bolt.async_app import AsyncApp
        except ImportError:
            raise ImportError(
                "slack-bolt is required for the Slack channel. "
                "Install it with: pip install zenki[slack]"
            )

        bot_token = self.config.get("bot_token")
        signing_secret = self.config.get("signing_secret")

        if not bot_token or not signing_secret:
            raise ValueError(
                "Slack bot_token and signing_secret are required. "
                "Set ZENKI_SLACK_BOT_TOKEN and ZENKI_SLACK_SIGNING_SECRET environment variables."
            )

        self._bolt_app = AsyncApp(
            token=bot_token,
            signing_secret=signing_secret,
        )
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register Slack event handlers."""
        if self._bolt_app is None:
            return

        @self._bolt_app.event("app_mention")
        async def handle_mention(event: dict, say: Any) -> None:
            await self._handle_message(event, say)

        @self._bolt_app.event("message")
        async def handle_dm(event: dict, say: Any) -> None:
            # Only handle DMs (channel type "im")
            if event.get("channel_type") == "im":
                await self._handle_message(event, say)

    async def _handle_message(self, event: dict, say: Any) -> None:
        """Process an incoming Slack message."""
        if self._message_handler is None:
            logger.warning("No message handler set for Slack channel")
            return

        text = event.get("text", "").strip()
        # Remove bot mention if present
        if "<@" in text:
            import re
            text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()

        incoming = IncomingMessage(
            content=text,
            user_id=event.get("user", "unknown"),
            channel_type="slack",
            channel_id=event.get("channel", ""),
            thread_id=event.get("thread_ts") or event.get("ts"),
            metadata={
                "event_ts": event.get("ts"),
                "team": event.get("team"),
            },
        )

        response = await self._message_handler(incoming)
        if response:
            formatted = self.format_content(response.content, response.format_type)
            await say(text=formatted, thread_ts=incoming.thread_id)

    async def start(self) -> None:
        """Start the Slack channel (Bolt app handlers are ready)."""
        self._ensure_bolt()
        self._running = True
        logger.info("Slack channel started")

    async def stop(self) -> None:
        """Stop the Slack channel."""
        self._running = False
        logger.info("Slack channel stopped")

    async def send_message(self, message: OutgoingMessage) -> None:
        """Send a message to Slack."""
        if self._bolt_app is None:
            logger.warning("Slack app not initialized")
            return

        formatted = self.format_content(message.content, message.format_type)
        try:
            await self._bolt_app.client.chat_postMessage(
                channel=message.channel_id,
                text=formatted,
                thread_ts=message.thread_id,
            )
        except Exception as e:
            logger.error("Failed to send Slack message: %s", e)

    async def send_notification(self, notification: Notification, user_id: str) -> None:
        """Send a notification as a DM to the user on Slack."""
        if self._bolt_app is None:
            logger.warning("Slack app not initialized")
            return

        try:
            # Open DM channel
            result = await self._bolt_app.client.conversations_open(users=[user_id])
            dm_channel = result["channel"]["id"]

            prefix = ""
            if notification.priority in ("high", "urgent"):
                prefix = f":warning: *[{notification.priority.upper()}]* "

            title = f"*{notification.title}*\n" if notification.title else ""
            text = f"{prefix}{title}{notification.content}"

            await self._bolt_app.client.chat_postMessage(
                channel=dm_channel,
                text=text,
            )
        except Exception as e:
            logger.error("Failed to send Slack notification: %s", e)

    def format_content(self, content: str, format_type: str = "markdown") -> str:
        """Convert markdown to Slack mrkdwn format."""
        if format_type == "plain":
            return content

        # Basic markdown → Slack mrkdwn conversions
        text = content
        # Bold: **text** → *text*
        import re
        text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
        # Italic: _text_ stays the same in Slack
        # Code blocks: ```lang\ncode``` → ```code```
        text = re.sub(r"```\w*\n", "```\n", text)
        # Inline code: `code` stays the same
        # Headers: ## Header → *Header*
        text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)
        # Links: [text](url) → <url|text>
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)

        return text

    def get_session_rules(self) -> dict[str, Any]:
        """Slack session rules: one session per thread."""
        return {
            "session_per": "thread_id",
            "auto_close_after_minutes": 60,
            "include_channel_context": True,
        }

    def get_bolt_app(self) -> Any:
        """Get the underlying Slack Bolt app for webhook registration."""
        self._ensure_bolt()
        return self._bolt_app
