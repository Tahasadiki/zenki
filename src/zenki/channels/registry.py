"""Channel registry - manages all active channels."""

from __future__ import annotations

import logging

from zenki.channels.base import BaseChannel, MessageHandler
from zenki.channels.message import Notification

logger = logging.getLogger(__name__)


class ChannelRegistry:
    """Manages all registered communication channels."""

    def __init__(self) -> None:
        self._channels: dict[str, BaseChannel] = {}
        self._default_channel: str | None = None

    def register(self, channel: BaseChannel) -> None:
        """Register a channel."""
        self._channels[channel.channel_type] = channel
        logger.info("Registered channel: %s", channel.channel_type)
        if self._default_channel is None:
            self._default_channel = channel.channel_type

    def unregister(self, channel_type: str) -> None:
        """Unregister a channel."""
        if channel_type in self._channels:
            del self._channels[channel_type]
            if self._default_channel == channel_type:
                self._default_channel = next(iter(self._channels), None)

    def get(self, channel_type: str) -> BaseChannel | None:
        """Get a channel by type."""
        return self._channels.get(channel_type)

    def get_default(self) -> BaseChannel | None:
        """Get the default notification channel."""
        if self._default_channel:
            return self._channels.get(self._default_channel)
        return None

    def set_default(self, channel_type: str) -> None:
        """Set the default notification channel."""
        if channel_type not in self._channels:
            raise ValueError(f"Channel '{channel_type}' not registered")
        self._default_channel = channel_type

    @property
    def default_channel_type(self) -> str | None:
        """The type of the default channel."""
        return self._default_channel

    def list_channels(self) -> list[str]:
        """List all registered channel types."""
        return list(self._channels.keys())

    def set_message_handler(self, handler: MessageHandler) -> None:
        """Set the message handler on all channels."""
        for channel in self._channels.values():
            channel.set_message_handler(handler)

    async def start_all(self) -> None:
        """Start all registered channels."""
        for channel_type, channel in self._channels.items():
            try:
                await channel.start()
                logger.info("Started channel: %s", channel_type)
            except Exception as e:
                logger.error("Failed to start channel %s: %s", channel_type, e)

    async def stop_all(self) -> None:
        """Stop all registered channels."""
        for channel_type, channel in self._channels.items():
            try:
                await channel.stop()
                logger.info("Stopped channel: %s", channel_type)
            except Exception as e:
                logger.error("Failed to stop channel %s: %s", channel_type, e)

    async def send_notification(
        self,
        notification: Notification,
        user_id: str,
        channel_type: str | None = None,
    ) -> None:
        """Send a notification via the specified or default channel."""
        target_type = channel_type or self._default_channel
        if target_type is None:
            logger.warning("No channel available for notification")
            return

        channel = self._channels.get(target_type)
        if channel is None:
            logger.warning("Channel '%s' not found for notification", target_type)
            return

        await channel.send_notification(notification, user_id)
