"""Tests for the channel system."""

import pytest

from zenki.channels.cli_channel.adapter import CLIChannel
from zenki.channels.message import IncomingMessage, Notification, OutgoingMessage
from zenki.channels.registry import ChannelRegistry


class TestIncomingMessage:
    def test_create_message(self):
        msg = IncomingMessage(content="hello", user_id="u1", channel_type="cli")
        assert msg.content == "hello"
        assert msg.user_id == "u1"
        assert msg.channel_type == "cli"

    def test_defaults(self):
        msg = IncomingMessage(content="test", user_id="u1", channel_type="cli")
        assert msg.channel_id == ""
        assert msg.thread_id is None
        assert msg.session_id is None
        assert msg.metadata == {}

    def test_is_threaded(self):
        msg1 = IncomingMessage(content="test", user_id="u1", channel_type="cli")
        assert not msg1.is_threaded

        msg2 = IncomingMessage(
            content="test", user_id="u1", channel_type="slack", thread_id="t1"
        )
        assert msg2.is_threaded

    def test_timestamp_set(self):
        msg = IncomingMessage(content="test", user_id="u1", channel_type="cli")
        assert msg.timestamp is not None


class TestOutgoingMessage:
    def test_create(self):
        msg = OutgoingMessage(content="hello")
        assert msg.content == "hello"
        assert msg.format_type == "markdown"

    def test_with_thread(self):
        msg = OutgoingMessage(content="reply", channel_id="c1", thread_id="t1")
        assert msg.channel_id == "c1"
        assert msg.thread_id == "t1"


class TestNotification:
    def test_create(self):
        notif = Notification(content="Task done")
        assert notif.content == "Task done"
        assert notif.priority == "normal"
        assert notif.title == ""

    def test_with_priority(self):
        notif = Notification(content="Error!", priority="urgent", title="Alert")
        assert notif.priority == "urgent"
        assert notif.title == "Alert"


class TestCLIChannel:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        channel = CLIChannel()
        assert not channel.is_running
        await channel.start()
        assert channel.is_running
        await channel.stop()
        assert not channel.is_running

    @pytest.mark.asyncio
    async def test_send_message(self):
        channel = CLIChannel()
        await channel.start()
        msg = OutgoingMessage(content="Hello from agent")
        await channel.send_message(msg)
        output = channel.get_output()
        assert len(output) == 1
        assert "Hello from agent" in output[0]

    @pytest.mark.asyncio
    async def test_send_notification(self):
        channel = CLIChannel()
        await channel.start()
        notif = Notification(content="Task completed", title="Done")
        await channel.send_notification(notif, "user1")
        output = channel.get_output()
        assert len(output) == 1
        assert "Task completed" in output[0]

    @pytest.mark.asyncio
    async def test_send_urgent_notification(self):
        channel = CLIChannel()
        await channel.start()
        notif = Notification(content="Error!", priority="urgent")
        await channel.send_notification(notif, "user1")
        output = channel.get_output()
        assert "URGENT" in output[0]

    def test_format_content_passthrough(self):
        channel = CLIChannel()
        assert channel.format_content("**bold**") == "**bold**"

    def test_session_rules(self):
        channel = CLIChannel()
        rules = channel.get_session_rules()
        assert rules["session_per"] == "invocation"

    def test_create_incoming_message(self):
        channel = CLIChannel()
        msg = channel.create_incoming_message("hello", "user1", "session1")
        assert msg.content == "hello"
        assert msg.user_id == "user1"
        assert msg.channel_type == "cli"
        assert msg.session_id == "session1"

    @pytest.mark.asyncio
    async def test_get_output_clears_buffer(self):
        channel = CLIChannel()
        await channel.start()
        await channel.send_message(OutgoingMessage(content="msg1"))
        await channel.send_message(OutgoingMessage(content="msg2"))
        output1 = channel.get_output()
        assert len(output1) == 2
        output2 = channel.get_output()
        assert len(output2) == 0

    def test_channel_type(self):
        channel = CLIChannel()
        assert channel.channel_type == "cli"


class TestChannelRegistry:
    def test_register_channel(self):
        registry = ChannelRegistry()
        channel = CLIChannel()
        registry.register(channel)
        assert "cli" in registry.list_channels()

    def test_first_registered_becomes_default(self):
        registry = ChannelRegistry()
        channel = CLIChannel()
        registry.register(channel)
        assert registry.default_channel_type == "cli"

    def test_get_channel(self):
        registry = ChannelRegistry()
        channel = CLIChannel()
        registry.register(channel)
        assert registry.get("cli") is channel
        assert registry.get("nonexistent") is None

    def test_get_default(self):
        registry = ChannelRegistry()
        channel = CLIChannel()
        registry.register(channel)
        assert registry.get_default() is channel

    def test_set_default(self):
        registry = ChannelRegistry()
        cli = CLIChannel()
        registry.register(cli)
        # Can't set nonexistent channel as default
        with pytest.raises(ValueError):
            registry.set_default("nonexistent")

    def test_unregister(self):
        registry = ChannelRegistry()
        channel = CLIChannel()
        registry.register(channel)
        registry.unregister("cli")
        assert "cli" not in registry.list_channels()
        assert registry.default_channel_type is None

    def test_set_message_handler(self):
        registry = ChannelRegistry()
        channel = CLIChannel()
        registry.register(channel)

        async def handler(msg):
            return None

        registry.set_message_handler(handler)
        assert channel._message_handler is handler

    @pytest.mark.asyncio
    async def test_start_stop_all(self):
        registry = ChannelRegistry()
        channel = CLIChannel()
        registry.register(channel)
        await registry.start_all()
        assert channel.is_running
        await registry.stop_all()
        assert not channel.is_running

    @pytest.mark.asyncio
    async def test_send_notification(self):
        registry = ChannelRegistry()
        channel = CLIChannel()
        registry.register(channel)
        await channel.start()
        notif = Notification(content="test notification")
        await registry.send_notification(notif, "user1")
        output = channel.get_output()
        assert len(output) == 1

    @pytest.mark.asyncio
    async def test_notification_no_channel(self):
        registry = ChannelRegistry()
        notif = Notification(content="test")
        # Should not raise
        await registry.send_notification(notif, "user1")

    def test_list_channels_empty(self):
        registry = ChannelRegistry()
        assert registry.list_channels() == []
