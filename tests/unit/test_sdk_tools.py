"""Tests for SDK-native custom tools.

Validates tool definitions, MCP server creation, and the tool naming
conventions expected by the Claude Agent SDK.
"""

from __future__ import annotations

import pytest

from zenki.sdk.tools import (
    ZENKI_TOOL_NAMES,
    create_zenki_tools,
    get_session_history,
    list_scheduled_tasks,
    memory_read_core,
    memory_search,
    memory_store,
    memory_update_core,
    schedule_task,
    send_notification,
)
from zenki.sdk._context import clear_all


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_context() -> None:
    """Ensure a clean context for each test."""
    clear_all()


# ---------------------------------------------------------------------------
# Tool naming conventions
# ---------------------------------------------------------------------------


class TestToolNamingConventions:
    """Verify MCP tool naming follows the SDK convention."""

    def test_all_names_have_zenki_prefix(self) -> None:
        for name in ZENKI_TOOL_NAMES:
            assert name.startswith("mcp__zenki__"), (
                f"Tool name '{name}' should start with 'mcp__zenki__'"
            )

    def test_expected_tool_count(self) -> None:
        assert len(ZENKI_TOOL_NAMES) == 8

    def test_expected_tools_present(self) -> None:
        expected = {
            "mcp__zenki__memory_search",
            "mcp__zenki__memory_store",
            "mcp__zenki__memory_read_core",
            "mcp__zenki__memory_update_core",
            "mcp__zenki__schedule_task",
            "mcp__zenki__list_scheduled_tasks",
            "mcp__zenki__send_notification",
            "mcp__zenki__get_session_history",
        }
        assert expected == set(ZENKI_TOOL_NAMES)


# ---------------------------------------------------------------------------
# MCP server creation
# ---------------------------------------------------------------------------


class TestCreateZenkiTools:
    """Test the MCP server factory."""

    def test_returns_dict_with_zenki_key(self) -> None:
        servers = create_zenki_tools()
        assert "zenki" in servers

    def test_server_is_not_none(self) -> None:
        servers = create_zenki_tools()
        assert servers["zenki"] is not None


# ---------------------------------------------------------------------------
# Tool graceful degradation (no services registered)
# ---------------------------------------------------------------------------


class TestToolsWithoutContext:
    """Tools should handle missing services gracefully.

    Note: The ``@tool`` decorator wraps functions into ``SdkMcpTool`` objects.
    To test the underlying handler, call ``.handler(args)`` on each tool.
    """

    @pytest.mark.asyncio
    async def test_memory_search_without_db(self) -> None:
        result = await memory_search.handler({"query": "test", "category": "", "limit": 5})
        assert result["content"][0]["text"] == "Memory system not initialized."

    @pytest.mark.asyncio
    async def test_memory_store_without_db(self) -> None:
        result = await memory_store.handler({
            "content": "test fact",
            "category": "knowledge",
            "tags": "test",
            "importance": 0.5,
        })
        assert result["content"][0]["text"] == "Memory system not initialized."

    @pytest.mark.asyncio
    async def test_memory_read_core_without_manager(self) -> None:
        result = await memory_read_core.handler({"section": "identity"})
        assert result["content"][0]["text"] == "Memory system not initialized."

    @pytest.mark.asyncio
    async def test_memory_update_core_without_manager(self) -> None:
        result = await memory_update_core.handler({
            "section": "identity",
            "content": "test",
            "mode": "append",
        })
        assert result["content"][0]["text"] == "Memory system not initialized."

    @pytest.mark.asyncio
    async def test_schedule_task_without_scheduler(self) -> None:
        result = await schedule_task.handler({
            "description": "test task",
            "schedule": "0 * * * *",
            "task_type": "reminder",
        })
        assert result["content"][0]["text"] == "Scheduler not initialized."

    @pytest.mark.asyncio
    async def test_list_scheduled_tasks_without_scheduler(self) -> None:
        result = await list_scheduled_tasks.handler({"enabled_only": True})
        assert result["content"][0]["text"] == "Scheduler not initialized."

    @pytest.mark.asyncio
    async def test_send_notification_without_registry(self) -> None:
        result = await send_notification.handler({
            "message": "test",
            "channel": "default",
            "urgency": "normal",
        })
        assert result["content"][0]["text"] == "Channel system not initialized."

    @pytest.mark.asyncio
    async def test_get_session_history_without_db(self) -> None:
        result = await get_session_history.handler({"session_id": "test", "limit": 10})
        assert result["content"][0]["text"] == "Database not initialized."
