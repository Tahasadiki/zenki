"""Custom MCP tools for Zenki-specific operations.

These tools extend Claude's built-in capabilities with Zenki's unique features:
memory management, skill management, scheduling, and notifications.

Tools are defined using the Claude Agent SDK's native ``@tool`` decorator and
bundled into in-process MCP servers via ``create_sdk_mcp_server()``.

Reference: docs/claude-agent-sdk/custom-tools.md
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------

@tool(
    "memory_search",
    "Search Zenki's semantic memory for relevant knowledge, facts, preferences, or past learnings",
    {"query": str, "category": str, "limit": int},
)
async def memory_search(args: dict[str, Any]) -> dict[str, Any]:
    """Search semantic memory by query and optional category."""
    query = args["query"]
    category = args.get("category", "")
    limit = args.get("limit", 5)

    # Access the shared database via the tool context
    # In production, this would be injected or accessed via a global registry
    from zenki.sdk._context import get_database

    db = get_database()
    if db is None:
        return {"content": [{"type": "text", "text": "Memory system not initialized."}]}

    memories = db.search_semantic_memories(query=query, category=category, limit=limit)
    if not memories:
        return {"content": [{"type": "text", "text": f"No memories found for: {query}"}]}

    results = []
    for mem in memories:
        results.append(f"- [{mem.category}] {mem.content} (importance: {mem.importance:.1f})")

    return {
        "content": [
            {"type": "text", "text": f"Found {len(results)} memories:\n" + "\n".join(results)}
        ]
    }


@tool(
    "memory_store",
    "Store a new fact, preference, insight, or knowledge item in Zenki's semantic memory",
    {
        "content": str,
        "category": str,
        "tags": str,
        "importance": float,
    },
)
async def memory_store(args: dict[str, Any]) -> dict[str, Any]:
    """Store a new semantic memory entry."""
    from zenki.sdk._context import get_database

    db = get_database()
    if db is None:
        return {"content": [{"type": "text", "text": "Memory system not initialized."}]}

    tags = [t.strip() for t in args.get("tags", "").split(",") if t.strip()]
    importance = min(1.0, max(0.0, args.get("importance", 0.5)))

    db.store_semantic_memory(
        content=args["content"],
        category=args.get("category", "knowledge"),
        tags=tags,
        importance=importance,
    )

    return {
        "content": [
            {"type": "text", "text": f"Stored memory: {args['content'][:100]}..."}
        ]
    }


@tool(
    "memory_read_core",
    "Read Zenki's core memory files (identity, preferences, projects, relationships, patterns)",
    {"section": str},
)
async def memory_read_core(args: dict[str, Any]) -> dict[str, Any]:
    """Read a core memory section."""
    from zenki.sdk._context import get_memory_manager

    manager = get_memory_manager()
    if manager is None:
        return {"content": [{"type": "text", "text": "Memory system not initialized."}]}

    section = args.get("section", "all")
    content = manager.read_core_section(section)

    return {"content": [{"type": "text", "text": content or f"No content in section: {section}"}]}


@tool(
    "memory_update_core",
    "Update a section of Zenki's core memory (identity, preferences, projects, relationships, patterns)",
    {"section": str, "content": str, "mode": str},
)
async def memory_update_core(args: dict[str, Any]) -> dict[str, Any]:
    """Update a core memory section."""
    from zenki.sdk._context import get_memory_manager

    manager = get_memory_manager()
    if manager is None:
        return {"content": [{"type": "text", "text": "Memory system not initialized."}]}

    section = args["section"]
    content = args["content"]
    mode = args.get("mode", "append")  # "append" or "replace"

    manager.update_core_section(section, content, mode=mode)

    return {
        "content": [
            {"type": "text", "text": f"Updated core memory section '{section}' ({mode})."}
        ]
    }


# ---------------------------------------------------------------------------
# Scheduling tools
# ---------------------------------------------------------------------------

@tool(
    "schedule_task",
    "Schedule a recurring or one-time task for Zenki to execute later",
    {
        "description": str,
        "schedule": str,
        "task_type": str,
    },
)
async def schedule_task(args: dict[str, Any]) -> dict[str, Any]:
    """Create a scheduled task."""
    from zenki.sdk._context import get_scheduler

    scheduler = get_scheduler()
    if scheduler is None:
        return {"content": [{"type": "text", "text": "Scheduler not initialized."}]}

    task_id = scheduler.add_task(
        description=args["description"],
        schedule=args["schedule"],
        task_type=args.get("task_type", "reminder"),
    )

    return {
        "content": [
            {
                "type": "text",
                "text": f"Scheduled task '{args['description']}' (ID: {task_id}, schedule: {args['schedule']})",
            }
        ]
    }


@tool(
    "list_scheduled_tasks",
    "List all scheduled tasks with their status, schedule, and next run time",
    {"enabled_only": bool},
)
async def list_scheduled_tasks(args: dict[str, Any]) -> dict[str, Any]:
    """List scheduled tasks."""
    from zenki.sdk._context import get_scheduler

    scheduler = get_scheduler()
    if scheduler is None:
        return {"content": [{"type": "text", "text": "Scheduler not initialized."}]}

    enabled_only = args.get("enabled_only", True)
    tasks = scheduler.list_tasks(enabled_only=enabled_only)

    if not tasks:
        return {"content": [{"type": "text", "text": "No scheduled tasks found."}]}

    lines = []
    for task in tasks:
        status = "enabled" if task.enabled else "disabled"
        lines.append(f"- [{task.id[:8]}] {task.description} ({task.cron_expression}) [{status}]")

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


# ---------------------------------------------------------------------------
# Notification tools
# ---------------------------------------------------------------------------

@tool(
    "send_notification",
    "Send a notification message to the user via a specific channel (Slack, CLI, etc.)",
    {"message": str, "channel": str, "urgency": str},
)
async def send_notification(args: dict[str, Any]) -> dict[str, Any]:
    """Send a notification through a channel."""
    from zenki.sdk._context import get_channel_registry

    registry = get_channel_registry()
    if registry is None:
        return {"content": [{"type": "text", "text": "Channel system not initialized."}]}

    channel = args.get("channel", "default")
    urgency = args.get("urgency", "normal")

    await registry.send_notification(
        message=args["message"],
        channel=channel,
        urgency=urgency,
    )

    return {
        "content": [
            {"type": "text", "text": f"Notification sent via {channel}: {args['message'][:80]}..."}
        ]
    }


# ---------------------------------------------------------------------------
# Session / conversation context tools
# ---------------------------------------------------------------------------

@tool(
    "get_session_history",
    "Retrieve recent conversation history from the current or a past session",
    {"session_id": str, "limit": int},
)
async def get_session_history(args: dict[str, Any]) -> dict[str, Any]:
    """Get conversation history for a session."""
    from zenki.sdk._context import get_database

    db = get_database()
    if db is None:
        return {"content": [{"type": "text", "text": "Database not initialized."}]}

    session_id = args.get("session_id", "")
    limit = args.get("limit", 10)

    messages = db.get_messages(session_id=session_id, limit=limit)
    if not messages:
        return {"content": [{"type": "text", "text": "No messages found for this session."}]}

    lines = []
    for msg in messages:
        role = msg.role.upper()
        preview = msg.content[:120].replace("\n", " ")
        lines.append(f"[{role}] {preview}")

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


# ---------------------------------------------------------------------------
# MCP Server creation
# ---------------------------------------------------------------------------

def create_zenki_tools() -> dict[str, Any]:
    """Create the Zenki custom MCP tools server.

    Returns a dict suitable for passing to ``ClaudeAgentOptions.mcp_servers``:

        >>> tools_server = create_zenki_tools()
        >>> options = ClaudeAgentOptions(mcp_servers={"zenki": tools_server})
    """
    server = create_sdk_mcp_server(
        name="zenki",
        version="1.0.0",
        tools=[
            # Memory tools
            memory_search,
            memory_store,
            memory_read_core,
            memory_update_core,
            # Scheduling tools
            schedule_task,
            list_scheduled_tasks,
            # Notification tools
            send_notification,
            # Session tools
            get_session_history,
        ],
    )
    return {"zenki": server}


# Convenience: list of all Zenki MCP tool names for allowed_tools configuration.
ZENKI_TOOL_NAMES = [
    "mcp__zenki__memory_search",
    "mcp__zenki__memory_store",
    "mcp__zenki__memory_read_core",
    "mcp__zenki__memory_update_core",
    "mcp__zenki__schedule_task",
    "mcp__zenki__list_scheduled_tasks",
    "mcp__zenki__send_notification",
    "mcp__zenki__get_session_history",
]
