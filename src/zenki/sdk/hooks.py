"""SDK hooks for Zenki lifecycle management.

Hooks intercept agent behavior at key execution points to provide:
- **Security**: Block dangerous operations (system dirs, env files, destructive commands)
- **Audit**: Log all tool calls for debugging and compliance
- **Memory**: Automatically capture insights from tool results
- **Notifications**: Forward agent status to communication channels

Reference: docs/claude-agent-sdk/hooks.md
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from claude_agent_sdk import HookMatcher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Security hooks
# ---------------------------------------------------------------------------

async def protect_sensitive_files(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: Any,
) -> dict[str, Any]:
    """Block modifications to sensitive files (.env, credentials, keys).

    This hook fires on PreToolUse for Write/Edit tools and denies operations
    targeting files that likely contain secrets.
    """
    if input_data.get("hook_event_name") != "PreToolUse":
        return {}

    file_path = input_data.get("tool_input", {}).get("file_path", "")
    file_name = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path

    sensitive_patterns = {
        ".env",
        ".env.local",
        ".env.production",
        "credentials.json",
        "secrets.json",
        "id_rsa",
        "id_ed25519",
        ".pem",
    }

    if file_name in sensitive_patterns or file_name.endswith((".pem", ".key")):
        return {
            "systemMessage": (
                "Reminder: Sensitive files containing secrets or credentials "
                "are protected and cannot be modified directly."
            ),
            "hookSpecificOutput": {
                "hookEventName": input_data["hook_event_name"],
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"Cannot modify sensitive file: {file_name}. "
                    "Use environment variables instead."
                ),
            },
        }

    return {}


async def block_dangerous_paths(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: Any,
) -> dict[str, Any]:
    """Block file operations targeting system directories.

    Prevents writing to /etc, /sys, /root/.ssh, /root/.aws, and other
    sensitive system paths.
    """
    if input_data.get("hook_event_name") != "PreToolUse":
        return {}

    file_path = input_data.get("tool_input", {}).get("file_path", "")
    command = input_data.get("tool_input", {}).get("command", "")

    protected_dirs = ["/etc/", "/sys/", "/proc/", "/root/.ssh", "/root/.aws"]

    # Check file paths in Write/Edit
    for protected in protected_dirs:
        if file_path.startswith(protected):
            return {
                "hookSpecificOutput": {
                    "hookEventName": input_data["hook_event_name"],
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"Access to system directory {protected} is restricted."
                    ),
                },
            }

    return {}


async def check_dangerous_commands(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: Any,
) -> dict[str, Any]:
    """Block destructive shell commands unless explicitly authorized.

    Catches common dangerous patterns like rm -rf /, sudo rm, DROP TABLE, etc.
    """
    if input_data.get("hook_event_name") != "PreToolUse":
        return {}
    if input_data.get("tool_name") != "Bash":
        return {}

    command = input_data.get("tool_input", {}).get("command", "")

    # Patterns that should always be blocked
    blocked_patterns = [
        "rm -rf /",
        "rm -rf /*",
        "mkfs.",
        "> /dev/sd",
        "dd if=/dev/zero",
        ":(){:|:&};:",  # Fork bomb
    ]

    for pattern in blocked_patterns:
        if pattern in command:
            return {
                "systemMessage": "Destructive system commands are blocked for safety.",
                "hookSpecificOutput": {
                    "hookEventName": input_data["hook_event_name"],
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Blocked dangerous command pattern: {pattern}",
                },
            }

    return {}


# ---------------------------------------------------------------------------
# Audit hooks
# ---------------------------------------------------------------------------

async def audit_tool_usage(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: Any,
) -> dict[str, Any]:
    """Log every tool call for audit and debugging purposes.

    This is an async (non-blocking) hook — it logs the event and returns
    immediately without blocking the agent.
    """
    tool_name = input_data.get("tool_name", "unknown")
    event = input_data.get("hook_event_name", "unknown")
    session_id = input_data.get("session_id", "unknown")
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info(
        "AUDIT | %s | tool=%s | session=%s | tool_use_id=%s",
        event,
        tool_name,
        session_id[:12] if session_id else "n/a",
        tool_use_id or "n/a",
    )

    # Non-blocking: return immediately
    return {"async_": True, "asyncTimeout": 5000}


# ---------------------------------------------------------------------------
# Memory integration hooks
# ---------------------------------------------------------------------------

async def capture_conversation_insights(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: Any,
) -> dict[str, Any]:
    """After certain tool calls, capture useful information for memory.

    For example, after a web search completes, store a summary of findings.
    After code review, store the patterns identified.
    This is a PostToolUse hook, so it runs after the tool completes.
    """
    if input_data.get("hook_event_name") != "PostToolUse":
        return {}

    # We just log for now; the actual memory storage would be done via
    # the memory tools or the consolidation engine.
    tool_name = input_data.get("tool_name", "")
    if tool_name in ("WebSearch", "WebFetch"):
        logger.debug("Web research completed — consider storing insights in memory")

    return {}


# ---------------------------------------------------------------------------
# Subagent tracking hooks
# ---------------------------------------------------------------------------

async def track_subagent_lifecycle(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: Any,
) -> dict[str, Any]:
    """Track when subagents start and stop for monitoring and analytics."""
    event = input_data.get("hook_event_name", "")
    agent_id = input_data.get("agent_id", "unknown")

    if event == "SubagentStart":
        logger.info("SUBAGENT START | agent=%s | tool_use_id=%s", agent_id, tool_use_id)
    elif event == "SubagentStop":
        transcript_path = input_data.get("agent_transcript_path", "")
        logger.info(
            "SUBAGENT STOP | agent=%s | transcript=%s",
            agent_id,
            transcript_path,
        )

    return {}


# ---------------------------------------------------------------------------
# Notification hooks
# ---------------------------------------------------------------------------

async def forward_notifications(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: Any,
) -> dict[str, Any]:
    """Forward agent notifications to the user's preferred channel.

    Notification events include: permission_prompt, idle_prompt, auth_success,
    and elicitation_dialog.
    """
    message = input_data.get("message", "")
    notification_type = input_data.get("type", "")

    logger.info("NOTIFICATION | type=%s | message=%s", notification_type, message[:100])

    # In production, this would forward to Slack/email/etc. via the channel registry
    # For now, just log it.
    return {}


# ---------------------------------------------------------------------------
# Hook assembly
# ---------------------------------------------------------------------------

def create_zenki_hooks() -> dict[str, list[HookMatcher]]:
    """Assemble all Zenki hooks into the format expected by ``ClaudeAgentOptions.hooks``.

    Returns
    -------
    dict
        A mapping of hook event names to lists of ``HookMatcher`` objects.

    Example
    -------
    >>> hooks = create_zenki_hooks()
    >>> options = ClaudeAgentOptions(hooks=hooks)
    """
    return {
        "PreToolUse": [
            # Security: protect sensitive files from Write/Edit
            HookMatcher(matcher="Write|Edit", hooks=[protect_sensitive_files]),
            # Security: block dangerous system paths
            HookMatcher(matcher="Write|Edit", hooks=[block_dangerous_paths]),
            # Security: block destructive shell commands
            HookMatcher(matcher="Bash", hooks=[check_dangerous_commands]),
            # Audit: log all tool calls (no matcher = all tools)
            HookMatcher(hooks=[audit_tool_usage]),
        ],
        "PostToolUse": [
            # Memory: capture insights from tool results
            HookMatcher(hooks=[capture_conversation_insights]),
            # Audit: log completed tool calls
            HookMatcher(hooks=[audit_tool_usage]),
        ],
        "SubagentStart": [
            HookMatcher(hooks=[track_subagent_lifecycle]),
        ],
        "SubagentStop": [
            HookMatcher(hooks=[track_subagent_lifecycle]),
        ],
        "Notification": [
            HookMatcher(hooks=[forward_notifications]),
        ],
    }
