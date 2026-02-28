# Claude Agent SDK - Hooks

> Reference: https://platform.claude.com/docs/en/agent-sdk/hooks

## Overview

Hooks are callback functions that run at key agent execution points.
They can block operations, log/audit, transform inputs/outputs, require approval, etc.

## Available Hook Events

| Hook Event | Python | Description |
|------------|--------|-------------|
| `PreToolUse` | Yes | Before tool execution (can block/modify) |
| `PostToolUse` | Yes | After tool execution |
| `PostToolUseFailure` | Yes | Tool execution failure |
| `UserPromptSubmit` | Yes | User prompt submission |
| `Stop` | Yes | Agent execution stop |
| `SubagentStart` | Yes | Subagent initialization |
| `SubagentStop` | Yes | Subagent completion |
| `PreCompact` | Yes | Conversation compaction |
| `PermissionRequest` | Yes | Permission dialog |
| `Notification` | Yes | Agent status messages |

## Configuration

```python
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

options = ClaudeAgentOptions(
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="Write|Edit", hooks=[protect_env_files]),
            HookMatcher(hooks=[auto_approve_reads]),  # No matcher = all tools
        ],
        "PostToolUse": [
            HookMatcher(hooks=[audit_logger]),
        ],
        "SubagentStop": [
            HookMatcher(hooks=[subagent_tracker]),
        ],
    }
)
```

## Callback Signature

```python
async def my_hook(input_data, tool_use_id, context):
    # input_data: dict with event details (tool_name, tool_input, session_id, etc.)
    # tool_use_id: str | None - correlates Pre/Post events for same call
    # context: reserved for future use in Python

    # Return {} to allow operation without changes
    return {}
```

## Hook Outputs

### Allow operation (default)
```python
return {}
```

### Block operation
```python
return {
    "hookSpecificOutput": {
        "hookEventName": input_data["hook_event_name"],
        "permissionDecision": "deny",
        "permissionDecisionReason": "Reason for blocking",
    }
}
```

### Auto-approve
```python
return {
    "hookSpecificOutput": {
        "hookEventName": input_data["hook_event_name"],
        "permissionDecision": "allow",
    }
}
```

### Modify input
```python
return {
    "hookSpecificOutput": {
        "hookEventName": input_data["hook_event_name"],
        "permissionDecision": "allow",
        "updatedInput": {**input_data["tool_input"], "file_path": "/sandbox/file.txt"},
    }
}
```

### Inject context
```python
return {
    "systemMessage": "Remember: system dirs are protected.",
    "hookSpecificOutput": {
        "hookEventName": input_data["hook_event_name"],
        "permissionDecision": "deny",
        "permissionDecisionReason": "Not allowed",
    },
}
```

### Async (non-blocking side effect)
```python
return {"async_": True, "asyncTimeout": 30000}
```

## Matchers

Regex patterns matching tool names:
- `"Write|Edit"` - file modification tools
- `"^mcp__"` - all MCP tools
- `"Bash"` - shell commands
- No matcher = match everything

## Priority

When multiple hooks apply: **deny > ask > allow**
