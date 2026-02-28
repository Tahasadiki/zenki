# Claude Agent SDK - Agents (Subagents)

> Reference: https://platform.claude.com/docs/en/agent-sdk/subagents

## Overview

Subagents are separate agent instances spawned by the main agent to handle focused subtasks.
They are defined using `AgentDefinition` and invoked via the `Task` tool.

### Three Ways to Create Subagents

1. **Programmatically** - `agents` parameter in `ClaudeAgentOptions` (recommended for SDK apps)
2. **Filesystem-based** - Markdown files in `.claude/agents/` directories
3. **Built-in** - Claude can use the `general-purpose` subagent via Task tool without defining anything

## AgentDefinition

```python
from claude_agent_sdk import AgentDefinition

agent = AgentDefinition(
    description="Expert code reviewer for quality and security reviews.",
    prompt="Analyze code quality and suggest improvements.",
    tools=["Read", "Glob", "Grep"],  # Optional: restrict tools
    model="sonnet",                   # Optional: override model
)
```

### Fields

| Field | Type | Required | Description |
|:------|:-----|:---------|:------------|
| `description` | `str` | Yes | Natural language description of WHEN to use this agent |
| `prompt` | `str` | Yes | The agent's system prompt (role, behavior, expertise) |
| `tools` | `str[]` | No | Allowed tools. Omit = inherit all from parent |
| `model` | `'sonnet'\|'opus'\|'haiku'\|'inherit'` | No | Model override |

### Important Notes

- **`Task` must be in `allowed_tools`** for the main agent to invoke subagents
- **Subagents cannot spawn their own subagents** - don't include `Task` in subagent tools
- Claude decides when to invoke based on `description` field
- Explicit invocation: mention agent by name in prompt

## Example: Multi-Agent System

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition

async def main():
    async for message in query(
        prompt="Review this codebase for security issues",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Grep", "Glob", "Task"],
            agents={
                "code-reviewer": AgentDefinition(
                    description="Expert code review specialist.",
                    prompt="You are a code review specialist...",
                    tools=["Read", "Grep", "Glob"],
                    model="sonnet",
                ),
                "test-runner": AgentDefinition(
                    description="Runs and analyzes test suites.",
                    prompt="You are a test execution specialist...",
                    tools=["Bash", "Read", "Grep"],
                ),
            },
        ),
    ):
        if hasattr(message, "result"):
            print(message.result)

asyncio.run(main())
```

## Dynamic Agent Configuration

```python
def create_security_agent(security_level: str) -> AgentDefinition:
    is_strict = security_level == "strict"
    return AgentDefinition(
        description="Security code reviewer",
        prompt=f"You are a {'strict' if is_strict else 'balanced'} security reviewer...",
        tools=["Read", "Grep", "Glob"],
        model="opus" if is_strict else "sonnet",
    )
```

## Common Tool Combinations

| Use case | Tools | Description |
|:---------|:------|:------------|
| Read-only analysis | `Read`, `Grep`, `Glob` | Examine code, no modifications |
| Test execution | `Bash`, `Read`, `Grep` | Run commands, analyze output |
| Code modification | `Read`, `Edit`, `Write`, `Grep`, `Glob` | Full read/write, no commands |
| Full access | All tools | Inherit all (omit `tools` field) |
