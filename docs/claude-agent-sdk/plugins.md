# Claude Agent SDK - Plugins

> Reference: https://platform.claude.com/docs/en/agent-sdk/plugins

## Overview

Plugins package Claude Code extensions including:
- **Commands**: Custom slash commands
- **Agents**: Specialized subagents
- **Skills**: Model-invoked capabilities
- **Hooks**: Event handlers
- **MCP servers**: External tool integrations

## Plugin Structure

```
my-plugin/
├── .claude-plugin/
│   └── plugin.json          # Required: plugin manifest
├── commands/                 # Custom slash commands
│   └── custom-cmd.md
├── agents/                   # Custom agents
│   └── specialist.md
├── skills/                   # Agent Skills
│   └── my-skill/
│       └── SKILL.md
├── hooks/                    # Event handlers
│   └── hooks.json
└── .mcp.json                # MCP server definitions
```

## Loading Plugins

```python
from claude_agent_sdk import query, ClaudeAgentOptions

async for message in query(
    prompt="Hello",
    options=ClaudeAgentOptions(
        plugins=[
            {"type": "local", "path": "./my-plugin"},
            {"type": "local", "path": "/absolute/path/to/another-plugin"},
        ]
    ),
):
    pass
```

## Using Plugin Commands

Commands are namespaced: `plugin-name:command-name`

```python
async for message in query(
    prompt="/my-plugin:greet",
    options=ClaudeAgentOptions(
        plugins=[{"type": "local", "path": "./my-plugin"}]
    ),
):
    print(message)
```
