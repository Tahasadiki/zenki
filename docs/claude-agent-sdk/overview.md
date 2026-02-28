# Claude Agent SDK - Overview

> Reference: https://platform.claude.com/docs/en/agent-sdk/overview

## Installation

```bash
pip install claude-agent-sdk
```

## Core Concepts

The Claude Agent SDK (formerly Claude Code SDK) exposes the same infrastructure powering
Claude Code as a programmable library for building autonomous AI agents.

### Two Interaction Models

| Feature             | `query()`                     | `ClaudeSDKClient`                  |
| :------------------ | :---------------------------- | :--------------------------------- |
| **Session**         | Creates new session each time | Reuses same session                |
| **Conversation**    | Single exchange               | Multiple exchanges in same context |
| **Connection**      | Managed automatically         | Manual control                     |
| **Streaming Input** | Supported                     | Supported                          |
| **Interrupts**      | Not supported                 | Supported                          |
| **Hooks**           | Supported                     | Supported                          |
| **Custom Tools**    | Supported                     | Supported                          |
| **Continue Chat**   | New session each time         | Maintains conversation             |

### Built-in Tools

| Tool | What it does |
|------|--------------|
| **Read** | Read any file in the working directory |
| **Write** | Create new files |
| **Edit** | Make precise edits to existing files |
| **Bash** | Run terminal commands, scripts, git operations |
| **Glob** | Find files by pattern |
| **Grep** | Search file contents with regex |
| **WebSearch** | Search the web for current information |
| **WebFetch** | Fetch and parse web page content |
| **Task** | Invoke subagents |
| **Skill** | Invoke filesystem-based skills |
| **AskUserQuestion** | Ask the user clarifying questions |

### Capabilities

1. **Subagents** - Define specialized agents via `AgentDefinition`
2. **Custom Tools** - `@tool` decorator + `create_sdk_mcp_server()`
3. **Hooks** - Callback functions at key execution points
4. **Skills** - Filesystem-based SKILL.md files (auto-discovered)
5. **MCP** - Connect to external MCP servers
6. **Plugins** - Extend with commands, agents, skills, hooks
7. **Sessions** - Maintain context across exchanges
8. **Permissions** - Control tool access per agent

### Filesystem-based Configuration

| Feature | Description | Location |
|---------|-------------|----------|
| Skills | Specialized capabilities in Markdown | `.claude/skills/SKILL.md` |
| Slash commands | Custom commands for common tasks | `.claude/commands/*.md` |
| Memory | Project context and instructions | `CLAUDE.md` |
| Plugins | Custom commands, agents, MCP servers | Programmatic via `plugins` option |
