# Claude Agent SDK - Custom Tools

> Reference: https://platform.claude.com/docs/en/agent-sdk/custom-tools

## Overview

Custom tools extend Claude's capabilities through in-process MCP servers.
Define tools with the `@tool` decorator and bundle them in an MCP server.

## Creating Custom Tools

### @tool Decorator

```python
from claude_agent_sdk import tool
from typing import Any

@tool(
    "tool_name",           # Unique identifier
    "Tool description",    # What it does
    {"param": str},        # Input schema (simple types or JSON Schema)
)
async def my_tool(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [
            {"type": "text", "text": f"Result: {args['param']}"}
        ]
    }
```

### Input Schema Options

1. **Simple type mapping** (recommended):
   ```python
   {"text": str, "count": int, "enabled": bool}
   ```

2. **JSON Schema** (complex validation):
   ```python
   {
       "type": "object",
       "properties": {
           "text": {"type": "string"},
           "count": {"type": "integer", "minimum": 0},
       },
       "required": ["text"],
   }
   ```

### create_sdk_mcp_server()

Bundle tools into an MCP server:

```python
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool("add", "Add two numbers", {"a": float, "b": float})
async def add(args):
    return {"content": [{"type": "text", "text": f"Sum: {args['a'] + args['b']}"}]}

@tool("multiply", "Multiply two numbers", {"a": float, "b": float})
async def multiply(args):
    return {"content": [{"type": "text", "text": f"Product: {args['a'] * args['b']}"}]}

calculator = create_sdk_mcp_server(
    name="calculator",
    version="2.0.0",
    tools=[add, multiply],
)
```

## Using Custom Tools

### Tool Naming Convention

MCP tools follow the pattern: `mcp__{server_name}__{tool_name}`

Example: `mcp__calculator__add`

### Configuration

```python
options = ClaudeAgentOptions(
    mcp_servers={"calculator": calculator},
    allowed_tools=[
        "mcp__calculator__add",
        "mcp__calculator__multiply",
    ],
)
```

### With ClaudeSDKClient

```python
async def main():
    async with ClaudeSDKClient(options=options) as client:
        await client.query("What is 5 + 3?")
        async for msg in client.receive_response():
            print(msg)
```

## Error Handling

```python
@tool("fetch_data", "Fetch data from API", {"endpoint": str})
async def fetch_data(args: dict[str, Any]) -> dict[str, Any]:
    try:
        # ... implementation
        return {"content": [{"type": "text", "text": json.dumps(data)}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {str(e)}"}]}
```
