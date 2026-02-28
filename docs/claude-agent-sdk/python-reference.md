# Claude Agent SDK - Python Reference

> Reference: https://platform.claude.com/docs/en/agent-sdk/python

## Functions

### query()

```python
async def query(
    *,
    prompt: str | AsyncIterable[dict[str, Any]],
    options: ClaudeAgentOptions | None = None,
    transport: Transport | None = None
) -> AsyncIterator[Message]
```

### tool()

```python
def tool(
    name: str,
    description: str,
    input_schema: type | dict[str, Any],
    annotations: ToolAnnotations | None = None
) -> Callable[[Callable[[Any], Awaitable[dict[str, Any]]]], SdkMcpTool[Any]]
```

### create_sdk_mcp_server()

```python
def create_sdk_mcp_server(
    name: str,
    version: str = "1.0.0",
    tools: list[SdkMcpTool[Any]] | None = None
) -> McpSdkServerConfig
```

## Classes

### ClaudeSDKClient

```python
class ClaudeSDKClient:
    def __init__(self, options: ClaudeAgentOptions | None = None, transport: Transport | None = None)
    async def connect(self, prompt: str | AsyncIterable[dict] | None = None) -> None
    async def query(self, prompt: str | AsyncIterable[dict], session_id: str = "default") -> None
    async def receive_messages(self) -> AsyncIterator[Message]
    async def receive_response(self) -> AsyncIterator[Message]
    async def interrupt(self) -> None
    async def set_permission_mode(self, mode: str) -> None
    async def set_model(self, model: str | None = None) -> None
    async def get_mcp_status(self) -> dict[str, Any]
    async def get_server_info(self) -> dict[str, Any] | None
    async def disconnect(self) -> None
```

### ClaudeAgentOptions

Key fields:
- `prompt` / `system_prompt` - System prompt override
- `allowed_tools` - List of tool names agent can use
- `permission_mode` - "default" | "acceptEdits" | "bypassPermissions"
- `cwd` - Working directory
- `model` - Default model ("sonnet", "opus", "haiku")
- `fallback_model` - Fallback when primary unavailable
- `max_turns` - Maximum agent loop iterations
- `max_budget_usd` - Spending limit
- `mcp_servers` - Dict of MCP server configs
- `hooks` - Dict of hook event -> HookMatcher arrays
- `agents` - Dict of agent name -> AgentDefinition
- `setting_sources` - ["user", "project"] to load filesystem config
- `plugins` - List of plugin configs
- `resume` - Session ID to resume
- `output_format` - "text" | "json" | "stream-json"

### AgentDefinition

```python
@dataclass
class AgentDefinition:
    description: str       # When to use this agent
    prompt: str           # System prompt for the agent
    tools: list[str] | None = None  # Allowed tools (None = inherit all)
    model: str | None = None        # Model override
```

### HookMatcher

```python
@dataclass
class HookMatcher:
    hooks: list[HookCallback]      # Callback functions
    matcher: str | None = None     # Regex pattern for tool names
    timeout: int = 60              # Timeout in seconds
```

### SdkMcpTool

```python
@dataclass
class SdkMcpTool(Generic[T]):
    name: str
    description: str
    input_schema: type[T] | dict[str, Any]
    handler: Callable[[T], Awaitable[dict[str, Any]]]
    annotations: ToolAnnotations | None = None
```

## Message Types

- `AssistantMessage` - Claude's text responses
- `ResultMessage` - Final result with `result` field
- `TextBlock` - Text content block in messages

## Key Imports

```python
from claude_agent_sdk import (
    query,
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AgentDefinition,
    HookMatcher,
    tool,
    create_sdk_mcp_server,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)
```
