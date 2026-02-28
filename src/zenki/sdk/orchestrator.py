"""Main Zenki orchestrator using the Claude Agent SDK.

The ``ZenkiOrchestrator`` replaces the previous custom agent loop with the
SDK's native ``ClaudeSDKClient``.  It wires together:

- **Agents**: Specialized subagents (software engineer, web researcher, etc.)
- **Tools**: Custom MCP tools (memory, scheduling, notifications)
- **Hooks**: Security, audit, and memory integration hooks
- **Skills**: Filesystem-based skills loaded via ``setting_sources``
- **Sessions**: Multi-turn conversations with full context retention

Reference: docs/claude-agent-sdk/python-reference.md
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncIterator

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    query,
)

from zenki.config.settings import ZenkiSettings
from zenki.core.prompts import build_system_prompt
from zenki.db.database import ZenkiDatabase
from zenki.sdk import _context
from zenki.sdk.agents import ZENKI_AGENTS, get_agents
from zenki.sdk.hooks import create_zenki_hooks
from zenki.sdk.tools import ZENKI_TOOL_NAMES, create_zenki_tools

logger = logging.getLogger(__name__)


class ZenkiOrchestrator:
    """SDK-native orchestrator for Zenki.

    Uses ``ClaudeSDKClient`` for multi-turn conversations with subagent
    delegation, custom tools, and lifecycle hooks.

    Parameters
    ----------
    settings:
        Zenki configuration (loaded from ``~/.config/zenki/config.json``).
    db:
        The database instance for storing conversations, memories, etc.
    memory_manager:
        Optional memory manager for the 4-tier memory system.
    scheduler:
        Optional task scheduler for background jobs.
    channel_registry:
        Optional channel registry for notifications.
    """

    def __init__(
        self,
        settings: ZenkiSettings,
        db: ZenkiDatabase,
        memory_manager: Any | None = None,
        scheduler: Any | None = None,
        channel_registry: Any | None = None,
    ) -> None:
        self.settings = settings
        self.db = db

        # Register services so custom tools can access them
        _context.set_database(db)
        if memory_manager:
            _context.set_memory_manager(memory_manager)
        if scheduler:
            _context.set_scheduler(scheduler)
        if channel_registry:
            _context.set_channel_registry(channel_registry)

        # Pre-build the options components
        self._agents = get_agents()
        self._mcp_servers = create_zenki_tools()
        self._hooks = create_zenki_hooks()
        self._system_prompt = self._build_system_prompt()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the Zenki system prompt from settings and memory."""
        personality = self.settings.personality.model_dump()
        base_prompt = build_system_prompt(personality=personality)

        # Append Zenki-specific context about available agents and tools
        agent_descriptions = "\n".join(
            f"- **{name}**: {agent.description}"
            for name, agent in self._agents.items()
        )

        zenki_context = f"""

## Available Specialist Agents
You can delegate tasks to these specialist agents using the Task tool:
{agent_descriptions}

## Zenki Custom Tools
You have access to Zenki-specific tools for memory management, scheduling,
and notifications. These are available as MCP tools with the `mcp__zenki__` prefix.
"""
        return base_prompt + zenki_context

    def _build_options(
        self,
        *,
        max_turns: int | None = None,
        cwd: str | Path | None = None,
        extra_agents: dict[str, AgentDefinition] | None = None,
        permission_mode: str = "acceptEdits",
    ) -> ClaudeAgentOptions:
        """Build ``ClaudeAgentOptions`` with all Zenki components wired in.

        Parameters
        ----------
        max_turns:
            Maximum agent loop iterations.
        cwd:
            Working directory for the agent.
        extra_agents:
            Additional agents to merge with built-in ones.
        permission_mode:
            SDK permission mode (default, acceptEdits, bypassPermissions).
        """
        agents = get_agents(extra_agents)

        # Built-in tools + Zenki MCP tools + Task for subagents + Skill for skills
        allowed_tools = [
            "Read", "Write", "Edit", "Bash", "Glob", "Grep",
            "WebSearch", "WebFetch",
            "Task",  # Required for subagent delegation
            "Skill",  # Required for skill invocation
            "AskUserQuestion",
        ] + ZENKI_TOOL_NAMES

        # Determine model from settings
        model_map = {
            "haiku": "haiku",
            "sonnet": "sonnet",
            "opus": "opus",
        }
        default_model = model_map.get(self.settings.llm.default_model, "sonnet")

        return ClaudeAgentOptions(
            system_prompt=self._system_prompt,
            allowed_tools=allowed_tools,
            permission_mode=permission_mode,
            cwd=str(cwd) if cwd else None,
            model=default_model,
            max_turns=max_turns,
            mcp_servers=self._mcp_servers,
            hooks=self._hooks,
            agents=agents,
            # Load skills from filesystem (project and user directories)
            setting_sources=["user", "project"],
        )

    # ------------------------------------------------------------------
    # Public API: One-shot query
    # ------------------------------------------------------------------

    async def run_query(
        self,
        prompt: str,
        *,
        max_turns: int | None = None,
        cwd: str | Path | None = None,
    ) -> str:
        """Run a one-shot query and return the final result.

        Creates a new session for each call. Best for independent tasks.

        Parameters
        ----------
        prompt:
            The user's message or task description.
        max_turns:
            Maximum agent loop iterations.
        cwd:
            Working directory for the agent.

        Returns
        -------
        str
            The agent's final response text.
        """
        options = self._build_options(max_turns=max_turns, cwd=cwd)
        result_text = ""

        async for message in query(prompt=prompt, options=options):
            if hasattr(message, "result") and message.result:
                result_text = message.result

        return result_text or "Task completed."

    # ------------------------------------------------------------------
    # Public API: Streaming query
    # ------------------------------------------------------------------

    async def run_query_streaming(
        self,
        prompt: str,
        *,
        max_turns: int | None = None,
        cwd: str | Path | None = None,
    ) -> AsyncIterator[str]:
        """Run a query and yield response text chunks as they arrive.

        Parameters
        ----------
        prompt:
            The user's message or task description.
        max_turns:
            Maximum agent loop iterations.
        cwd:
            Working directory for the agent.

        Yields
        ------
        str
            Text chunks from the agent's response.
        """
        options = self._build_options(max_turns=max_turns, cwd=cwd)

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage) and message.content:
                for block in message.content:
                    if isinstance(block, TextBlock):
                        yield block.text

    # ------------------------------------------------------------------
    # Public API: Multi-turn conversation
    # ------------------------------------------------------------------

    async def create_conversation(
        self,
        *,
        max_turns: int | None = None,
        cwd: str | Path | None = None,
    ) -> ZenkiConversation:
        """Create a new multi-turn conversation.

        Returns a ``ZenkiConversation`` that wraps ``ClaudeSDKClient`` for
        continuous context-preserving exchanges.

        Parameters
        ----------
        max_turns:
            Maximum agent loop iterations per query.
        cwd:
            Working directory for the agent.
        """
        options = self._build_options(max_turns=max_turns, cwd=cwd)
        return ZenkiConversation(options)


class ZenkiConversation:
    """A multi-turn conversation session using ``ClaudeSDKClient``.

    Maintains context across multiple exchanges so Claude remembers
    files read, analysis done, and previous conversation history.

    Usage
    -----
    ::

        conv = await orchestrator.create_conversation()
        async with conv:
            response = await conv.send("What files are in this project?")
            print(response)
            response = await conv.send("Now refactor the main module")
            print(response)
    """

    def __init__(self, options: ClaudeAgentOptions) -> None:
        self._options = options
        self._client: ClaudeSDKClient | None = None
        self._session_id: str | None = None

    async def __aenter__(self) -> ZenkiConversation:
        self._client = ClaudeSDKClient(options=self._options)
        await self._client.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._client:
            await self._client.disconnect()
            self._client = None

    async def send(self, message: str) -> str:
        """Send a message and return the response text.

        Parameters
        ----------
        message:
            The user's message.

        Returns
        -------
        str
            The agent's response text.
        """
        if self._client is None:
            raise RuntimeError("Conversation not started. Use 'async with' context manager.")

        await self._client.query(message)

        result_text = ""
        text_parts: list[str] = []

        async for msg in self._client.receive_response():
            # Capture session ID from result messages
            if hasattr(msg, "session_id") and msg.session_id:
                self._session_id = msg.session_id

            # Collect text from assistant messages
            if isinstance(msg, AssistantMessage) and msg.content:
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)

            # Capture the final result
            if isinstance(msg, ResultMessage) and hasattr(msg, "result") and msg.result:
                result_text = msg.result

        return result_text or "\n".join(text_parts) or "Done."

    async def send_streaming(self, message: str) -> AsyncIterator[str]:
        """Send a message and yield response chunks as they arrive.

        Parameters
        ----------
        message:
            The user's message.

        Yields
        ------
        str
            Text chunks from the agent's response.
        """
        if self._client is None:
            raise RuntimeError("Conversation not started. Use 'async with' context manager.")

        await self._client.query(message)

        async for msg in self._client.receive_response():
            if hasattr(msg, "session_id") and msg.session_id:
                self._session_id = msg.session_id

            if isinstance(msg, AssistantMessage) and msg.content:
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        yield block.text

    @property
    def session_id(self) -> str | None:
        """The current session ID (available after the first exchange)."""
        return self._session_id
