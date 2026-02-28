"""Main Zenki orchestrator using the Claude Agent SDK.

The ``ZenkiOrchestrator`` is the single entry-point for all Zenki operations.
It wires together:

- **Agents**: Specialized subagents (software engineer, web researcher, etc.)
- **Tools**: Custom MCP tools (memory, scheduling, notifications)
- **Hooks**: Security, audit, and memory integration hooks
- **Skills**: Filesystem-based skills loaded via ``setting_sources``
- **Sessions**: Multi-turn conversations with full context retention
- **Session management**: Create/resume sessions and persist messages
- **Model routing**: Dynamic per-query model selection based on complexity
- **Error handling**: Smart retry/escalation for transient failures

Reference: docs/claude-agent-sdk/python-reference.md
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

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
from zenki.core.errors import SmartErrorHandler
from zenki.core.prompts import build_system_prompt
from zenki.core.session import SessionManager
from zenki.db.database import ZenkiDatabase
from zenki.sdk import _context
from zenki.sdk.agents import get_agents
from zenki.sdk.hooks import create_zenki_hooks
from zenki.sdk.router import ModelRouter
from zenki.sdk.tools import ZENKI_TOOL_NAMES, create_zenki_tools

logger = logging.getLogger(__name__)


class ZenkiOrchestrator:
    """SDK-native orchestrator for Zenki.

    Uses ``ClaudeSDKClient`` for multi-turn conversations with subagent
    delegation, custom tools, and lifecycle hooks.  Also manages sessions,
    persists messages, and routes queries to the right model.

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

        # Session management and error handling
        self.session_manager = SessionManager(db)
        self.error_handler = SmartErrorHandler()
        self._router = ModelRouter()

        # Keep a direct reference for memory-aware prompts
        self._memory_manager = memory_manager

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

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        query: str | None = None,
        conversation_history: str | None = None,
    ) -> str:
        """Build the Zenki system prompt from settings, memory, and history.

        Parameters
        ----------
        query:
            The current user message. When provided (and a memory manager is
            available), relevant memories are retrieved and injected into
            the prompt.
        conversation_history:
            Formatted prior conversation to include for session resumption.
        """
        personality = self.settings.personality.model_dump()

        core_memory: dict[str, str] | None = None
        retrieved_memories: dict[str, str] | None = None

        if self._memory_manager and query:
            try:
                core_memory = self._memory_manager.core.load_all()

                context = self._memory_manager.build_context(query)

                episodic_lines = [
                    f"- {entry['summary']} (topics: {', '.join(entry.get('topics', []))})"
                    for entry in context.get("episodic", [])
                ]
                semantic_lines = [
                    f"- [{entry['category']}] {entry['content']} (tags: {', '.join(entry.get('tags', []))})"
                    for entry in context.get("semantic", [])
                ]
                retrieved_memories = {
                    "episodic": "\n".join(episodic_lines) if episodic_lines else "",
                    "semantic": "\n".join(semantic_lines) if semantic_lines else "",
                }
            except Exception:
                logger.warning("Memory retrieval failed; proceeding without memory context", exc_info=True)

        base_prompt = build_system_prompt(
            personality=personality,
            core_memory=core_memory,
            retrieved_memories=retrieved_memories,
            conversation_history=conversation_history,
        )

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
        model_override: str | None = None,
        query: str | None = None,
        conversation_history: str | None = None,
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
        model_override:
            Override the default model (e.g. from the model router).
        query:
            The current user message for memory-aware prompt building.
        conversation_history:
            Formatted prior conversation for session resumption.
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

        # Use model override (from router) or fall back to settings default
        model = model_override or self.settings.llm.default_model

        system_prompt = self._build_system_prompt(
            query=query,
            conversation_history=conversation_history,
        )

        return ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            permission_mode=permission_mode,
            cwd=str(cwd) if cwd else None,
            model=model,
            max_turns=max_turns,
            mcp_servers=self._mcp_servers,
            hooks=self._hooks,
            agents=agents,
            # Load skills from filesystem (project and user directories)
            setting_sources=["user", "project"],
        )

    # ------------------------------------------------------------------
    # Public API: One-shot query (low-level)
    # ------------------------------------------------------------------

    async def run_query(
        self,
        prompt: str,
        *,
        max_turns: int | None = None,
        cwd: str | Path | None = None,
        model_override: str | None = None,
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
        model_override:
            Override the default model for this query.

        Returns
        -------
        str
            The agent's final response text.
        """
        options = self._build_options(
            max_turns=max_turns, cwd=cwd, model_override=model_override,
            query=prompt,
        )
        result_text = ""

        async for message in query(prompt=prompt, options=options):
            if hasattr(message, "result") and message.result:
                result_text = message.result

        return result_text or "Task completed."

    # ------------------------------------------------------------------
    # Public API: Streaming query (low-level)
    # ------------------------------------------------------------------

    async def run_query_streaming(
        self,
        prompt: str,
        *,
        max_turns: int | None = None,
        cwd: str | Path | None = None,
        model_override: str | None = None,
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
        model_override:
            Override the default model for this query.

        Yields
        ------
        str
            Text chunks from the agent's response.
        """
        options = self._build_options(
            max_turns=max_turns, cwd=cwd, model_override=model_override,
            query=prompt,
        )

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
        initial_query: str | None = None,
        conversation_history: str | None = None,
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
        initial_query:
            Used for memory retrieval in the initial system prompt.
        conversation_history:
            Formatted prior conversation for session resumption.
        """
        options = self._build_options(
            max_turns=max_turns,
            cwd=cwd,
            query=initial_query,
            conversation_history=conversation_history,
        )
        return ZenkiConversation(options)

    # ------------------------------------------------------------------
    # Public API: Session-aware message processing (high-level)
    # ------------------------------------------------------------------

    async def process_message(
        self,
        message: str,
        session_id: str | None = None,
        user_id: str = "default",
        channel_type: str = "cli",
        channel_id: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        """Process an incoming user message and return the assistant's reply.

        This is the primary API for channel adapters. It handles:
        1. Session resolution/creation
        2. Message persistence
        3. Dynamic model routing based on message complexity
        4. SDK delegation
        5. Error handling with smart retry/escalation

        Parameters
        ----------
        message:
            The user's message text.
        session_id:
            Existing session to continue, or ``None`` to auto-resolve.
        user_id:
            The user identifier.
        channel_type:
            Channel type (``"cli"``, ``"slack"``, etc.).
        channel_id:
            Channel-specific identifier.
        thread_id:
            Thread within the channel.

        Returns
        -------
        str
            The assistant's response text.
        """
        # 1. Resolve session.
        if session_id is not None:
            session = self.session_manager.get_session(session_id)
            if session is None:
                session = self.session_manager.create_session(
                    user_id=user_id,
                    channel_type=channel_type,
                    channel_id=channel_id,
                    thread_id=thread_id,
                )
        else:
            session = self.session_manager.get_or_create_session(
                user_id=user_id,
                channel_type=channel_type,
                channel_id=channel_id,
                thread_id=thread_id,
            )

        # 2. Store user message.
        self.session_manager.add_message(
            session_id=session.id,
            role="user",
            content=message,
        )

        # 3. Route to appropriate model and delegate to SDK.
        model = self._router.classify_complexity(message)

        try:
            response_text = await self.run_query(message, model_override=model)
        except Exception as exc:
            logger.exception("SDK agent call failed")
            result = await self.error_handler.handle(exc)
            error_text = result.user_message or f"I encountered an error: {exc}"
            self.session_manager.add_message(
                session_id=session.id,
                role="assistant",
                content=error_text,
            )
            return error_text

        # 4. Store assistant response.
        self.session_manager.add_message(
            session_id=session.id,
            role="assistant",
            content=response_text,
        )

        return response_text

    async def process_message_streaming(
        self,
        message: str,
        session_id: str | None = None,
        user_id: str = "default",
        channel_type: str = "cli",
        channel_id: str | None = None,
        thread_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Process a message and yield response chunks as they arrive.

        Same as ``process_message`` but streams the response for real-time
        display in chat interfaces.
        """
        # Resolve session
        if session_id is not None:
            session = self.session_manager.get_session(session_id)
            if session is None:
                session = self.session_manager.create_session(
                    user_id=user_id,
                    channel_type=channel_type,
                    channel_id=channel_id,
                    thread_id=thread_id,
                )
        else:
            session = self.session_manager.get_or_create_session(
                user_id=user_id,
                channel_type=channel_type,
                channel_id=channel_id,
                thread_id=thread_id,
            )

        self.session_manager.add_message(
            session_id=session.id,
            role="user",
            content=message,
        )

        model = self._router.classify_complexity(message)
        full_response: list[str] = []

        async for chunk in self.run_query_streaming(message, model_override=model):
            full_response.append(chunk)
            yield chunk

        # Store the complete response
        response_text = "\n".join(full_response) if full_response else "Done."
        self.session_manager.add_message(
            session_id=session.id,
            role="assistant",
            content=response_text,
        )

    async def close_session(self, session_id: str) -> None:
        """Close an active session."""
        self.session_manager.close_session(session_id)


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
        """Send a message and return the response text."""
        if self._client is None:
            raise RuntimeError("Conversation not started. Use 'async with' context manager.")

        await self._client.query(message)

        result_text = ""
        text_parts: list[str] = []

        async for msg in self._client.receive_response():
            if hasattr(msg, "session_id") and msg.session_id:
                self._session_id = msg.session_id

            if isinstance(msg, AssistantMessage) and msg.content:
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)

            if isinstance(msg, ResultMessage) and hasattr(msg, "result") and msg.result:
                result_text = msg.result

        return result_text or "\n".join(text_parts) or "Done."

    async def send_streaming(self, message: str) -> AsyncIterator[str]:
        """Send a message and yield response chunks as they arrive."""
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
