"""Main agent orchestrator for Zenki.

The :class:`ZenkiAgent` provides a high-level interface that delegates to the
Claude Agent SDK via :class:`~zenki.sdk.orchestrator.ZenkiOrchestrator`.

For simple, single-exchange tasks it wraps ``ZenkiOrchestrator.run_query()``.
For multi-turn conversations it provides a ``create_conversation()`` method
that returns a ``ZenkiConversation`` context manager.

The legacy ``process_message`` method is preserved for backward compatibility
with existing channel adapters, but internally delegates to the SDK.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from typing import TYPE_CHECKING

from zenki.config.settings import ZenkiSettings
from zenki.core.errors import SmartErrorHandler
from zenki.core.session import SessionManager
from zenki.db.database import ZenkiDatabase

if TYPE_CHECKING:
    from zenki.sdk.orchestrator import ZenkiConversation, ZenkiOrchestrator

logger = logging.getLogger(__name__)


class ZenkiAgent:
    """Top-level agent orchestrator.

    Now delegates to the Claude Agent SDK's native orchestration via
    :class:`ZenkiOrchestrator` for LLM calls, tool execution, subagent
    delegation, and hooks.  Session management and error handling remain
    as Zenki-specific concerns.

    Parameters
    ----------
    settings:
        Zenki configuration.
    db:
        Database instance.
    memory_manager:
        Optional memory manager (passed to SDK tools).
    scheduler:
        Optional task scheduler (passed to SDK tools).
    channel_registry:
        Optional channel registry (passed to SDK tools).
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
        self.session_manager = SessionManager(db)
        self.error_handler = SmartErrorHandler()

        # Lazy import to break circular dependency: core -> sdk -> core
        from zenki.sdk.orchestrator import ZenkiOrchestrator

        # The SDK-native orchestrator handles LLM calls, agents, tools, hooks
        self.orchestrator = ZenkiOrchestrator(
            settings=settings,
            db=db,
            memory_manager=memory_manager,
            scheduler=scheduler,
            channel_registry=channel_registry,
        )

    # ------------------------------------------------------------------
    # Public API
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

        This method preserves backward compatibility with existing channel
        adapters.  Internally it delegates to the Claude Agent SDK via
        ``ZenkiOrchestrator.run_query()``.

        Steps
        -----
        1. Resolve or create a session.
        2. Store the user message in the database.
        3. Delegate to the SDK orchestrator (which handles the full agent
           loop: prompt building, model routing, tool execution, subagent
           delegation, and hooks).
        4. Store the assistant response in the database.
        5. Return the response text.
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

        # 2. Store the user message.
        self.session_manager.add_message(
            session_id=session.id,
            role="user",
            content=message,
        )

        # 3. Delegate to the SDK orchestrator.
        try:
            response_text = await self.orchestrator.run_query(message)
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

        # 5. Return the response content.
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

        full_response: list[str] = []

        async for chunk in self.orchestrator.run_query_streaming(message):
            full_response.append(chunk)
            yield chunk

        # Store the complete response
        response_text = "\n".join(full_response) if full_response else "Done."
        self.session_manager.add_message(
            session_id=session.id,
            role="assistant",
            content=response_text,
        )

    async def create_conversation(self) -> Any:
        """Create a new multi-turn conversation.

        Returns a ``ZenkiConversation`` for use as an async context manager.

        Example
        -------
        ::

            conv = await agent.create_conversation()
            async with conv:
                answer = await conv.send("What's in this repo?")
                followup = await conv.send("Explain the main module")
        """
        return await self.orchestrator.create_conversation()

    async def close_session(self, session_id: str) -> None:
        """Close an active session."""
        self.session_manager.close_session(session_id)
