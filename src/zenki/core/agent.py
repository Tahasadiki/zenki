"""Main agent orchestrator for Zenki.

The :class:`ZenkiAgent` ties together session management, model routing,
LLM interaction, error handling, and prompt construction into a single
high-level ``process_message`` entry point.
"""

from __future__ import annotations

import logging

from zenki.config.settings import ZenkiSettings
from zenki.core.errors import SmartErrorHandler
from zenki.core.prompts import build_system_prompt
from zenki.core.session import SessionManager
from zenki.db.database import ZenkiDatabase
from zenki.llm.base import BaseLLMProvider, LLMMessage
from zenki.llm.router import ModelRouter

logger = logging.getLogger(__name__)


class ZenkiAgent:
    """Top-level agent orchestrator.

    Coordinates session management, prompt building, model routing,
    LLM calls, and error handling for each user message.
    """

    def __init__(
        self,
        settings: ZenkiSettings,
        db: ZenkiDatabase,
        llm_provider: BaseLLMProvider,
    ) -> None:
        self.settings = settings
        self.db = db
        self.llm_provider = llm_provider
        self.session_manager = SessionManager(db)
        self.router = ModelRouter()
        self.error_handler = SmartErrorHandler()

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

        Steps
        -----
        1. Resolve or create a session.
        2. Store the user message in the database.
        3. Build the system prompt from personality settings.
        4. Collect conversation history from the session.
        5. Route the message to an appropriate model.
        6. Send the conversation to the LLM provider.
        7. Store the assistant response in the database.
        8. Return the response text.

        Any LLM errors are wrapped by :class:`SmartErrorHandler` so that
        the caller always receives a meaningful string.
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

        # 3. Build system prompt.
        personality = self.settings.personality.model_dump()
        system_prompt = build_system_prompt(personality=personality)

        # 4. Collect conversation history.
        history = self.session_manager.get_messages(session.id)
        llm_messages: list[LLMMessage] = [
            LLMMessage(role=msg.role, content=msg.content)
            for msg in history
        ]

        # 5. Route to model.
        model = self.router.route(message)

        # 6. Call the LLM provider (with error handling).
        try:
            response = await self.llm_provider.send(
                messages=llm_messages,
                system=system_prompt,
                model=model,
            )
        except Exception as exc:
            logger.exception("LLM call failed")
            result = await self.error_handler.handle(exc)
            error_text = result.user_message or f"I encountered an error: {exc}"
            # Store the error as an assistant message so the session log is complete.
            self.session_manager.add_message(
                session_id=session.id,
                role="assistant",
                content=error_text,
            )
            return error_text

        # 7. Store assistant response.
        self.session_manager.add_message(
            session_id=session.id,
            role="assistant",
            content=response.content,
            model_used=response.model,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            tool_calls=response.tool_calls if response.tool_calls else None,
        )

        # 8. Return the response content.
        return response.content

    async def close_session(self, session_id: str) -> None:
        """Close an active session."""
        self.session_manager.close_session(session_id)
