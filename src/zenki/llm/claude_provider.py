"""Claude (Anthropic) LLM provider for Zenki."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

from zenki.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from zenki.llm.config import MODEL_MAP, get_model_id

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

_DEFAULT_MODEL = "sonnet"


class ClaudeProvider(BaseLLMProvider):
    """LLM provider backed by the Anthropic Messages API.

    Parameters
    ----------
    api_key:
        Anthropic API key.  When *None* the provider falls back to the
        ``ANTHROPIC_API_KEY`` environment variable.
    default_model:
        Model shortname (``"haiku"``, ``"sonnet"``, or ``"opus"``) used when
        no explicit model is supplied to :meth:`send` / :meth:`stream`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = _DEFAULT_MODEL,
    ) -> None:
        if anthropic is None:
            raise ImportError(
                "The 'anthropic' package is required for ClaudeProvider. "
                "Install it with:  pip install 'zenki[claude]'  or  pip install anthropic"
            )

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = anthropic.AsyncAnthropic(api_key=resolved_key)
        self._default_model = default_model

    # --------------------------------------------------------------------- #
    # Helper to convert internal messages to the Anthropic API format
    # --------------------------------------------------------------------- #

    @staticmethod
    def _to_api_messages(messages: list[LLMMessage]) -> list[dict]:
        """Convert a list of :class:`LLMMessage` objects to Anthropic API dicts."""
        api_msgs: list[dict] = []
        for msg in messages:
            if msg.role == "system":
                # System messages are handled separately in the API call.
                continue
            entry: dict = {"role": msg.role, "content": msg.content}
            api_msgs.append(entry)
        return api_msgs

    # --------------------------------------------------------------------- #
    # Public interface
    # --------------------------------------------------------------------- #

    async def send(
        self,
        messages: list[LLMMessage],
        system: str = "",
        model: str | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send messages and return the full response."""
        model_id = get_model_id(model or self._default_model)
        api_messages = self._to_api_messages(messages)

        kwargs: dict = {
            "model": model_id,
            "max_tokens": 8192,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = await self._client.messages.create(**kwargs)

        # Extract text content from response blocks.
        content_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        return LLMResponse(
            content="\n".join(content_parts),
            model=response.model,
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "",
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        system: str = "",
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens as an async iterator."""
        model_id = get_model_id(model or self._default_model)
        api_messages = self._to_api_messages(messages)

        kwargs: dict = {
            "model": model_id,
            "max_tokens": 8192,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    def get_available_models(self) -> list[str]:
        """Return the list of supported model shortnames."""
        return list(MODEL_MAP.keys())
