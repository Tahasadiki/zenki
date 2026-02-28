"""Abstract LLM provider interface for Zenki."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class LLMMessage:
    """A single message in a conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    tool_calls: list[dict] | None = None
    tool_results: list[dict] | None = None


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    stop_reason: str = ""


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def send(
        self,
        messages: list[LLMMessage],
        system: str = "",
        model: str | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send a list of messages and return a complete response."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        system: str = "",
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a response token by token."""
        ...

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Return a list of available model shortnames."""
        ...
