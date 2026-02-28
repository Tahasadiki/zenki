"""Tier 1: Working Memory - active conversation context management."""

from __future__ import annotations


class WorkingMemory:
    """Manages the active conversation context within a single session.

    Working memory acts as the short-term buffer for the current conversation,
    keeping track of messages and enforcing a token budget so that the context
    window is not exceeded.
    """

    def __init__(self, max_tokens: int = 60000) -> None:
        self.max_tokens = max_tokens
        self._messages: list[dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """Append a message to working memory.

        Args:
            role: The message role (e.g. ``"user"``, ``"assistant"``).
            content: The message text.
        """
        self._messages.append({"role": role, "content": content})

    def get_context(self) -> list[dict[str, str]]:
        """Return messages that fit within the token budget.

        Messages are returned in chronological order. If the total token
        count exceeds :attr:`max_tokens`, older messages are dropped from
        the front until the budget is satisfied.

        Returns:
            A list of message dicts (``{"role": ..., "content": ...}``).
        """
        total_tokens = 0
        result: list[dict[str, str]] = []

        # Walk backwards from the most recent message, accumulating tokens
        for msg in reversed(self._messages):
            msg_tokens = self.estimate_tokens(msg["content"])
            if total_tokens + msg_tokens > self.max_tokens:
                break
            total_tokens += msg_tokens
            result.append(msg)

        # Reverse to restore chronological order
        result.reverse()
        return result

    def clear(self) -> None:
        """Remove all messages from working memory."""
        self._messages.clear()

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate for a piece of text.

        Uses the heuristic of ``len(text) / 4``, which is a reasonable
        approximation for English text with typical LLM tokenisers.

        Args:
            text: The text to estimate.

        Returns:
            Estimated token count (minimum 1).
        """
        return max(1, len(text) // 4)

    def get_summary(self) -> str:
        """Return a brief summary of the conversation so far.

        Produces a simple textual overview by listing roles and truncated
        content for each message. For a production system this would be
        replaced with an LLM-generated summary.

        Returns:
            A string summarising the conversation.
        """
        if not self._messages:
            return "No conversation yet."

        lines: list[str] = []
        for msg in self._messages:
            content = msg["content"]
            truncated = content[:100] + "..." if len(content) > 100 else content
            lines.append(f"{msg['role']}: {truncated}")

        return f"Conversation ({len(self._messages)} messages):\n" + "\n".join(lines)

    @property
    def message_count(self) -> int:
        """Return the number of messages currently stored."""
        return len(self._messages)

    @property
    def total_tokens(self) -> int:
        """Return estimated total tokens across all stored messages."""
        return sum(self.estimate_tokens(m["content"]) for m in self._messages)
