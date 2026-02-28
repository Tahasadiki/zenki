"""Smart model routing for Zenki.

Classifies user messages by complexity and selects an appropriate model
shortname so that simple interactions use a cheaper/faster model while
complex reasoning tasks are routed to a more capable one.
"""

from __future__ import annotations

import re

from zenki.llm.config import get_model_id

# -------------------------------------------------------------------------
# Pattern definitions
# -------------------------------------------------------------------------

# Patterns that indicate a *simple* request best served by haiku.
_HAIKU_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*(hi|hello|hey|yo|sup|greetings|good\s*(morning|afternoon|evening))\b", re.I),
    re.compile(r"^\s*(thanks|thank\s*you|thx|ty)\b", re.I),
    re.compile(r"^\s*(ok|okay|sure|got\s*it|sounds\s*good|great|cool|nice)\s*[.!]?\s*$", re.I),
    re.compile(r"^\s*(yes|no|yep|nope|yeah|nah)\s*[.!]?\s*$", re.I),
    re.compile(r"^\s*what\s*(is|'s)\s*(the\s+)?(time|date|day)\s*\??\s*$", re.I),
    re.compile(r"^\s*how\s+are\s+you\b", re.I),
    re.compile(r"^\s*(status|ping|health)\s*\??\s*$", re.I),
    re.compile(r"^\s*format\s+(this|that|it)\b", re.I),
]

# Keyword / phrases that indicate a *complex* request best served by opus.
_OPUS_KEYWORDS: list[str] = [
    "write code",
    "generate code",
    "implement",
    "refactor",
    "debug",
    "fix the bug",
    "fix this bug",
    "architecture",
    "design pattern",
    "complex analysis",
    "analyze the",
    "deep dive",
    "skill generation",
    "generate a skill",
    "create a skill",
    "memory consolidation",
    "consolidate memories",
    "explain in detail",
    "step by step",
    "write a function",
    "write a class",
    "write a module",
    "code review",
    "security audit",
    "performance optimization",
    "optimize",
    "algorithm",
    "data structure",
    "system design",
    "technical design",
]

_OPUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(write|create|build|implement|generate)\b.*\b(function|class|module|api|service)\b",
        re.I,
    ),
    re.compile(r"\b(debug|fix|troubleshoot)\b.*\b(error|bug|issue|crash|exception)\b", re.I),
    re.compile(r"\b(architect|design)\b.*\b(system|application|service)\b", re.I),
    re.compile(r"\bcode\s*(review|generation|gen)\b", re.I),
]


class ModelRouter:
    """Routes messages to the most appropriate model based on complexity."""

    def classify_complexity(
        self,
        message: str,
        context: dict | None = None,
    ) -> str:
        """Classify *message* complexity and return a model shortname.

        Parameters
        ----------
        message:
            The user's message text.
        context:
            Optional contextual hints (e.g. conversation length, active tools).

        Returns
        -------
        str
            One of ``"haiku"``, ``"sonnet"``, or ``"opus"``.
        """
        # Check for simple / haiku patterns first.
        for pattern in _HAIKU_PATTERNS:
            if pattern.search(message):
                return "haiku"

        # Check for complex / opus indicators.
        lower_msg = message.lower()
        for keyword in _OPUS_KEYWORDS:
            if keyword in lower_msg:
                return "opus"
        for pattern in _OPUS_PATTERNS:
            if pattern.search(message):
                return "opus"

        # Default to sonnet for everything else.
        return "sonnet"

    def route(
        self,
        message: str,
        context: dict | None = None,
    ) -> str:
        """Return the full model ID for the given message.

        This is a convenience wrapper that classifies the message and then
        resolves the shortname to a full model identifier via
        :func:`~zenki.llm.config.get_model_id`.

        Parameters
        ----------
        message:
            The user's message text.
        context:
            Optional contextual hints.

        Returns
        -------
        str
            A full Anthropic model identifier (e.g. ``"claude-sonnet-4-6"``).
        """
        shortname = self.classify_complexity(message, context=context)
        return get_model_id(shortname)
