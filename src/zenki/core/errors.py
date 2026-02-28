"""Smart error escalation system for Zenki.

Classifies errors as known (auto-recoverable) or novel (escalate to user).
Known patterns are retried with exponential backoff or handled via fallback strategies.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ESCALATE = "escalate"


@dataclass
class ErrorPattern:
    """A known error pattern with a recovery strategy."""

    name: str
    match_fn: Callable[[Exception], bool]
    strategy: RecoveryStrategy
    max_retries: int = 3
    base_delay: float = 1.0
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    fallback_fn: Callable[..., Awaitable[Any]] | None = None
    message_template: str = ""


@dataclass
class ErrorResult:
    """Result of error handling."""

    recovered: bool
    strategy_used: RecoveryStrategy
    error: Exception
    attempts: int = 1
    user_message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class SmartErrorHandler:
    """Handles errors with smart escalation.

    Known error patterns are auto-recovered. Novel errors are escalated to the user.
    """

    def __init__(self) -> None:
        self._patterns: list[ErrorPattern] = []
        self._register_default_patterns()

    def _register_default_patterns(self) -> None:
        """Register built-in error patterns."""
        self.register_pattern(ErrorPattern(
            name="rate_limit",
            match_fn=lambda e: "rate" in str(e).lower() and "limit" in str(e).lower(),
            strategy=RecoveryStrategy.RETRY,
            max_retries=5,
            base_delay=2.0,
            severity=ErrorSeverity.LOW,
            message_template="Rate limited. Retrying in {delay}s...",
        ))
        self.register_pattern(ErrorPattern(
            name="network_timeout",
            match_fn=lambda e: any(
                kw in str(e).lower()
                for kw in ["timeout", "timed out", "connection reset", "connection refused"]
            ),
            strategy=RecoveryStrategy.RETRY,
            max_retries=3,
            base_delay=1.0,
            severity=ErrorSeverity.MEDIUM,
            message_template="Network error. Retrying in {delay}s...",
        ))
        self.register_pattern(ErrorPattern(
            name="file_not_found",
            match_fn=lambda e: isinstance(e, FileNotFoundError),
            strategy=RecoveryStrategy.ESCALATE,
            severity=ErrorSeverity.MEDIUM,
            message_template="File not found: {error}. Would you like me to search for it?",
        ))
        self.register_pattern(ErrorPattern(
            name="permission_denied",
            match_fn=lambda e: isinstance(e, PermissionError),
            strategy=RecoveryStrategy.ESCALATE,
            severity=ErrorSeverity.HIGH,
            message_template="Permission denied: {error}. Please check file permissions.",
        ))
        self.register_pattern(ErrorPattern(
            name="api_auth_error",
            match_fn=lambda e: any(
                kw in str(e).lower()
                for kw in ["unauthorized", "authentication", "invalid api key", "401"]
            ),
            strategy=RecoveryStrategy.ESCALATE,
            severity=ErrorSeverity.CRITICAL,
            message_template="Authentication failed. Please check your API key configuration.",
        ))

    def register_pattern(self, pattern: ErrorPattern) -> None:
        """Register a new error pattern."""
        self._patterns.append(pattern)

    def _find_pattern(self, error: Exception) -> ErrorPattern | None:
        """Find a matching error pattern."""
        for pattern in self._patterns:
            try:
                if pattern.match_fn(error):
                    return pattern
            except Exception:
                continue
        return None

    async def handle(
        self,
        error: Exception,
        retry_fn: Callable[..., Awaitable[Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ErrorResult:
        """Handle an error using smart escalation.

        Args:
            error: The exception to handle.
            retry_fn: Async function to retry if strategy is RETRY.
            context: Additional context for error handling.

        Returns:
            ErrorResult with recovery status and details.
        """
        pattern = self._find_pattern(error)

        if pattern is None:
            logger.warning("Novel error encountered: %s", error)
            return ErrorResult(
                recovered=False,
                strategy_used=RecoveryStrategy.ESCALATE,
                error=error,
                user_message=f"I encountered an unexpected error: {error}. How should I proceed?",
            )

        logger.info("Matched error pattern: %s (strategy: %s)", pattern.name, pattern.strategy)

        if pattern.strategy == RecoveryStrategy.RETRY and retry_fn:
            return await self._retry(error, pattern, retry_fn)
        elif pattern.strategy == RecoveryStrategy.FALLBACK and pattern.fallback_fn:
            return await self._fallback(error, pattern)
        elif pattern.strategy == RecoveryStrategy.SKIP:
            return ErrorResult(
                recovered=True,
                strategy_used=RecoveryStrategy.SKIP,
                error=error,
                user_message=pattern.message_template.format(error=error),
            )
        else:
            msg = pattern.message_template.format(error=error, delay=0)
            return ErrorResult(
                recovered=False,
                strategy_used=RecoveryStrategy.ESCALATE,
                error=error,
                user_message=msg,
            )

    async def _retry(
        self,
        original_error: Exception,
        pattern: ErrorPattern,
        retry_fn: Callable[..., Awaitable[Any]],
    ) -> ErrorResult:
        """Retry with exponential backoff."""
        for attempt in range(1, pattern.max_retries + 1):
            delay = pattern.base_delay * (2 ** (attempt - 1))
            logger.info(
                "Retry attempt %d/%d for %s (delay: %.1fs)",
                attempt, pattern.max_retries, pattern.name, delay,
            )
            await asyncio.sleep(delay)
            try:
                await retry_fn()
                return ErrorResult(
                    recovered=True,
                    strategy_used=RecoveryStrategy.RETRY,
                    error=original_error,
                    attempts=attempt + 1,
                    user_message=f"Recovered after {attempt + 1} attempts.",
                )
            except Exception as e:
                if not pattern.match_fn(e):
                    return ErrorResult(
                        recovered=False,
                        strategy_used=RecoveryStrategy.ESCALATE,
                        error=e,
                        attempts=attempt + 1,
                        user_message=f"Error changed during retry: {e}",
                    )

        return ErrorResult(
            recovered=False,
            strategy_used=RecoveryStrategy.RETRY,
            error=original_error,
            attempts=pattern.max_retries + 1,
            user_message=f"Failed after {pattern.max_retries + 1} attempts: {original_error}",
        )

    async def _fallback(
        self,
        error: Exception,
        pattern: ErrorPattern,
    ) -> ErrorResult:
        """Execute fallback strategy."""
        try:
            await pattern.fallback_fn()  # type: ignore[misc]
            return ErrorResult(
                recovered=True,
                strategy_used=RecoveryStrategy.FALLBACK,
                error=error,
                user_message="Recovered using fallback strategy.",
            )
        except Exception as fallback_error:
            return ErrorResult(
                recovered=False,
                strategy_used=RecoveryStrategy.ESCALATE,
                error=fallback_error,
                user_message=f"Fallback also failed: {fallback_error}",
            )
