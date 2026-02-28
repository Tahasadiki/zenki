"""Tests for the smart error escalation system."""

import pytest

from zenki.core.errors import (
    ErrorPattern,
    ErrorResult,
    ErrorSeverity,
    RecoveryStrategy,
    SmartErrorHandler,
)


@pytest.fixture
def handler():
    return SmartErrorHandler()


class TestErrorPatternMatching:
    def test_rate_limit_detected(self, handler):
        error = Exception("Rate limit exceeded")
        pattern = handler._find_pattern(error)
        assert pattern is not None
        assert pattern.name == "rate_limit"

    def test_timeout_detected(self, handler):
        error = Exception("Connection timed out")
        pattern = handler._find_pattern(error)
        assert pattern is not None
        assert pattern.name == "network_timeout"

    def test_connection_refused_detected(self, handler):
        error = Exception("Connection refused")
        pattern = handler._find_pattern(error)
        assert pattern is not None
        assert pattern.name == "network_timeout"

    def test_file_not_found_detected(self, handler):
        error = FileNotFoundError("/path/to/file")
        pattern = handler._find_pattern(error)
        assert pattern is not None
        assert pattern.name == "file_not_found"

    def test_permission_denied_detected(self, handler):
        error = PermissionError("Access denied")
        pattern = handler._find_pattern(error)
        assert pattern is not None
        assert pattern.name == "permission_denied"

    def test_auth_error_detected(self, handler):
        error = Exception("401 Unauthorized: Invalid API key")
        pattern = handler._find_pattern(error)
        assert pattern is not None
        assert pattern.name == "api_auth_error"

    def test_unknown_error_no_pattern(self, handler):
        error = Exception("Something completely unexpected")
        pattern = handler._find_pattern(error)
        assert pattern is None

    def test_custom_pattern_registered(self, handler):
        handler.register_pattern(ErrorPattern(
            name="custom_error",
            match_fn=lambda e: "custom" in str(e).lower(),
            strategy=RecoveryStrategy.SKIP,
        ))
        error = Exception("Custom error occurred")
        pattern = handler._find_pattern(error)
        assert pattern is not None
        assert pattern.name == "custom_error"


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_novel_error_escalates(self, handler):
        error = Exception("Never seen this before")
        result = await handler.handle(error)
        assert not result.recovered
        assert result.strategy_used == RecoveryStrategy.ESCALATE
        assert "unexpected error" in result.user_message

    @pytest.mark.asyncio
    async def test_file_not_found_escalates(self, handler):
        error = FileNotFoundError("missing.txt")
        result = await handler.handle(error)
        assert not result.recovered
        assert result.strategy_used == RecoveryStrategy.ESCALATE

    @pytest.mark.asyncio
    async def test_auth_error_escalates(self, handler):
        error = Exception("401 Unauthorized")
        result = await handler.handle(error)
        assert not result.recovered
        assert result.strategy_used == RecoveryStrategy.ESCALATE
        assert "API key" in result.user_message or "Authentication" in result.user_message

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self, handler):
        call_count = 0

        async def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Rate limit exceeded")

        error = Exception("Rate limit exceeded")
        # Override base_delay for fast tests
        pattern = handler._find_pattern(error)
        pattern.base_delay = 0.01

        result = await handler.handle(error, retry_fn=flaky_fn)
        assert result.recovered
        assert result.strategy_used == RecoveryStrategy.RETRY
        # attempts counts: 1 original + 1 retry that failed + 1 retry that succeeded = 3
        assert result.attempts >= 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, handler):
        async def always_fail():
            raise Exception("Rate limit exceeded")

        error = Exception("Rate limit exceeded")
        pattern = handler._find_pattern(error)
        pattern.base_delay = 0.01
        pattern.max_retries = 2

        result = await handler.handle(error, retry_fn=always_fail)
        assert not result.recovered
        assert result.strategy_used == RecoveryStrategy.RETRY
        assert result.attempts == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_retry_different_error_escalates(self, handler):
        async def change_error():
            raise ValueError("Different error entirely")

        error = Exception("Rate limit exceeded")
        pattern = handler._find_pattern(error)
        pattern.base_delay = 0.01

        result = await handler.handle(error, retry_fn=change_error)
        assert not result.recovered
        assert result.strategy_used == RecoveryStrategy.ESCALATE

    @pytest.mark.asyncio
    async def test_fallback_succeeds(self, handler):
        fallback_called = False

        async def fallback():
            nonlocal fallback_called
            fallback_called = True

        handler.register_pattern(ErrorPattern(
            name="test_fallback",
            match_fn=lambda e: "fallback_test" in str(e),
            strategy=RecoveryStrategy.FALLBACK,
            fallback_fn=fallback,
        ))

        error = Exception("fallback_test error")
        result = await handler.handle(error)
        assert result.recovered
        assert result.strategy_used == RecoveryStrategy.FALLBACK
        assert fallback_called

    @pytest.mark.asyncio
    async def test_fallback_fails_escalates(self, handler):
        async def bad_fallback():
            raise Exception("Fallback also broken")

        handler.register_pattern(ErrorPattern(
            name="bad_fallback",
            match_fn=lambda e: "bad_fallback_test" in str(e),
            strategy=RecoveryStrategy.FALLBACK,
            fallback_fn=bad_fallback,
        ))

        error = Exception("bad_fallback_test error")
        result = await handler.handle(error)
        assert not result.recovered
        assert result.strategy_used == RecoveryStrategy.ESCALATE

    @pytest.mark.asyncio
    async def test_skip_strategy(self, handler):
        handler.register_pattern(ErrorPattern(
            name="skippable",
            match_fn=lambda e: "skip_this" in str(e),
            strategy=RecoveryStrategy.SKIP,
            message_template="Skipped: {error}",
        ))

        error = Exception("skip_this error")
        result = await handler.handle(error)
        assert result.recovered
        assert result.strategy_used == RecoveryStrategy.SKIP

    @pytest.mark.asyncio
    async def test_retry_without_fn_escalates(self, handler):
        error = Exception("Rate limit exceeded")
        result = await handler.handle(error, retry_fn=None)
        assert not result.recovered
        assert result.strategy_used == RecoveryStrategy.ESCALATE


class TestErrorResult:
    def test_error_result_defaults(self):
        error = Exception("test")
        result = ErrorResult(
            recovered=True,
            strategy_used=RecoveryStrategy.RETRY,
            error=error,
        )
        assert result.recovered
        assert result.attempts == 1
        assert result.user_message == ""
        assert result.details == {}


class TestErrorSeverity:
    def test_severity_values(self):
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestRecoveryStrategy:
    def test_strategy_values(self):
        assert RecoveryStrategy.RETRY.value == "retry"
        assert RecoveryStrategy.FALLBACK.value == "fallback"
        assert RecoveryStrategy.SKIP.value == "skip"
        assert RecoveryStrategy.ESCALATE.value == "escalate"
