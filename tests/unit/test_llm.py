"""Unit tests for the Zenki LLM abstraction layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from zenki.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from zenki.llm.config import MODEL_CONTEXT_WINDOWS, MODEL_MAP, get_model_id
from zenki.llm.router import ModelRouter

# ---------------------------------------------------------------------------
# LLMMessage dataclass
# ---------------------------------------------------------------------------


class TestLLMMessage:
    """Tests for the LLMMessage dataclass."""

    def test_basic_creation(self) -> None:
        msg = LLMMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_default_optional_fields(self) -> None:
        msg = LLMMessage(role="assistant", content="Hi there")
        assert msg.tool_calls is None
        assert msg.tool_results is None

    def test_with_tool_calls(self) -> None:
        tool_calls = [{"id": "tc_1", "name": "search", "input": {"q": "test"}}]
        msg = LLMMessage(role="assistant", content="", tool_calls=tool_calls)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "search"

    def test_with_tool_results(self) -> None:
        tool_results = [{"tool_use_id": "tc_1", "content": "result data"}]
        msg = LLMMessage(role="user", content="", tool_results=tool_results)
        assert msg.tool_results is not None
        assert len(msg.tool_results) == 1

    def test_system_role(self) -> None:
        msg = LLMMessage(role="system", content="You are a helpful assistant.")
        assert msg.role == "system"

    def test_equality(self) -> None:
        msg1 = LLMMessage(role="user", content="hello")
        msg2 = LLMMessage(role="user", content="hello")
        assert msg1 == msg2

    def test_inequality(self) -> None:
        msg1 = LLMMessage(role="user", content="hello")
        msg2 = LLMMessage(role="assistant", content="hello")
        assert msg1 != msg2


# ---------------------------------------------------------------------------
# LLMResponse dataclass
# ---------------------------------------------------------------------------


class TestLLMResponse:
    """Tests for the LLMResponse dataclass."""

    def test_basic_creation(self) -> None:
        resp = LLMResponse(content="Hello!", model="claude-sonnet-4-6")
        assert resp.content == "Hello!"
        assert resp.model == "claude-sonnet-4-6"

    def test_default_fields(self) -> None:
        resp = LLMResponse(content="test", model="test-model")
        assert resp.tokens_in == 0
        assert resp.tokens_out == 0
        assert resp.tool_calls == []
        assert resp.stop_reason == ""

    def test_with_token_counts(self) -> None:
        resp = LLMResponse(
            content="answer",
            model="claude-sonnet-4-6",
            tokens_in=150,
            tokens_out=300,
        )
        assert resp.tokens_in == 150
        assert resp.tokens_out == 300

    def test_with_tool_calls(self) -> None:
        tool_calls = [{"id": "tc_1", "name": "calculator", "input": {"expr": "2+2"}}]
        resp = LLMResponse(
            content="",
            model="claude-sonnet-4-6",
            tool_calls=tool_calls,
            stop_reason="tool_use",
        )
        assert len(resp.tool_calls) == 1
        assert resp.stop_reason == "tool_use"

    def test_tool_calls_default_is_independent_list(self) -> None:
        """Each instance should get its own list for tool_calls."""
        resp1 = LLMResponse(content="a", model="m")
        resp2 = LLMResponse(content="b", model="m")
        resp1.tool_calls.append({"id": "x"})
        assert resp2.tool_calls == []


# ---------------------------------------------------------------------------
# ModelRouter.classify_complexity
# ---------------------------------------------------------------------------


class TestModelRouterClassifyComplexity:
    """Tests for ModelRouter.classify_complexity()."""

    def setup_method(self) -> None:
        self.router = ModelRouter()

    # -- haiku (simple) --

    @pytest.mark.parametrize(
        "message",
        [
            "hi",
            "Hello",
            "Hey there",
            "good morning",
            "Good Evening",
            "thanks",
            "Thank you!",
            "ok",
            "sure",
            "yes",
            "no",
            "nope",
            "status?",
            "ping",
            "how are you",
            "format this",
        ],
    )
    def test_haiku_simple_messages(self, message: str) -> None:
        assert self.router.classify_complexity(message) == "haiku"

    # -- opus (complex) --

    @pytest.mark.parametrize(
        "message",
        [
            "write code to parse CSV files",
            "generate code for a REST API",
            "implement a binary search tree",
            "refactor the database module",
            "debug this error in the auth flow",
            "fix the bug in the payment system",
            "design the architecture for our new microservice",
            "do a complex analysis of our user data",
            "I need skill generation for a new capability",
            "consolidate memories from the last week",
            "explain in detail how transformers work",
            "step by step guide to deploying on AWS",
            "write a function to validate emails",
            "write a class for managing connections",
            "do a code review of this PR",
            "security audit of the authentication module",
            "performance optimization for the database queries",
            "implement an algorithm for shortest path",
            "create a skill for web scraping",
            "write a module for handling webhooks",
            "build a function for data processing",
            "debug the error in the login system",
            "architect a system for real-time notifications",
        ],
    )
    def test_opus_complex_messages(self, message: str) -> None:
        assert self.router.classify_complexity(message) == "opus"

    # -- sonnet (default / medium) --

    @pytest.mark.parametrize(
        "message",
        [
            "what is the capital of France?",
            "summarize this article for me",
            "translate this to Spanish",
            "tell me about quantum computing",
            "list three benefits of exercise",
            "how does photosynthesis work?",
        ],
    )
    def test_sonnet_default_messages(self, message: str) -> None:
        assert self.router.classify_complexity(message) == "sonnet"

    def test_context_parameter_accepted(self) -> None:
        """classify_complexity should accept a context dict without error."""
        result = self.router.classify_complexity(
            "hello", context={"conversation_length": 5}
        )
        assert result == "haiku"


# ---------------------------------------------------------------------------
# ModelRouter.route
# ---------------------------------------------------------------------------


class TestModelRouterRoute:
    """Tests for ModelRouter.route()."""

    def setup_method(self) -> None:
        self.router = ModelRouter()

    def test_route_returns_full_model_id_for_haiku(self) -> None:
        model_id = self.router.route("hi")
        assert model_id == MODEL_MAP["haiku"]

    def test_route_returns_full_model_id_for_sonnet(self) -> None:
        model_id = self.router.route("what is the capital of France?")
        assert model_id == MODEL_MAP["sonnet"]

    def test_route_returns_full_model_id_for_opus(self) -> None:
        model_id = self.router.route("write code to parse JSON files")
        assert model_id == MODEL_MAP["opus"]

    def test_route_with_context(self) -> None:
        model_id = self.router.route("hello", context={"key": "value"})
        assert model_id == MODEL_MAP["haiku"]


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Tests for model configuration (config.py)."""

    def test_model_map_has_all_shortnames(self) -> None:
        assert "haiku" in MODEL_MAP
        assert "sonnet" in MODEL_MAP
        assert "opus" in MODEL_MAP

    def test_model_map_values(self) -> None:
        assert MODEL_MAP["haiku"] == "claude-haiku-4-5-20251001"
        assert MODEL_MAP["sonnet"] == "claude-sonnet-4-6"
        assert MODEL_MAP["opus"] == "claude-opus-4-6"

    def test_get_model_id_known_shortname(self) -> None:
        assert get_model_id("haiku") == "claude-haiku-4-5-20251001"
        assert get_model_id("sonnet") == "claude-sonnet-4-6"
        assert get_model_id("opus") == "claude-opus-4-6"

    def test_get_model_id_unknown_returns_input(self) -> None:
        """Unknown shortnames should be returned as-is (assumed full model ID)."""
        assert get_model_id("claude-sonnet-4-6") == "claude-sonnet-4-6"
        assert get_model_id("some-custom-model") == "some-custom-model"

    def test_context_windows_defined(self) -> None:
        assert "haiku" in MODEL_CONTEXT_WINDOWS
        assert "sonnet" in MODEL_CONTEXT_WINDOWS
        assert "opus" in MODEL_CONTEXT_WINDOWS

    def test_context_windows_are_positive_integers(self) -> None:
        for name, size in MODEL_CONTEXT_WINDOWS.items():
            assert isinstance(size, int), f"{name} context window should be int"
            assert size > 0, f"{name} context window should be positive"


# ---------------------------------------------------------------------------
# Claude provider initialization (mocked)
# ---------------------------------------------------------------------------


class TestClaudeProviderInit:
    """Tests for ClaudeProvider initialization -- no real API calls."""

    def test_import_error_when_anthropic_missing(self) -> None:
        """ClaudeProvider should raise ImportError with a helpful message
        when the anthropic package is not installed."""
        with patch.dict("sys.modules", {"anthropic": None}):
            # We need to re-import the module so it picks up the patched import.
            import zenki.llm.claude_provider as cp_module

            original_anthropic = cp_module.anthropic
            cp_module.anthropic = None  # type: ignore[assignment]
            try:
                with pytest.raises(ImportError, match="anthropic"):
                    cp_module.ClaudeProvider(api_key="test-key")
            finally:
                cp_module.anthropic = original_anthropic

    def test_provider_creates_with_api_key(self) -> None:
        """ClaudeProvider should initialise when given an API key."""
        mock_anthropic_module = MagicMock()
        mock_client = MagicMock()
        mock_anthropic_module.AsyncAnthropic.return_value = mock_client

        import zenki.llm.claude_provider as cp_module

        original_anthropic = cp_module.anthropic
        cp_module.anthropic = mock_anthropic_module
        try:
            provider = cp_module.ClaudeProvider(api_key="sk-test-123")
            assert provider._client is mock_client
            mock_anthropic_module.AsyncAnthropic.assert_called_once_with(
                api_key="sk-test-123"
            )
        finally:
            cp_module.anthropic = original_anthropic

    def test_provider_falls_back_to_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ClaudeProvider should use ANTHROPIC_API_KEY env var when no key is passed."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-key")
        mock_anthropic_module = MagicMock()
        mock_client = MagicMock()
        mock_anthropic_module.AsyncAnthropic.return_value = mock_client

        import zenki.llm.claude_provider as cp_module

        original_anthropic = cp_module.anthropic
        cp_module.anthropic = mock_anthropic_module
        try:
            cp_module.ClaudeProvider()
            mock_anthropic_module.AsyncAnthropic.assert_called_once_with(
                api_key="sk-env-key"
            )
        finally:
            cp_module.anthropic = original_anthropic

    def test_get_available_models(self) -> None:
        """get_available_models should return the shortnames from MODEL_MAP."""
        mock_anthropic_module = MagicMock()
        mock_anthropic_module.AsyncAnthropic.return_value = MagicMock()

        import zenki.llm.claude_provider as cp_module

        original_anthropic = cp_module.anthropic
        cp_module.anthropic = mock_anthropic_module
        try:
            provider = cp_module.ClaudeProvider(api_key="sk-test")
            models = provider.get_available_models()
            assert "haiku" in models
            assert "sonnet" in models
            assert "opus" in models
        finally:
            cp_module.anthropic = original_anthropic

    def test_to_api_messages_skips_system(self) -> None:
        """_to_api_messages should skip system messages."""
        from zenki.llm.claude_provider import ClaudeProvider

        messages = [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi!"),
        ]
        api_msgs = ClaudeProvider._to_api_messages(messages)
        assert len(api_msgs) == 2
        assert api_msgs[0]["role"] == "user"
        assert api_msgs[1]["role"] == "assistant"

    def test_to_api_messages_preserves_content(self) -> None:
        """_to_api_messages should preserve message content."""
        from zenki.llm.claude_provider import ClaudeProvider

        messages = [
            LLMMessage(role="user", content="What is 2+2?"),
        ]
        api_msgs = ClaudeProvider._to_api_messages(messages)
        assert len(api_msgs) == 1
        assert api_msgs[0]["content"] == "What is 2+2?"


# ---------------------------------------------------------------------------
# Package-level exports
# ---------------------------------------------------------------------------


class TestLLMPackageExports:
    """Test that the llm package exports the expected symbols."""

    def test_base_provider_importable(self) -> None:
        from zenki.llm import BaseLLMProvider as Imported

        assert Imported is BaseLLMProvider

    def test_llm_message_importable(self) -> None:
        from zenki.llm import LLMMessage as Imported

        assert Imported is LLMMessage

    def test_llm_response_importable(self) -> None:
        from zenki.llm import LLMResponse as Imported

        assert Imported is LLMResponse

    def test_model_router_importable(self) -> None:
        from zenki.llm import ModelRouter as Imported

        assert Imported is ModelRouter

    def test_base_provider_is_abstract(self) -> None:
        """BaseLLMProvider should not be directly instantiable."""
        with pytest.raises(TypeError):
            BaseLLMProvider()  # type: ignore[abstract]
