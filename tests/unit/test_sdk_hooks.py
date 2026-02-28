"""Tests for SDK-native hooks.

Validates hook behavior including security blocking, audit logging,
and proper hook output format.
"""

from __future__ import annotations

import pytest

from zenki.sdk.hooks import (
    audit_tool_usage,
    block_dangerous_paths,
    check_dangerous_commands,
    create_zenki_hooks,
    forward_notifications,
    protect_sensitive_files,
    track_subagent_lifecycle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pretool_input(tool_name: str, **tool_input: str) -> dict:
    """Create a PreToolUse hook input dict."""
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "session_id": "test-session",
    }


def _posttool_input(tool_name: str) -> dict:
    """Create a PostToolUse hook input dict."""
    return {
        "hook_event_name": "PostToolUse",
        "tool_name": tool_name,
        "tool_input": {},
        "session_id": "test-session",
    }


# ---------------------------------------------------------------------------
# Security hooks
# ---------------------------------------------------------------------------


class TestProtectSensitiveFiles:
    """Test the protect_sensitive_files hook."""

    @pytest.mark.asyncio
    async def test_blocks_env_file(self) -> None:
        result = await protect_sensitive_files(
            _pretool_input("Write", file_path="/project/.env"),
            tool_use_id="test-123",
            context=None,
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert ".env" in result["hookSpecificOutput"]["permissionDecisionReason"]

    @pytest.mark.asyncio
    async def test_blocks_credentials_json(self) -> None:
        result = await protect_sensitive_files(
            _pretool_input("Edit", file_path="/app/credentials.json"),
            tool_use_id="test-123",
            context=None,
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    @pytest.mark.asyncio
    async def test_blocks_pem_files(self) -> None:
        result = await protect_sensitive_files(
            _pretool_input("Write", file_path="/certs/server.pem"),
            tool_use_id="test-123",
            context=None,
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    @pytest.mark.asyncio
    async def test_allows_normal_files(self) -> None:
        result = await protect_sensitive_files(
            _pretool_input("Write", file_path="/project/src/main.py"),
            tool_use_id="test-123",
            context=None,
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_ignores_non_pretooluse(self) -> None:
        result = await protect_sensitive_files(
            {"hook_event_name": "PostToolUse", "tool_input": {"file_path": ".env"}},
            tool_use_id="test-123",
            context=None,
        )
        assert result == {}


class TestBlockDangerousPaths:
    """Test the block_dangerous_paths hook."""

    @pytest.mark.asyncio
    async def test_blocks_etc(self) -> None:
        result = await block_dangerous_paths(
            _pretool_input("Write", file_path="/etc/passwd"),
            tool_use_id="test-123",
            context=None,
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    @pytest.mark.asyncio
    async def test_blocks_sys(self) -> None:
        result = await block_dangerous_paths(
            _pretool_input("Edit", file_path="/sys/kernel/something"),
            tool_use_id="test-123",
            context=None,
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    @pytest.mark.asyncio
    async def test_blocks_ssh(self) -> None:
        result = await block_dangerous_paths(
            _pretool_input("Write", file_path="/root/.ssh/authorized_keys"),
            tool_use_id="test-123",
            context=None,
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    @pytest.mark.asyncio
    async def test_allows_safe_paths(self) -> None:
        result = await block_dangerous_paths(
            _pretool_input("Write", file_path="/home/user/project/main.py"),
            tool_use_id="test-123",
            context=None,
        )
        assert result == {}


class TestCheckDangerousCommands:
    """Test the check_dangerous_commands hook."""

    @pytest.mark.asyncio
    async def test_blocks_rm_rf_root(self) -> None:
        result = await check_dangerous_commands(
            _pretool_input("Bash", command="rm -rf /"),
            tool_use_id="test-123",
            context=None,
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    @pytest.mark.asyncio
    async def test_blocks_fork_bomb(self) -> None:
        result = await check_dangerous_commands(
            _pretool_input("Bash", command=":(){:|:&};:"),
            tool_use_id="test-123",
            context=None,
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    @pytest.mark.asyncio
    async def test_blocks_dd_zero(self) -> None:
        result = await check_dangerous_commands(
            _pretool_input("Bash", command="dd if=/dev/zero of=/dev/sda"),
            tool_use_id="test-123",
            context=None,
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    @pytest.mark.asyncio
    async def test_allows_normal_commands(self) -> None:
        result = await check_dangerous_commands(
            _pretool_input("Bash", command="ls -la"),
            tool_use_id="test-123",
            context=None,
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_allows_safe_rm(self) -> None:
        result = await check_dangerous_commands(
            _pretool_input("Bash", command="rm temp.txt"),
            tool_use_id="test-123",
            context=None,
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_ignores_non_bash(self) -> None:
        result = await check_dangerous_commands(
            _pretool_input("Read", command="rm -rf /"),
            tool_use_id="test-123",
            context=None,
        )
        assert result == {}


# ---------------------------------------------------------------------------
# Audit hooks
# ---------------------------------------------------------------------------


class TestAuditToolUsage:
    """Test the audit_tool_usage hook."""

    @pytest.mark.asyncio
    async def test_returns_async_output(self) -> None:
        result = await audit_tool_usage(
            _pretool_input("Read", file_path="/test"),
            tool_use_id="test-123",
            context=None,
        )
        assert result["async_"] is True
        assert "asyncTimeout" in result


# ---------------------------------------------------------------------------
# Hook assembly
# ---------------------------------------------------------------------------


class TestCreateZenkiHooks:
    """Test the hook assembly function."""

    def test_returns_all_hook_events(self) -> None:
        hooks = create_zenki_hooks()
        assert "PreToolUse" in hooks
        assert "PostToolUse" in hooks
        assert "SubagentStart" in hooks
        assert "SubagentStop" in hooks
        assert "Notification" in hooks

    def test_pretooluse_has_multiple_matchers(self) -> None:
        hooks = create_zenki_hooks()
        # Security + audit hooks
        assert len(hooks["PreToolUse"]) >= 3

    def test_matchers_have_hooks(self) -> None:
        hooks = create_zenki_hooks()
        for event, matchers in hooks.items():
            for matcher in matchers:
                assert len(matcher.hooks) > 0, (
                    f"Matcher in {event} has no hook callbacks"
                )
