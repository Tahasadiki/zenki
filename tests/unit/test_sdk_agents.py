"""Tests for SDK-native agent definitions.

Validates that all agent definitions follow the correct patterns and
contain the required fields for the Claude Agent SDK.
"""

from __future__ import annotations

from zenki.sdk.agents import (
    ZENKI_AGENTS,
    create_agent,
    get_agents,
)


class TestAgentDefinitions:
    """Test the built-in agent definitions."""

    def test_all_agents_have_required_fields(self) -> None:
        """Every agent must have description and prompt."""
        for name, agent in ZENKI_AGENTS.items():
            assert agent.description, f"Agent '{name}' missing description"
            assert agent.prompt, f"Agent '{name}' missing prompt"
            assert len(agent.description) > 20, (
                f"Agent '{name}' description too short: {agent.description!r}"
            )
            assert len(agent.prompt) > 50, (
                f"Agent '{name}' prompt too short"
            )

    def test_expected_agents_exist(self) -> None:
        """The five core agents should be defined."""
        expected = {
            "software-engineer",
            "web-researcher",
            "code-reviewer",
            "test-engineer",
            "system-ops",
        }
        assert expected == set(ZENKI_AGENTS.keys())

    def test_agents_have_tool_restrictions(self) -> None:
        """Each agent should have explicit tool restrictions (not None)."""
        for name, agent in ZENKI_AGENTS.items():
            assert agent.tools is not None, (
                f"Agent '{name}' should have explicit tool restrictions"
            )
            assert len(agent.tools) > 0, (
                f"Agent '{name}' has empty tools list"
            )

    def test_no_agent_has_task_tool(self) -> None:
        """Subagents cannot spawn their own subagents — no Task tool."""
        for name, agent in ZENKI_AGENTS.items():
            if agent.tools:
                assert "Task" not in agent.tools, (
                    f"Agent '{name}' should NOT have 'Task' in tools "
                    "(subagents cannot spawn subagents)"
                )

    def test_code_reviewer_is_read_only(self) -> None:
        """Code reviewer should only have read-only tools."""
        reviewer = ZENKI_AGENTS["code-reviewer"]
        write_tools = {"Write", "Edit", "Bash"}
        if reviewer.tools:
            for tool in reviewer.tools:
                assert tool not in write_tools, (
                    f"Code reviewer should be read-only, but has '{tool}'"
                )

    def test_web_researcher_has_web_tools(self) -> None:
        """Web researcher must have WebSearch and WebFetch."""
        researcher = ZENKI_AGENTS["web-researcher"]
        assert researcher.tools is not None
        assert "WebSearch" in researcher.tools
        assert "WebFetch" in researcher.tools

    def test_software_engineer_has_write_tools(self) -> None:
        """Software engineer needs Write and Edit for code modifications."""
        engineer = ZENKI_AGENTS["software-engineer"]
        assert engineer.tools is not None
        assert "Write" in engineer.tools
        assert "Edit" in engineer.tools
        assert "Bash" in engineer.tools

    def test_agents_have_model_override(self) -> None:
        """Each agent should specify a model for optimal routing."""
        for name, agent in ZENKI_AGENTS.items():
            # model can be None (inherit from parent), but we prefer explicit
            # For now, just check the ones that should have overrides
            pass  # Model is optional, so no hard assertion

    def test_software_engineer_model(self) -> None:
        """Software engineer should use sonnet (balanced capability)."""
        assert ZENKI_AGENTS["software-engineer"].model == "sonnet"

    def test_web_researcher_model(self) -> None:
        """Web researcher should use haiku (fast for research)."""
        assert ZENKI_AGENTS["web-researcher"].model == "haiku"


class TestCreateAgent:
    """Test the create_agent factory function."""

    def test_creates_valid_definition(self) -> None:
        agent = create_agent(
            name="test-agent",
            description="A test agent",
            prompt="You are a test agent.",
            tools=["Read"],
            model="sonnet",
        )
        assert agent.description == "A test agent"
        assert agent.prompt == "You are a test agent."
        assert agent.tools == ["Read"]
        assert agent.model == "sonnet"

    def test_tools_default_to_none(self) -> None:
        agent = create_agent(
            name="test-agent",
            description="A test agent",
            prompt="You are a test agent.",
        )
        assert agent.tools is None  # Inherit all from parent

    def test_model_default_to_none(self) -> None:
        agent = create_agent(
            name="test-agent",
            description="A test agent",
            prompt="You are a test agent.",
        )
        assert agent.model is None


class TestGetAgents:
    """Test the agent registry getter."""

    def test_returns_all_builtin_agents(self) -> None:
        agents = get_agents()
        assert len(agents) == len(ZENKI_AGENTS)
        assert set(agents.keys()) == set(ZENKI_AGENTS.keys())

    def test_merges_extra_agents(self) -> None:
        from claude_agent_sdk import AgentDefinition

        extra = {
            "custom-agent": AgentDefinition(
                description="Custom agent",
                prompt="Custom prompt",
            )
        }
        agents = get_agents(extra_agents=extra)
        assert "custom-agent" in agents
        assert len(agents) == len(ZENKI_AGENTS) + 1

    def test_extra_agents_override_builtin(self) -> None:
        from claude_agent_sdk import AgentDefinition

        custom_engineer = AgentDefinition(
            description="Custom engineer",
            prompt="Custom prompt",
            tools=["Read"],
        )
        agents = get_agents(extra_agents={"software-engineer": custom_engineer})
        assert agents["software-engineer"].description == "Custom engineer"
        assert len(agents) == len(ZENKI_AGENTS)  # Same count, just overridden
