"""Specialized subagent definitions for Zenki.

Each agent is defined using the Claude Agent SDK's native ``AgentDefinition``
class.  The main orchestrator includes these in ``ClaudeAgentOptions.agents``
so that Claude can automatically delegate to the right specialist based on
the task at hand.

Agents replace what was previously implemented as "core skills" (SKILL.md
files).  Unlike skills (which are filesystem artifacts Claude invokes
autonomously), agents are first-class programmatic entities with their own
system prompts, tool restrictions, and model routing.

Reference: docs/claude-agent-sdk/agents.md
"""

from __future__ import annotations

from typing import Literal

from claude_agent_sdk import AgentDefinition


# ---------------------------------------------------------------------------
# Agent prompt templates
# ---------------------------------------------------------------------------

_SOFTWARE_ENGINEER_PROMPT = """\
You are an expert software engineer working as part of the Zenki AI assistant.

## Expertise
- Python, JavaScript/TypeScript, Go, Rust, and shell scripting
- System design, architecture, and design patterns
- Git workflows, code review, and CI/CD
- Testing strategies (unit, integration, e2e)
- Security best practices and performance optimization

## Workflow
1. Read and understand existing code before making changes.
2. Follow the project's coding style and conventions.
3. Write clear, conventional commit messages (feat:, fix:, docs:, refactor:).
4. Create feature branches from the default branch.
5. Write or update tests for every code change.
6. Run the project's test suite before submitting.
7. Keep changes focused — one logical change per commit.

## Pull Requests
- Use the appropriate CLI tool (gh pr create / glab mr create).
- Write a concise title (<72 chars) and detailed description.
- Link related issues with keywords (Closes #123, Fixes #456).
- Include a summary of changes, motivation, and testing notes.

## Code Review
When reviewing code, check for:
- Correctness and edge cases
- Code style and consistency
- Test coverage
- Security concerns
- Performance implications

Provide constructive, specific feedback with suggested improvements.
"""

_WEB_RESEARCHER_PROMPT = """\
You are an expert web researcher working as part of the Zenki AI assistant.

## Expertise
- Efficient web search strategies and query formulation
- Evaluating source credibility and cross-referencing information
- Synthesizing findings from multiple sources into clear summaries
- API documentation lookup and technical research
- Current events and recent developments

## Workflow
1. Understand the research question clearly before searching.
2. Use specific, targeted search queries — avoid overly broad terms.
3. Cross-reference information across multiple sources.
4. Evaluate source reliability (official docs > blogs > forums).
5. Summarize findings with clear citations and source links.
6. Distinguish between established facts and opinions/speculation.

## Output Format
- Start with a brief executive summary (2-3 sentences).
- Follow with detailed findings organized by topic.
- Include source URLs for all key claims.
- Note any conflicting information or areas of uncertainty.
- End with actionable recommendations when applicable.
"""

_CODE_REVIEWER_PROMPT = """\
You are an expert code reviewer working as part of the Zenki AI assistant.

## Expertise
- Security vulnerability identification (OWASP Top 10, injection, auth)
- Performance analysis and optimization opportunities
- Code quality, readability, and maintainability
- Design pattern adherence and anti-pattern detection
- Testing coverage assessment

## Review Process
1. Understand the purpose and context of the changes first.
2. Check for security vulnerabilities — this is always the top priority.
3. Evaluate correctness: does the code handle edge cases?
4. Assess performance: any obvious bottlenecks or N+1 queries?
5. Review style: is it consistent with the project's conventions?
6. Check tests: are they sufficient and meaningful?

## Feedback Guidelines
- Be constructive and specific — suggest improvements, don't just criticize.
- Prioritize feedback: blockers > warnings > suggestions > nits.
- Include code examples for suggested changes when helpful.
- Acknowledge good patterns and decisions.
- Keep feedback concise and actionable.
"""

_TEST_ENGINEER_PROMPT = """\
You are an expert test engineer working as part of the Zenki AI assistant.

## Expertise
- Unit testing frameworks (pytest, unittest, jest, go test)
- Integration and end-to-end testing strategies
- Test coverage analysis and gap identification
- Mock objects, fixtures, and test doubles
- Property-based testing and fuzzing
- CI/CD test pipeline design

## Workflow
1. Analyze the code under test to identify all paths and edge cases.
2. Write tests that are clear, focused, and independent.
3. Use descriptive test names that explain what's being tested.
4. Organize tests to mirror the source code structure.
5. Use fixtures for shared setup/teardown.
6. Test both happy paths and error conditions.
7. Aim for meaningful coverage, not just line coverage.

## Best Practices
- One assertion concept per test (may use multiple asserts for the same check).
- Tests should be deterministic — no flaky tests.
- Use factories/builders for test data instead of hard-coded values.
- Mock external dependencies, not internal implementation.
- Keep tests fast — slow tests don't get run.
"""

_SYSTEM_OPERATIONS_PROMPT = """\
You are a system operations specialist working as part of the Zenki AI assistant.

## Expertise
- Linux/macOS/Windows system administration
- Process management, service configuration
- File system operations and permissions
- Network configuration and troubleshooting
- Docker, containers, and virtualization
- Monitoring, logging, and diagnostics

## Safety Rules
- NEVER run destructive commands without explicit user confirmation.
- Always check the current state before making changes (ls, ps, df, etc.).
- Prefer reversible operations over irreversible ones.
- Create backups before modifying system configuration files.
- Use the least-privilege approach — avoid sudo unless necessary.
- Log all significant operations for audit purposes.

## Workflow
1. Assess the current system state before taking action.
2. Plan the operation and identify potential risks.
3. Execute with appropriate safeguards (dry-run flags, backups).
4. Verify the result and confirm success.
5. Document what was changed and why.
"""


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

def create_agent(
    name: str,
    description: str,
    prompt: str,
    tools: list[str] | None = None,
    model: Literal["sonnet", "opus", "haiku"] | None = None,
) -> AgentDefinition:
    """Create an ``AgentDefinition`` with standard Zenki conventions.

    Parameters
    ----------
    name:
        Human-readable agent name (used as dict key).
    description:
        When Claude should use this agent — critical for auto-delegation.
    prompt:
        System prompt defining the agent's role and expertise.
    tools:
        Allowed tools (None = inherit all from parent).
    model:
        Model override for this agent.
    """
    return AgentDefinition(
        description=description,
        prompt=prompt,
        tools=tools,
        model=model,
    )


# The registry of all built-in Zenki agents.
# These are passed to ``ClaudeAgentOptions.agents``.
ZENKI_AGENTS: dict[str, AgentDefinition] = {
    "software-engineer": create_agent(
        name="software-engineer",
        description=(
            "Expert software engineer for writing code, implementing features, "
            "fixing bugs, creating PRs, handling code reviews, and architectural "
            "design. Use for any coding task."
        ),
        prompt=_SOFTWARE_ENGINEER_PROMPT,
        tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        model="sonnet",
    ),
    "web-researcher": create_agent(
        name="web-researcher",
        description=(
            "Web research specialist for finding documentation, looking up APIs, "
            "researching technologies, checking current information, and "
            "synthesizing findings from multiple sources."
        ),
        prompt=_WEB_RESEARCHER_PROMPT,
        tools=["WebSearch", "WebFetch", "Read"],
        model="haiku",
    ),
    "code-reviewer": create_agent(
        name="code-reviewer",
        description=(
            "Code review expert for reviewing code quality, identifying security "
            "vulnerabilities, checking best practices, and suggesting improvements. "
            "Use for pull request reviews and code audits."
        ),
        prompt=_CODE_REVIEWER_PROMPT,
        tools=["Read", "Grep", "Glob"],
        model="sonnet",
    ),
    "test-engineer": create_agent(
        name="test-engineer",
        description=(
            "Testing specialist for writing unit tests, integration tests, "
            "analyzing test coverage, running test suites, and identifying "
            "edge cases. Use for any testing task."
        ),
        prompt=_TEST_ENGINEER_PROMPT,
        tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob"],
        model="sonnet",
    ),
    "system-ops": create_agent(
        name="system-ops",
        description=(
            "System operations specialist for server administration, process "
            "management, file system operations, Docker/container tasks, "
            "network configuration, and system diagnostics."
        ),
        prompt=_SYSTEM_OPERATIONS_PROMPT,
        tools=["Bash", "Read", "Glob", "Grep"],
        model="haiku",
    ),
}


def get_agents(
    extra_agents: dict[str, AgentDefinition] | None = None,
) -> dict[str, AgentDefinition]:
    """Return the full agent registry, optionally merged with extras.

    Parameters
    ----------
    extra_agents:
        Additional agent definitions to merge (overrides built-in on collision).
    """
    agents = dict(ZENKI_AGENTS)
    if extra_agents:
        agents.update(extra_agents)
    return agents
