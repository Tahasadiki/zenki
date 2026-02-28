"""Claude Agent SDK integration for Zenki.

This module provides the native SDK-based architecture for Zenki, using
the Claude Agent SDK's built-in patterns for agents, tools, hooks, and
orchestration instead of custom implementations.

Key components:
- ``agents``: Specialized subagent definitions (software engineer, web researcher, etc.)
- ``tools``: Custom MCP tools for Zenki-specific operations (memory, scheduling, etc.)
- ``hooks``: Lifecycle hooks for security, audit logging, and memory integration
- ``orchestrator``: Main orchestrator using ``ClaudeSDKClient`` for multi-turn conversations

Note: Imports are kept minimal at package level to avoid circular imports
with ``zenki.core``.  Import from submodules directly when needed:

    from zenki.sdk.agents import ZENKI_AGENTS
    from zenki.sdk.orchestrator import ZenkiOrchestrator
"""

from zenki.sdk.agents import ZENKI_AGENTS, create_agent
from zenki.sdk.hooks import create_zenki_hooks
from zenki.sdk.tools import create_zenki_tools

__all__ = [
    "ZENKI_AGENTS",
    "create_agent",
    "create_zenki_hooks",
    "create_zenki_tools",
]
