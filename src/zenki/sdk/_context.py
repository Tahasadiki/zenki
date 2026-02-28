"""Shared context for Zenki SDK tools.

Custom MCP tools need access to Zenki's internal services (database, memory
manager, scheduler, channel registry).  This module provides a simple
thread-safe registry so that tools defined with ``@tool`` can access these
services without circular imports or global singletons.

Usage
-----
At application startup (in ``ZenkiOrchestrator.__init__``), register services::

    from zenki.sdk._context import set_database, set_memory_manager
    set_database(db)
    set_memory_manager(manager)

In tool functions, retrieve them::

    from zenki.sdk._context import get_database
    db = get_database()
"""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_services: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Setters (called at startup)
# ---------------------------------------------------------------------------


def set_database(db: Any) -> None:
    with _lock:
        _services["database"] = db


def set_memory_manager(manager: Any) -> None:
    with _lock:
        _services["memory_manager"] = manager


def set_scheduler(scheduler: Any) -> None:
    with _lock:
        _services["scheduler"] = scheduler


def set_channel_registry(registry: Any) -> None:
    with _lock:
        _services["channel_registry"] = registry


# ---------------------------------------------------------------------------
# Getters (called from tools)
# ---------------------------------------------------------------------------


def get_database() -> Any | None:
    with _lock:
        return _services.get("database")


def get_memory_manager() -> Any | None:
    with _lock:
        return _services.get("memory_manager")


def get_scheduler() -> Any | None:
    with _lock:
        return _services.get("scheduler")


def get_channel_registry() -> Any | None:
    with _lock:
        return _services.get("channel_registry")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def clear_all() -> None:
    """Remove all registered services (for testing)."""
    with _lock:
        _services.clear()
