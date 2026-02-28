"""Tests for the SDK service context registry."""

from __future__ import annotations

import pytest

from zenki.sdk._context import (
    clear_all,
    get_channel_registry,
    get_database,
    get_memory_manager,
    get_scheduler,
    set_channel_registry,
    set_database,
    set_memory_manager,
    set_scheduler,
)


@pytest.fixture(autouse=True)
def _clean_context() -> None:
    """Ensure a clean context for each test."""
    clear_all()
    yield
    clear_all()


class TestServiceContext:
    """Test the service registry for SDK tool dependency injection."""

    def test_database_roundtrip(self) -> None:
        assert get_database() is None
        sentinel = object()
        set_database(sentinel)
        assert get_database() is sentinel

    def test_memory_manager_roundtrip(self) -> None:
        assert get_memory_manager() is None
        sentinel = object()
        set_memory_manager(sentinel)
        assert get_memory_manager() is sentinel

    def test_scheduler_roundtrip(self) -> None:
        assert get_scheduler() is None
        sentinel = object()
        set_scheduler(sentinel)
        assert get_scheduler() is sentinel

    def test_channel_registry_roundtrip(self) -> None:
        assert get_channel_registry() is None
        sentinel = object()
        set_channel_registry(sentinel)
        assert get_channel_registry() is sentinel

    def test_clear_all_removes_everything(self) -> None:
        set_database("db")
        set_memory_manager("mm")
        set_scheduler("sched")
        set_channel_registry("cr")
        clear_all()
        assert get_database() is None
        assert get_memory_manager() is None
        assert get_scheduler() is None
        assert get_channel_registry() is None

    def test_overwrite_service(self) -> None:
        set_database("first")
        assert get_database() == "first"
        set_database("second")
        assert get_database() == "second"
