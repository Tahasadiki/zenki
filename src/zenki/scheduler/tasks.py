"""Task definitions and utilities for the scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TaskDefinition:
    """Definition of a task that can be scheduled."""

    task_type: str  # "reminder", "action", "self_improve", "monitor"
    description: str
    config: dict[str, Any]

    @staticmethod
    def reminder(message: str, notify_channel: str | None = None) -> TaskDefinition:
        """Create a reminder task."""
        return TaskDefinition(
            task_type="reminder",
            description=f"Reminder: {message}",
            config={"message": message, "notify_channel": notify_channel},
        )

    @staticmethod
    def action(prompt: str, skill: str | None = None) -> TaskDefinition:
        """Create an action task (execute a prompt or skill)."""
        return TaskDefinition(
            task_type="action",
            description=f"Action: {prompt[:50]}...",
            config={"prompt": prompt, "skill": skill},
        )

    @staticmethod
    def self_improve() -> TaskDefinition:
        """Create a self-improvement task."""
        return TaskDefinition(
            task_type="self_improve",
            description="Daily self-improvement cycle",
            config={"consolidate_memory": True, "generate_skills": True, "review_errors": True},
        )

    @staticmethod
    def monitor(target: str, check_prompt: str, interval_description: str = "") -> TaskDefinition:
        """Create a monitoring task."""
        return TaskDefinition(
            task_type="monitor",
            description=f"Monitor: {target}",
            config={"target": target, "check_prompt": check_prompt},
        )
