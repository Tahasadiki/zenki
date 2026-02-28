"""Task scheduler for Zenki - manages scheduled and recurring tasks."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from zenki.db.database import ZenkiDatabase
from zenki.db.models import ScheduledTask

logger = logging.getLogger(__name__)

TaskCallback = Callable[[ScheduledTask], Awaitable[str | None]]


class ZenkiScheduler:
    """Manages scheduled and recurring tasks."""

    def __init__(self, db: ZenkiDatabase) -> None:
        self.db = db
        self._scheduler = AsyncIOScheduler()
        self._task_callback: TaskCallback | None = None
        self._running = False

    def set_task_callback(self, callback: TaskCallback) -> None:
        """Set the callback for executing scheduled tasks."""
        self._task_callback = callback

    async def start(self) -> None:
        """Start the scheduler and load persisted tasks."""
        self._scheduler.start()
        self._running = True
        await self._load_persisted_tasks()
        logger.info("Scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        if self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Scheduler stopped")

    async def _load_persisted_tasks(self) -> None:
        """Load tasks from database and schedule them."""
        tasks = self.db.get_scheduled_tasks(enabled_only=True)
        for task in tasks:
            self._schedule_task(task)
        logger.info("Loaded %d persisted tasks", len(tasks))

    def _schedule_task(self, task: ScheduledTask) -> None:
        """Schedule a task with APScheduler."""
        try:
            trigger = CronTrigger.from_crontab(task.cron_expression)
            self._scheduler.add_job(
                self._execute_task,
                trigger=trigger,
                id=task.id,
                args=[task.id],
                replace_existing=True,
                name=task.description,
            )
            logger.info("Scheduled task: %s (%s)", task.description, task.cron_expression)
        except Exception as e:
            logger.error("Failed to schedule task %s: %s", task.id, e)

    async def _execute_task(self, task_id: str) -> None:
        """Execute a scheduled task."""
        tasks = self.db.get_scheduled_tasks()
        task = next((t for t in tasks if t.id == task_id), None)
        if task is None:
            logger.warning("Task %s not found in database", task_id)
            return

        if not task.enabled:
            logger.info("Task %s is disabled, skipping", task_id)
            return

        logger.info("Executing scheduled task: %s", task.description)
        result = None
        try:
            if self._task_callback:
                result = await self._task_callback(task)
        except Exception as e:
            result = f"Error: {e}"
            logger.error("Task %s failed: %s", task_id, e)

        # Update task in database
        task.last_run_at = datetime.now(UTC)
        if result:
            task.last_result = result
        self.db.update_scheduled_task(task)

    def add_task(
        self,
        user_id: str,
        description: str,
        cron_expression: str,
        task_type: str = "action",
        task_config: dict[str, Any] | None = None,
        original_request: str = "",
        notify_channel: str | None = None,
    ) -> ScheduledTask:
        """Add a new scheduled task."""
        task = ScheduledTask(
            user_id=user_id,
            description=description,
            cron_expression=cron_expression,
            task_type=task_type,
            task_config=task_config or {},
            original_request=original_request or "",
            notify_channel=notify_channel,
        )
        task = self.db.create_scheduled_task(task)
        self._schedule_task(task)
        return task

    def remove_task(self, task_id: str) -> None:
        """Remove a scheduled task."""
        try:
            self._scheduler.remove_job(task_id)
        except Exception:
            pass
        # Fetch, update, save
        tasks = self.db.get_scheduled_tasks()
        for task in tasks:
            if task.id == task_id:
                task.enabled = False
                self.db.update_scheduled_task(task)
                break
        logger.info("Removed task: %s", task_id)

    def list_tasks(self, user_id: str | None = None) -> list[ScheduledTask]:
        """List all scheduled tasks."""
        tasks = self.db.get_scheduled_tasks(user_id=user_id, enabled_only=False)
        return tasks

    @property
    def is_running(self) -> bool:
        """Whether the scheduler is running."""
        return self._running
