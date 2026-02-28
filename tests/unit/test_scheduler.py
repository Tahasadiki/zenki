"""Tests for the scheduler and natural language parser."""


import pytest

from zenki.db.database import ZenkiDatabase
from zenki.scheduler.nl_parser import ParsedSchedule, parse_schedule
from zenki.scheduler.scheduler import ZenkiScheduler
from zenki.scheduler.tasks import TaskDefinition


@pytest.fixture
def db(tmp_path):
    from zenki.db.models import User
    database = ZenkiDatabase(tmp_path / "test.db")
    database.initialize()
    database.create_user(User(id="test-user"))
    return database


class TestNLParser:
    def test_every_30_minutes(self):
        result = parse_schedule("every 30 minutes")
        assert result is not None
        assert result.cron_expression == "*/30 * * * *"

    def test_every_2_hours(self):
        result = parse_schedule("every 2 hours")
        assert result is not None
        assert result.cron_expression == "0 */2 * * *"

    def test_daily_at_9am(self):
        result = parse_schedule("every day at 9am")
        assert result is not None
        assert result.cron_expression == "0 9 * * *"

    def test_daily_at_3_30_pm(self):
        result = parse_schedule("daily at 3:30 pm")
        assert result is not None
        assert result.cron_expression == "30 15 * * *"

    def test_daily_at_14_00(self):
        result = parse_schedule("daily at 14:00")
        assert result is not None
        assert result.cron_expression == "0 14 * * *"

    def test_every_morning(self):
        result = parse_schedule("every morning")
        assert result is not None
        assert result.cron_expression == "0 9 * * *"

    def test_every_evening(self):
        result = parse_schedule("every evening")
        assert result is not None
        assert result.cron_expression == "0 18 * * *"

    def test_every_night(self):
        result = parse_schedule("every night")
        assert result is not None
        assert result.cron_expression == "0 21 * * *"

    def test_every_monday(self):
        result = parse_schedule("every monday")
        assert result is not None
        assert "1" in result.cron_expression

    def test_every_friday_at_5pm(self):
        result = parse_schedule("every friday at 5pm")
        assert result is not None
        assert "17" in result.cron_expression
        assert "5" in result.cron_expression

    def test_every_weekday(self):
        result = parse_schedule("every weekday")
        assert result is not None
        assert "1-5" in result.cron_expression

    def test_every_hour(self):
        result = parse_schedule("every hour")
        assert result is not None
        assert result.cron_expression == "0 * * * *"

    def test_hourly(self):
        result = parse_schedule("hourly")
        assert result is not None
        assert result.cron_expression == "0 * * * *"

    def test_weekly(self):
        result = parse_schedule("weekly")
        assert result is not None

    def test_unparseable(self):
        result = parse_schedule("whenever you feel like it")
        assert result is None

    def test_confidence_scores(self):
        result = parse_schedule("every 5 minutes")
        assert result is not None
        assert result.confidence >= 0.7

    def test_case_insensitive(self):
        result = parse_schedule("Every Morning")
        assert result is not None

    def test_am_pm_handling(self):
        result_am = parse_schedule("daily at 12am")
        assert result_am is not None
        assert "0 0" in result_am.cron_expression  # midnight

        result_pm = parse_schedule("daily at 12pm")
        assert result_pm is not None
        assert "0 12" in result_pm.cron_expression  # noon


class TestTaskDefinition:
    def test_reminder(self):
        task = TaskDefinition.reminder("Check PR #42")
        assert task.task_type == "reminder"
        assert "Check PR #42" in task.description
        assert task.config["message"] == "Check PR #42"

    def test_action(self):
        task = TaskDefinition.action("Deploy to staging", skill="deploy")
        assert task.task_type == "action"
        assert task.config["skill"] == "deploy"

    def test_self_improve(self):
        task = TaskDefinition.self_improve()
        assert task.task_type == "self_improve"
        assert task.config["consolidate_memory"] is True

    def test_monitor(self):
        task = TaskDefinition.monitor("PR #42", "Check for new reviews")
        assert task.task_type == "monitor"
        assert task.config["target"] == "PR #42"


class TestZenkiScheduler:
    def test_add_task(self, db):
        scheduler = ZenkiScheduler(db)
        task = scheduler.add_task(
            user_id="test-user",
            description="Test task",
            cron_expression="0 9 * * *",
            task_type="reminder",
            task_config={"message": "Hello"},
        )
        assert task.description == "Test task"
        assert task.cron_expression == "0 9 * * *"

    def test_list_tasks(self, db):
        scheduler = ZenkiScheduler(db)
        scheduler.add_task(
            user_id="test-user",
            description="Task 1",
            cron_expression="0 9 * * *",
        )
        scheduler.add_task(
            user_id="test-user",
            description="Task 2",
            cron_expression="0 18 * * *",
        )
        tasks = scheduler.list_tasks()
        assert len(tasks) >= 2

    def test_remove_task(self, db):
        scheduler = ZenkiScheduler(db)
        task = scheduler.add_task(
            user_id="test-user",
            description="To remove",
            cron_expression="0 9 * * *",
        )
        scheduler.remove_task(task.id)
        # Task should be disabled
        tasks = db.get_scheduled_tasks(enabled_only=True)
        enabled_ids = [t.id for t in tasks]
        assert task.id not in enabled_ids

    def test_is_running(self, db):
        scheduler = ZenkiScheduler(db)
        assert not scheduler.is_running

    @pytest.mark.asyncio
    async def test_start_stop(self, db):
        scheduler = ZenkiScheduler(db)
        await scheduler.start()
        assert scheduler.is_running
        await scheduler.stop()
        assert not scheduler.is_running

    @pytest.mark.asyncio
    async def test_load_persisted_tasks(self, db):
        # Add task to DB directly using a model
        from zenki.db.models import ScheduledTask
        task_model = ScheduledTask(
            user_id="test-user",
            description="Persisted task",
            cron_expression="0 9 * * *",
            task_type="reminder",
            task_config={"message": "test"},
            original_request="",
        )
        db.create_scheduled_task(task_model)
        scheduler = ZenkiScheduler(db)
        await scheduler.start()
        # Task should be loaded
        jobs = scheduler._scheduler.get_jobs()
        assert len(jobs) >= 1
        await scheduler.stop()


class TestParsedSchedule:
    def test_dataclass(self):
        ps = ParsedSchedule(
            cron_expression="0 9 * * *",
            description="Daily at 9am",
            confidence=0.9,
        )
        assert ps.cron_expression == "0 9 * * *"
        assert ps.description == "Daily at 9am"
        assert ps.confidence == 0.9
