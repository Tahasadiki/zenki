"""Natural language to cron expression parser."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedSchedule:
    """Result of parsing a natural language schedule."""

    cron_expression: str
    description: str
    confidence: float  # 0.0 to 1.0


# Common schedule patterns
SCHEDULE_PATTERNS: list[tuple[str, str, str]] = [
    # Every N minutes/hours
    (r"every\s+(\d+)\s+minutes?", "0/{m} * * * *", "Every {m} minutes"),
    (r"every\s+(\d+)\s+hours?", "0 0/{h} * * *", "Every {h} hours"),
    # Daily patterns
    (r"every\s+day\s+at\s+(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)?", "", "Daily at {time}"),
    (r"daily\s+at\s+(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)?", "", "Daily at {time}"),
    (r"every\s+morning", "0 9 * * *", "Every morning at 9:00 AM"),
    (r"every\s+evening", "0 18 * * *", "Every evening at 6:00 PM"),
    (r"every\s+night", "0 21 * * *", "Every night at 9:00 PM"),
    # Weekly patterns
    (r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", "", "Every {day}"),
    (r"weekly", "0 9 * * 1", "Weekly on Monday at 9:00 AM"),
    (r"every\s+weekday", "0 9 * * 1-5", "Every weekday at 9:00 AM"),
    # Hourly
    (r"every\s+hour", "0 * * * *", "Every hour"),
    (r"hourly", "0 * * * *", "Every hour"),
]

DAY_MAP = {
    "monday": "1", "tuesday": "2", "wednesday": "3",
    "thursday": "4", "friday": "5", "saturday": "6", "sunday": "0",
    "mon": "1", "tue": "2", "wed": "3", "thu": "4",
    "fri": "5", "sat": "6", "sun": "0",
}


def parse_time(hour_str: str, minute_str: str | None, ampm: str | None) -> tuple[int, int]:
    """Parse time components into 24-hour format."""
    hour = int(hour_str)
    minute = int(minute_str) if minute_str else 0

    if ampm:
        ampm = ampm.lower()
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0

    return hour, minute


def parse_schedule(text: str) -> ParsedSchedule | None:
    """Parse a natural language schedule into a cron expression.

    Args:
        text: Natural language schedule description.

    Returns:
        ParsedSchedule or None if unable to parse.

    Examples:
        "every day at 9am" → "0 9 * * *"
        "every 30 minutes" → "0/30 * * * *"
        "every monday" → "0 9 * * 1"
        "every morning" → "0 9 * * *"
    """
    text_lower = text.lower().strip()

    # Every N minutes
    match = re.search(r"every\s+(\d+)\s+minutes?", text_lower)
    if match:
        minutes = int(match.group(1))
        if 1 <= minutes <= 59:
            return ParsedSchedule(
                cron_expression=f"*/{minutes} * * * *",
                description=f"Every {minutes} minutes",
                confidence=0.9,
            )

    # Every N hours
    match = re.search(r"every\s+(\d+)\s+hours?", text_lower)
    if match:
        hours = int(match.group(1))
        if 1 <= hours <= 23:
            return ParsedSchedule(
                cron_expression=f"0 */{hours} * * *",
                description=f"Every {hours} hours",
                confidence=0.9,
            )

    # Daily at specific time
    match = re.search(
        r"(?:every\s+day|daily)\s+at\s+(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)?",
        text_lower,
    )
    if match:
        hour, minute = parse_time(match.group(1), match.group(2), match.group(3))
        return ParsedSchedule(
            cron_expression=f"{minute} {hour} * * *",
            description=f"Daily at {hour:02d}:{minute:02d}",
            confidence=0.9,
        )

    # Every morning/evening/night
    for pattern, cron, desc in [
        (r"every\s+morning", "0 9 * * *", "Every morning at 9:00 AM"),
        (r"every\s+evening", "0 18 * * *", "Every evening at 6:00 PM"),
        (r"every\s+night", "0 21 * * *", "Every night at 9:00 PM"),
    ]:
        if re.search(pattern, text_lower):
            return ParsedSchedule(cron_expression=cron, description=desc, confidence=0.8)

    # Every specific day
    match = re.search(
        r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)",
        text_lower,
    )
    if match:
        day = match.group(1)
        day_num = DAY_MAP.get(day, "1")
        # Check for time
        time_match = re.search(r"at\s+(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)?", text_lower)
        if time_match:
            hour, minute = parse_time(time_match.group(1), time_match.group(2), time_match.group(3))
        else:
            hour, minute = 9, 0
        return ParsedSchedule(
            cron_expression=f"{minute} {hour} * * {day_num}",
            description=f"Every {day.capitalize()} at {hour:02d}:{minute:02d}",
            confidence=0.85,
        )

    # Every weekday
    if re.search(r"every\s+weekday", text_lower):
        return ParsedSchedule(
            cron_expression="0 9 * * 1-5",
            description="Every weekday at 9:00 AM",
            confidence=0.85,
        )

    # Every hour / hourly
    if re.search(r"every\s+hour|hourly", text_lower):
        return ParsedSchedule(
            cron_expression="0 * * * *",
            description="Every hour",
            confidence=0.9,
        )

    # Weekly
    if re.search(r"^weekly$", text_lower):
        return ParsedSchedule(
            cron_expression="0 9 * * 1",
            description="Weekly on Monday at 9:00 AM",
            confidence=0.7,
        )

    return None
