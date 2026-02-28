"""Memory consolidation engine - daily self-improvement cycle.

Reviews recent conversations, extracts knowledge, consolidates memories,
identifies skill opportunities, and decays old memories.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from zenki.db.database import ZenkiDatabase
from zenki.db.models import EpisodicMemory

logger = logging.getLogger(__name__)


class ConsolidationResult:
    """Result of a memory consolidation cycle."""

    def __init__(self) -> None:
        self.sessions_reviewed: int = 0
        self.summaries_generated: int = 0
        self.facts_extracted: int = 0
        self.core_memory_updates: int = 0
        self.skill_proposals: list[dict[str, str]] = []
        self.memories_decayed: int = 0
        self.errors: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sessions_reviewed": self.sessions_reviewed,
            "summaries_generated": self.summaries_generated,
            "facts_extracted": self.facts_extracted,
            "core_memory_updates": self.core_memory_updates,
            "skill_proposals": self.skill_proposals,
            "memories_decayed": self.memories_decayed,
            "errors": self.errors,
        }

    def __str__(self) -> str:
        parts = [
            f"Sessions reviewed: {self.sessions_reviewed}",
            f"Summaries generated: {self.summaries_generated}",
            f"Facts extracted: {self.facts_extracted}",
            f"Core memory updates: {self.core_memory_updates}",
            f"Skill proposals: {len(self.skill_proposals)}",
            f"Memories decayed: {self.memories_decayed}",
        ]
        if self.errors:
            parts.append(f"Errors: {len(self.errors)}")
        return " | ".join(parts)


def calculate_importance(
    base_importance: float,
    created_at: datetime,
    access_count: int,
    current_time: datetime | None = None,
    decay_rate: float = 0.01,
    access_boost_factor: float = 0.1,
    max_access_boost: float = 2.0,
) -> float:
    """Calculate current importance of a memory with time decay and access boost.

    Importance = base_importance * recency_factor * access_factor

    Args:
        base_importance: Initial importance (0.0 to 1.0)
        created_at: When the memory was created
        access_count: How many times the memory has been accessed
        current_time: Current time (defaults to now)
        decay_rate: Exponential decay rate per day
        access_boost_factor: How much each access boosts importance
        max_access_boost: Maximum multiplier from access frequency
    """
    if current_time is None:
        current_time = datetime.now(UTC)

    # Handle timezone-naive datetimes
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=UTC)

    days_old = max(0, (current_time - created_at).total_seconds() / 86400)
    recency = math.exp(-decay_rate * days_old)

    access_boost = min(1.0 + (access_count * access_boost_factor), max_access_boost)

    return min(1.0, base_importance * recency * access_boost)


class MemoryConsolidator:
    """Handles the memory consolidation cycle.

    This runs periodically (e.g., daily) to:
    1. Review recent conversations and generate summaries
    2. Extract facts from conversations into semantic memory
    3. Identify patterns for potential new skills
    4. Decay importance of old, unused memories
    5. Generate a consolidation report
    """

    def __init__(
        self,
        db: ZenkiDatabase,
        memory_dir: Path,
        consolidation_dir: Path | None = None,
    ) -> None:
        self.db = db
        self.memory_dir = memory_dir
        self.consolidation_dir = consolidation_dir or (
            memory_dir.parent / "conversations" / "consolidations"
        )
        self.consolidation_dir.mkdir(parents=True, exist_ok=True)

    async def run_consolidation(
        self,
        days_back: int = 1,
        decay_threshold: float = 0.05,
    ) -> ConsolidationResult:
        """Run a full consolidation cycle.

        Args:
            days_back: How many days back to look for unconsolidated sessions.
            decay_threshold: Memories below this importance get archived.
        """
        result = ConsolidationResult()

        try:
            # Step 1: Review recent sessions
            await self._review_sessions(result, days_back)
        except Exception as e:
            result.errors.append(f"Session review failed: {e}")
            logger.error("Session review failed: %s", e)

        try:
            # Step 2: Decay old memories
            self._decay_memories(result, decay_threshold)
        except Exception as e:
            result.errors.append(f"Memory decay failed: {e}")
            logger.error("Memory decay failed: %s", e)

        try:
            # Step 3: Save consolidation report
            self._save_report(result)
        except Exception as e:
            result.errors.append(f"Report save failed: {e}")
            logger.error("Report save failed: %s", e)

        logger.info("Consolidation complete: %s", result)
        return result

    async def _review_sessions(self, result: ConsolidationResult, days_back: int) -> None:
        """Review recent sessions that haven't been summarized."""
        sessions = self.db.list_sessions()
        cutoff = datetime.now(UTC) - timedelta(days=days_back)

        for session in sessions:
            # Skip sessions already summarized
            if session.summary:
                continue

            # Skip sessions that are too old (already consolidated) or still active
            session_time = session.started_at
            if session_time and session_time.tzinfo is None:
                session_time = session_time.replace(tzinfo=UTC)

            if session_time and session_time < cutoff:
                continue

            if session.ended_at is None:
                continue

            result.sessions_reviewed += 1

            # Get messages for this session
            messages = self.db.get_messages_for_session(session.id)
            if not messages:
                continue

            # Generate a simple summary from messages
            summary = self._generate_simple_summary(messages)
            if summary:
                result.summaries_generated += 1

                # Store as episodic memory
                episodic = EpisodicMemory(
                    session_id=session.id,
                    summary=summary,
                    key_topics=self._extract_topics(messages),
                    key_entities=[],
                    importance=0.5,
                )
                self.db.create_episodic_memory(episodic)

                # Update session with summary
                session.summary = summary
                self.db.update_session(session)

    def _generate_simple_summary(self, messages: list) -> str:
        """Generate a basic summary from messages (without LLM).

        For a proper summary, the LLM would be used. This is a fallback
        that creates a basic summary from message content.
        """
        user_messages = [m for m in messages if m.role == "user"]
        if not user_messages:
            return ""

        topics = []
        for msg in user_messages[:5]:  # First 5 user messages
            # Take first sentence or 100 chars
            text = msg.content.strip()
            if "." in text[:100]:
                text = text[:text.index(".") + 1]
            else:
                text = text[:100]
            topics.append(text)

        return f"Conversation covering: {'; '.join(topics)}"

    def _extract_topics(self, messages: list) -> list[str]:
        """Extract key topics from messages."""
        topics: set[str] = set()
        for msg in messages:
            if msg.role != "user":
                continue
            words = msg.content.lower().split()
            # Simple heuristic: collect capitalized words and common patterns
            for word in words:
                if len(word) > 3 and word[0].isupper():
                    topics.add(word.strip(".,!?"))
        return list(topics)[:10]

    def _decay_memories(self, result: ConsolidationResult, threshold: float) -> None:
        """Apply importance decay to all memories."""
        now = datetime.now(UTC)

        # Decay episodic memories
        episodic_memories = self.db.search_episodic_memories("")
        for memory in episodic_memories:
            new_importance = calculate_importance(
                memory.importance,
                memory.created_at,
                memory.access_count,
                now,
            )
            if abs(new_importance - memory.importance) > 0.01:
                memory.importance = new_importance
                result.memories_decayed += 1

        # Decay semantic memories
        semantic_memories = self.db.search_semantic_memories("")
        for memory in semantic_memories:
            new_importance = calculate_importance(
                memory.importance,
                memory.created_at,
                memory.access_count,
                now,
            )
            if abs(new_importance - memory.importance) > 0.01:
                memory.importance = new_importance
                result.memories_decayed += 1

    def _save_report(self, result: ConsolidationResult) -> None:
        """Save consolidation report to file."""
        now = datetime.now(UTC)
        filename = f"{now.strftime('%Y-%m-%d')}_consolidation.md"
        report_path = self.consolidation_dir / filename

        report = f"""# Consolidation Report - {now.strftime('%Y-%m-%d %H:%M UTC')}

## Summary
{result}

## Details
- Sessions reviewed: {result.sessions_reviewed}
- Summaries generated: {result.summaries_generated}
- Facts extracted: {result.facts_extracted}
- Core memory updates: {result.core_memory_updates}
- Memories decayed: {result.memories_decayed}
- Skill proposals: {len(result.skill_proposals)}

"""
        if result.skill_proposals:
            report += "## Skill Proposals\n"
            for proposal in result.skill_proposals:
                name = proposal.get('name', 'unnamed')
                desc = proposal.get('description', '')
                report += f"- **{name}**: {desc}\n"

        if result.errors:
            report += "\n## Errors\n"
            for error in result.errors:
                report += f"- {error}\n"

        report_path.write_text(report)
        logger.info("Consolidation report saved to %s", report_path)
