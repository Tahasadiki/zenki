"""Skill registry: syncs filesystem skills with the database and manages approval workflows."""

from __future__ import annotations

from datetime import UTC, datetime

from zenki.db.database import ZenkiDatabase
from zenki.db.models import Skill
from zenki.skills.loader import SkillLoader


class SkillRegistry:
    """Manages skill registration, approval, and availability.

    Parameters
    ----------
    db:
        The Zenki database instance for persisting skill state.
    loader:
        The SkillLoader used to discover skills on the filesystem.
    """

    def __init__(self, db: ZenkiDatabase, loader: SkillLoader) -> None:
        self.db = db
        self.loader = loader

    # ------------------------------------------------------------------
    # Sync filesystem -> database
    # ------------------------------------------------------------------

    def sync_skills(self) -> None:
        """Discover skills from the filesystem and sync them into the database.

        New skills are inserted. Core skills are automatically approved.
        Learned skills start in "pending" status (unless already in the DB).
        Existing DB entries are *not* overwritten so that user-set statuses
        (approved / rejected / disabled) are preserved.
        """
        discovered = self.loader.discover()
        existing = {s.name: s for s in self.db.list_skills()}

        for meta in discovered:
            if meta.name in existing:
                # Skill already tracked -- leave its status alone
                continue

            now = datetime.now(UTC)
            status = "approved" if meta.skill_type == "core" else "pending"

            skill = Skill(
                name=meta.name,
                skill_type=meta.skill_type,
                status=status,
                path=str(meta.path),
                description=meta.description,
                approved_at=now if status == "approved" else None,
                created_at=now,
                metadata={
                    "allowed_tools": meta.allowed_tools,
                    "disable_model_invocation": meta.disable_model_invocation,
                },
            )
            self.db.create_skill(skill)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_available_skills(self) -> list[Skill]:
        """Return all skills that are approved (core skills are always approved)."""
        return self.db.list_skills(status="approved")

    def get_pending_skills(self) -> list[Skill]:
        """Return skills awaiting approval."""
        return self.db.list_skills(status="pending")

    # ------------------------------------------------------------------
    # Approval workflow
    # ------------------------------------------------------------------

    def approve_skill(self, skill_name: str) -> Skill:
        """Approve a skill, making it available for use.

        Raises
        ------
        ValueError
            If the skill is not found.
        """
        skill = self._get_by_name(skill_name)
        skill.status = "approved"
        skill.approved_at = datetime.now(UTC)
        return self.db.update_skill(skill)

    def reject_skill(self, skill_name: str) -> None:
        """Reject a skill so it will not be used.

        Raises
        ------
        ValueError
            If the skill is not found.
        """
        skill = self._get_by_name(skill_name)
        skill.status = "rejected"
        self.db.update_skill(skill)

    def disable_skill(self, skill_name: str) -> None:
        """Disable a previously approved skill.

        Raises
        ------
        ValueError
            If the skill is not found.
        """
        skill = self._get_by_name(skill_name)
        skill.status = "disabled"
        self.db.update_skill(skill)

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def record_usage(self, skill_name: str) -> None:
        """Increment the usage count and update last-used timestamp.

        Raises
        ------
        ValueError
            If the skill is not found.
        """
        skill = self._get_by_name(skill_name)
        skill.usage_count += 1
        skill.last_used_at = datetime.now(UTC)
        self.db.update_skill(skill)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def get_skill_descriptions(self) -> str:
        """Return a formatted string of all available skills for use in prompts."""
        available = self.get_available_skills()
        if not available:
            return "No skills available."

        lines: list[str] = []
        for skill in available:
            desc = skill.description or "No description."
            lines.append(f"- **{skill.name}**: {desc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_by_name(self, skill_name: str) -> Skill:
        """Look up a skill by name, raising ValueError if not found."""
        all_skills = self.db.list_skills()
        for s in all_skills:
            if s.name == skill_name:
                return s
        raise ValueError(f"Skill not found: {skill_name}")
