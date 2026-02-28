"""Skill execution: loads a skill and prepares its content for LLM consumption."""

from __future__ import annotations

import shlex

from zenki.skills.loader import SkillLoader
from zenki.skills.registry import SkillRegistry


class SkillExecutor:
    """Loads and executes skills by preparing their content for the LLM.

    "Execution" means loading the skill's SKILL.md, applying argument
    substitutions, and returning the processed content as context that
    the LLM can act upon.

    Parameters
    ----------
    registry:
        The SkillRegistry used for availability checks and usage tracking.
    loader:
        The SkillLoader used to load skill content from the filesystem.
    """

    def __init__(self, registry: SkillRegistry, loader: SkillLoader) -> None:
        self.registry = registry
        self.loader = loader

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, skill_name: str, arguments: str = "") -> str:
        """Load a skill and return its processed content.

        Parameters
        ----------
        skill_name:
            The name of the skill to execute.
        arguments:
            A string of arguments passed to the skill (space-separated).

        Returns
        -------
        str
            The skill body with argument substitutions applied.

        Raises
        ------
        ValueError
            If the skill is not found or not available.
        """
        # Look up the skill in the registry
        skill_db = self.registry._get_by_name(skill_name)
        if skill_db.status != "approved":
            raise ValueError(
                f"Skill '{skill_name}' is not approved (status: {skill_db.status})"
            )

        # Load the full skill content from the filesystem
        from pathlib import Path

        skill_dir = Path(skill_db.path)
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            raise ValueError(f"SKILL.md not found at {skill_md}")

        content = self.loader.load_full(skill_md)

        # Apply substitutions
        processed_body = self.apply_substitutions(content.body, arguments)

        # Record usage
        self.registry.record_usage(skill_name)

        return processed_body

    # ------------------------------------------------------------------
    # Substitutions
    # ------------------------------------------------------------------

    @staticmethod
    def apply_substitutions(content: str, arguments: str) -> str:
        """Replace skill placeholders with actual argument values.

        Supported placeholders:
        - ``$ARGUMENTS``  -- the full arguments string
        - ``$ARGUMENTS[N]`` -- the N-th argument (0-indexed)
        - ``$0``, ``$1``, ... ``$N`` -- shorthand for positional arguments

        Parameters
        ----------
        content:
            The raw skill body text.
        arguments:
            Space-separated argument string.

        Returns
        -------
        str
            Content with all placeholders replaced.
        """
        # Parse arguments using shell-like splitting
        try:
            parts = shlex.split(arguments) if arguments else []
        except ValueError:
            # If shlex fails (unbalanced quotes), fall back to simple split
            parts = arguments.split() if arguments else []

        # Replace $ARGUMENTS[N] placeholders first (before $ARGUMENTS)
        result = content
        for i, part in enumerate(parts):
            result = result.replace(f"$ARGUMENTS[{i}]", part)

        # Replace $ARGUMENTS (the full string) -- must come after indexed replacements
        result = result.replace("$ARGUMENTS", arguments)

        # Replace positional $N placeholders (check longer numbers first)
        for i in range(len(parts) - 1, -1, -1):
            result = result.replace(f"${i}", parts[i])

        return result
