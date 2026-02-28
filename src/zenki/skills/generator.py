"""Skill generation: creates new skill directories from structured input or LLM responses."""

from __future__ import annotations

import re
from pathlib import Path


class SkillGenerator:
    """Generates new skill directories with SKILL.md files.

    Parameters
    ----------
    output_dir:
        The directory where new (pending) skill directories are created.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    # Generate from structured input
    # ------------------------------------------------------------------

    def generate_skill(
        self,
        name: str,
        description: str,
        instructions: str,
        allowed_tools: list[str] | None = None,
        templates: dict[str, str] | None = None,
    ) -> Path:
        """Create a skill directory with a SKILL.md file.

        Parameters
        ----------
        name:
            The skill name (used as the directory name and in frontmatter).
        description:
            A short description of what the skill does.
        instructions:
            The markdown body / instructions for the skill.
        allowed_tools:
            Optional list of tool names the skill is allowed to use.
        templates:
            Optional mapping of template filename -> content.

        Returns
        -------
        Path
            Path to the created skill directory.
        """
        # Sanitize the name for use as a directory name
        dir_name = self._sanitize_name(name)
        skill_dir = self.output_dir / dir_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Build YAML frontmatter
        frontmatter_lines = [
            "---",
            f"name: {name}",
            f"description: {description}",
        ]

        if allowed_tools:
            frontmatter_lines.append("allowed-tools:")
            for tool in allowed_tools:
                frontmatter_lines.append(f"  - {tool}")

        frontmatter_lines.append("---")

        # Compose the full SKILL.md content
        skill_md_content = "\n".join(frontmatter_lines) + "\n\n" + instructions.strip() + "\n"

        skill_md_path = skill_dir / "SKILL.md"
        skill_md_path.write_text(skill_md_content, encoding="utf-8")

        # Create templates if provided
        if templates:
            templates_dir = skill_dir / "templates"
            templates_dir.mkdir(exist_ok=True)
            for filename, content in templates.items():
                tpl_path = templates_dir / filename
                tpl_path.write_text(content, encoding="utf-8")

        return skill_dir

    # ------------------------------------------------------------------
    # Generate from LLM response
    # ------------------------------------------------------------------

    def generate_from_llm_response(self, response: str) -> Path | None:
        """Parse an LLM response that contains a skill definition and create it.

        The expected format is a fenced code block (```markdown or ```yaml or
        just ```) containing a full SKILL.md with YAML frontmatter.

        Parameters
        ----------
        response:
            The raw LLM response text.

        Returns
        -------
        Path | None
            Path to the created skill directory, or None if parsing fails.
        """
        # Try to extract a fenced code block containing SKILL.md content
        skill_content = self._extract_skill_block(response)
        if skill_content is None:
            return None

        # Parse the frontmatter to get the skill name
        name = self._extract_name_from_frontmatter(skill_content)
        if name is None:
            return None

        # Create the skill directory
        dir_name = self._sanitize_name(name)
        skill_dir = self.output_dir / dir_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Write the SKILL.md
        skill_md_path = skill_dir / "SKILL.md"
        skill_md_path.write_text(skill_content, encoding="utf-8")

        return skill_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Convert a skill name to a safe directory name."""
        # Replace non-alphanumeric chars (except hyphens) with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        return sanitized.strip("_").lower()

    @staticmethod
    def _extract_skill_block(response: str) -> str | None:
        """Extract SKILL.md content from a fenced code block in an LLM response."""
        # Match ```markdown ... ```, ```yaml ... ```, or ``` ... ```
        pattern = r"```(?:markdown|yaml|md)?\s*\n(---\n.*?\n---\n.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip() + "\n"

        # Fallback: look for raw frontmatter (--- ... --- followed by content)
        pattern_raw = r"(---\n.*?\n---\n.+)"
        match_raw = re.search(pattern_raw, response, re.DOTALL)
        if match_raw:
            return match_raw.group(1).strip() + "\n"

        return None

    @staticmethod
    def _extract_name_from_frontmatter(content: str) -> str | None:
        """Extract the 'name' field from YAML frontmatter."""
        match = re.search(r"^name:\s*(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None
