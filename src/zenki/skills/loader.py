"""Skill discovery and loading from the filesystem.

Each skill is a directory containing a SKILL.md file with YAML frontmatter
(metadata) and a markdown body (instructions).  The loader scans configured
directories, parses the frontmatter, and returns structured dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import frontmatter

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SkillMetadata:
    """Parsed YAML frontmatter from a SKILL.md file."""

    name: str
    description: str
    path: Path
    skill_type: Literal["core", "learned"]
    allowed_tools: list[str] = field(default_factory=list)
    disable_model_invocation: bool = False


@dataclass
class SkillContent:
    """Full parsed content of a SKILL.md file (metadata + body)."""

    metadata: SkillMetadata
    body: str
    templates: dict[str, str] = field(default_factory=dict)
    scripts: list[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

SKILL_FILENAME = "SKILL.md"


class SkillLoader:
    """Discovers and loads skills from the filesystem.

    Parameters
    ----------
    core_skills_dir:
        Path to the directory containing built-in core skills.
    user_skills_dir:
        Optional path to a directory containing user-created / learned skills.
    """

    def __init__(
        self,
        core_skills_dir: Path,
        user_skills_dir: Path | None = None,
    ) -> None:
        self.core_skills_dir = core_skills_dir
        self.user_skills_dir = user_skills_dir

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> list[SkillMetadata]:
        """Scan configured directories for SKILL.md files and return metadata."""
        results: list[SkillMetadata] = []

        # Core skills
        if self.core_skills_dir.is_dir():
            for skill_md in self.core_skills_dir.rglob(SKILL_FILENAME):
                try:
                    meta = self.load_metadata(skill_md)
                    results.append(meta)
                except Exception:
                    # Skip malformed skill files during discovery
                    continue

        # User / learned skills
        if self.user_skills_dir is not None and self.user_skills_dir.is_dir():
            for skill_md in self.user_skills_dir.rglob(SKILL_FILENAME):
                try:
                    meta = self.load_metadata(skill_md)
                    results.append(meta)
                except Exception:
                    continue

        return results

    # ------------------------------------------------------------------
    # Metadata-only parsing
    # ------------------------------------------------------------------

    def load_metadata(self, skill_path: Path) -> SkillMetadata:
        """Parse YAML frontmatter from a SKILL.md file.

        Parameters
        ----------
        skill_path:
            Path to the SKILL.md file itself.

        Returns
        -------
        SkillMetadata
            Parsed metadata from the frontmatter.
        """
        post = frontmatter.load(str(skill_path))
        fm = post.metadata

        # Determine skill_type based on directory location
        skill_dir = skill_path.parent
        skill_type: Literal["core", "learned"] = self._infer_skill_type(skill_dir)

        # Parse allowed-tools (may be a list or a comma-separated string)
        raw_tools = fm.get("allowed-tools", [])
        if isinstance(raw_tools, str):
            allowed_tools = [t.strip() for t in raw_tools.split(",") if t.strip()]
        else:
            allowed_tools = list(raw_tools)

        return SkillMetadata(
            name=fm.get("name", skill_dir.name),
            description=fm.get("description", ""),
            path=skill_dir,
            skill_type=skill_type,
            allowed_tools=allowed_tools,
            disable_model_invocation=bool(fm.get("disable-model-invocation", False)),
        )

    # ------------------------------------------------------------------
    # Full content parsing
    # ------------------------------------------------------------------

    def load_full(self, skill_path: Path) -> SkillContent:
        """Parse the full SKILL.md file: metadata, body, templates, and scripts.

        Parameters
        ----------
        skill_path:
            Path to the SKILL.md file itself.
        """
        post = frontmatter.load(str(skill_path))
        metadata = self.load_metadata(skill_path)
        body = post.content

        # Discover templates
        templates: dict[str, str] = {}
        skill_dir = skill_path.parent
        templates_dir = skill_dir / "templates"
        if templates_dir.is_dir():
            for tpl_file in templates_dir.iterdir():
                if tpl_file.is_file() and not tpl_file.name.startswith("__"):
                    templates[tpl_file.name] = tpl_file.read_text(encoding="utf-8")

        # Discover scripts
        scripts: list[Path] = []
        scripts_dir = skill_dir / "scripts"
        if scripts_dir.is_dir():
            for script_file in scripts_dir.iterdir():
                if script_file.is_file() and not script_file.name.startswith("__"):
                    scripts.append(script_file)

        return SkillContent(
            metadata=metadata,
            body=body,
            templates=templates,
            scripts=scripts,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_skill_type(self, skill_dir: Path) -> Literal["core", "learned"]:
        """Determine whether a skill is core or learned based on its location."""
        try:
            skill_dir.relative_to(self.core_skills_dir)
            return "core"
        except ValueError:
            return "learned"
