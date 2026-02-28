"""Tier 2: Core Memory - persistent user knowledge stored as markdown files."""

from __future__ import annotations

from pathlib import Path

# The canonical set of core memory files.
CORE_MEMORY_FILES = [
    "identity.md",
    "preferences.md",
    "projects.md",
    "relationships.md",
    "patterns.md",
    "personality.md",
]


class CoreMemory:
    """Manages persistent core-memory markdown files.

    Core memory is the long-term store for information about the user that
    should always be available: identity, preferences, active projects,
    relationships, behavioural patterns, and personality notes.

    Files are stored under ``<memory_dir>/`` (typically
    ``~/.config/zenki/memory/``).
    """

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_files(self) -> None:
        """Create the memory directory and default files if they do not exist."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        for filename in CORE_MEMORY_FILES:
            filepath = self.memory_dir / filename
            if not filepath.exists():
                filepath.write_text("")

    def load_all(self) -> dict[str, str]:
        """Load all core-memory files into a dictionary.

        Returns:
            Mapping of key name (filename without ``.md``) to file contents.
        """
        self.ensure_files()
        result: dict[str, str] = {}
        for filename in CORE_MEMORY_FILES:
            key = filename.removesuffix(".md")
            filepath = self.memory_dir / filename
            result[key] = filepath.read_text()
        return result

    def get(self, key: str) -> str:
        """Read the content of a single core-memory file.

        Args:
            key: The memory key (e.g. ``"identity"``). The ``.md`` extension
                is added automatically.

        Returns:
            The file content as a string. Returns an empty string if the
            file does not exist.
        """
        filepath = self.memory_dir / f"{key}.md"
        if not filepath.exists():
            return ""
        return filepath.read_text()

    def update(self, key: str, content: str) -> None:
        """Write content to a core-memory file, creating it if needed.

        Args:
            key: The memory key (e.g. ``"preferences"``).
            content: The new file content.
        """
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.memory_dir / f"{key}.md"
        filepath.write_text(content)

    def get_context_string(self) -> str:
        """Combine all core-memory files into a single context string.

        Each non-empty section is formatted with a markdown heading.

        Returns:
            A combined string suitable for inclusion in a prompt.
        """
        data = self.load_all()
        parts: list[str] = []
        for key, content in data.items():
            if content.strip():
                heading = key.replace("_", " ").title()
                parts.append(f"### {heading}\n{content.strip()}")
        if not parts:
            return ""
        return "\n\n".join(parts)
