"""Comprehensive tests for the Zenki skills system.

Covers:
- SkillLoader: discovery, metadata parsing, full loading
- SkillRegistry: sync, availability, approval/reject/disable workflows
- SkillExecutor: execution flow and argument substitution
- SkillGenerator: skill creation from structured input and LLM responses
- Core skills: all 4 SKILL.md files exist and parse correctly
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zenki.db.database import ZenkiDatabase
from zenki.skills.executor import SkillExecutor
from zenki.skills.generator import SkillGenerator
from zenki.skills.loader import SkillContent, SkillLoader, SkillMetadata
from zenki.skills.registry import SkillRegistry

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def core_skills_dir() -> Path:
    """Return the path to the real core_skills directory."""
    return Path(__file__).resolve().parents[2] / "src" / "zenki" / "skills" / "core_skills"


@pytest.fixture()
def db(tmp_path: Path) -> ZenkiDatabase:
    """Create a fresh in-memory-like database in tmp_path."""
    db = ZenkiDatabase(tmp_path / "test.db")
    db.initialize()
    return db


@pytest.fixture()
def sample_skill_dir(tmp_path: Path) -> Path:
    """Create a minimal skill directory with SKILL.md for testing."""
    skill_dir = tmp_path / "core" / "my_skill"
    skill_dir.mkdir(parents=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: my-skill\n"
        "description: A test skill.\n"
        "allowed-tools:\n"
        "  - Bash\n"
        "  - Read\n"
        "---\n"
        "\n"
        "# My Skill\n"
        "\n"
        "Do the thing with $ARGUMENTS.\n"
        "First arg: $0\n"
        "Second arg: $1\n",
        encoding="utf-8",
    )

    # Add a template
    tpl_dir = skill_dir / "templates"
    tpl_dir.mkdir()
    (tpl_dir / "example.txt").write_text("Hello {{ name }}", encoding="utf-8")

    # Add a script
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "setup.sh").write_text("#!/bin/bash\necho setup", encoding="utf-8")

    return skill_dir


@pytest.fixture()
def user_skill_dir(tmp_path: Path) -> Path:
    """Create a user/learned skill directory."""
    skill_dir = tmp_path / "learned" / "custom_skill"
    skill_dir.mkdir(parents=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: custom-skill\n"
        "description: A custom learned skill.\n"
        "---\n"
        "\n"
        "# Custom Skill\n"
        "\n"
        "Custom instructions here.\n",
        encoding="utf-8",
    )
    return skill_dir


# ======================================================================
# SkillLoader Tests
# ======================================================================


class TestSkillLoader:
    """Tests for SkillLoader: discover, load_metadata, load_full."""

    def test_discover_finds_core_skills(self, core_skills_dir: Path) -> None:
        """discover() should find all 4 core SKILL.md files."""
        loader = SkillLoader(core_skills_dir=core_skills_dir)
        discovered = loader.discover()

        names = {m.name for m in discovered}
        assert "software-engineering" in names
        assert "web-research" in names
        assert "file-management" in names
        assert "system-operations" in names
        assert len(discovered) >= 4

    def test_discover_both_directories(
        self, sample_skill_dir: Path, user_skill_dir: Path, tmp_path: Path
    ) -> None:
        """discover() should scan both core and user directories."""
        core_dir = sample_skill_dir.parent  # tmp_path / "core"
        user_dir = user_skill_dir.parent  # tmp_path / "learned"
        loader = SkillLoader(core_skills_dir=core_dir, user_skills_dir=user_dir)
        discovered = loader.discover()

        names = {m.name for m in discovered}
        assert "my-skill" in names
        assert "custom-skill" in names
        assert len(discovered) == 2

    def test_discover_empty_directory(self, tmp_path: Path) -> None:
        """discover() should return an empty list for an empty directory."""
        empty = tmp_path / "empty"
        empty.mkdir()
        loader = SkillLoader(core_skills_dir=empty)
        assert loader.discover() == []

    def test_discover_nonexistent_directory(self, tmp_path: Path) -> None:
        """discover() should handle nonexistent directories gracefully."""
        loader = SkillLoader(core_skills_dir=tmp_path / "does_not_exist")
        assert loader.discover() == []

    def test_load_metadata_parses_frontmatter(self, sample_skill_dir: Path, tmp_path: Path) -> None:
        """load_metadata() should correctly parse YAML frontmatter."""
        core_dir = sample_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir)
        meta = loader.load_metadata(sample_skill_dir / "SKILL.md")

        assert isinstance(meta, SkillMetadata)
        assert meta.name == "my-skill"
        assert meta.description == "A test skill."
        assert meta.path == sample_skill_dir
        assert meta.skill_type == "core"
        assert meta.allowed_tools == ["Bash", "Read"]
        assert meta.disable_model_invocation is False

    def test_load_metadata_skill_type_learned(
        self, user_skill_dir: Path, tmp_path: Path
    ) -> None:
        """load_metadata() should infer 'learned' type for user skills."""
        core_dir = tmp_path / "core"
        core_dir.mkdir(exist_ok=True)
        user_dir = user_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir, user_skills_dir=user_dir)
        meta = loader.load_metadata(user_skill_dir / "SKILL.md")

        assert meta.skill_type == "learned"

    def test_load_metadata_disable_model_invocation(self, tmp_path: Path) -> None:
        """load_metadata() should parse disable-model-invocation correctly."""
        skill_dir = tmp_path / "core" / "restricted"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: restricted\n"
            "description: Restricted skill.\n"
            "disable-model-invocation: true\n"
            "---\n"
            "\n"
            "Restricted.\n",
            encoding="utf-8",
        )
        loader = SkillLoader(core_skills_dir=tmp_path / "core")
        meta = loader.load_metadata(skill_dir / "SKILL.md")
        assert meta.disable_model_invocation is True

    def test_load_full_gets_body(self, sample_skill_dir: Path, tmp_path: Path) -> None:
        """load_full() should return the full content including body."""
        core_dir = sample_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir)
        content = loader.load_full(sample_skill_dir / "SKILL.md")

        assert isinstance(content, SkillContent)
        assert content.metadata.name == "my-skill"
        assert "# My Skill" in content.body
        assert "$ARGUMENTS" in content.body

    def test_load_full_finds_templates(self, sample_skill_dir: Path, tmp_path: Path) -> None:
        """load_full() should discover template files."""
        core_dir = sample_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir)
        content = loader.load_full(sample_skill_dir / "SKILL.md")

        assert "example.txt" in content.templates
        assert "Hello {{ name }}" in content.templates["example.txt"]

    def test_load_full_finds_scripts(self, sample_skill_dir: Path, tmp_path: Path) -> None:
        """load_full() should discover script files."""
        core_dir = sample_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir)
        content = loader.load_full(sample_skill_dir / "SKILL.md")

        script_names = [s.name for s in content.scripts]
        assert "setup.sh" in script_names


# ======================================================================
# SkillRegistry Tests
# ======================================================================


class TestSkillRegistry:
    """Tests for SkillRegistry: sync, queries, approval workflow."""

    def test_sync_skills_creates_db_entries(
        self, db: ZenkiDatabase, sample_skill_dir: Path, tmp_path: Path
    ) -> None:
        """sync_skills() should insert discovered skills into the database."""
        core_dir = sample_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir)
        registry = SkillRegistry(db=db, loader=loader)

        registry.sync_skills()

        skills = db.list_skills()
        assert len(skills) == 1
        assert skills[0].name == "my-skill"
        assert skills[0].skill_type == "core"
        assert skills[0].status == "approved"

    def test_sync_skills_learned_starts_pending(
        self, db: ZenkiDatabase, user_skill_dir: Path, tmp_path: Path
    ) -> None:
        """sync_skills() should set learned skills to 'pending' status."""
        core_dir = tmp_path / "empty_core"
        core_dir.mkdir()
        user_dir = user_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir, user_skills_dir=user_dir)
        registry = SkillRegistry(db=db, loader=loader)

        registry.sync_skills()

        skills = db.list_skills()
        assert len(skills) == 1
        assert skills[0].name == "custom-skill"
        assert skills[0].skill_type == "learned"
        assert skills[0].status == "pending"

    def test_sync_skills_idempotent(
        self, db: ZenkiDatabase, sample_skill_dir: Path, tmp_path: Path
    ) -> None:
        """sync_skills() should not duplicate entries on repeated calls."""
        core_dir = sample_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir)
        registry = SkillRegistry(db=db, loader=loader)

        registry.sync_skills()
        registry.sync_skills()

        skills = db.list_skills()
        assert len(skills) == 1

    def test_get_available_skills(
        self, db: ZenkiDatabase, sample_skill_dir: Path, tmp_path: Path
    ) -> None:
        """get_available_skills() returns only approved skills."""
        core_dir = sample_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir)
        registry = SkillRegistry(db=db, loader=loader)
        registry.sync_skills()

        available = registry.get_available_skills()
        assert len(available) == 1
        assert available[0].name == "my-skill"

    def test_get_pending_skills(
        self, db: ZenkiDatabase, user_skill_dir: Path, tmp_path: Path
    ) -> None:
        """get_pending_skills() returns only pending skills."""
        core_dir = tmp_path / "empty_core"
        core_dir.mkdir()
        user_dir = user_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir, user_skills_dir=user_dir)
        registry = SkillRegistry(db=db, loader=loader)
        registry.sync_skills()

        pending = registry.get_pending_skills()
        assert len(pending) == 1
        assert pending[0].name == "custom-skill"

    def test_approve_skill(
        self, db: ZenkiDatabase, user_skill_dir: Path, tmp_path: Path
    ) -> None:
        """approve_skill() should move a skill from pending to approved."""
        core_dir = tmp_path / "empty_core"
        core_dir.mkdir()
        user_dir = user_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir, user_skills_dir=user_dir)
        registry = SkillRegistry(db=db, loader=loader)
        registry.sync_skills()

        skill = registry.approve_skill("custom-skill")
        assert skill.status == "approved"
        assert skill.approved_at is not None

        # Should now appear in available
        available = registry.get_available_skills()
        assert any(s.name == "custom-skill" for s in available)

    def test_reject_skill(
        self, db: ZenkiDatabase, user_skill_dir: Path, tmp_path: Path
    ) -> None:
        """reject_skill() should mark a skill as rejected."""
        core_dir = tmp_path / "empty_core"
        core_dir.mkdir()
        user_dir = user_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir, user_skills_dir=user_dir)
        registry = SkillRegistry(db=db, loader=loader)
        registry.sync_skills()

        registry.reject_skill("custom-skill")

        pending = registry.get_pending_skills()
        assert len(pending) == 0

        available = registry.get_available_skills()
        assert not any(s.name == "custom-skill" for s in available)

    def test_disable_skill(
        self, db: ZenkiDatabase, sample_skill_dir: Path, tmp_path: Path
    ) -> None:
        """disable_skill() should remove a skill from available list."""
        core_dir = sample_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir)
        registry = SkillRegistry(db=db, loader=loader)
        registry.sync_skills()

        registry.disable_skill("my-skill")

        available = registry.get_available_skills()
        assert len(available) == 0

    def test_record_usage(
        self, db: ZenkiDatabase, sample_skill_dir: Path, tmp_path: Path
    ) -> None:
        """record_usage() should increment usage count and update timestamp."""
        core_dir = sample_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir)
        registry = SkillRegistry(db=db, loader=loader)
        registry.sync_skills()

        registry.record_usage("my-skill")
        registry.record_usage("my-skill")

        skill = registry._get_by_name("my-skill")
        assert skill.usage_count == 2
        assert skill.last_used_at is not None

    def test_get_by_name_raises_for_unknown(
        self, db: ZenkiDatabase, tmp_path: Path
    ) -> None:
        """_get_by_name() should raise ValueError for unknown skills."""
        core_dir = tmp_path / "empty_core"
        core_dir.mkdir()
        loader = SkillLoader(core_skills_dir=core_dir)
        registry = SkillRegistry(db=db, loader=loader)

        with pytest.raises(ValueError, match="Skill not found"):
            registry._get_by_name("nonexistent")

    def test_get_skill_descriptions(
        self, db: ZenkiDatabase, sample_skill_dir: Path, tmp_path: Path
    ) -> None:
        """get_skill_descriptions() should return a formatted string."""
        core_dir = sample_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir)
        registry = SkillRegistry(db=db, loader=loader)
        registry.sync_skills()

        desc = registry.get_skill_descriptions()
        assert "my-skill" in desc
        assert "A test skill." in desc

    def test_get_skill_descriptions_empty(
        self, db: ZenkiDatabase, tmp_path: Path
    ) -> None:
        """get_skill_descriptions() should handle no skills gracefully."""
        core_dir = tmp_path / "empty_core"
        core_dir.mkdir()
        loader = SkillLoader(core_skills_dir=core_dir)
        registry = SkillRegistry(db=db, loader=loader)

        desc = registry.get_skill_descriptions()
        assert desc == "No skills available."


# ======================================================================
# SkillExecutor Tests
# ======================================================================


class TestSkillExecutor:
    """Tests for SkillExecutor: execute and apply_substitutions."""

    def test_execute_loads_and_returns_content(
        self, db: ZenkiDatabase, sample_skill_dir: Path, tmp_path: Path
    ) -> None:
        """execute() should load a skill and return its processed content."""
        core_dir = sample_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir)
        registry = SkillRegistry(db=db, loader=loader)
        registry.sync_skills()

        executor = SkillExecutor(registry=registry, loader=loader)
        result = executor.execute("my-skill", "hello world")

        assert "# My Skill" in result
        assert "hello world" in result  # $ARGUMENTS replaced

    def test_execute_increments_usage(
        self, db: ZenkiDatabase, sample_skill_dir: Path, tmp_path: Path
    ) -> None:
        """execute() should record usage after loading a skill."""
        core_dir = sample_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir)
        registry = SkillRegistry(db=db, loader=loader)
        registry.sync_skills()

        executor = SkillExecutor(registry=registry, loader=loader)
        executor.execute("my-skill")

        skill = registry._get_by_name("my-skill")
        assert skill.usage_count == 1

    def test_execute_raises_for_unapproved(
        self, db: ZenkiDatabase, user_skill_dir: Path, tmp_path: Path
    ) -> None:
        """execute() should raise ValueError for unapproved skills."""
        core_dir = tmp_path / "empty_core"
        core_dir.mkdir()
        user_dir = user_skill_dir.parent
        loader = SkillLoader(core_skills_dir=core_dir, user_skills_dir=user_dir)
        registry = SkillRegistry(db=db, loader=loader)
        registry.sync_skills()

        executor = SkillExecutor(registry=registry, loader=loader)

        with pytest.raises(ValueError, match="not approved"):
            executor.execute("custom-skill")

    def test_execute_raises_for_unknown(
        self, db: ZenkiDatabase, tmp_path: Path
    ) -> None:
        """execute() should raise ValueError for unknown skills."""
        core_dir = tmp_path / "empty_core"
        core_dir.mkdir()
        loader = SkillLoader(core_skills_dir=core_dir)
        registry = SkillRegistry(db=db, loader=loader)

        executor = SkillExecutor(registry=registry, loader=loader)

        with pytest.raises(ValueError, match="Skill not found"):
            executor.execute("nonexistent")

    def test_apply_substitutions_full_arguments(self) -> None:
        """apply_substitutions() should replace $ARGUMENTS with the full string."""
        content = "Do this: $ARGUMENTS"
        result = SkillExecutor.apply_substitutions(content, "foo bar baz")
        assert result == "Do this: foo bar baz"

    def test_apply_substitutions_indexed_arguments(self) -> None:
        """apply_substitutions() should replace $ARGUMENTS[N] placeholders."""
        content = "First: $ARGUMENTS[0], Second: $ARGUMENTS[1]"
        result = SkillExecutor.apply_substitutions(content, "alpha beta")
        assert result == "First: alpha, Second: beta"

    def test_apply_substitutions_positional(self) -> None:
        """apply_substitutions() should replace $N positional placeholders."""
        content = "Repo: $0, Branch: $1"
        result = SkillExecutor.apply_substitutions(content, "myrepo main")
        assert result == "Repo: myrepo, Branch: main"

    def test_apply_substitutions_empty_arguments(self) -> None:
        """apply_substitutions() should handle empty arguments gracefully."""
        content = "Do: $ARGUMENTS (nothing)"
        result = SkillExecutor.apply_substitutions(content, "")
        assert result == "Do:  (nothing)"

    def test_apply_substitutions_quoted_arguments(self) -> None:
        """apply_substitutions() should handle quoted arguments via shlex."""
        content = "Path: $0"
        result = SkillExecutor.apply_substitutions(content, '"my file.txt"')
        assert result == "Path: my file.txt"


# ======================================================================
# SkillGenerator Tests
# ======================================================================


class TestSkillGenerator:
    """Tests for SkillGenerator: generate_skill and generate_from_llm_response."""

    def test_generate_skill_creates_directory(self, tmp_path: Path) -> None:
        """generate_skill() should create a skill directory."""
        gen = SkillGenerator(output_dir=tmp_path)
        skill_dir = gen.generate_skill(
            name="test-skill",
            description="A test skill.",
            instructions="# Test\n\nDo things.",
        )

        assert skill_dir.is_dir()
        assert (skill_dir / "SKILL.md").exists()

    def test_generate_skill_creates_valid_skill_md(self, tmp_path: Path) -> None:
        """generate_skill() should create a valid SKILL.md with frontmatter."""
        gen = SkillGenerator(output_dir=tmp_path)
        skill_dir = gen.generate_skill(
            name="test-skill",
            description="A test skill.",
            instructions="# Test\n\nDo things.",
            allowed_tools=["Bash", "Read"],
        )

        skill_md = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert "name: test-skill" in skill_md
        assert "description: A test skill." in skill_md
        assert "- Bash" in skill_md
        assert "- Read" in skill_md
        assert "# Test" in skill_md

    def test_generate_skill_with_templates(self, tmp_path: Path) -> None:
        """generate_skill() should create template files."""
        gen = SkillGenerator(output_dir=tmp_path)
        templates = {"pr.md": "## PR\n\n{{ title }}"}
        skill_dir = gen.generate_skill(
            name="test-skill",
            description="A test skill.",
            instructions="# Test",
            templates=templates,
        )

        tpl_path = skill_dir / "templates" / "pr.md"
        assert tpl_path.exists()
        assert "{{ title }}" in tpl_path.read_text(encoding="utf-8")

    def test_generate_skill_sanitizes_name(self, tmp_path: Path) -> None:
        """generate_skill() should sanitize the skill name for directory use."""
        gen = SkillGenerator(output_dir=tmp_path)
        skill_dir = gen.generate_skill(
            name="My Cool Skill!",
            description="Cool.",
            instructions="# Cool",
        )

        # Directory name should be sanitized (trailing special chars stripped)
        assert skill_dir.name == "my_cool_skill"

    def test_generate_skill_without_tools(self, tmp_path: Path) -> None:
        """generate_skill() should work without allowed_tools."""
        gen = SkillGenerator(output_dir=tmp_path)
        skill_dir = gen.generate_skill(
            name="simple",
            description="Simple skill.",
            instructions="# Simple",
        )

        skill_md = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert "allowed-tools" not in skill_md

    def test_generate_from_llm_response_success(self, tmp_path: Path) -> None:
        """generate_from_llm_response() should parse and create a skill."""
        gen = SkillGenerator(output_dir=tmp_path)
        response = (
            "Here is a new skill:\n\n"
            "```markdown\n"
            "---\n"
            "name: llm-skill\n"
            "description: Generated by LLM.\n"
            "---\n"
            "\n"
            "# LLM Skill\n"
            "\n"
            "Do LLM things.\n"
            "```\n"
        )

        skill_dir = gen.generate_from_llm_response(response)

        assert skill_dir is not None
        assert skill_dir.is_dir()
        assert (skill_dir / "SKILL.md").exists()

        content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert "name: llm-skill" in content
        assert "# LLM Skill" in content

    def test_generate_from_llm_response_no_code_block(self, tmp_path: Path) -> None:
        """generate_from_llm_response() should return None for invalid input."""
        gen = SkillGenerator(output_dir=tmp_path)
        result = gen.generate_from_llm_response("No skill definition here.")
        assert result is None

    def test_generate_from_llm_response_raw_frontmatter(self, tmp_path: Path) -> None:
        """generate_from_llm_response() should handle raw frontmatter without code fences."""
        gen = SkillGenerator(output_dir=tmp_path)
        response = (
            "---\n"
            "name: raw-skill\n"
            "description: Raw frontmatter.\n"
            "---\n"
            "\n"
            "# Raw Skill\n"
            "\n"
            "Instructions.\n"
        )

        skill_dir = gen.generate_from_llm_response(response)
        assert skill_dir is not None
        assert (skill_dir / "SKILL.md").exists()


# ======================================================================
# Core Skills Integration Tests
# ======================================================================


class TestCoreSkills:
    """Tests that all 4 core SKILL.md files exist and parse correctly."""

    CORE_SKILLS = [
        ("software_engineering", "software-engineering"),
        ("web_research", "web-research"),
        ("file_management", "file-management"),
        ("system_operations", "system-operations"),
    ]

    @pytest.fixture()
    def loader(self, core_skills_dir: Path) -> SkillLoader:
        return SkillLoader(core_skills_dir=core_skills_dir)

    @pytest.mark.parametrize("dir_name,expected_name", CORE_SKILLS)
    def test_skill_md_exists(
        self, core_skills_dir: Path, dir_name: str, expected_name: str
    ) -> None:
        """Each core skill directory should contain a SKILL.md file."""
        skill_md = core_skills_dir / dir_name / "SKILL.md"
        assert skill_md.exists(), f"Missing: {skill_md}"

    @pytest.mark.parametrize("dir_name,expected_name", CORE_SKILLS)
    def test_skill_metadata_parses(
        self, loader: SkillLoader, core_skills_dir: Path, dir_name: str, expected_name: str
    ) -> None:
        """Each core SKILL.md should have valid frontmatter with the expected name."""
        skill_md = core_skills_dir / dir_name / "SKILL.md"
        meta = loader.load_metadata(skill_md)

        assert meta.name == expected_name
        assert meta.description  # non-empty
        assert meta.skill_type == "core"

    @pytest.mark.parametrize("dir_name,expected_name", CORE_SKILLS)
    def test_skill_has_allowed_tools(
        self, loader: SkillLoader, core_skills_dir: Path, dir_name: str, expected_name: str
    ) -> None:
        """Each core skill should declare at least one allowed tool."""
        skill_md = core_skills_dir / dir_name / "SKILL.md"
        meta = loader.load_metadata(skill_md)
        assert len(meta.allowed_tools) >= 1

    @pytest.mark.parametrize("dir_name,expected_name", CORE_SKILLS)
    def test_skill_body_not_empty(
        self, loader: SkillLoader, core_skills_dir: Path, dir_name: str, expected_name: str
    ) -> None:
        """Each core skill should have a non-empty body."""
        skill_md = core_skills_dir / dir_name / "SKILL.md"
        content = loader.load_full(skill_md)
        assert content.body.strip()

    def test_system_operations_disables_model_invocation(
        self, loader: SkillLoader, core_skills_dir: Path
    ) -> None:
        """system-operations should have disable-model-invocation: true."""
        skill_md = core_skills_dir / "system_operations" / "SKILL.md"
        meta = loader.load_metadata(skill_md)
        assert meta.disable_model_invocation is True

    def test_software_engineering_has_all_tools(
        self, loader: SkillLoader, core_skills_dir: Path
    ) -> None:
        """software-engineering should allow Bash, Read, Write, Edit, Glob, Grep."""
        skill_md = core_skills_dir / "software_engineering" / "SKILL.md"
        meta = loader.load_metadata(skill_md)
        expected_tools = {"Bash", "Read", "Write", "Edit", "Glob", "Grep"}
        assert set(meta.allowed_tools) == expected_tools
