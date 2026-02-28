"""Tests for the Zenki CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zenki import __version__
from zenki.cli.app import app
from zenki.config.settings import ZenkiSettings
from zenki.db.database import ZenkiDatabase

runner = CliRunner()


# ---------------------------------------------------------------------------
# zenki --version
# ---------------------------------------------------------------------------


class TestVersion:
    """Tests for the --version flag."""

    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_short_flag(self) -> None:
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert __version__ in result.output


# ---------------------------------------------------------------------------
# zenki (no args)
# ---------------------------------------------------------------------------


class TestHelp:
    """Tests for the help output."""

    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        # Typer's no_args_is_help returns exit code 0 or 2 depending on version.
        assert result.exit_code in (0, 2)
        assert "Zenki" in result.output

    def test_help_flag(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "setup" in result.output
        assert "chat" in result.output
        assert "status" in result.output
        assert "config" in result.output
        assert "memory" in result.output
        assert "skills" in result.output


# ---------------------------------------------------------------------------
# zenki setup
# ---------------------------------------------------------------------------


class TestSetup:
    """Tests for the setup wizard."""

    def test_setup_creates_config(self, tmp_path: Path) -> None:
        """Setup should create the config directory, config.json, and core memory files."""
        config_dir = tmp_path / "zenki_config"
        result = runner.invoke(
            app,
            ["setup", "--config-dir", str(config_dir)],
            input="TestUser\nANTHROPIC_API_KEY\nsonnet\n",
        )
        assert result.exit_code == 0
        assert "Setup complete" in result.output

        # Verify config.json was created.
        config_path = config_dir / "config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data["user"]["display_name"] == "TestUser"
        assert data["llm"]["default_model"] == "sonnet"

        # Verify directory structure.
        assert (config_dir / "memory" / "core").is_dir()
        assert (config_dir / "skills").is_dir()
        assert (config_dir / "logs").is_dir()

        # Verify core memory files.
        core_dir = config_dir / "memory" / "core"
        assert (core_dir / "identity.md").exists()
        assert (core_dir / "preferences.md").exists()
        assert (core_dir / "projects.md").exists()
        assert (core_dir / "relationships.md").exists()
        assert (core_dir / "patterns.md").exists()

    def test_setup_overwrite_cancel(self, tmp_path: Path) -> None:
        """If config exists and user declines overwrite, setup should cancel."""
        config_dir = tmp_path / "zenki_config"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text("{}")

        result = runner.invoke(
            app,
            ["setup", "--config-dir", str(config_dir)],
            input="n\n",
        )
        assert result.exit_code == 0
        assert "cancelled" in result.output.lower()


# ---------------------------------------------------------------------------
# zenki config list / get / set
# ---------------------------------------------------------------------------


class TestConfig:
    """Tests for config commands."""

    def test_config_list(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config list should display configuration values."""
        config_dir = tmp_path / "zenki_config"
        config_dir.mkdir(parents=True)
        settings = ZenkiSettings()
        settings.save(config_dir / "config.json")

        monkeypatch.setattr(ZenkiSettings, "get_config_dir", lambda: config_dir)

        result = runner.invoke(app, ["config", "list"])
        assert result.exit_code == 0
        assert "llm.default_model" in result.output

    def test_config_get(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config get should return a specific value."""
        config_dir = tmp_path / "zenki_config"
        config_dir.mkdir(parents=True)
        settings = ZenkiSettings()
        settings.save(config_dir / "config.json")

        monkeypatch.setattr(ZenkiSettings, "get_config_dir", lambda: config_dir)

        result = runner.invoke(app, ["config", "get", "llm.default_model"])
        assert result.exit_code == 0
        assert "sonnet" in result.output

    def test_config_get_unknown_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config get with an unknown key should exit with error."""
        config_dir = tmp_path / "zenki_config"
        config_dir.mkdir(parents=True)
        settings = ZenkiSettings()
        settings.save(config_dir / "config.json")

        monkeypatch.setattr(ZenkiSettings, "get_config_dir", lambda: config_dir)

        result = runner.invoke(app, ["config", "get", "nonexistent.key"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# zenki status
# ---------------------------------------------------------------------------


class TestStatus:
    """Tests for the status command."""

    def test_status_shows_info(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Status should display system information."""
        config_dir = tmp_path / "zenki_config"
        config_dir.mkdir(parents=True)
        settings = ZenkiSettings()
        settings.save(config_dir / "config.json")

        monkeypatch.setattr(ZenkiSettings, "get_config_dir", lambda: config_dir)

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Zenki Status" in result.output
        assert __version__ in result.output

    def test_status_without_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Status should work even when no config file exists."""
        config_dir = tmp_path / "zenki_config"
        config_dir.mkdir(parents=True)

        monkeypatch.setattr(ZenkiSettings, "get_config_dir", lambda: config_dir)

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "No" in result.output


# ---------------------------------------------------------------------------
# zenki skills list
# ---------------------------------------------------------------------------


class TestSkills:
    """Tests for skill commands."""

    def test_skills_list_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Skills list with no skills should indicate none registered."""
        config_dir = tmp_path / "zenki_config"
        config_dir.mkdir(parents=True)

        # Create an empty database.
        db_path = config_dir / "zenki.db"
        db = ZenkiDatabase(db_path)
        db.initialize()
        db.close()

        monkeypatch.setattr(ZenkiSettings, "get_config_dir", lambda: config_dir)

        result = runner.invoke(app, ["skills", "list"])
        assert result.exit_code == 0
        assert "No skills" in result.output

    def test_skills_list_with_skills(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Skills list should display registered skills."""
        from zenki.db.models import Skill

        config_dir = tmp_path / "zenki_config"
        config_dir.mkdir(parents=True)

        db_path = config_dir / "zenki.db"
        db = ZenkiDatabase(db_path)
        db.initialize()
        db.create_skill(
            Skill(
                name="test-skill",
                skill_type="core",
                status="approved",
                path="/skills/test.md",
                description="A test skill",
            )
        )
        db.close()

        monkeypatch.setattr(ZenkiSettings, "get_config_dir", lambda: config_dir)

        result = runner.invoke(app, ["skills", "list"])
        assert result.exit_code == 0
        assert "test-skill" in result.output


# ---------------------------------------------------------------------------
# zenki memory inspect
# ---------------------------------------------------------------------------


class TestMemory:
    """Tests for memory commands."""

    def test_memory_inspect(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Memory inspect should show memory statistics."""
        config_dir = tmp_path / "zenki_config"
        config_dir.mkdir(parents=True)
        (config_dir / "memory" / "core").mkdir(parents=True)
        (config_dir / "memory" / "core" / "identity.md").write_text("# Identity")

        # Create database.
        db_path = config_dir / "zenki.db"
        db = ZenkiDatabase(db_path)
        db.initialize()
        db.close()

        monkeypatch.setattr(ZenkiSettings, "get_config_dir", lambda: config_dir)

        result = runner.invoke(app, ["memory", "inspect"])
        assert result.exit_code == 0
        assert "Memory Statistics" in result.output
