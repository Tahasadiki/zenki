"""Comprehensive tests for the Zenki configuration system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from zenki.config import ZenkiSettings, get_config_dir
from zenki.config.defaults import DEFAULT_CONFIG
from zenki.config.settings import (
    ContextBudgetConfig,
    DaemonConfig,
    LLMConfig,
    PersonalityConfig,
    RetrievalConfig,
)

# ---------------------------------------------------------------------------
# Default config generation
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    """Tests for the DEFAULT_CONFIG dictionary."""

    def test_default_config_is_a_dict(self) -> None:
        assert isinstance(DEFAULT_CONFIG, dict)

    def test_default_config_has_all_top_level_keys(self) -> None:
        expected_keys = {
            "version",
            "user",
            "llm",
            "memory",
            "channels",
            "daemon",
            "skills",
            "personality",
        }
        assert set(DEFAULT_CONFIG.keys()) == expected_keys

    def test_default_version(self) -> None:
        assert DEFAULT_CONFIG["version"] == "1.0.0"

    def test_default_user(self) -> None:
        assert DEFAULT_CONFIG["user"]["id"] == "default"
        assert DEFAULT_CONFIG["user"]["display_name"] == ""

    def test_default_llm(self) -> None:
        llm = DEFAULT_CONFIG["llm"]
        assert llm["provider"] == "claude"
        assert llm["api_key_env"] == "ANTHROPIC_API_KEY"
        assert llm["default_model"] == "sonnet"
        assert llm["smart_routing"] is True
        assert llm["models"]["haiku"] == "claude-haiku-4-5-20251001"
        assert llm["models"]["sonnet"] == "claude-sonnet-4-6"
        assert llm["models"]["opus"] == "claude-opus-4-6"

    def test_default_memory_embeddings(self) -> None:
        emb = DEFAULT_CONFIG["memory"]["embeddings"]
        assert emb["provider"] == "local"
        assert emb["model"] == "all-MiniLM-L6-v2"

    def test_default_memory_retrieval(self) -> None:
        ret = DEFAULT_CONFIG["memory"]["retrieval"]
        assert ret["episodic_top_k"] == 5
        assert ret["semantic_top_k"] == 10
        assert ret["min_relevance_score"] == 0.3

    def test_default_memory_consolidation(self) -> None:
        con = DEFAULT_CONFIG["memory"]["consolidation"]
        assert con["enabled"] is True
        assert con["schedule"] == "0 3 * * *"
        assert con["weekly"] is True
        assert con["monthly"] is True

    def test_default_memory_context_budget(self) -> None:
        cb = DEFAULT_CONFIG["memory"]["context_budget"]
        assert cb["core_memory_pct"] == 10
        assert cb["episodic_pct"] == 15
        assert cb["semantic_pct"] == 15
        assert cb["working_pct"] == 60

    def test_default_channels(self) -> None:
        ch = DEFAULT_CONFIG["channels"]
        assert ch["default_notification_channel"] == "cli"
        assert ch["slack"]["enabled"] is False
        assert ch["slack"]["bot_token_env"] == "ZENKI_SLACK_BOT_TOKEN"
        assert ch["slack"]["signing_secret_env"] == "ZENKI_SLACK_SIGNING_SECRET"

    def test_default_daemon(self) -> None:
        d = DEFAULT_CONFIG["daemon"]
        assert d["host"] == "0.0.0.0"
        assert d["port"] == 8420
        assert d["log_level"] == "INFO"

    def test_default_skills(self) -> None:
        s = DEFAULT_CONFIG["skills"]
        assert s["auto_discover"] is True
        assert s["require_approval"] is True

    def test_default_personality(self) -> None:
        p = DEFAULT_CONFIG["personality"]
        assert p["tone"] == "professional"
        assert p["verbosity"] == "balanced"
        assert p["proactivity"] == "moderate"
        assert p["custom_instructions"] == ""


# ---------------------------------------------------------------------------
# Settings instantiation with defaults
# ---------------------------------------------------------------------------


class TestZenkiSettingsDefaults:
    """Test that ZenkiSettings() produces correct defaults without any input."""

    def test_default_instantiation(self) -> None:
        settings = ZenkiSettings()
        assert settings.version == "1.0.0"

    def test_default_user(self) -> None:
        settings = ZenkiSettings()
        assert settings.user.id == "default"
        assert settings.user.display_name == ""

    def test_default_llm(self) -> None:
        settings = ZenkiSettings()
        assert settings.llm.provider == "claude"
        assert settings.llm.smart_routing is True
        assert settings.llm.models.sonnet == "claude-sonnet-4-6"

    def test_default_memory(self) -> None:
        settings = ZenkiSettings()
        assert settings.memory.embeddings.provider == "local"
        assert settings.memory.retrieval.episodic_top_k == 5
        assert settings.memory.consolidation.enabled is True
        assert settings.memory.context_budget.working_pct == 60

    def test_default_channels(self) -> None:
        settings = ZenkiSettings()
        assert settings.channels.default_notification_channel == "cli"
        assert settings.channels.slack.enabled is False

    def test_default_daemon(self) -> None:
        settings = ZenkiSettings()
        assert settings.daemon.port == 8420
        assert settings.daemon.log_level == "INFO"

    def test_default_skills(self) -> None:
        settings = ZenkiSettings()
        assert settings.skills.auto_discover is True
        assert settings.skills.require_approval is True

    def test_default_personality(self) -> None:
        settings = ZenkiSettings()
        assert settings.personality.tone == "professional"
        assert settings.personality.verbosity == "balanced"
        assert settings.personality.proactivity == "moderate"

    def test_model_dump_matches_default_config(self) -> None:
        """ZenkiSettings() dumped to dict should match DEFAULT_CONFIG."""
        settings = ZenkiSettings()
        dumped = settings.model_dump()
        assert dumped == DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Loading configuration from file
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Test ZenkiSettings.load() from a JSON file."""

    def test_load_returns_defaults_when_file_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"
        settings = ZenkiSettings.load(path)
        assert settings.version == "1.0.0"
        assert settings.llm.provider == "claude"

    def test_load_from_file(self, tmp_path: Path) -> None:
        config = {"version": "2.0.0", "daemon": {"port": 9999}}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        settings = ZenkiSettings.load(path)
        assert settings.version == "2.0.0"
        assert settings.daemon.port == 9999
        # Other defaults should still be present
        assert settings.llm.provider == "claude"
        assert settings.memory.retrieval.episodic_top_k == 5

    def test_load_partial_nested_config(self, tmp_path: Path) -> None:
        config = {
            "llm": {"default_model": "haiku"},
            "personality": {"tone": "casual"},
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        settings = ZenkiSettings.load(path)
        assert settings.llm.default_model == "haiku"
        # Unchanged nested defaults preserved
        assert settings.llm.provider == "claude"
        assert settings.llm.smart_routing is True
        assert settings.personality.tone == "casual"
        assert settings.personality.verbosity == "balanced"

    def test_load_empty_json_object(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text("{}")
        settings = ZenkiSettings.load(path)
        assert settings.version == "1.0.0"
        assert settings == ZenkiSettings()


# ---------------------------------------------------------------------------
# Saving configuration to file
# ---------------------------------------------------------------------------


class TestSaveConfig:
    """Test ZenkiSettings.save() to a JSON file."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        settings = ZenkiSettings()
        path = tmp_path / "config.json"
        returned = settings.save(path)
        assert returned == path
        assert path.exists()

    def test_save_content_is_valid_json(self, tmp_path: Path) -> None:
        settings = ZenkiSettings()
        path = tmp_path / "config.json"
        settings.save(path)
        data = json.loads(path.read_text())
        assert data["version"] == "1.0.0"

    def test_save_roundtrip(self, tmp_path: Path) -> None:
        """Save then load should produce identical settings."""
        original = ZenkiSettings(
            version="2.0.0",
            daemon=DaemonConfig(port=1234, log_level="DEBUG"),
        )
        path = tmp_path / "config.json"
        original.save(path)

        loaded = ZenkiSettings.load(path)
        assert loaded.version == "2.0.0"
        assert loaded.daemon.port == 1234
        assert loaded.daemon.log_level == "DEBUG"
        assert loaded.model_dump() == original.model_dump()

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "config.json"
        settings = ZenkiSettings()
        settings.save(path)
        assert path.exists()

    def test_save_overwrites_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        ZenkiSettings(version="1.0.0").save(path)
        ZenkiSettings(version="3.0.0").save(path)

        data = json.loads(path.read_text())
        assert data["version"] == "3.0.0"


# ---------------------------------------------------------------------------
# Config validation (invalid values rejected)
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Test that Pydantic validation rejects invalid configuration values."""

    def test_invalid_default_model(self) -> None:
        with pytest.raises(ValidationError, match="default_model"):
            LLMConfig(default_model="invalid-model")

    def test_invalid_log_level(self) -> None:
        with pytest.raises(ValidationError, match="log_level"):
            DaemonConfig(log_level="TRACE")

    def test_invalid_port_too_high(self) -> None:
        with pytest.raises(ValidationError):
            DaemonConfig(port=70000)

    def test_invalid_port_too_low(self) -> None:
        with pytest.raises(ValidationError):
            DaemonConfig(port=0)

    def test_invalid_tone(self) -> None:
        with pytest.raises(ValidationError, match="tone"):
            PersonalityConfig(tone="aggressive")

    def test_invalid_verbosity(self) -> None:
        with pytest.raises(ValidationError, match="verbosity"):
            PersonalityConfig(verbosity="extreme")

    def test_invalid_proactivity(self) -> None:
        with pytest.raises(ValidationError, match="proactivity"):
            PersonalityConfig(proactivity="max")

    def test_invalid_min_relevance_score_too_high(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(min_relevance_score=1.5)

    def test_invalid_min_relevance_score_negative(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(min_relevance_score=-0.1)

    def test_invalid_episodic_top_k_zero(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(episodic_top_k=0)

    def test_invalid_context_budget_pct_negative(self) -> None:
        with pytest.raises(ValidationError):
            ContextBudgetConfig(core_memory_pct=-1)

    def test_invalid_context_budget_pct_over_100(self) -> None:
        with pytest.raises(ValidationError):
            ContextBudgetConfig(working_pct=101)

    def test_valid_log_level_case_insensitive(self) -> None:
        """Log level should be uppercased automatically."""
        cfg = DaemonConfig(log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_invalid_config_from_file(self, tmp_path: Path) -> None:
        """Loading an invalid config from file should raise ValidationError."""
        bad_config = {"daemon": {"port": -1}}
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad_config))
        with pytest.raises(ValidationError):
            ZenkiSettings.load(path)


# ---------------------------------------------------------------------------
# Config merging (user values override defaults)
# ---------------------------------------------------------------------------


class TestConfigMerging:
    """Test the merge_with_defaults logic."""

    def test_empty_override_returns_defaults(self) -> None:
        merged = ZenkiSettings.merge_with_defaults({})
        assert merged == DEFAULT_CONFIG

    def test_top_level_override(self) -> None:
        merged = ZenkiSettings.merge_with_defaults({"version": "9.9.9"})
        assert merged["version"] == "9.9.9"
        # Everything else should remain the same
        assert merged["llm"] == DEFAULT_CONFIG["llm"]

    def test_nested_override_preserves_siblings(self) -> None:
        merged = ZenkiSettings.merge_with_defaults(
            {"llm": {"default_model": "opus"}}
        )
        assert merged["llm"]["default_model"] == "opus"
        # Siblings unchanged
        assert merged["llm"]["provider"] == "claude"
        assert merged["llm"]["smart_routing"] is True
        assert merged["llm"]["models"] == DEFAULT_CONFIG["llm"]["models"]

    def test_deeply_nested_override(self) -> None:
        merged = ZenkiSettings.merge_with_defaults(
            {"memory": {"retrieval": {"episodic_top_k": 20}}}
        )
        assert merged["memory"]["retrieval"]["episodic_top_k"] == 20
        # Other retrieval keys unchanged
        assert merged["memory"]["retrieval"]["semantic_top_k"] == 10
        assert merged["memory"]["retrieval"]["min_relevance_score"] == 0.3
        # Other memory sections unchanged
        assert merged["memory"]["embeddings"] == DEFAULT_CONFIG["memory"]["embeddings"]

    def test_multiple_sections_override(self) -> None:
        overrides = {
            "version": "3.0.0",
            "daemon": {"port": 5555, "log_level": "DEBUG"},
            "skills": {"require_approval": False},
        }
        merged = ZenkiSettings.merge_with_defaults(overrides)
        assert merged["version"] == "3.0.0"
        assert merged["daemon"]["port"] == 5555
        assert merged["daemon"]["log_level"] == "DEBUG"
        assert merged["daemon"]["host"] == "0.0.0.0"  # default preserved
        assert merged["skills"]["require_approval"] is False
        assert merged["skills"]["auto_discover"] is True  # default preserved

    def test_merge_does_not_mutate_defaults(self) -> None:
        """Merging should not modify the original DEFAULT_CONFIG."""
        import copy

        original = copy.deepcopy(DEFAULT_CONFIG)
        ZenkiSettings.merge_with_defaults({"version": "changed"})
        assert DEFAULT_CONFIG == original


# ---------------------------------------------------------------------------
# Config directory creation
# ---------------------------------------------------------------------------


class TestConfigDirectory:
    """Test get_config_dir and ensure_config_dir."""

    def test_get_config_dir_returns_path(self) -> None:
        config_dir = ZenkiSettings.get_config_dir()
        assert isinstance(config_dir, Path)
        assert config_dir.name == "zenki"
        assert config_dir.parent.name == ".config"

    def test_get_config_dir_via_package_export(self) -> None:
        config_dir = get_config_dir()
        assert isinstance(config_dir, Path)
        assert config_dir == ZenkiSettings.get_config_dir()

    def test_ensure_config_dir_creates_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        config_dir = ZenkiSettings.ensure_config_dir()
        assert config_dir.exists()
        assert config_dir.is_dir()
        assert config_dir == fake_home / ".config" / "zenki"

    def test_ensure_config_dir_idempotent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        dir1 = ZenkiSettings.ensure_config_dir()
        dir2 = ZenkiSettings.ensure_config_dir()
        assert dir1 == dir2
        assert dir1.exists()


# ---------------------------------------------------------------------------
# Package-level exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Test that the config package exports the expected symbols."""

    def test_zenki_settings_importable(self) -> None:
        from zenki.config import ZenkiSettings as Imported
        assert Imported is ZenkiSettings

    def test_get_config_dir_importable(self) -> None:
        from zenki.config import get_config_dir as imported_fn
        assert callable(imported_fn)
