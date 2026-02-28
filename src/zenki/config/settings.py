"""Settings management for Zenki using Pydantic models."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from zenki.config.defaults import DEFAULT_CONFIG

# --- Nested Pydantic models ---


class UserConfig(BaseModel):
    """User identity configuration."""

    id: str = "default"
    display_name: str = ""


class LLMModelsConfig(BaseModel):
    """Available LLM model identifiers."""

    haiku: str = "claude-haiku-4-5-20251001"
    sonnet: str = "claude-sonnet-4-6"
    opus: str = "claude-opus-4-6"


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "claude"
    api_key_env: str = "ANTHROPIC_API_KEY"
    default_model: str = "sonnet"
    models: LLMModelsConfig = Field(default_factory=LLMModelsConfig)

    @field_validator("default_model")
    @classmethod
    def validate_default_model(cls, v: str) -> str:
        allowed = {"haiku", "sonnet", "opus"}
        if v not in allowed:
            raise ValueError(f"default_model must be one of {allowed}, got '{v}'")
        return v


class EmbeddingsConfig(BaseModel):
    """Embeddings provider configuration."""

    provider: str = "local"
    model: str = "all-MiniLM-L6-v2"


class RetrievalConfig(BaseModel):
    """Memory retrieval configuration."""

    episodic_top_k: int = Field(default=5, ge=1)
    semantic_top_k: int = Field(default=10, ge=1)
    min_relevance_score: float = Field(default=0.3, ge=0.0, le=1.0)


class ConsolidationConfig(BaseModel):
    """Memory consolidation configuration."""

    enabled: bool = True
    schedule: str = "0 3 * * *"
    weekly: bool = True
    monthly: bool = True


class ContextBudgetConfig(BaseModel):
    """Context budget allocation (percentages)."""

    core_memory_pct: int = Field(default=10, ge=0, le=100)
    episodic_pct: int = Field(default=15, ge=0, le=100)
    semantic_pct: int = Field(default=15, ge=0, le=100)
    working_pct: int = Field(default=60, ge=0, le=100)


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    context_budget: ContextBudgetConfig = Field(default_factory=ContextBudgetConfig)


class SlackConfig(BaseModel):
    """Slack channel configuration."""

    enabled: bool = False
    bot_token_env: str = "ZENKI_SLACK_BOT_TOKEN"
    signing_secret_env: str = "ZENKI_SLACK_SIGNING_SECRET"


class ChannelConfig(BaseModel):
    """Communication channels configuration."""

    default_notification_channel: str = "cli"
    slack: SlackConfig = Field(default_factory=SlackConfig)


class SessionConfig(BaseModel):
    """Session lifecycle configuration."""

    timeout_minutes: int = Field(default=60, ge=1)


class DaemonConfig(BaseModel):
    """Daemon server configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=8420, ge=1, le=65535)
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return upper


class PersonalityConfig(BaseModel):
    """Personality and interaction style configuration."""

    tone: str = "professional"
    verbosity: str = "balanced"
    proactivity: str = "moderate"
    custom_instructions: str = ""

    @field_validator("tone")
    @classmethod
    def validate_tone(cls, v: str) -> str:
        allowed = {"professional", "casual", "friendly", "formal"}
        if v not in allowed:
            raise ValueError(f"tone must be one of {allowed}, got '{v}'")
        return v

    @field_validator("verbosity")
    @classmethod
    def validate_verbosity(cls, v: str) -> str:
        allowed = {"minimal", "balanced", "verbose"}
        if v not in allowed:
            raise ValueError(f"verbosity must be one of {allowed}, got '{v}'")
        return v

    @field_validator("proactivity")
    @classmethod
    def validate_proactivity(cls, v: str) -> str:
        allowed = {"low", "moderate", "high"}
        if v not in allowed:
            raise ValueError(f"proactivity must be one of {allowed}, got '{v}'")
        return v


# --- Main settings class ---


class ZenkiSettings(BaseModel):
    """Root settings model for the Zenki AI assistant."""

    version: str = "1.0.0"
    user: UserConfig = Field(default_factory=UserConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    channels: ChannelConfig = Field(default_factory=ChannelConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)

    @staticmethod
    def get_config_dir() -> Path:
        """Return the path to the Zenki configuration directory."""
        return Path.home() / ".config" / "zenki"

    @staticmethod
    def ensure_config_dir() -> Path:
        """Create the configuration directory structure if it doesn't exist.

        Returns the path to the config directory.
        """
        config_dir = ZenkiSettings.get_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    @classmethod
    def _merge_dicts(cls, base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        """Deep-merge two dicts. Values from *overrides* take precedence."""
        merged = copy.deepcopy(base)
        for key, value in overrides.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = cls._merge_dicts(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    @classmethod
    def merge_with_defaults(cls, user_config: dict[str, Any]) -> dict[str, Any]:
        """Merge user configuration with defaults.

        User values take precedence over defaults.
        """
        return cls._merge_dicts(DEFAULT_CONFIG, user_config)

    @classmethod
    def load(cls, path: Path | None = None) -> ZenkiSettings:
        """Load settings from a JSON config file.

        If *path* is ``None``, the default config location is used
        (``~/.config/zenki/config.json``).  If the file does not exist,
        default settings are returned.
        """
        if path is None:
            path = cls.get_config_dir() / "config.json"

        if not path.exists():
            return cls()

        with open(path) as f:
            user_config = json.load(f)

        merged = cls.merge_with_defaults(user_config)
        return cls.model_validate(merged)

    def save(self, path: Path | None = None) -> Path:
        """Persist the current settings to a JSON file.

        If *path* is ``None``, the default config location is used.
        Returns the path the file was written to.
        """
        if path is None:
            config_dir = self.ensure_config_dir()
            path = config_dir / "config.json"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

        return path
