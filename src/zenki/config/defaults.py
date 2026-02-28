"""Default configuration values for Zenki."""

DEFAULT_CONFIG: dict = {
    "version": "1.0.0",
    "user": {
        "id": "default",
        "display_name": "",
    },
    "llm": {
        "provider": "claude",
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "sonnet",
        "models": {
            "haiku": "claude-haiku-4-5-20251001",
            "sonnet": "claude-sonnet-4-6",
            "opus": "claude-opus-4-6",
        },
    },
    "memory": {
        "embeddings": {
            "provider": "local",
            "model": "all-MiniLM-L6-v2",
        },
        "retrieval": {
            "episodic_top_k": 5,
            "semantic_top_k": 10,
            "min_relevance_score": 0.3,
        },
        "consolidation": {
            "enabled": True,
            "schedule": "0 3 * * *",
            "weekly": True,
            "monthly": True,
        },
        "context_budget": {
            "core_memory_pct": 10,
            "episodic_pct": 15,
            "semantic_pct": 15,
            "working_pct": 60,
        },
    },
    "session": {
        "timeout_minutes": 60,
    },
    "channels": {
        "default_notification_channel": "cli",
        "slack": {
            "enabled": False,
            "bot_token_env": "ZENKI_SLACK_BOT_TOKEN",
            "signing_secret_env": "ZENKI_SLACK_SIGNING_SECRET",
        },
    },
    "daemon": {
        "host": "0.0.0.0",
        "port": 8420,
        "log_level": "INFO",
    },
    "personality": {
        "tone": "professional",
        "verbosity": "balanced",
        "proactivity": "moderate",
        "custom_instructions": "",
    },
}
