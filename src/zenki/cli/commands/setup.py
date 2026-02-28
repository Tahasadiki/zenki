"""First-time setup wizard for Zenki."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from zenki.config.settings import ZenkiSettings

console = Console()

# Default core memory files to create during setup.
_CORE_MEMORY_FILES: dict[str, str] = {
    "identity.md": "# User Identity\n\n<!-- Zenki will learn about you over time -->\n",
    "preferences.md": "# User Preferences\n\n<!-- Your preferences will be recorded here -->\n",
    "projects.md": "# Projects\n\n<!-- Active projects and context -->\n",
    "relationships.md": "# Relationships\n\n<!-- Key people and contacts -->\n",
    "patterns.md": "# Patterns\n\n<!-- Recurring workflows and habits -->\n",
}


def _get_config_dir_override() -> Path | None:
    """Return the overridden config directory, if set.

    This allows tests to redirect config to a temporary directory by setting
    ``_config_dir_override`` on this module.
    """
    return getattr(_get_config_dir_override, "_override", None)


def _resolve_config_dir() -> Path:
    """Return the effective config directory (supports test overrides)."""
    override = _get_config_dir_override()
    if override is not None:
        return override
    return ZenkiSettings.get_config_dir()


def setup(
    config_dir: Path | None = typer.Option(None, "--config-dir", help="Override config directory."),
) -> None:
    """Run the first-time setup wizard."""
    console.print(
        Panel(
            "[bold cyan]Welcome to Zenki![/bold cyan]\n"
            "Let's get you set up with your adaptive AI assistant.",
            title="Setup Wizard",
            border_style="cyan",
        )
    )

    # Determine config directory.
    effective_dir = config_dir if config_dir is not None else _resolve_config_dir()
    config_path = effective_dir / "config.json"

    # Check for existing config.
    if config_path.exists():
        overwrite = Confirm.ask(
            "[yellow]Configuration already exists. Overwrite?[/yellow]",
            default=False,
        )
        if not overwrite:
            console.print("[dim]Setup cancelled.[/dim]")
            raise typer.Exit()

    # Prompt for settings.
    display_name = Prompt.ask("Your display name", default="User")
    api_key_env = Prompt.ask(
        "API key environment variable name",
        default="ANTHROPIC_API_KEY",
    )
    default_model = Prompt.ask(
        "Default model [haiku/sonnet/opus]",
        choices=["haiku", "sonnet", "opus"],
        default="sonnet",
    )

    # Build settings.
    settings = ZenkiSettings()
    settings.user.display_name = display_name
    settings.llm.api_key_env = api_key_env
    settings.llm.default_model = default_model

    # Create directory structure.
    effective_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = effective_dir / "memory" / "core"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (effective_dir / "skills").mkdir(parents=True, exist_ok=True)
    (effective_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Create default core memory files.
    for filename, content in _CORE_MEMORY_FILES.items():
        filepath = memory_dir / filename
        if not filepath.exists():
            filepath.write_text(content)

    # Save config.
    settings.save(config_path)

    console.print()
    console.print(
        Panel(
            f"[bold green]Setup complete![/bold green]\n\n"
            f"  Config directory: [cyan]{effective_dir}[/cyan]\n"
            f"  Display name:     [cyan]{display_name}[/cyan]\n"
            f"  API key env:      [cyan]{api_key_env}[/cyan]\n"
            f"  Default model:    [cyan]{default_model}[/cyan]\n\n"
            f"[dim]Next steps:[/dim]\n"
            f"  1. Set your API key: export {api_key_env}=your-key\n"
            f"  2. Start chatting:   zenki chat\n"
            f"  3. Check status:     zenki status",
            title="Done",
            border_style="green",
        )
    )
