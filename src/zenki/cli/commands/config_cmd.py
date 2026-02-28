"""Configuration management commands for Zenki."""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from zenki.config.settings import ZenkiSettings

console = Console()

config_app = typer.Typer(help="Manage configuration.")


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    """Flatten a nested dict into dot-notation key-value pairs."""
    items: list[tuple[str, str]] = []
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, full_key))
        else:
            items.append((full_key, str(value)))
    return items


def _get_nested(d: dict[str, Any], dotted_key: str) -> Any:
    """Get a value from a nested dict using dot notation."""
    keys = dotted_key.split(".")
    current: Any = d
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return None
        current = current[k]
    return current


def _set_nested(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot notation."""
    keys = dotted_key.split(".")
    current = d
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    # Attempt to coerce the value to the right type.
    current[keys[-1]] = _coerce_value(value)


def _coerce_value(value: str) -> Any:
    """Coerce a string value to an appropriate Python type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


@config_app.command("list")
def config_list() -> None:
    """Show all configuration values."""
    settings = ZenkiSettings.load()
    data = settings.model_dump()

    table = Table(title="Zenki Configuration", show_lines=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    for key, value in _flatten_dict(data):
        table.add_row(key, value)

    console.print(table)


@config_app.command("get")
def config_get(key: str = typer.Argument(..., help="Config key in dot notation.")) -> None:
    """Get a configuration value."""
    settings = ZenkiSettings.load()
    data = settings.model_dump()
    value = _get_nested(data, key)

    if value is None:
        console.print(f"[red]Key not found:[/red] {key}")
        raise typer.Exit(code=1)

    if isinstance(value, dict):
        # Print sub-keys as a table.
        table = Table(title=key, show_lines=False)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        for k, v in _flatten_dict(value, key):
            table.add_row(k, v)
        console.print(table)
    else:
        console.print(f"[cyan]{key}[/cyan] = [green]{value}[/green]")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key in dot notation."),
    value: str = typer.Argument(..., help="Value to set."),
) -> None:
    """Set a configuration value."""
    settings = ZenkiSettings.load()
    data = settings.model_dump()

    # Verify the key exists (to prevent typos creating new keys).
    existing = _get_nested(data, key)
    if existing is None:
        console.print(f"[red]Unknown config key:[/red] {key}")
        raise typer.Exit(code=1)

    _set_nested(data, key, value)

    try:
        new_settings = ZenkiSettings.model_validate(data)
    except Exception as e:
        console.print(f"[red]Invalid value:[/red] {e}")
        raise typer.Exit(code=1)

    new_settings.save()
    console.print(f"[green]Set[/green] [cyan]{key}[/cyan] = [green]{value}[/green]")
