"""Status display command for Zenki."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from zenki import __version__
from zenki.config.settings import ZenkiSettings

console = Console()


def _file_size_str(path: Path) -> str:
    """Return a human-readable file size, or 'N/A' if the file is missing."""
    if not path.exists():
        return "N/A"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _count_rows(db_path: Path, table_name: str) -> int:
    """Count rows in a SQLite table without importing the full database module."""
    if not db_path.exists():
        return 0
    import sqlite3

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")  # noqa: S608
        count: int = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def _check_daemon_running(port: int = 8420) -> bool:
    """Check whether the Zenki daemon is listening on the expected port."""
    import socket

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def status() -> None:
    """Show Zenki system status."""
    config_dir = ZenkiSettings.get_config_dir()
    config_path = config_dir / "config.json"
    db_path = config_dir / "zenki.db"

    # Load settings (defaults if not configured).
    settings = ZenkiSettings.load()

    # Gather information.
    config_exists = config_path.exists()
    db_size = _file_size_str(db_path)
    session_count = _count_rows(db_path, "sessions")
    skill_count = _count_rows(db_path, "skills")
    daemon_running = _check_daemon_running(settings.daemon.port)

    # Build table.
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Version", __version__)
    table.add_row("Config directory", str(config_dir))
    table.add_row("Config exists", "Yes" if config_exists else "[red]No[/red] (run zenki setup)")
    table.add_row("Database size", db_size)
    table.add_row("Sessions", str(session_count))
    table.add_row("Skills", str(skill_count))
    table.add_row(
        "Daemon",
        "[green]Running[/green]" if daemon_running else "[dim]Not running[/dim]",
    )

    console.print(
        Panel(table, title="Zenki Status", border_style="cyan")
    )
