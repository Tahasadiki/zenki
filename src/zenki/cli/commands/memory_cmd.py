"""Memory management commands for Zenki."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from zenki.config.settings import ZenkiSettings
from zenki.db.database import ZenkiDatabase

console = Console()

memory_app = typer.Typer(help="Memory management.")


def _get_db() -> ZenkiDatabase:
    """Open (and initialise) the Zenki database."""
    config_dir = ZenkiSettings.get_config_dir()
    db_path = config_dir / "zenki.db"
    db = ZenkiDatabase(db_path)
    db.initialize()
    return db


def _count_core_files(config_dir: Path) -> int:
    """Count core memory markdown files."""
    core_dir = config_dir / "memory" / "core"
    if not core_dir.exists():
        return 0
    return len(list(core_dir.glob("*.md")))


@memory_app.command("inspect")
def memory_inspect() -> None:
    """Show memory statistics."""
    config_dir = ZenkiSettings.get_config_dir()
    db = _get_db()

    try:
        episodic_count = len(db.search_episodic_memories(""))
        semantic_count = len(db.search_semantic_memories(""))
    except Exception:
        episodic_count = 0
        semantic_count = 0
    finally:
        db.close()

    core_count = _count_core_files(config_dir)

    table = Table(title="Memory Statistics", show_lines=False)
    table.add_column("Type", style="cyan")
    table.add_column("Count", style="green", justify="right")

    table.add_row("Core memory files", str(core_count))
    table.add_row("Episodic memories", str(episodic_count))
    table.add_row("Semantic memories", str(semantic_count))

    console.print(Panel(table, border_style="cyan"))


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search query."),
) -> None:
    """Search memories."""
    db = _get_db()

    try:
        episodic = db.search_episodic_memories(query)
        semantic = db.search_semantic_memories(query)
    finally:
        db.close()

    if not episodic and not semantic:
        console.print(f"[dim]No memories found matching '{query}'.[/dim]")
        return

    if episodic:
        table = Table(title="Episodic Memories", show_lines=True)
        table.add_column("ID", style="dim", max_width=12)
        table.add_column("Summary", style="cyan")
        table.add_column("Topics", style="green")
        table.add_column("Importance", justify="right")

        for mem in episodic:
            table.add_row(
                mem.id[:12],
                mem.summary[:80],
                ", ".join(mem.key_topics[:3]),
                f"{mem.importance:.2f}",
            )
        console.print(table)

    if semantic:
        table = Table(title="Semantic Memories", show_lines=True)
        table.add_column("ID", style="dim", max_width=12)
        table.add_column("Content", style="cyan")
        table.add_column("Category", style="green")
        table.add_column("Importance", justify="right")

        for mem in semantic:
            table.add_row(
                mem.id[:12],
                mem.content[:80],
                mem.category,
                f"{mem.importance:.2f}",
            )
        console.print(table)
