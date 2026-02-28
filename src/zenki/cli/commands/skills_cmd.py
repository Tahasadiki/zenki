"""Skill management commands for Zenki."""

from __future__ import annotations

from datetime import UTC, datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from zenki.config.settings import ZenkiSettings
from zenki.db.database import ZenkiDatabase

console = Console()

skills_app = typer.Typer(help="Skill management.")


def _get_db() -> ZenkiDatabase:
    """Open (and initialise) the Zenki database."""
    config_dir = ZenkiSettings.get_config_dir()
    db_path = config_dir / "zenki.db"
    db = ZenkiDatabase(db_path)
    db.initialize()
    return db


def _find_skill_by_name(db: ZenkiDatabase, name: str):  # noqa: ANN201
    """Find a skill by name. Returns the Skill model or exits with an error."""
    skills = db.list_skills()
    for skill in skills:
        if skill.name == name:
            return skill
    console.print(f"[red]Skill not found:[/red] {name}")
    raise typer.Exit(code=1)


@skills_app.command("list")
def skills_list() -> None:
    """List all skills with status."""
    db = _get_db()

    try:
        skills = db.list_skills()
    finally:
        db.close()

    if not skills:
        console.print("[dim]No skills registered.[/dim]")
        return

    table = Table(title="Skills", show_lines=False)
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column("Status", style="green")
    table.add_column("Uses", justify="right")
    table.add_column("Description")

    for skill in skills:
        status_style = {
            "approved": "[green]approved[/green]",
            "pending": "[yellow]pending[/yellow]",
            "rejected": "[red]rejected[/red]",
            "disabled": "[dim]disabled[/dim]",
        }.get(skill.status, skill.status)

        table.add_row(
            skill.name,
            skill.skill_type,
            status_style,
            str(skill.usage_count),
            (skill.description or "")[:60],
        )

    console.print(table)


@skills_app.command("approve")
def skills_approve(
    name: str = typer.Argument(..., help="Name of the skill to approve."),
) -> None:
    """Approve a pending skill."""
    db = _get_db()

    try:
        skill = _find_skill_by_name(db, name)
        if skill.status == "approved":
            console.print(f"[yellow]Skill '{name}' is already approved.[/yellow]")
            return
        skill.status = "approved"
        skill.approved_at = datetime.now(UTC)
        db.update_skill(skill)
        console.print(f"[green]Skill '{name}' approved.[/green]")
    finally:
        db.close()


@skills_app.command("reject")
def skills_reject(
    name: str = typer.Argument(..., help="Name of the skill to reject."),
) -> None:
    """Reject a pending skill."""
    db = _get_db()

    try:
        skill = _find_skill_by_name(db, name)
        if skill.status == "rejected":
            console.print(f"[yellow]Skill '{name}' is already rejected.[/yellow]")
            return
        skill.status = "rejected"
        db.update_skill(skill)
        console.print(f"[green]Skill '{name}' rejected.[/green]")
    finally:
        db.close()


@skills_app.command("info")
def skills_info(
    name: str = typer.Argument(..., help="Name of the skill to inspect."),
) -> None:
    """Show detailed information about a skill."""
    db = _get_db()

    try:
        skill = _find_skill_by_name(db, name)
    finally:
        db.close()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="green")

    table.add_row("Name", skill.name)
    table.add_row("ID", skill.id)
    table.add_row("Type", skill.skill_type)
    table.add_row("Status", skill.status)
    table.add_row("Path", skill.path)
    table.add_row("Description", skill.description or "N/A")
    table.add_row("Generated from", skill.generated_from or "N/A")
    table.add_row("Usage count", str(skill.usage_count))
    table.add_row("Last used", str(skill.last_used_at or "Never"))
    table.add_row("Approved at", str(skill.approved_at or "N/A"))
    table.add_row("Created at", str(skill.created_at))

    console.print(Panel(table, title=f"Skill: {skill.name}", border_style="cyan"))
