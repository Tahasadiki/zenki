"""Interactive chat command for Zenki."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

_SPECIAL_COMMANDS = {"/quit", "/exit", "/clear", "/history"}


def _handle_special(command: str) -> bool:
    """Handle a special slash command. Return True if the REPL should exit."""
    cmd = command.strip().lower()
    if cmd in ("/quit", "/exit"):
        console.print("[dim]Goodbye![/dim]")
        return True
    if cmd == "/clear":
        console.clear()
        console.print("[dim]Screen cleared.[/dim]")
        return False
    if cmd == "/history":
        console.print("[dim]Session history is not yet available.[/dim]")
        return False
    return False


def _init_orchestrator():
    """Initialise the Zenki orchestrator stack."""
    from zenki.config.settings import ZenkiSettings
    from zenki.db.database import ZenkiDatabase
    from zenki.memory.manager import MemoryManager
    from zenki.sdk.orchestrator import ZenkiOrchestrator

    settings = ZenkiSettings.load()
    config_dir = settings.ensure_config_dir()

    db_path = config_dir / "zenki.db"
    db = ZenkiDatabase(db_path)
    db.initialize()

    memory_manager = MemoryManager(settings=settings, db=db, config_dir=config_dir)

    orchestrator = ZenkiOrchestrator(
        settings=settings,
        db=db,
        memory_manager=memory_manager,
    )
    return orchestrator, settings


def chat(
    resume: str | None = typer.Option(
        None, "--resume", help="Resume an existing session by ID."
    ),
) -> None:
    """Start an interactive chat session."""
    console.print(
        Panel(
            "[bold cyan]Zenki Chat[/bold cyan]\n"
            "[dim]Type your message and press Enter. "
            "Commands: /quit, /exit, /clear, /history[/dim]",
            border_style="cyan",
        )
    )

    try:
        orchestrator, settings = _init_orchestrator()
    except Exception as exc:
        console.print(f"[red]Failed to initialise Zenki: {exc}[/red]")
        console.print("[dim]Run 'zenki setup' first if this is a new installation.[/dim]")
        raise typer.Exit(code=1)

    session_id = resume
    if resume:
        console.print(f"[dim]Resuming session: {resume}[/dim]")

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Handle special commands.
        if user_input.startswith("/"):
            if _handle_special(user_input):
                break
            continue

        # Send to the SDK orchestrator.
        try:
            response = asyncio.run(
                orchestrator.process_message(
                    message=user_input,
                    session_id=session_id,
                    user_id=settings.user.id,
                    channel_type="cli",
                )
            )
        except Exception as exc:
            console.print(f"\n[red]Error: {exc}[/red]\n")
            continue

        console.print()
        console.print(
            Panel(
                Markdown(response),
                title="[bold blue]Zenki[/bold blue]",
                border_style="blue",
            )
        )
        console.print()
