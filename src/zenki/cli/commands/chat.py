"""Interactive chat command for Zenki."""

from __future__ import annotations

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


def _generate_response(user_input: str) -> str:
    """Generate a response to the user's input.

    This is a placeholder that echoes the input. When the real agent is
    ready, replace this function with an actual LLM call.
    """
    return f"You said: {user_input}\n\n*(Agent not connected -- echo mode)*"


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

        # Generate and display response.
        response = _generate_response(user_input)
        console.print()
        console.print(
            Panel(
                Markdown(response),
                title="[bold blue]Zenki[/bold blue]",
                border_style="blue",
            )
        )
        console.print()
