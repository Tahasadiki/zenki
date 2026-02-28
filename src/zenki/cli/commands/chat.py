"""Interactive chat command for Zenki."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

_SPECIAL_COMMANDS = {"/quit", "/exit", "/clear", "/history"}

# Maximum characters per message when replaying history for context.
_HISTORY_MSG_MAX_CHARS = 500
# Maximum number of prior messages to replay into the system prompt.
_HISTORY_MAX_MESSAGES = 20


def _handle_special(command: str, session_manager=None, session_id=None) -> bool:
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
        if session_manager and session_id:
            messages = session_manager.get_messages(session_id, limit=20)
            if messages:
                for msg in messages:
                    role_label = "[bold green]You[/bold green]" if msg.role == "user" else "[bold blue]Zenki[/bold blue]"
                    console.print(f"{role_label}: {msg.content[:200]}")
            else:
                console.print("[dim]No messages in this session yet.[/dim]")
        else:
            console.print("[dim]No active session.[/dim]")
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


def _format_history(messages) -> str:
    """Format prior messages into a conversation history string for the system prompt."""
    # Take the last N messages to stay within token budget.
    recent = messages[-_HISTORY_MAX_MESSAGES:]
    lines: list[str] = []
    for msg in recent:
        if msg.role not in ("user", "assistant"):
            continue
        label = "User" if msg.role == "user" else "Zenki"
        content = msg.content
        if len(content) > _HISTORY_MSG_MAX_CHARS:
            content = content[:_HISTORY_MSG_MAX_CHARS] + "..."
        lines.append(f"**{label}**: {content}")
    return "\n\n".join(lines)


async def _chat_loop(orchestrator, settings, resume: str | None) -> None:
    """Async REPL loop using ZenkiConversation for multi-turn context."""
    sm = orchestrator.session_manager

    # Close stale sessions before starting.
    timeout = settings.session.timeout_minutes
    stale = sm.close_stale_sessions(timeout_minutes=timeout)
    if stale:
        console.print(f"[dim]Closed {len(stale)} stale session(s).[/dim]")

    # Resolve or create the DB session.
    conversation_history: str | None = None
    if resume:
        session = sm.get_session(resume)
        if session and session.ended_at is None:
            session_id = session.id
            # Load prior messages for replay.
            prior_messages = sm.get_messages(session_id, limit=50)
            if prior_messages:
                conversation_history = _format_history(prior_messages)
                console.print(
                    f"[dim]Resumed session {resume} with "
                    f"{len(prior_messages)} prior message(s).[/dim]"
                )
            else:
                console.print(f"[dim]Resumed session {resume} (no prior messages).[/dim]")
        else:
            console.print(
                f"[dim]Session {resume} not found or already closed. "
                "Starting new session.[/dim]"
            )
            session = sm.create_session(
                user_id=settings.user.id, channel_type="cli",
            )
            session_id = session.id
    else:
        session = sm.get_or_create_session(
            user_id=settings.user.id, channel_type="cli",
        )
        session_id = session.id

    # Create a multi-turn conversation with optional history context.
    conv = await orchestrator.create_conversation(
        conversation_history=conversation_history,
    )

    async with conv:
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
                if _handle_special(user_input, session_manager=sm, session_id=session_id):
                    break
                continue

            # Persist the user message.
            sm.add_message(session_id=session_id, role="user", content=user_input)

            # Stream the response with progressive rendering.
            try:
                full_parts: list[str] = []
                console.print()
                with Live(
                    Panel(
                        Markdown(""),
                        title="[bold blue]Zenki[/bold blue]",
                        border_style="blue",
                    ),
                    console=console,
                    refresh_per_second=8,
                ) as live:
                    async for chunk in conv.send_streaming(user_input):
                        full_parts.append(chunk)
                        live.update(
                            Panel(
                                Markdown("".join(full_parts)),
                                title="[bold blue]Zenki[/bold blue]",
                                border_style="blue",
                            )
                        )

                response = "".join(full_parts) or "Done."
            except Exception as exc:
                response = f"Error: {exc}"
                console.print(f"\n[red]{response}[/red]\n")
                sm.add_message(
                    session_id=session_id, role="assistant", content=response,
                )
                continue

            # Persist the assistant response.
            sm.add_message(session_id=session_id, role="assistant", content=response)
            console.print()


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

    asyncio.run(_chat_loop(orchestrator, settings, resume))
