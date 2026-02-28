"""Main CLI application for Zenki."""

from __future__ import annotations

import typer

from zenki import __version__

app = typer.Typer(
    name="zenki",
    help="Zenki - Adaptive AI Assistant",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"zenki {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Zenki - Adaptive AI Assistant."""


# --- Register subcommands ---

from zenki.cli.commands.chat import chat  # noqa: E402
from zenki.cli.commands.config_cmd import config_app  # noqa: E402
from zenki.cli.commands.memory_cmd import memory_app  # noqa: E402
from zenki.cli.commands.setup import setup  # noqa: E402
from zenki.cli.commands.skills_cmd import skills_app  # noqa: E402
from zenki.cli.commands.status import status  # noqa: E402

app.command()(setup)
app.command()(chat)
app.command()(status)
app.add_typer(config_app, name="config", help="Manage configuration.")
app.add_typer(memory_app, name="memory", help="Memory management.")
app.add_typer(skills_app, name="skills", help="Skill management.")
