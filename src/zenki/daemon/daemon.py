"""Zenki daemon lifecycle management."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Any

from zenki.config.settings import ZenkiSettings
from zenki.daemon.server import ZenkiServer

logger = logging.getLogger(__name__)


class ZenkiDaemon:
    """Manages the Zenki daemon lifecycle."""

    def __init__(self, settings: ZenkiSettings) -> None:
        self.settings = settings
        self._server: ZenkiServer | None = None
        self._running = False
        self._pid_file = self._get_pid_file()

    def _get_pid_file(self) -> Path:
        """Get the path to the PID file."""
        config_dir = ZenkiSettings.get_config_dir()
        return config_dir / "zenki.pid"

    def is_running(self) -> bool:
        """Check if the daemon is already running."""
        if not self._pid_file.exists():
            return False

        try:
            pid = int(self._pid_file.read_text().strip())
            # Check if process exists
            os.kill(pid, 0)
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            # PID file exists but process is dead
            self._pid_file.unlink(missing_ok=True)
            return False

    def get_pid(self) -> int | None:
        """Get the PID of the running daemon."""
        if not self._pid_file.exists():
            return None
        try:
            return int(self._pid_file.read_text().strip())
        except (ValueError, FileNotFoundError):
            return None

    def _write_pid(self) -> None:
        """Write the current PID to the PID file."""
        self._pid_file.parent.mkdir(parents=True, exist_ok=True)
        self._pid_file.write_text(str(os.getpid()))

    def _remove_pid(self) -> None:
        """Remove the PID file."""
        self._pid_file.unlink(missing_ok=True)

    async def start(self, foreground: bool = False) -> None:
        """Start the daemon."""
        if self.is_running():
            logger.warning("Daemon is already running (PID: %s)", self.get_pid())
            return

        self._write_pid()
        self._running = True

        # Set up signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        # Start HTTP server
        self._server = ZenkiServer(
            host=self.settings.daemon.host,
            port=self.settings.daemon.port,
        )
        await self._server.start()

        logger.info("Zenki daemon started (PID: %d)", os.getpid())

        if foreground:
            # Keep running until stopped
            try:
                while self._running:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            finally:
                await self.stop()

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        if not self._running:
            return

        logger.info("Stopping Zenki daemon...")
        self._running = False

        if self._server:
            await self._server.stop()

        self._remove_pid()
        logger.info("Zenki daemon stopped")

    def stop_remote(self) -> bool:
        """Stop a remotely running daemon by sending SIGTERM."""
        pid = self.get_pid()
        if pid is None:
            return False

        try:
            os.kill(pid, signal.SIGTERM)
            # Wait briefly for process to stop
            import time
            for _ in range(10):
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    self._remove_pid()
                    return True
            return False
        except ProcessLookupError:
            self._remove_pid()
            return True
        except PermissionError:
            logger.error("Permission denied to stop daemon (PID: %d)", pid)
            return False

    def get_status(self) -> dict[str, Any]:
        """Get daemon status information."""
        running = self.is_running()
        status: dict[str, Any] = {
            "running": running,
            "pid": self.get_pid() if running else None,
            "config_dir": str(ZenkiSettings.get_config_dir()),
            "host": self.settings.daemon.host,
            "port": self.settings.daemon.port,
        }
        return status
