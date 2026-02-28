"""HTTP server for Zenki daemon - handles webhooks and health checks."""

from __future__ import annotations

import json
import logging
from typing import Any

from aiohttp import web

logger = logging.getLogger(__name__)


class ZenkiServer:
    """HTTP server for webhook endpoints and health checks."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8420) -> None:
        self.host = host
        self.port = port
        self._app = web.Application()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._slack_handler: Any = None
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Set up HTTP routes."""
        self._app.router.add_get("/health", self._health_check)
        self._app.router.add_get("/api/v1/status", self._api_status)
        self._app.router.add_post("/slack/events", self._slack_events)
        self._app.router.add_post("/slack/interactions", self._slack_interactions)

    def set_slack_handler(self, handler: Any) -> None:
        """Set the Slack Bolt app for handling events."""
        self._slack_handler = handler

    async def _health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "healthy", "service": "zenki"})

    async def _api_status(self, request: web.Request) -> web.Response:
        """Status API endpoint."""
        status = {
            "status": "running",
            "version": "0.1.0",
            "channels": [],
        }
        return web.json_response(status)

    async def _slack_events(self, request: web.Request) -> web.Response:
        """Handle Slack Events API webhook."""
        try:
            body = await request.text()
            data = json.loads(body)

            # Handle Slack URL verification challenge
            if data.get("type") == "url_verification":
                return web.json_response({"challenge": data["challenge"]})

            if self._slack_handler is None:
                logger.warning("Slack event received but no handler configured")
                return web.Response(status=200)

            # Forward to Slack Bolt handler
            from slack_bolt.adapter.aiohttp import to_aiohttp_response
            from slack_bolt.request import BoltRequest

            bolt_req = BoltRequest(
                body=body,
                headers={k: v for k, v in request.headers.items()},
            )
            bolt_resp = await self._slack_handler.async_dispatch(bolt_req)
            return to_aiohttp_response(bolt_resp)

        except ImportError:
            # slack-bolt not installed, just acknowledge
            return web.Response(status=200)
        except Exception as e:
            logger.error("Error handling Slack event: %s", e)
            return web.Response(status=500, text=str(e))

    async def _slack_interactions(self, request: web.Request) -> web.Response:
        """Handle Slack interactive components (buttons, modals)."""
        try:
            body = await request.text()

            if self._slack_handler is None:
                return web.Response(status=200)

            from slack_bolt.adapter.aiohttp import to_aiohttp_response
            from slack_bolt.request import BoltRequest

            bolt_req = BoltRequest(
                body=body,
                headers={k: v for k, v in request.headers.items()},
            )
            bolt_resp = await self._slack_handler.async_dispatch(bolt_req)
            return to_aiohttp_response(bolt_resp)

        except ImportError:
            return web.Response(status=200)
        except Exception as e:
            logger.error("Error handling Slack interaction: %s", e)
            return web.Response(status=500, text=str(e))

    async def start(self) -> None:
        """Start the HTTP server."""
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        logger.info("Zenki server started on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Stop the HTTP server."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        logger.info("Zenki server stopped")

    @property
    def app(self) -> web.Application:
        """Get the aiohttp application (for testing)."""
        return self._app
