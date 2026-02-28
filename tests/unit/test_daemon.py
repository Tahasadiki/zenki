"""Tests for the daemon server and lifecycle."""


import pytest

from zenki.config.settings import ZenkiSettings
from zenki.daemon.daemon import ZenkiDaemon
from zenki.daemon.server import ZenkiServer


class TestZenkiServer:
    def test_server_creation(self):
        server = ZenkiServer(host="127.0.0.1", port=9999)
        assert server.host == "127.0.0.1"
        assert server.port == 9999

    def test_app_has_routes(self):
        server = ZenkiServer()
        routes = [
            r.resource.canonical
            for r in server.app.router.routes()
            if hasattr(r, "resource") and r.resource
        ]
        assert "/health" in routes
        assert "/api/v1/status" in routes
        assert "/slack/events" in routes

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        server = ZenkiServer()
        from aiohttp.test_utils import TestClient, TestServer
        async with TestClient(TestServer(server.app)) as client:
            resp = await client.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "healthy"
            assert data["service"] == "zenki"

    @pytest.mark.asyncio
    async def test_status_endpoint(self):
        server = ZenkiServer()
        from aiohttp.test_utils import TestClient, TestServer
        async with TestClient(TestServer(server.app)) as client:
            resp = await client.get("/api/v1/status")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "running"
            assert "version" in data

    @pytest.mark.asyncio
    async def test_slack_url_verification(self):
        server = ZenkiServer()
        from aiohttp.test_utils import TestClient, TestServer
        async with TestClient(TestServer(server.app)) as client:
            resp = await client.post(
                "/slack/events",
                json={"type": "url_verification", "challenge": "test_challenge"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["challenge"] == "test_challenge"

    @pytest.mark.asyncio
    async def test_slack_event_no_handler(self):
        server = ZenkiServer()
        from aiohttp.test_utils import TestClient, TestServer
        async with TestClient(TestServer(server.app)) as client:
            resp = await client.post(
                "/slack/events",
                json={"type": "event_callback", "event": {"type": "message"}},
            )
            assert resp.status == 200


class TestZenkiDaemon:
    def test_daemon_creation(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ZenkiSettings, "get_config_dir", staticmethod(lambda: tmp_path))
        settings = ZenkiSettings()
        daemon = ZenkiDaemon(settings)
        assert daemon._pid_file == tmp_path / "zenki.pid"

    def test_not_running_initially(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ZenkiSettings, "get_config_dir", staticmethod(lambda: tmp_path))
        settings = ZenkiSettings()
        daemon = ZenkiDaemon(settings)
        assert not daemon.is_running()

    def test_get_pid_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ZenkiSettings, "get_config_dir", staticmethod(lambda: tmp_path))
        settings = ZenkiSettings()
        daemon = ZenkiDaemon(settings)
        assert daemon.get_pid() is None

    def test_write_and_read_pid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ZenkiSettings, "get_config_dir", staticmethod(lambda: tmp_path))
        settings = ZenkiSettings()
        daemon = ZenkiDaemon(settings)
        daemon._write_pid()
        import os
        assert daemon.get_pid() == os.getpid()

    def test_remove_pid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ZenkiSettings, "get_config_dir", staticmethod(lambda: tmp_path))
        settings = ZenkiSettings()
        daemon = ZenkiDaemon(settings)
        daemon._write_pid()
        daemon._remove_pid()
        assert daemon.get_pid() is None
        assert not daemon._pid_file.exists()

    def test_stale_pid_cleaned(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ZenkiSettings, "get_config_dir", staticmethod(lambda: tmp_path))
        settings = ZenkiSettings()
        daemon = ZenkiDaemon(settings)
        # Write a PID that doesn't exist
        daemon._pid_file.write_text("99999999")
        assert not daemon.is_running()
        # PID file should be cleaned up
        assert not daemon._pid_file.exists()

    def test_get_status(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ZenkiSettings, "get_config_dir", staticmethod(lambda: tmp_path))
        settings = ZenkiSettings()
        daemon = ZenkiDaemon(settings)
        status = daemon.get_status()
        assert status["running"] is False
        assert status["pid"] is None
        assert "config_dir" in status
        assert status["port"] == 8420
