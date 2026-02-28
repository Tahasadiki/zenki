"""Systemd service management for Linux."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SERVICE_NAME = "zenki"
SERVICE_TEMPLATE = """\
[Unit]
Description=Zenki AI Assistant Daemon
After=network.target

[Service]
Type=simple
User={user}
ExecStart={exec_path} start --foreground
Restart=on-failure
RestartSec=5
Environment=ZENKI_CONFIG_DIR={config_dir}
WorkingDirectory={home_dir}

[Install]
WantedBy=multi-user.target
"""


def get_service_path() -> Path:
    """Get the systemd service file path."""
    return Path(f"/etc/systemd/system/{SERVICE_NAME}.service")


def get_user_service_path() -> Path:
    """Get the user-level systemd service path."""
    config_dir = Path.home() / ".config" / "systemd" / "user"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / f"{SERVICE_NAME}.service"


def generate_service_file(exec_path: str | None = None, config_dir: str | None = None) -> str:
    """Generate the systemd service file content."""
    import shutil
    if exec_path is None:
        exec_path = shutil.which("zenki") or "/usr/local/bin/zenki"
    if config_dir is None:
        config_dir = str(Path.home() / ".config" / "zenki")

    return SERVICE_TEMPLATE.format(
        user=os.environ.get("USER", "root"),
        exec_path=exec_path,
        config_dir=config_dir,
        home_dir=str(Path.home()),
    )


def install(user_level: bool = True) -> str:
    """Install the systemd service."""
    content = generate_service_file()
    service_path = get_user_service_path() if user_level else get_service_path()
    service_path.write_text(content)

    if user_level:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", SERVICE_NAME], check=True)
        return (
            f"User service installed at {service_path}. "
            f"Start with: systemctl --user start {SERVICE_NAME}"
        )
    else:
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "enable", SERVICE_NAME], check=True)
        return (
            f"System service installed at {service_path}. "
            f"Start with: sudo systemctl start {SERVICE_NAME}"
        )


def uninstall(user_level: bool = True) -> str:
    """Uninstall the systemd service."""
    service_path = get_user_service_path() if user_level else get_service_path()

    if user_level:
        subprocess.run(["systemctl", "--user", "stop", SERVICE_NAME], check=False)
        subprocess.run(["systemctl", "--user", "disable", SERVICE_NAME], check=False)
    else:
        subprocess.run(["systemctl", "stop", SERVICE_NAME], check=False)
        subprocess.run(["systemctl", "disable", SERVICE_NAME], check=False)

    if service_path.exists():
        service_path.unlink()
    return f"Service uninstalled from {service_path}"


def status(user_level: bool = True) -> str:
    """Get the systemd service status."""
    cmd = (
        ["systemctl", "--user", "status", SERVICE_NAME]
        if user_level
        else ["systemctl", "status", SERVICE_NAME]
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout or result.stderr
