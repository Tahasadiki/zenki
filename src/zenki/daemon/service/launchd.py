"""Launchd service management for macOS."""

from __future__ import annotations

import subprocess
from pathlib import Path

SERVICE_LABEL = "com.zenki.daemon"
PLIST_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exec_path}</string>
        <string>start</string>
        <string>--foreground</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>ZENKI_CONFIG_DIR</key>
        <string>{config_dir}</string>
    </dict>
    <key>StandardOutPath</key>
    <string>{log_dir}/zenki.out.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/zenki.err.log</string>
    <key>WorkingDirectory</key>
    <string>{home_dir}</string>
</dict>
</plist>
"""


def get_plist_path() -> Path:
    """Get the launchd plist file path."""
    return Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_LABEL}.plist"


def generate_plist(exec_path: str | None = None, config_dir: str | None = None) -> str:
    """Generate the launchd plist content."""
    import shutil
    if exec_path is None:
        exec_path = shutil.which("zenki") or "/usr/local/bin/zenki"
    if config_dir is None:
        config_dir = str(Path.home() / ".config" / "zenki")

    log_dir = str(Path(config_dir) / "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    return PLIST_TEMPLATE.format(
        label=SERVICE_LABEL,
        exec_path=exec_path,
        config_dir=config_dir,
        log_dir=log_dir,
        home_dir=str(Path.home()),
    )


def install() -> str:
    """Install the launchd service."""
    plist_path = get_plist_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(generate_plist())

    subprocess.run(["launchctl", "load", str(plist_path)], check=True)
    return f"Service installed at {plist_path}. It will start automatically on login."


def uninstall() -> str:
    """Uninstall the launchd service."""
    plist_path = get_plist_path()

    subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
    if plist_path.exists():
        plist_path.unlink()
    return f"Service uninstalled from {plist_path}"


def status() -> str:
    """Get the launchd service status."""
    result = subprocess.run(
        ["launchctl", "list", SERVICE_LABEL],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return f"Service is loaded:\n{result.stdout}"
    return "Service is not loaded"
