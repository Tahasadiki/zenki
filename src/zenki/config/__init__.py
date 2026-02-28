"""Zenki configuration package."""

from zenki.config.settings import ZenkiSettings

__all__ = ["ZenkiSettings", "get_config_dir"]


def get_config_dir():
    """Return the path to the Zenki configuration directory."""
    return ZenkiSettings.get_config_dir()
