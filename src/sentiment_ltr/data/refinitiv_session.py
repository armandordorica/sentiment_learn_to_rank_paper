"""Helpers for opening a local LSEG/Refinitiv Workspace desktop session."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_FILENAME = "lseg-data.config.json"
PLACEHOLDER_APP_KEY = "PASTE_YOUR_APP_KEY_HERE"


def resolve_config_path(project_root: Path, config_path: str | Path | None = None) -> Path:
    """Resolve the local LSEG config file path."""
    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    env_path = os.environ.get("LSEG_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    return (project_root / DEFAULT_CONFIG_FILENAME).resolve()


def load_app_key(project_root: Path, config_path: str | Path | None = None) -> str:
    """Load the Workspace App Key from env or the local JSON config file."""
    env_key = os.environ.get("LSEG_APP_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    path = resolve_config_path(project_root, config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"LSEG app key config not found at {path}. "
            "Copy lseg-data.config.example.json to lseg-data.config.json and paste your App Key."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        app_key = data["sessions"]["desktop"]["workspace"]["app-key"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Could not read sessions.desktop.workspace.app-key from {path}"
        ) from exc

    app_key = str(app_key).strip()
    if not app_key or app_key == PLACEHOLDER_APP_KEY:
        raise ValueError(f"Replace the placeholder App Key in {path} before connecting.")
    return app_key


def configure_workspace_app_key(
    project_root: Path,
    ld_module: Any,
    config_path: str | Path | None = None,
) -> str:
    """Apply the Workspace App Key to the active LSEG config."""
    app_key = load_app_key(project_root, config_path)
    config = ld_module.get_config()
    config.set_param("sessions.desktop.workspace.app-key", app_key)
    return app_key


def open_workspace_session(
    project_root: Path,
    ld_module: Any,
    config_path: str | Path | None = None,
):
    """Configure the App Key and open a local Workspace desktop session."""
    configure_workspace_app_key(project_root, ld_module, config_path)
    session = ld_module.open_session()
    session_state = str(getattr(session, "state", "")).lower()
    if session_state and "closed" in session_state:
        raise RuntimeError(
            "LSEG session did not open. Confirm Workspace is running, signed in, "
            "and your App Key is valid."
        )
    return session
