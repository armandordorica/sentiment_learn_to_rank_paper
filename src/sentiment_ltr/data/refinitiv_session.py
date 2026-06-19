"""Helpers for opening LSEG/Refinitiv desktop or cloud platform sessions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from sentiment_ltr.data.secrets import get_env_or_secret


DEFAULT_CONFIG_FILENAME = "lseg-data.config.json"
PLACEHOLDER_APP_KEY = "PASTE_YOUR_APP_KEY_HERE"
SessionMode = Literal["platform", "desktop"]

# Populated by the most recent open_refinitiv_session() call.
LAST_SESSION_MODE: SessionMode | None = None
LAST_SESSION_FALLBACK = False
LAST_SESSION_MESSAGE: str | None = None


def get_last_refinitiv_session_info() -> dict[str, object]:
    """Return metadata about how the latest Refinitiv session was opened."""
    return {
        "mode": LAST_SESSION_MODE,
        "fallback": LAST_SESSION_FALLBACK,
        "message": LAST_SESSION_MESSAGE,
    }


def platform_credentials_available(project_root: Path, config_path: str | Path | None = None) -> bool:
    """Return whether LSEG cloud platform credentials are configured."""
    try:
        load_platform_credentials(project_root, config_path)
        return True
    except (FileNotFoundError, ValueError, OSError):
        return False


def _platform_fallback_enabled() -> bool:
    flag = (get_env_or_secret("LSEG_ALLOW_PLATFORM_FALLBACK") or "true").lower()
    return flag not in {"0", "false", "no", "off"}


def _reset_last_session_info(
    mode: SessionMode | None,
    *,
    fallback: bool = False,
    message: str | None = None,
) -> None:
    global LAST_SESSION_MODE, LAST_SESSION_FALLBACK, LAST_SESSION_MESSAGE
    LAST_SESSION_MODE = mode
    LAST_SESSION_FALLBACK = fallback
    LAST_SESSION_MESSAGE = message


def _close_ld_session(ld_module: Any) -> None:
    try:
        ld_module.close_session()
    except Exception:
        pass


def is_huggingface_space() -> bool:
    """Return whether the app is running on a Hugging Face Space."""
    return bool(os.environ.get("SPACE_ID")) or os.environ.get("SYSTEM") == "spaces"


def resolve_config_path(project_root: Path, config_path: str | Path | None = None) -> Path:
    """Resolve the local LSEG config file path."""
    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    env_path = get_env_or_secret("LSEG_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    return (project_root / DEFAULT_CONFIG_FILENAME).resolve()


def load_app_key(project_root: Path, config_path: str | Path | None = None) -> str:
    """Load the LSEG App Key from env, Streamlit secrets, or the local JSON config file."""
    env_key = get_env_or_secret("LSEG_APP_KEY")
    if env_key:
        return env_key

    path = resolve_config_path(project_root, config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"LSEG app key config not found at {path}. "
            "Copy lseg-data.config.example.json to lseg-data.config.json and paste your App Key, "
            "or set LSEG_APP_KEY."
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


def load_platform_credentials(project_root: Path, config_path: str | Path | None = None) -> tuple[str, str, str]:
    """Load username, password, and app key for the LSEG cloud platform session."""
    username = get_env_or_secret("LSEG_USERNAME") or get_env_or_secret("RDP_USERNAME")
    password = get_env_or_secret("LSEG_PASSWORD") or get_env_or_secret("RDP_PASSWORD")
    app_key = load_app_key(project_root, config_path)

    missing = [name for name, value in [("LSEG_USERNAME", username), ("LSEG_PASSWORD", password)] if not value]
    if missing:
        raise ValueError(
            "Cloud Refinitiv access requires "
            + " and ".join(missing)
            + ". Ask U of T for LSEG Data Platform (RDP) credentials."
        )
    return str(username), str(password), app_key


def refinitiv_session_mode(project_root: Path, config_path: str | Path | None = None) -> SessionMode | None:
    """Choose desktop Workspace or cloud platform session based on runtime environment."""
    forced_mode = (get_env_or_secret("LSEG_SESSION_MODE") or "").lower()
    if forced_mode in {"desktop", "platform"}:
        try:
            if forced_mode == "platform":
                load_platform_credentials(project_root, config_path)
            else:
                load_app_key(project_root, config_path)
            return forced_mode  # type: ignore[return-value]
        except (FileNotFoundError, ValueError, OSError):
            return None

    if is_huggingface_space():
        try:
            load_platform_credentials(project_root, config_path)
            return "platform"
        except (FileNotFoundError, ValueError, OSError):
            return None

    try:
        load_platform_credentials(project_root, config_path)
        return "platform"
    except (FileNotFoundError, ValueError, OSError):
        pass

    try:
        load_app_key(project_root, config_path)
        return "desktop"
    except (FileNotFoundError, ValueError, OSError):
        return None


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


def open_platform_session(
    project_root: Path,
    ld_module: Any,
    config_path: str | Path | None = None,
):
    """Open an LSEG Data Platform cloud session using RDP credentials."""
    username, password, app_key = load_platform_credentials(project_root, config_path)
    definition = ld_module.session.platform.Definition(
        app_key=app_key,
        grant=ld_module.session.platform.GrantPassword(username=username, password=password),
        signon_control=True,
    )
    session = definition.get_session()
    session.open()
    ld_module.session.set_default(session)

    session_state = str(getattr(session, "state", "")).lower()
    if session_state and "closed" in session_state:
        raise RuntimeError(
            "LSEG cloud session did not open. Confirm your RDP username, password, and App Key "
            "are valid for the LSEG Data Platform."
        )
    return session


def open_workspace_session(
    project_root: Path,
    ld_module: Any,
    config_path: str | Path | None = None,
):
    """Configure the App Key and open a local Workspace desktop session."""
    configure_workspace_app_key(project_root, ld_module, config_path)
    session = ld_module.open_session()

    try:
        ld_module.get_data(universe=["AAPL.O"], fields=["TR.PriceClose"])
    except Exception as exc:
        message = str(exc).lower()
        if "application key is not valid" in message or "session is not opened" in message:
            raise RuntimeError(
                "LSEG Workspace rejected the desktop API connection (localhost:9006 handshake failed). "
                "Keep Workspace running and signed in, run APPKEY to confirm the key, then retry. "
                "If desktop keeps failing, cloud fallback will be attempted when LSEG_USERNAME and "
                "LSEG_PASSWORD are configured."
            ) from exc
        raise RuntimeError(
            "LSEG desktop session did not open. Keep Workspace running and signed in, then verify "
            "your App Key in APPKEY."
        ) from exc
    return session


def open_refinitiv_session(
    project_root: Path,
    ld_module: Any,
    config_path: str | Path | None = None,
    *,
    allow_platform_fallback: bool | None = None,
):
    """Open the best available Refinitiv session for the current runtime."""
    if allow_platform_fallback is None:
        allow_platform_fallback = _platform_fallback_enabled()

    preferred = refinitiv_session_mode(project_root, config_path)
    if preferred == "platform":
        _reset_last_session_info("platform")
        return open_platform_session(project_root, ld_module, config_path)

    if preferred == "desktop":
        try:
            session = open_workspace_session(project_root, ld_module, config_path)
            _reset_last_session_info("desktop")
            return session
        except RuntimeError as desktop_exc:
            can_fallback = (
                allow_platform_fallback
                and not is_huggingface_space()
                and platform_credentials_available(project_root, config_path)
            )
            if not can_fallback:
                _reset_last_session_info("desktop", message=str(desktop_exc))
                raise

            _close_ld_session(ld_module)
            try:
                session = open_platform_session(project_root, ld_module, config_path)
            except RuntimeError:
                _reset_last_session_info("desktop", message=str(desktop_exc))
                raise desktop_exc from None

            _reset_last_session_info(
                "platform",
                fallback=True,
                message=(
                    "Workspace desktop API failed, so this query used the LSEG cloud API instead. "
                    "Prices should work like Hugging Face; news may still require desktop Workspace "
                    "or the U of T cloud news scope."
                ),
            )
            return session

    if is_huggingface_space():
        _reset_last_session_info(None)
        raise RuntimeError(
            "Hosted Refinitiv requires LSEG cloud credentials. Add LSEG_APP_KEY, "
            "LSEG_USERNAME, and LSEG_PASSWORD as Hugging Face Space secrets."
        )

    _reset_last_session_info(None)
    raise RuntimeError(
        "Refinitiv is not configured. Set LSEG_APP_KEY for local Workspace access, "
        "or add LSEG_USERNAME and LSEG_PASSWORD for cloud platform access."
    )
