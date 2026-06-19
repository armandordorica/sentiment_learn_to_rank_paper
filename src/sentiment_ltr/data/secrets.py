"""Read runtime secrets from environment variables or Streamlit secrets."""

from __future__ import annotations

import os


def get_env_or_secret(name: str) -> str | None:
    """Return a config value from env vars or Streamlit secrets when available."""
    env_value = os.environ.get(name)
    if env_value and str(env_value).strip():
        return str(env_value).strip()

    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
    except Exception:
        secret_value = None

    if secret_value and str(secret_value).strip():
        return str(secret_value).strip()
    return None
