"""Classify LSEG / Refinitiv API errors for retry and session policy."""

from __future__ import annotations


def is_refinitiv_rate_limit_error(exc: BaseException) -> bool:
    """Return True for HTTP 429 / throttle responses."""
    message = str(exc).lower()
    return (
        "too many requests" in message
        or "error code 429" in message
        or " 429 " in f" {message} "
        or "throttle" in message
        or "rate limit" in message
    )


def is_refinitiv_scope_error(exc: BaseException) -> bool:
    """Return True when the session lacks a required news/data scope."""
    message = str(exc).lower()
    return (
        "insufficient scope" in message
        or "no user scope" in message
        or "missing scopes" in message
        or "trapi.data.news.read" in message
    )
