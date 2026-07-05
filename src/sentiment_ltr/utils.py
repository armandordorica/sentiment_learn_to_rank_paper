"""Pure utility functions — standard library only, no domain dependencies.

These helpers are intentionally dependency-free so they can be imported by any
layer of the project (data, models, viz, provenance, scripts, web app) without
introducing import cycles.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Iterable


def get_git_info(repo_path: Path = Path(".")) -> dict[str, Any]:
    """Return git commit hash, branch, and dirty-tree flag for a repository.

    Parameters
    ----------
    repo_path:
        Any directory inside the target git repository.

    Returns
    -------
    dict with keys:
        ``commit_hash``, ``commit_hash_short``, ``branch``,
        ``is_dirty``, ``dirty_files``.
        All values fall back to ``None`` / ``False`` / ``[]`` when git is
        unavailable or the path is not inside a repository.
    """

    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=repo_path, text=True).strip()
        except Exception:
            return None

    dirty_output = _run(["git", "status", "--porcelain"])
    return {
        "commit_hash": _run(["git", "rev-parse", "HEAD"]),
        "commit_hash_short": _run(["git", "rev-parse", "--short", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "is_dirty": bool(dirty_output),
        "dirty_files": dirty_output.splitlines() if dirty_output else [],
    }


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes.

    Reads in ``chunk_size``-byte chunks so large weight files (hundreds of MB)
    are never loaded entirely into memory.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_text_label_pairs(texts: Iterable, labels: Iterable) -> str:
    """Return a stable SHA-256 over ``(text, label)`` pairs.

    Useful for detecting silent upstream changes to a labeled dataset between
    runs — the hash changes whenever any text or label changes, or the order
    of rows changes.  Pairs are null-byte / SOH delimited so adjacent values
    cannot be concatenated to produce boundary collisions.

    Parameters
    ----------
    texts:
        Iterable of text strings (e.g. headlines or sentences).
    labels:
        Iterable of labels aligned with *texts*.
    """
    h = hashlib.sha256()
    for text, label in zip(texts, labels):
        h.update(str(text).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(label).encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()
