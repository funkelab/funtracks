"""Record exactly which code produced a history log."""

from __future__ import annotations

import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

_GIT_TIMEOUT = 2.0  # seconds


def package_provenance(name: str) -> dict[str, Any]:
    """Version and source of an installed package.

    The source is read from the distribution's PEP 610 ``direct_url.json``, never
    from where the package directory sits: a uv ``.venv`` usually lives inside a git
    repository, so a wheel in its ``site-packages`` would otherwise report that
    repository's commit.

    - Editable install: ``source="git"``, with ``commit`` and ``dirty`` from git in
      the installed directory (``None`` if git fails or it is not a repository).
    - Installed from a VCS URL: ``source="vcs"``, ``commit`` from ``direct_url.json``.
    - Otherwise: ``source="wheel"``; the exact version identifies the code.

    Args:
        name: The distribution name, e.g. ``"funtracks"``.

    Returns:
        dict[str, Any]: ``version``, ``commit``, ``dirty``, ``editable`` and
            ``source``. All ``None`` (and ``editable=False``) if the package is not
            installed.
    """
    info: dict[str, Any] = {
        "version": None,
        "commit": None,
        "dirty": None,
        "editable": False,
        "source": None,
    }
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return info
    info["version"] = dist.version
    info["source"] = "wheel"

    direct_url_text = dist.read_text("direct_url.json")
    if not direct_url_text:
        return info
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return info

    if direct_url.get("dir_info", {}).get("editable"):
        info["editable"] = True
        info["source"] = "git"
        directory = _url_to_path(direct_url.get("url", ""))
        if directory is not None:
            info["commit"], info["dirty"] = _git_state(directory)
    elif "vcs_info" in direct_url:
        info["source"] = "vcs"
        info["commit"] = direct_url["vcs_info"].get("commit_id")
    return info


def _url_to_path(url: str) -> Path | None:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    return Path(url2pathname(unquote(parsed.path)))


def _git_state(directory: Path) -> tuple[str | None, bool | None]:
    """The HEAD commit and whether the work tree has changes, or Nones on failure."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(directory), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit, bool(status.strip())
