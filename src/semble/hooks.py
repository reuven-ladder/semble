"""Git hook installation helpers."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

HOOK_MARKER = "# semble-managed hook"
HOOK_TEMPLATE = """#!/usr/bin/env bash
{marker}
# Re-index the repo after each commit so MCP/CLI search reflects the new state.
# Remove this hook with: semble uninstall-hooks
set -e
ROOT="$(git rev-parse --show-toplevel)"
exec semble reindex "$ROOT" >/dev/null 2>&1 || true
"""


def _git_dir(repo: Path) -> Path:
    """Return the .git directory for ``repo`` (handles worktrees and gitdir files)."""
    result = subprocess.run(
        ["git", "rev-parse", "--git-path", "hooks"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return (repo / result.stdout.strip()).resolve()


def install_hook(repo: str | Path, hook_name: str = "post-commit", force: bool = False) -> Path:
    """Write a semble-managed git hook into ``repo``. Returns the path written.

    :raises FileExistsError: If a non-semble hook already exists and ``force`` is False.
    :raises RuntimeError: If ``repo`` is not a git repository.
    """
    repo_path = Path(repo).resolve()
    if not (repo_path / ".git").exists() and not (repo_path / ".git").is_file():
        raise RuntimeError(f"{repo_path} is not a git repository")
    hooks_dir = _git_dir(repo_path)
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / hook_name
    if hook_path.exists() and not force:
        existing = hook_path.read_text(encoding="utf-8", errors="ignore")
        if HOOK_MARKER not in existing:
            raise FileExistsError(
                f"{hook_path} already exists and is not semble-managed. Re-run with --force to overwrite."
            )
    hook_path.write_text(HOOK_TEMPLATE.format(marker=HOOK_MARKER), encoding="utf-8")
    mode = hook_path.stat().st_mode
    hook_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return hook_path


def uninstall_hook(repo: str | Path, hook_name: str = "post-commit") -> bool:
    """Remove the semble-managed hook if present. Returns True if removed."""
    repo_path = Path(repo).resolve()
    try:
        hooks_dir = _git_dir(repo_path)
    except subprocess.CalledProcessError:
        return False
    hook_path = hooks_dir / hook_name
    if not hook_path.exists():
        return False
    content = hook_path.read_text(encoding="utf-8", errors="ignore")
    if HOOK_MARKER not in content:
        return False
    os.unlink(hook_path)
    return True
