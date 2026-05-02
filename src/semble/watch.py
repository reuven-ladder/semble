"""Local file watcher: re-index changed files into ``<root>/.semble``."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from semble.index.file_walker import DEFAULT_IGNORED_DIRS, filter_extensions

if TYPE_CHECKING:
    from semble.index.index import SembleIndex


def watch(
    index: SembleIndex,
    *,
    debounce_seconds: float = 0.5,
    on_refresh: "callable | None" = None,  # type: ignore[name-defined]
) -> None:
    """Block, re-indexing on file change events. Ctrl-C to stop.

    :param index: A SembleIndex with an associated root and cache_dir.
    :param debounce_seconds: Coalesce events within this window into one refresh.
    :param on_refresh: Optional callback ``fn(changed_paths)`` invoked after each refresh.
    :raises RuntimeError: If ``index`` has no root, or if ``watchdog`` is not installed.
    """
    try:
        from watchdog.events import FileSystemEvent, FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError as exc:
        raise RuntimeError(
            "watchdog is required for `semble watch`. Install with: pip install watchdog"
        ) from exc

    if index._root is None:
        raise RuntimeError("watch() requires an index built from a local path")
    root = index._root
    ext_set = filter_extensions(index._extensions, include_text_files=index._include_text_files)
    ignore_dirs = DEFAULT_IGNORED_DIRS | (index._ignore or frozenset())

    pending: set[str] = set()
    lock = threading.Lock()
    timer: threading.Timer | None = None

    def _is_relevant(path_str: str) -> bool:
        try:
            p = Path(path_str).resolve()
            rel = p.relative_to(root)
        except ValueError:
            return False
        # Skip ignored directories anywhere in path.
        for part in rel.parts[:-1]:
            if part in ignore_dirs:
                return False
        if p.suffix.lower() not in ext_set:
            # Allow deletes of previously-indexed files even if extension filter excludes them now.
            return rel.as_posix() in index._file_state
        return True

    def _flush() -> None:
        nonlocal timer
        with lock:
            batch = list(pending)
            pending.clear()
            timer = None
        if not batch:
            return
        try:
            index.refresh(batch)
        except Exception as exc:
            print(f"[semble watch] refresh failed: {exc}")
            return
        if on_refresh is not None:
            on_refresh(batch)
        print(f"[semble watch] refreshed {len(batch)} file(s)")

    class _Handler(FileSystemEventHandler):
        def on_any_event(self, event: FileSystemEvent) -> None:
            nonlocal timer
            if event.is_directory:
                return
            if not _is_relevant(event.src_path):
                return
            with lock:
                pending.add(str(Path(event.src_path).resolve()))
                if timer is not None:
                    timer.cancel()
                timer = threading.Timer(debounce_seconds, _flush)
                timer.daemon = True
                timer.start()

    observer = Observer()
    observer.schedule(_Handler(), str(root), recursive=True)
    observer.start()
    print(f"[semble watch] watching {root} (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
