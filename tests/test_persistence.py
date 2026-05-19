"""Concurrency + atomicity tests for ``semble.persistence``."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from semble import SembleIndex
from semble.persistence import (
    _atomic_write_bytes,
    _atomic_write_text,
    _writer_lock,
    cache_dir_for,
    load_index,
    save_index,
)


def test_atomic_write_text_no_tmp_left_behind(tmp_path: Path) -> None:
    """A successful atomic write leaves no ``.tmp`` sidecar behind."""
    target = tmp_path / "meta.json"
    _atomic_write_text(target, json.dumps({"k": "v"}))
    assert target.read_text() == json.dumps({"k": "v"})
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith("meta.json.") and p.name.endswith(".tmp")]
    assert leftovers == []


def test_atomic_write_concurrent_no_filenotfound(tmp_path: Path) -> None:
    """Concurrent writers to the same target must not raise FileNotFoundError on rename.

    Reproduces the pre-fix bug where a shared ``meta.json.tmp`` name caused one
    writer's ``replace`` to fail after another's already renamed.
    """
    target = tmp_path / "meta.json"
    errors: list[BaseException] = []

    def writer(i: int) -> None:
        try:
            for _ in range(20):
                _atomic_write_text(target, json.dumps({"writer": i}))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent writers raised: {errors!r}"
    assert json.loads(target.read_text())["writer"] in range(8)


def test_writer_lock_serializes(tmp_path: Path) -> None:
    """``_writer_lock`` provides mutual exclusion within a single process."""
    cache_dir = tmp_path / ".semble"
    inside = 0
    max_inside = 0
    enter_lock = threading.Lock()

    def worker() -> None:
        nonlocal inside, max_inside
        with _writer_lock(cache_dir):
            with enter_lock:
                inside += 1
                max_inside = max(max_inside, inside)
            # Tiny window to expose overlap if the lock were broken.
            for _ in range(1000):
                pass
            with enter_lock:
                inside -= 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_inside == 1


def test_save_index_concurrent_yields_valid_cache(mock_model: Any, tmp_project: Path) -> None:
    """Two threads calling ``save_index`` on the same cache must leave it loadable.

    Pre-fix this could leave a torn ``chunks.jsonl`` or fail with FileNotFoundError
    on the ``meta.json.tmp`` rename. Post-fix the cache always loads cleanly.
    """
    index = SembleIndex.from_path(tmp_project, model=mock_model)
    cache_dir = cache_dir_for(tmp_project)
    index._cache_dir = cache_dir

    errors: list[BaseException] = []

    def saver() -> None:
        try:
            for _ in range(10):
                save_index(index, cache_dir)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=saver) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"save_index raised under concurrency: {errors!r}"

    # No stale per-writer tmp files left behind.
    tmps = [p for p in cache_dir.iterdir() if p.name.endswith(".tmp")]
    assert tmps == [], f"leftover tmp files: {tmps!r}"

    # Cache loads back cleanly and round-trips chunk count.
    reloaded = load_index(cache_dir, model=mock_model)
    assert reloaded is not None
    assert len(reloaded.chunks) == len(index.chunks)


def test_save_index_atomic_under_writer_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If a write fails mid-flight, the previous good cache must still be loadable."""
    target = tmp_path / "meta.json"
    _atomic_write_text(target, json.dumps({"schema_version": 1}))
    good = target.read_text()

    original_replace = Path.replace

    def boom(self: Path, target: Any) -> None:  # type: ignore[override]
        raise OSError("simulated rename failure")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError):
        _atomic_write_text(target, json.dumps({"schema_version": 99}))
    monkeypatch.setattr(Path, "replace", original_replace)

    # Previous good content untouched.
    assert target.read_text() == good
    # And no leftover tmp.
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_atomic_write_bytes_roundtrip(tmp_path: Path) -> None:
    """Bytes written atomically round-trip through ``np.load`` unchanged."""
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    import io as _io

    buf = _io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    _atomic_write_bytes(tmp_path / "embeddings.npy", buf.getvalue())

    loaded = np.load(tmp_path / "embeddings.npy", allow_pickle=False)
    assert np.array_equal(loaded, arr)
