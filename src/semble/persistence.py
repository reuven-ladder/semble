"""Disk persistence for SembleIndex.

Layout under ``<root>/.semble/``::

    chunks.jsonl    one JSON object per chunk
    embeddings.npy  float32 array, row order matches chunks.jsonl
    meta.json       index metadata + per-file mtime map
    .lock           cross-process advisory lock for writers

BM25 and the vicinity backend are not persisted directly — they are rebuilt
from chunks/embeddings on load, which is the cheap step in indexing.

Writers are serialized cross-process via an ``fcntl`` advisory lock on
``.lock``. Each of the three payload files is replaced via a per-writer
unique ``.tmp`` plus ``rename(2)`` so a partial write or a concurrent writer
can never leave the cache half-updated.
"""

from __future__ import annotations

import io
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import bm25s
import numpy as np
import numpy.typing as npt
from vicinity.backends.basic import BasicArgs

from semble.index.dense import SelectableBasicBackend
from semble.index.sparse import enrich_for_bm25
from semble.tokens import tokenize
from semble.types import Chunk

if TYPE_CHECKING:
    from semble.index.index import SembleIndex

CACHE_DIRNAME = ".semble"
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class IndexMeta:
    schema_version: int
    model_name: str
    root: str
    extensions: list[str] | None
    ignore: list[str] | None
    include_text_files: bool
    file_state: dict[str, float]  # display-relative path -> mtime


def cache_dir_for(root: Path) -> Path:
    """Return the .semble cache dir under root."""
    return Path(root) / CACHE_DIRNAME


def _chunks_path(d: Path) -> Path:
    return d / "chunks.jsonl"


def _embeddings_path(d: Path) -> Path:
    return d / "embeddings.npy"


def _meta_path(d: Path) -> Path:
    return d / "meta.json"


def _lock_path(d: Path) -> Path:
    return d / ".lock"


def meta_mtime(cache_dir: Path) -> float | None:
    """Return mtime of meta.json or None if absent."""
    p = _meta_path(cache_dir)
    return p.stat().st_mtime if p.exists() else None


@contextmanager
def _writer_lock(cache_dir: Path) -> Iterator[None]:
    """Exclusive cross-process advisory lock for index writers.

    Multiple processes (e.g. ``semble watch`` and a ``post-commit`` hook
    invoking ``semble reindex``) can otherwise step on each other's
    rename-into-place, corrupting the cache. ``fcntl.flock`` serializes them.
    """
    import fcntl

    cache_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(_lock_path(cache_dir)), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically via a per-writer unique tmp file."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except BaseException:
        # Best-effort cleanup; tmp may already have been renamed.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def save_index(index: SembleIndex, cache_dir: Path) -> None:
    """Persist chunks, embeddings, and meta to ``cache_dir``.

    Serialized cross-process and atomic per-file: a concurrent writer either
    sees the full pre-state or the full post-state, never a torn mix.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.asarray(index._semantic_index._vectors, dtype=np.float32)

    emb_buf = io.BytesIO()
    np.save(emb_buf, embeddings, allow_pickle=False)
    emb_bytes = emb_buf.getvalue()

    chunks_buf = io.StringIO()
    for c in index.chunks:
        chunks_buf.write(
            json.dumps(
                {
                    "content": c.content,
                    "file_path": c.file_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "language": c.language,
                },
                ensure_ascii=False,
            )
        )
        chunks_buf.write("\n")
    chunks_text = chunks_buf.getvalue()

    meta = {
        "schema_version": SCHEMA_VERSION,
        "model_name": index._model_name,
        "root": str(index._root),
        "extensions": sorted(index._extensions) if index._extensions is not None else None,
        "ignore": sorted(index._ignore) if index._ignore is not None else None,
        "include_text_files": index._include_text_files,
        "file_state": index._file_state,
    }
    meta_text = json.dumps(meta, indent=2)

    with _writer_lock(cache_dir):
        _atomic_write_bytes(_embeddings_path(cache_dir), emb_bytes)
        _atomic_write_text(_chunks_path(cache_dir), chunks_text)
        _atomic_write_text(_meta_path(cache_dir), meta_text)


def _load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            chunks.append(
                Chunk(
                    content=d["content"],
                    file_path=d["file_path"],
                    start_line=d["start_line"],
                    end_line=d["end_line"],
                    language=d.get("language"),
                )
            )
    return chunks


def _build_bm25(chunks: list[Chunk]) -> bm25s.BM25:
    bm25_index = bm25s.BM25()
    bm25_index.index([tokenize(enrich_for_bm25(c)) for c in chunks], show_progress=False)
    return bm25_index


def _build_semantic(embeddings: npt.NDArray[np.float32]) -> SelectableBasicBackend:
    return SelectableBasicBackend(embeddings, BasicArgs())


def load_meta(cache_dir: Path) -> IndexMeta | None:
    """Read meta.json or return None if missing/incompatible."""
    p = _meta_path(cache_dir)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if d.get("schema_version") != SCHEMA_VERSION:
        return None
    return IndexMeta(
        schema_version=d["schema_version"],
        model_name=d["model_name"],
        root=d["root"],
        extensions=d.get("extensions"),
        ignore=d.get("ignore"),
        include_text_files=d.get("include_text_files", False),
        file_state=d.get("file_state", {}),
    )


def load_index(cache_dir: Path, model: object | None = None):  # type: ignore[no-untyped-def]
    """Load a persisted index. Returns SembleIndex or None if cache absent/invalid.

    If ``model`` is None, the model named in meta.json is loaded.
    """
    from semble.index.dense import load_model
    from semble.index.index import SembleIndex

    meta = load_meta(cache_dir)
    if meta is None:
        return None
    chunks_p = _chunks_path(cache_dir)
    emb_p = _embeddings_path(cache_dir)
    if not chunks_p.exists() or not emb_p.exists():
        return None

    chunks = _load_chunks(chunks_p)
    embeddings = np.load(emb_p, allow_pickle=False).astype(np.float32, copy=False)
    if len(chunks) != len(embeddings):
        return None

    if model is None:
        model = load_model(meta.model_name)

    bm25_index = _build_bm25(chunks)
    semantic_index = _build_semantic(embeddings)

    index = SembleIndex(model, bm25_index, semantic_index, chunks)  # type: ignore[arg-type]
    index._embeddings = embeddings
    index._root = Path(meta.root)
    index._cache_dir = cache_dir
    index._model_name = meta.model_name
    index._extensions = frozenset(meta.extensions) if meta.extensions is not None else None
    index._ignore = frozenset(meta.ignore) if meta.ignore is not None else None
    index._include_text_files = meta.include_text_files
    index._file_state = dict(meta.file_state)
    return index
