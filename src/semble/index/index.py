from __future__ import annotations

import contextlib
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
from bm25s import BM25

from semble.index.chunker import chunk_source
from semble.index.create import create_index_from_path
from semble.index.dense import SelectableBasicBackend, embed_chunks, load_model
from semble.index.file_walker import filter_extensions, language_for_path, walk_files
from semble.search import search_bm25, search_hybrid, search_semantic
from semble.types import Chunk, Encoder, IndexStats, SearchMode, SearchResult


class SembleIndex:
    """Fast local code index with hybrid search."""

    def __init__(
        self,
        model: Encoder,
        bm25_index: BM25,
        semantic_index: SelectableBasicBackend,
        chunks: list[Chunk],
    ) -> None:
        """Internal constructor — use :meth:`from_path` or :meth:`from_git`.

        :param model: Embedding model to use.
        :param bm25_index: The bm25 index.
        :param semantic_index: The semantic index.
        :param chunks: The found chunks.
        """
        self.model: Encoder = model
        self.chunks: list[Chunk] = chunks
        self._bm25_index: BM25 = bm25_index
        self._semantic_index: SelectableBasicBackend = semantic_index
        self._file_mapping, self._language_mapping = self._populate_mapping()
        # Live-refresh state. Populated by from_path / load_index; absent for from_git.
        self._embeddings: npt.NDArray[np.float32] | None = None
        self._root: Path | None = None
        self._cache_dir: Path | None = None
        self._model_name: str | None = None
        self._extensions: frozenset[str] | None = None
        self._ignore: frozenset[str] | None = None
        self._include_text_files: bool = False
        self._file_state: dict[str, float] = {}

    def _populate_mapping(self) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        """Build (file → chunk indices, language → chunk indices) mappings, in that order."""
        language_to_id = defaultdict(list)
        file_to_id = defaultdict(list)
        for i, chunk in enumerate(self.chunks):
            language = chunk.language
            if language:
                language_to_id[language].append(i)
            file_to_id[chunk.file_path].append(i)

        return dict(file_to_id), dict(language_to_id)

    @property
    def stats(self) -> IndexStats:
        """Stats of an index."""
        language_counts: dict[str, int] = defaultdict(int)
        for chunk in self.chunks:
            if chunk.language:
                language_counts[chunk.language] += 1

        return IndexStats(
            indexed_files=len(self._file_mapping),
            total_chunks=len(self.chunks),
            languages=dict(language_counts),
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        model: Encoder | None = None,
        extensions: frozenset[str] | None = None,
        ignore: frozenset[str] | None = None,
        include_text_files: bool = False,
        cache_dir: str | Path | None = None,
        model_name: str | None = None,
    ) -> SembleIndex:
        """Create and index a SembleIndex from a directory.

        :param path: Root directory to index.
        :param model: Embedding model to use. Defaults to potion-code-16M.
        :param extensions: File extensions to include. Defaults to a standard set of code extensions.
        :param ignore: Directory names to skip. Defaults to common VCS and build dirs.
        :param include_text_files: If True, also index non-code text files (.md, .yaml, .json, etc.).
        :param cache_dir: If set, persist index to this directory after build (enables refresh).
        :param model_name: Name of the embedding model (recorded in cache for later reload).
        :return: An indexed SembleIndex. Chunk file paths are relative to ``path``.
        :raises FileNotFoundError: If `path` does not exist.
        :raises NotADirectoryError: If `path` exists but is not a directory.
        """
        from semble.persistence import save_index

        model = model or load_model()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
        path = path.resolve()
        bm25, vicinity, chunks = create_index_from_path(
            path,
            model=model,
            extensions=extensions,
            ignore=ignore,
            include_text_files=include_text_files,
            display_root=path,
        )

        index = cls(model, bm25, vicinity, chunks)
        index._embeddings = np.asarray(vicinity._vectors, dtype=np.float32)
        index._root = path
        index._cache_dir = Path(cache_dir) if cache_dir is not None else None
        index._model_name = model_name
        index._extensions = extensions
        index._ignore = ignore
        index._include_text_files = include_text_files
        index._file_state = _scan_file_state(path, extensions, ignore, include_text_files)

        if index._cache_dir is not None:
            save_index(index, index._cache_dir)

        return index

    @classmethod
    def from_git(
        cls,
        url: str,
        ref: str | None = None,
        model: Encoder | None = None,
        extensions: frozenset[str] | None = None,
        ignore: frozenset[str] | None = None,
        include_text_files: bool = False,
    ) -> SembleIndex:
        """Clone a git repository and index it.

        The repository is cloned into a temporary directory that is removed once
        indexing finishes. Chunk content is preserved in-memory, but
        ``chunk.file_path`` will not point to a readable file after this call
        returns — it is a repo-relative label, not a filesystem path.

        :param url: URL of the git repository to clone (any git provider).
        :param ref: Branch or tag to check out. Defaults to the remote HEAD.
        :param model: Embedding model to use. Defaults to potion-code-16M.
        :param extensions: File extensions to include. Defaults to a standard set of code extensions.
        :param ignore: Directory names to skip. Defaults to common VCS and build dirs.
        :param include_text_files: If True, also index non-code text files (.md, .yaml, .json, etc.).
        :return: An indexed SembleIndex. Chunk file paths are repo-relative (e.g. ``src/foo.py``).
        :raises RuntimeError: If git is not on PATH or the clone fails.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # `--` prevents `url` from being interpreted as a git option (e.g. `--upload-pack=...`).
            cmd = ["git", "clone", "--depth", "1", *(["--branch", ref] if ref else []), "--", url, tmp_dir]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
            except FileNotFoundError:
                raise RuntimeError("git is not installed or not on PATH") from None
            if result.returncode != 0:
                raise RuntimeError(f"git clone failed for {url!r}:\n{result.stderr.strip()}")
            model = model or load_model()
            resolved_path = Path(tmp_dir).resolve()
            bm25, vicinity, chunks = create_index_from_path(
                resolved_path,
                model=model,
                extensions=extensions,
                ignore=ignore,
                include_text_files=include_text_files,
                display_root=resolved_path,
            )

            index = cls(model, bm25, vicinity, chunks)
            index._embeddings = np.asarray(vicinity._vectors, dtype=np.float32)
            return index

    def refresh(self, changed_paths: Iterable[str | Path] | None = None) -> None:
        """Re-index the codebase, in full or for a subset of files.

        :param changed_paths: If None, perform a full rebuild. Otherwise, paths
            (absolute or root-relative) of files added/modified/deleted since
            the last build; only those files are re-chunked and re-embedded.
        :raises RuntimeError: If the index has no associated root (e.g. built via from_git).
        """
        from semble.persistence import save_index

        if self._root is None:
            raise RuntimeError("refresh() requires an index built with a root path")

        if changed_paths is None:
            self._full_rebuild()
        else:
            self._incremental(changed_paths)

        if self._cache_dir is not None:
            save_index(self, self._cache_dir)

    def _full_rebuild(self) -> None:
        assert self._root is not None
        bm25, vicinity, chunks = create_index_from_path(
            self._root,
            model=self.model,
            extensions=self._extensions,
            ignore=self._ignore,
            include_text_files=self._include_text_files,
            display_root=self._root,
        )
        self._install(bm25, vicinity, chunks, np.asarray(vicinity._vectors, dtype=np.float32))
        self._file_state = _scan_file_state(
            self._root, self._extensions, self._ignore, self._include_text_files
        )

    def _incremental(self, changed_paths: Iterable[str | Path]) -> None:
        assert self._root is not None
        rel_changed: set[str] = set()
        for p in changed_paths:
            rel_changed.add(_to_rel(self._root, p))

        kept_pairs = [(c, i) for i, c in enumerate(self.chunks) if c.file_path not in rel_changed]
        kept_chunks = [c for c, _ in kept_pairs]
        kept_idx = [i for _, i in kept_pairs]
        embeddings = self._embeddings if self._embeddings is not None else np.empty((0, 0), dtype=np.float32)
        kept_embeddings = embeddings[kept_idx] if kept_idx else np.empty((0, embeddings.shape[1] if embeddings.ndim == 2 else 0), dtype=np.float32)

        ext_set = filter_extensions(self._extensions, include_text_files=self._include_text_files)
        new_chunks: list[Chunk] = []
        for rel in rel_changed:
            abs_path = (self._root / rel).resolve()
            if not abs_path.is_file() or abs_path.suffix.lower() not in ext_set:
                self._file_state.pop(rel, None)
                continue
            language = language_for_path(abs_path)
            with contextlib.suppress(OSError):
                source = abs_path.read_text(encoding="utf-8", errors="replace")
                new_chunks.extend(chunk_source(source, rel, language))
                with contextlib.suppress(OSError):
                    self._file_state[rel] = abs_path.stat().st_mtime

        new_embeddings = embed_chunks(self.model, new_chunks)
        all_chunks = kept_chunks + new_chunks
        if not all_chunks:
            # Empty index: keep prior structures as-is to avoid breaking search() invariants.
            # bm25s.BM25.index() requires at least one document; refuse to collapse.
            raise ValueError("Refresh would leave the index empty; aborting.")

        if kept_embeddings.size and new_embeddings.size:
            all_embeddings = np.vstack([kept_embeddings, new_embeddings]).astype(np.float32, copy=False)
        elif new_embeddings.size:
            all_embeddings = new_embeddings.astype(np.float32, copy=False)
        else:
            all_embeddings = kept_embeddings.astype(np.float32, copy=False)

        from semble.persistence import _build_bm25, _build_semantic

        bm25 = _build_bm25(all_chunks)
        semantic = _build_semantic(all_embeddings)
        self._install(bm25, semantic, all_chunks, all_embeddings)

    def _install(
        self,
        bm25: BM25,
        semantic: SelectableBasicBackend,
        chunks: list[Chunk],
        embeddings: npt.NDArray[np.float32],
    ) -> None:
        self._bm25_index = bm25
        self._semantic_index = semantic
        self.chunks = chunks
        self._embeddings = embeddings
        self._file_mapping, self._language_mapping = self._populate_mapping()

    def find_related(self, source: Chunk | SearchResult, *, top_k: int = 5) -> list[SearchResult]:
        """Return chunks semantically similar to the given chunk or search result.

        :param source: A SearchResult or Chunk to use as the seed.
        :param top_k: Number of similar chunks to return.
        :return: Ranked list of SearchResult objects, most similar first.
        """
        target = source.chunk if isinstance(source, SearchResult) else source
        selector = self._get_selector_vector(filter_languages=[target.language]) if target.language else None
        results = search_semantic(target.content, self.model, self._semantic_index, self.chunks, top_k + 1, selector)
        return [r for r in results if r.chunk != target][:top_k]

    def _get_selector_vector(
        self, filter_languages: list[str] | None = None, filter_paths: list[str] | None = None
    ) -> npt.NDArray[np.int_] | None:
        """Create a vector of chunk indices to restrict retrieval to."""
        selector = []
        for language in filter_languages or []:
            selector.extend(self._language_mapping.get(language, []))
        for filename in filter_paths or []:
            selector.extend(self._file_mapping.get(filename, []))

        return np.unique(selector) if selector else None

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: SearchMode | str = SearchMode.HYBRID,
        alpha: float | None = None,
        filter_languages: list[str] | None = None,
        filter_paths: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search the index and return the top-k most relevant chunks.

        :param query: Natural-language or keyword query string.
        :param top_k: Maximum number of results to return.
        :param mode: Search strategy — "hybrid" (default), "semantic", or "bm25".
        :param alpha: Blend weight for hybrid score combination; 1.0 = full semantic
            weight, 0.0 = full BM25 weight. File-path penalties and diversity reranking
            are applied regardless. ``None`` auto-detects from query type.
        :param filter_languages: Optional list of language codes; if set, only chunks in
            these languages are returned.
        :param filter_paths: Optional list of repo-relative file paths; if set, only
            chunks from these files are returned.
        :return: Ranked list of :class:`SearchResult` objects, best match first.
        :raises ValueError: If `mode` is not a recognised search strategy.
        """
        bm25_index, semantic_index = self._bm25_index, self._semantic_index
        if not self.chunks or not query.strip():
            return []

        selector = self._get_selector_vector(filter_languages, filter_paths)

        if mode == SearchMode.BM25:
            return search_bm25(query, bm25_index, self.chunks, top_k, selector=selector)
        if mode == SearchMode.SEMANTIC:
            return search_semantic(query, self.model, semantic_index, self.chunks, top_k, selector=selector)
        if mode == SearchMode.HYBRID:
            return search_hybrid(
                query, self.model, semantic_index, bm25_index, self.chunks, top_k, alpha=alpha, selector=selector
            )
        raise ValueError(f"Unknown search mode: {mode!r}")


def _to_rel(root: Path, p: str | Path) -> str:
    """Normalize a path to a forward-slash root-relative string."""
    pp = Path(p)
    if pp.is_absolute():
        try:
            return pp.resolve().relative_to(root).as_posix()
        except ValueError:
            return pp.resolve().as_posix()
    return Path(p).as_posix()


def _scan_file_state(
    root: Path,
    extensions: frozenset[str] | None,
    ignore: frozenset[str] | None,
    include_text_files: bool,
) -> dict[str, float]:
    """Map root-relative posix path -> mtime for every file currently indexable."""
    ext_set = filter_extensions(extensions, include_text_files=include_text_files)
    state: dict[str, float] = {}
    for fp in walk_files(root, ext_set, ignore):
        rel = fp.relative_to(root).as_posix()
        with contextlib.suppress(OSError):
            state[rel] = fp.stat().st_mtime
    return state
