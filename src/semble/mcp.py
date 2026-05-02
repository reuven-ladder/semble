from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from semble.index import SembleIndex
from semble.index.dense import _DEFAULT_MODEL_NAME, load_model
from semble.persistence import CACHE_DIRNAME, cache_dir_for, load_index, meta_mtime
from semble.types import Encoder
from semble.utils import _format_results, _is_git_url, _resolve_chunk

_REPO_DESCRIPTION = (
    "Git URL (e.g. https://github.com/org/repo) or local path to index and search. "
    "Required when no default index was configured at startup. "
    "The index is cached after the first call, so repeat queries are fast."
)


def create_server(cache: _IndexCache, default_source: str | None = None) -> FastMCP:
    """Build and return a configured FastMCP server backed by the given cache."""
    server = FastMCP(
        "semble",
        instructions=(
            "Instant code search for any local or GitHub repository. "
            "Call `search` to find relevant code; call `find_related` on a result to discover similar code elsewhere. "
            "For questions about a library (e.g. a PyPI/npm package), resolve the GitHub URL from your training "
            "knowledge and pass it as `repo`. "
            "Prefer these tools over Grep, Glob, or Read for any question about how code works."
        ),
    )

    @server.tool()
    async def search(
        query: Annotated[str, Field(description="Natural language or code query.")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        mode: Annotated[
            Literal["hybrid", "semantic", "bm25"],
            Field(description="Search mode. 'hybrid' is best for most queries."),
        ] = "hybrid",
        top_k: Annotated[int, Field(description="Number of results to return.", ge=1)] = 5,
    ) -> str:
        """Search a codebase with a natural-language or code query.

        Pass a git URL or local path as `repo` to index it on demand; indexes are cached for the session.
        Use this to find where something is implemented, understand a library, or locate related code.
        """
        source = repo or default_source
        if not source:
            return (
                "No repo specified and no default index. "
                "Pass a git URL (https://github.com/...) or local path as `repo`."
            )
        try:
            index = await cache.get(source)
        except Exception as exc:
            return f"Failed to index {source!r}: {exc}"
        results = index.search(query, top_k=top_k, mode=mode)
        if not results:
            return "No results found."
        return _format_results(f"Search results for: {query!r} (mode={mode})", results)

    @server.tool()
    async def find_related(
        file_path: Annotated[
            str,
            Field(description="Path to the file as stored in the index (use file_path from a search result)."),
        ],
        line: Annotated[int, Field(description="Line number (1-indexed).")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        top_k: Annotated[int, Field(description="Number of similar chunks to return.", ge=1)] = 5,
    ) -> str:
        """Find code chunks semantically similar to a specific location in a file.

        Use after `search` to explore related implementations or callers.
        Pass file_path and line from a prior search result.
        """
        source = repo or default_source
        if not source:
            return (
                "No repo specified and no default index. "
                "Pass a git URL (https://github.com/...) or local path as `repo`."
            )
        try:
            index = await cache.get(source)
        except Exception as exc:
            return f"Failed to index {source!r}: {exc}"
        chunk = _resolve_chunk(index.chunks, file_path, line)
        if chunk is None:
            return (
                f"No chunk found at {file_path}:{line}. "
                "Make sure the file is indexed and the line number is within a known chunk."
            )
        results = index.find_related(chunk, top_k=top_k)
        if not results:
            return f"No related chunks found for {file_path}:{line}."
        return _format_results(f"Chunks related to {file_path}:{line}", results)

    return server


async def serve(path: str | None = None, ref: str | None = None, use_cache: bool = True) -> None:
    """Start an MCP stdio server, optionally pre-indexing a default source.

    :param path: Default source to pre-index (local dir or git URL).
    :param ref: Branch/tag for git URLs.
    :param use_cache: If True (default), local paths persist and reload from
        ``<path>/.semble``; the cache reloads automatically when ``meta.json``
        mtime advances (e.g. after ``semble reindex`` or ``semble watch``).
    """
    model = await asyncio.to_thread(load_model)
    cache = _IndexCache(model=model, use_cache=use_cache)
    if path:
        await cache.get(path, ref=ref)

    server = create_server(cache, default_source=path)
    await server.run_stdio_async()


class _IndexCache:
    """Cache of indexed repos and local paths for the lifetime of the MCP server process.

    For local paths, the index is also persisted to ``<path>/.semble`` so that
    external refresh (git hook, file watcher) can update it. ``get()`` re-checks
    ``meta.json`` mtime and transparently reloads when it advances.
    """

    def __init__(self, model: Encoder, use_cache: bool = True) -> None:
        """Initialise an empty cache with a shared embedding model."""
        self._model = model
        self._use_cache = use_cache
        self._tasks: dict[str, asyncio.Task[SembleIndex]] = {}
        self._meta_mtimes: dict[str, float] = {}

    async def get(self, source: str, ref: str | None = None) -> SembleIndex:
        """Return an index for the requested source, building and caching it on first access."""
        is_git = _is_git_url(source)
        cache_key = (f"{source}@{ref}" if ref else source) if is_git else str(Path(source).resolve())

        if not is_git and self._use_cache:
            cdir = cache_dir_for(Path(cache_key))
            current = meta_mtime(cdir)
            cached = self._meta_mtimes.get(cache_key)
            if current is not None and cached is not None and current > cached:
                # On-disk index has been refreshed by an external process; drop our cached task.
                self._tasks.pop(cache_key, None)
                self._meta_mtimes.pop(cache_key, None)

        if cache_key not in self._tasks:
            if is_git:
                self._tasks[cache_key] = asyncio.create_task(
                    asyncio.to_thread(SembleIndex.from_git, source, ref=ref, model=self._model)
                )
            else:
                self._tasks[cache_key] = asyncio.create_task(
                    asyncio.to_thread(self._load_or_build, cache_key)
                )
        task = self._tasks[cache_key]
        try:
            index = await asyncio.shield(task)
        except asyncio.CancelledError:  # pragma: no cover
            if task.done():
                self._tasks.pop(cache_key, None)
            raise
        except Exception:
            # Build failed: evict so the next caller can retry.
            self._tasks.pop(cache_key, None)
            raise

        if not is_git and self._use_cache:
            cdir = cache_dir_for(Path(cache_key))
            mt = meta_mtime(cdir)
            if mt is not None:
                self._meta_mtimes[cache_key] = mt
        return index

    def _load_or_build(self, path: str) -> SembleIndex:
        """Load persisted index from ``path/.semble`` if present; otherwise build and persist."""
        cdir = cache_dir_for(Path(path))
        if self._use_cache:
            existing = load_index(cdir, model=self._model)
            if existing is not None:
                return existing
        return SembleIndex.from_path(
            path,
            model=self._model,
            cache_dir=cdir if self._use_cache else None,
            model_name=_DEFAULT_MODEL_NAME,
        )
