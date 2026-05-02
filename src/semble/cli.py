import argparse
import asyncio
import sys
from importlib.resources import files
from pathlib import Path

from semble.index import SembleIndex
from semble.utils import _format_results, _is_git_url, _resolve_chunk

_CLAUDE_FILE_PATH = Path(".claude") / "agents" / "semble-search.md"
_CLI_DISPATCH_ARGS = frozenset(
    {
        "search",
        "find-related",
        "init",
        "reindex",
        "watch",
        "install-hooks",
        "uninstall-hooks",
        "-h",
        "--help",
    }
)


def main() -> None:
    """Entry point for the semble command-line tool."""
    if len(sys.argv) > 1 and sys.argv[1] in _CLI_DISPATCH_ARGS:
        _cli_main()
    else:
        _mcp_main()


def _mcp_main() -> None:
    parser = argparse.ArgumentParser(
        prog="semble",
        description="Instant local code search for agents.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Local directory or git URL to pre-index at startup (optional).",
    )
    parser.add_argument("--ref", default=None, help="Branch or tag to check out (git URLs only).")
    args = parser.parse_args()
    from semble.mcp import serve

    asyncio.run(serve(args.path, ref=args.ref))


def _run_init(*, force: bool = False) -> None:
    """Write the Claude Code sub-agent file into the current project."""
    dest = _CLAUDE_FILE_PATH
    if dest.exists() and not force:
        print(f"{dest} already exists. Run with --force to overwrite.", file=sys.stderr)
        sys.exit(1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = files("semble").joinpath("agents/semble-search.md").read_text(encoding="utf-8")
    dest.write_text(content, encoding="utf-8")
    print(f"Created {dest}")


def _cli_main() -> None:
    parser = argparse.ArgumentParser(prog="semble")
    sub = parser.add_subparsers(dest="command")

    search_p = sub.add_parser("search", help="Search a codebase.")
    search_p.add_argument("query", help="Natural language or code query.")
    search_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    search_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    search_p.add_argument(
        "-m", "--mode", default="hybrid", choices=["hybrid", "semantic", "bm25"], help="Search mode (default: hybrid)."
    )

    related_p = sub.add_parser("find-related", help="Find code similar to a specific location.")
    related_p.add_argument("file_path", help="File path as shown in search results.")
    related_p.add_argument("line", type=int, help="Line number (1-indexed).")
    related_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    related_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")

    init_p = sub.add_parser("init", help="Write .claude/agents/semble-search.md for Claude Code sub-agent support.")
    init_p.add_argument("--force", action="store_true", help="Overwrite if the file already exists.")

    reindex_p = sub.add_parser("reindex", help="Re-index a local path; persists to <path>/.semble.")
    reindex_p.add_argument("path", nargs="?", default=".", help="Local path to (re-)index (default: current directory).")
    reindex_p.add_argument(
        "--full",
        action="store_true",
        help="Force a full rebuild even if a cached index exists.",
    )

    watch_p = sub.add_parser("watch", help="Watch a local path and re-index on file changes.")
    watch_p.add_argument("path", nargs="?", default=".", help="Local path to watch (default: current directory).")
    watch_p.add_argument(
        "--debounce",
        type=float,
        default=0.5,
        help="Seconds to coalesce events before refresh (default: 0.5).",
    )

    hooks_p = sub.add_parser("install-hooks", help="Install a post-commit hook that runs `semble reindex`.")
    hooks_p.add_argument("path", nargs="?", default=".", help="Repository root (default: current directory).")
    hooks_p.add_argument("--force", action="store_true", help="Overwrite a non-semble hook if present.")

    unhooks_p = sub.add_parser("uninstall-hooks", help="Remove a previously installed semble post-commit hook.")
    unhooks_p.add_argument("path", nargs="?", default=".", help="Repository root (default: current directory).")

    args = parser.parse_args()

    if args.command == "init":
        _run_init(force=args.force)
        return

    if args.command == "install-hooks":
        from semble.hooks import install_hook

        try:
            written = install_hook(args.path, force=args.force)
        except (FileExistsError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(f"Installed semble post-commit hook at {written}")
        return

    if args.command == "uninstall-hooks":
        from semble.hooks import uninstall_hook

        removed = uninstall_hook(args.path)
        print("Removed semble post-commit hook." if removed else "No semble-managed hook found.")
        return

    if args.command == "reindex":
        from semble.index.dense import _DEFAULT_MODEL_NAME, load_model
        from semble.persistence import cache_dir_for, load_index

        root = Path(args.path).resolve()
        cdir = cache_dir_for(root)
        model = load_model()
        if args.full:
            SembleIndex.from_path(root, model=model, cache_dir=cdir, model_name=_DEFAULT_MODEL_NAME)
            print(f"Full reindex of {root} → {cdir}")
        else:
            existing = load_index(cdir, model=model)
            if existing is None:
                SembleIndex.from_path(root, model=model, cache_dir=cdir, model_name=_DEFAULT_MODEL_NAME)
                print(f"Built initial index for {root} → {cdir}")
            else:
                existing.refresh()  # full refresh; incremental is exposed only programmatically.
                print(f"Refreshed index at {cdir}")
        return

    if args.command == "watch":
        from semble.index.dense import _DEFAULT_MODEL_NAME, load_model
        from semble.persistence import cache_dir_for, load_index
        from semble.watch import watch as run_watch

        root = Path(args.path).resolve()
        cdir = cache_dir_for(root)
        model = load_model()
        index = load_index(cdir, model=model)
        if index is None:
            index = SembleIndex.from_path(root, model=model, cache_dir=cdir, model_name=_DEFAULT_MODEL_NAME)
            print(f"Built initial index for {root}")
        run_watch(index, debounce_seconds=args.debounce)
        return

    index = SembleIndex.from_git(args.path) if _is_git_url(args.path) else SembleIndex.from_path(args.path)

    if args.command == "search":
        results = index.search(args.query, top_k=args.top_k, mode=args.mode)
        if not results:
            print("No results found.")
        else:
            print(_format_results(f"Search results for: {args.query!r} (mode={args.mode})", results))

    elif args.command == "find-related":
        chunk = _resolve_chunk(index.chunks, args.file_path, args.line)
        if chunk is None:
            print(f"No chunk found at {args.file_path}:{args.line}.", file=sys.stderr)
            sys.exit(1)
        results = index.find_related(chunk, top_k=args.top_k)
        if not results:
            print(f"No related chunks found for {args.file_path}:{args.line}.")
        else:
            print(_format_results(f"Chunks related to {args.file_path}:{args.line}", results))
