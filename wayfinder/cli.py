"""``wayfinder`` command line.

    wayfinder ask "How much is the Team plan?" --url https://example.com [--backend rules]
    wayfinder ask "..." --corpus ./saved-site --json
    wayfinder index ./saved-site
    wayfinder bench --tasks fixtures/tasks.json --backend rules
    wayfinder serve --backend transformers      # resident server
    wayfinder studio --backend stub             # web UI on http://127.0.0.1:8770
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def _ask(argv) -> int:
    ap = argparse.ArgumentParser(prog="wayfinder ask")
    ap.add_argument("question")
    ap.add_argument("--url", default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--backend", default="rules", help="rules | stub | openai | transformers | server")
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--json", action="store_true", help="print the output JSON only")
    ap.add_argument("--trace", action="store_true", help="print each agent step to stderr")
    a = ap.parse_args(argv)
    from .schema import Task

    task = Task(id="query", question=a.question, url=a.url, corpus=a.corpus)

    def trace(kind, **d):
        if a.trace:
            brief = {k: v for k, v in d.items() if k not in ("passages", "result", "candidates")}
            print(f"  · {kind:11} {json.dumps(brief, ensure_ascii=False)[:200]}", file=sys.stderr)

    t0 = time.monotonic()
    if a.backend == "server":
        from .client import solve_via_server

        res = solve_via_server(task, a.budget, on_trace=lambda k, d: trace(k, **d))
        if res is None:
            print("resident server unavailable", file=sys.stderr)
            return 1
    else:
        from .agent import Agent
        from .backends import get_backend

        res = Agent(get_backend(a.backend)).run(task, deadline=time.monotonic() + a.budget, trace=trace)
    out = res.to_output()
    if a.json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(f"answer     : {out['answer'] or '(no answer found)'}")
        print(f"citations  : {', '.join(out['citations']) or '-'}")
        print(f"source_url : {out['source_url'] or '-'}")
        print(f"confidence : {out['confidence']}   ({res.method}, {time.monotonic() - t0:.2f}s)")
    return 0


def _index(argv) -> int:
    ap = argparse.ArgumentParser(prog="wayfinder index")
    ap.add_argument("folder")
    a = ap.parse_args(argv)
    from .corpus import index_path, load_pages
    from pathlib import Path

    t = time.monotonic()
    pages = load_pages(a.folder)
    print(f"{len(pages)} pages indexed in {time.monotonic() - t:.2f}s -> {index_path(Path(a.folder))}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "ask":
        return _ask(rest)
    if cmd == "index":
        return _index(rest)
    if cmd == "bench":
        from .bench import main as bench_main

        return bench_main(rest)
    if cmd in ("serve", "server"):
        from .server import main as server_main

        return server_main(rest)
    if cmd == "supervise":
        from .supervisor import main as sup_main

        return sup_main(rest)
    if cmd == "studio":
        from .studio.server import main as studio_main

        return studio_main(rest)
    if cmd == "version":
        from . import __version__

        print(__version__)
        return 0
    # anything else: behave like app.py (contract flags)
    from .entry import main as entry_main

    return entry_main(argv)


if __name__ == "__main__":
    sys.exit(main())
