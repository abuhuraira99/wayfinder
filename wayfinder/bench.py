"""Bench: run a JSON task list end to end and write ``reports/<name>.json``.

    python -m wayfinder bench --tasks fixtures/tasks.json --backend rules
    python -m wayfinder bench --tasks fixtures/tasks.json --backend transformers --transport http
    python -m wayfinder bench --tasks fixtures/tasks.json --via app      # real app.py subprocesses

Task files: either the fixture format ({"site": "site", "tasks": [{"id", "question", "start",
"expected", "citations"}]}) or a plain list of {"id", "question", "url"|"corpus", "expected",
"citations"}. ``expected`` may be a string, a list of accepted strings, or null (not scored).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import functools
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .schema import Result, Task
from .textutil import grade_normalize, loose_normalize


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def serve_dir(root: str) -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_QuietHandler, directory=root))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/"


def load_spec(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        data = {"name": Path(path).stem, "tasks": data}
    base = Path(path).resolve().parent
    if data.get("site"):
        data["site_dir"] = str((base / data["site"]).resolve())
    for t in data.get("tasks", []):
        if t.get("corpus") and not os.path.isabs(t["corpus"]) and not str(t["corpus"]).startswith("http"):
            t["corpus"] = str((base / t["corpus"]).resolve())
    return data


def _norm_cite(c: str, base: str | None) -> str:
    c = str(c)
    if base and c.startswith(base):
        c = c[len(base):]
    c = c.split("#", 1)[0]
    while c.startswith(("./", "/")):
        c = c[1:] if c.startswith("/") else c[2:]
    return c


def score_item(expected, got: str, exp_cites, got_cites, base: str | None) -> dict:
    if expected is None:
        return {"scored": False, "answer_pass": None, "citation_pass": None, "pass": None}
    accepted = expected if isinstance(expected, list) else [expected]
    a_pass = any(grade_normalize(e) == grade_normalize(got) for e in accepted)
    loose = any(loose_normalize(e) == loose_normalize(got) for e in accepted)
    if exp_cites is None:
        c_pass = None
    else:
        c_pass = sorted({_norm_cite(c, base) for c in got_cites}) == sorted({_norm_cite(c, None) for c in exp_cites})
        if not any(accepted) or accepted == [""]:
            c_pass = not got_cites
    return {"scored": True, "answer_pass": a_pass, "answer_pass_loose": loose, "citation_pass": c_pass,
            "pass": bool(a_pass and (c_pass is not False))}


def _device_info(backend_obj, via: str) -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    if via == "server":
        try:
            from . import client

            evs = client.request({"op": "stats"}, timeout=5)
            info.update((evs[-1] if evs else {}).get("device", {}))
        except Exception as e:
            info["server_error"] = str(e)
    elif backend_obj is not None:
        try:
            info.update(backend_obj.device_info())
        except Exception as e:
            info["device_error"] = str(e)
    return info


def run_bench(tasks_path: str, backend: str = "rules", transport: str = "offline", via: str = "inproc",
              name: str | None = None, out_dir: str = "reports", budget: float = 25.0, limit: int | None = None,
              only: list[str] | None = None, quiet: bool = False) -> dict:
    spec = load_spec(tasks_path)
    items = spec.get("tasks", [])
    if only:
        items = [t for t in items if t.get("id") in only]
    if limit:
        items = items[:limit]
    site_dir = spec.get("site_dir")
    srv, base = (None, None)
    if transport == "http" and site_dir:
        srv, base = serve_dir(site_dir)

    agent = backend_obj = None
    if via == "inproc":
        from .agent import Agent
        from .backends import get_backend

        backend_obj = get_backend(backend)
        agent = Agent(backend_obj)

    rows = []
    out_tmp = tempfile.mkdtemp(prefix="wayfinder-bench-")
    for t in items:
        qid = t.get("id") or f"q{len(rows) + 1:02d}"
        url, corpus = t.get("url"), t.get("corpus")
        if site_dir:
            if transport == "http":
                url, corpus = base + (t.get("start") or "index.html"), None
            else:
                url, corpus = None, site_dir
        task = Task(id=qid, question=t["question"], url=url, corpus=corpus)
        trace_counts = {"fetch": 0, "hops": 0, "follow": 0}

        def trace(kind, **data):
            if kind == "fetch":
                trace_counts["fetch"] += 1
            elif kind == "decision":
                trace_counts["hops"] += 1
            elif kind == "follow":
                trace_counts["follow"] += 1

        t0 = time.monotonic()
        if via == "inproc":
            res = agent.run(task, deadline=time.monotonic() + budget, trace=trace)
            out = res.to_output()
            method = res.method
        elif via == "server":
            from .client import solve_via_server

            res = solve_via_server(task, budget, on_trace=lambda k, d: trace(k, **d)) or Result(method="server-unavailable")
            out, method = res.to_output(), res.method
        else:  # app: a real subprocess per question, exactly like the harness
            app = str(Path(__file__).resolve().parent.parent / "app" / "app.py")
            if not os.path.exists(app):
                app = "/app/app.py"
            cmd = [sys.executable, app, "--query-id", qid, "--query", t["question"], "--output-dir", out_tmp]
            if url:
                cmd += ["--url", url]
            if corpus:
                cmd += ["--corpus", corpus]
            if backend and backend not in ("auto", "server"):
                cmd += ["--backend", backend]
            try:
                subprocess.run(cmd, capture_output=True, timeout=budget + 10)
            except subprocess.TimeoutExpired:
                pass
            try:
                with open(os.path.join(out_tmp, f"{qid}_output.json"), encoding="utf-8") as fh:
                    out = json.load(fh)
            except (OSError, ValueError):
                out = {"answer": "", "citations": [], "malformed": True}
            method = "app.py"
        latency = time.monotonic() - t0
        sc = score_item(t.get("expected"), out.get("answer", ""), t.get("citations"), out.get("citations", []), base)
        row = {"id": qid, "question": t["question"], "kind": t.get("kind", ""), "expected": t.get("expected"),
               "got": out.get("answer", ""), "expected_citations": t.get("citations"),
               "citations": [_norm_cite(c, base) for c in out.get("citations", [])],
               "source_url": out.get("source_url", ""), "confidence": out.get("confidence", 0.0),
               "latency_s": round(latency, 3), "method": method, "pages_fetched": trace_counts["fetch"],
               "model_steps": trace_counts["hops"], "follows": trace_counts["follow"], **sc}
        rows.append(row)
        if not quiet:
            mark = "PASS" if row["pass"] else ("----" if row["pass"] is None else "FAIL")
            print(f"{mark} {qid:5} {latency:6.2f}s  got={row['got']!r:40.40} cites={row['citations']}", flush=True)
    if srv:
        srv.shutdown()

    scored = [r for r in rows if r["scored"]]
    lat = [r["latency_s"] for r in rows] or [0.0]
    summary = {
        "n": len(rows), "scored": len(scored),
        "pass": sum(1 for r in scored if r["pass"]),
        "answer_pass": sum(1 for r in scored if r["answer_pass"]),
        "citation_pass": sum(1 for r in scored if r["citation_pass"]),
        "accuracy": round(sum(1 for r in scored if r["pass"]) / len(scored), 4) if scored else None,
        "mean_latency_s": round(statistics.fmean(lat), 3),
        "p50_latency_s": round(statistics.median(lat), 3),
        "p95_latency_s": round(sorted(lat)[max(0, int(round(0.95 * len(lat))) - 1)], 3),
        "max_latency_s": round(max(lat), 3),
    }
    name = name or f"{Path(tasks_path).stem}-{backend}-{transport}"
    report = {
        "name": name, "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "tasks_file": os.path.basename(tasks_path), "backend": backend,
        "backend_is_stub": backend == "stub", "transport": transport, "via": via, "budget_s": budget,
        "device": _device_info(backend_obj, via), "summary": summary, "items": rows,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    if not quiet:
        print(f"\n{summary['pass']}/{summary['scored']} pass (answer {summary['answer_pass']}, "
              f"citations {summary['citation_pass']}); mean {summary['mean_latency_s']}s, "
              f"max {summary['max_latency_s']}s -> {path}")
    report["path"] = path
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wayfinder bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="fixtures/tasks.json")
    ap.add_argument("--backend", default=os.environ.get("WAYFINDER_BACKEND", "rules"),
                    help="rules | stub | openai | transformers (inproc/app) ; ignored with --via server")
    ap.add_argument("--transport", choices=["offline", "http"], default="offline",
                    help="fixture site from disk, or served over local HTTP to exercise the crawler")
    ap.add_argument("--via", choices=["inproc", "server", "app"], default="inproc")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="task ids to run")
    a = ap.parse_args(argv)
    run_bench(a.tasks, backend=a.backend, transport=a.transport, via=a.via, name=a.name, out_dir=a.out,
              budget=a.budget, limit=a.limit, only=a.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
