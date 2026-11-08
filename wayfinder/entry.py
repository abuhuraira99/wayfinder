"""What ``/app/app.py`` runs. Standard library only; never raises; always exits 0.

Per question:
1. write a valid empty placeholder ``<id>_output.json`` immediately;
2. solve on the resident server (model) - or locally with rules when it is not up;
3. persist every provisional answer atomically, so a hard stop still leaves the best
   answer known so far;
4. a watchdog finalises the file at the budget (default 25 s) whatever is still running.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

from .contract import (EMPTY_OUTPUT, Args, load_task_file, output_path, parse_args, pick, safe_id,
                       write_json_atomic, ID_KEYS, QUESTION_KEYS, URL_KEYS, CORPUS_KEYS)
from .schema import Result, Task

DEFAULT_BUDGET = float(os.environ.get("WAYFINDER_BUDGET", "25.0"))


def log(*a):
    print("[wayfinder]", *a, file=sys.stderr, flush=True)


def _tasks_from_args(a: Args) -> list[Task]:
    if a.input:
        out = []
        for d in load_task_file(a.input):
            t = Task.from_dict(d)
            t.url = t.url or a.url
            t.corpus = t.corpus or a.corpus
            out.append(t)
        return out
    return [Task(id=safe_id(a.qid), question=a.question or "", url=a.url, corpus=a.corpus)]


def solve(task: Task, budget: float, backend: str | None, on_update) -> Result:
    """Server first (unless a local backend was requested), then the local rules path."""
    t0 = time.monotonic()
    name = (backend or os.environ.get("WAYFINDER_CLIENT_BACKEND") or "auto").lower()
    if name in ("auto", "server"):
        from .client import solve_via_server

        try:
            res = solve_via_server(task, budget, on_update=on_update)
        except Exception as e:
            log("server path failed:", e)
            res = None
        if res is not None:
            return res
        if name == "server":
            return Result(method="server-unavailable")
        log("resident server unavailable; using the local rules path")
        name = "rules"
    from .agent import Agent
    from .backends import get_backend

    left = budget - (time.monotonic() - t0)
    return Agent(get_backend(name)).run(task, deadline=time.monotonic() + max(0.5, left), on_update=on_update)


def run_one(task: Task, out_dir: str, budget: float, backend: str | None) -> dict:
    path = output_path(out_dir, task.id)
    t_end = time.monotonic() + budget
    lock = threading.Lock()
    box = {"best": Result(), "final": None, "written": None}

    def persist(r: Result, final: bool = False):
        with lock:
            if box["written"] == "final":
                return
            obj = r.to_output()
            write_json_atomic(path, obj)
            box["written"] = "final" if final else "provisional"

    def on_update(r: Result):
        if r is not None and r.answer and time.monotonic() < t_end:
            box["best"] = r
            persist(r)

    def worker():
        try:
            box["final"] = solve(task, budget - 0.4, backend, on_update)
        except BaseException as e:  # noqa: BLE001 - the file must still be finalised
            log("solve crashed:", type(e).__name__, e)

    th = threading.Thread(target=worker, name=f"solve-{task.id}", daemon=True)
    th.start()
    th.join(max(0.0, t_end - time.monotonic()))
    res = box["final"] if box["final"] is not None else box["best"]
    if box["final"] is None:
        log(f"{task.id}: budget reached; finalising with the best answer so far")
    persist(res, final=True)
    return res.to_output()


def do_index(a: Args) -> int:
    """``--index <folder>``: parse and persist the saved site, then make the server preload it."""
    from .corpus import load_pages

    t = time.monotonic()
    pages = load_pages(a.index)
    log(f"indexed {len(pages)} pages from {a.index} in {time.monotonic() - t:.1f}s")
    try:
        from . import client

        wait_s = float(os.environ.get("WAYFINDER_INDEX_WAIT", "480"))
        # give the supervisor a moment to publish its status, then wait for the model
        t_status = time.monotonic() + 10.0
        while client.read_status() is None and client.ping() is None and time.monotonic() < t_status:
            time.sleep(0.25)
        p = client.wait_ready(max_wait=wait_s)
        if p:
            evs = client.request({"op": "index", "root": os.path.abspath(a.index)}, timeout=120)
            log("server preloaded:", evs[-1] if evs else None)
        else:
            log("no resident server reachable; queries will use the local rules path")
    except Exception as e:
        log("server preload skipped:", e)
    print(json.dumps({"indexed": len(pages), "root": a.index}))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        a = parse_args(argv)
    except Exception as e:  # pragma: no cover - parse_args is total, but never trust that
        log("argument parsing failed:", e)
        a = Args()
    if a.index and not a.question and not a.input:
        try:
            return do_index(a)
        except Exception as e:
            log("index failed:", e)
            return 0
    budget = a.budget if a.budget and a.budget > 0 else DEFAULT_BUDGET
    try:
        tasks = _tasks_from_args(a)
    except Exception as e:
        log("could not build tasks:", e)
        tasks = [Task(id=safe_id(a.qid))]
    if a.index and a.corpus is None:
        for t in tasks:
            t.corpus = t.corpus or a.index
    for t in tasks:
        t.id = safe_id(t.id)
        write_json_atomic(output_path(a.output_dir, t.id), dict(EMPTY_OUTPUT))
    for t in tasks:
        try:
            out = run_one(t, a.output_dir, budget, a.backend)
        except Exception as e:
            log(t.id, "failed:", e)
            out = dict(EMPTY_OUTPUT)
            write_json_atomic(output_path(a.output_dir, t.id), out)
        print(json.dumps(dict(out, id=t.id), ensure_ascii=False), flush=True)
    return 0


def emergency(argv: list[str]) -> None:
    """Last resort when even importing failed: write an empty answer for the id in argv."""
    qid, out_dir = "query", None
    for i, tok in enumerate(argv):
        k = tok.split("=", 1)[0].lstrip("-").lower().replace("_", "-")
        v = tok.split("=", 1)[1] if "=" in tok else (argv[i + 1] if i + 1 < len(argv) else "")
        if k in [x.replace("_", "-") for x in ID_KEYS]:
            qid = v or qid
        if k in ("output-dir", "outdir", "out-dir", "output"):
            out_dir = v
    try:
        write_json_atomic(output_path(out_dir or ("/app/output" if os.path.isdir("/app") else "output"), qid),
                          dict(EMPTY_OUTPUT))
    except Exception:
        pass


__all__ = ["main", "emergency", "run_one", "solve", "pick", "QUESTION_KEYS", "URL_KEYS", "CORPUS_KEYS"]
