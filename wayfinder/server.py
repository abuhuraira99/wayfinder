"""Resident server: holds the model warm and runs the agent for every ``app.py`` call.

    python -m wayfinder.server [--backend transformers|openai|rules|stub]

Listens on ``unix:/tmp/wayfinder.sock`` (``WAYFINDER_ADDR`` / ``WAYFINDER_SOCKET``
override; TCP where AF_UNIX is missing) and writes its state to a status file
next to the socket so clients know whether to wait.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import sys
import threading
import time
import traceback

from . import client
from .agent import Agent
from .backends import RulesBackend, get_backend
from .schema import Result, Task


def log(*a):
    print("[wayfinder-server]", *a, file=sys.stderr, flush=True)


class State:
    backend = None
    agent: Agent | None = None
    ready = False
    state = "starting"
    error = ""
    started = time.time()
    requests = 0


def write_status(addr: str, **extra):
    data = {"state": State.state, "ready": State.ready, "pid": os.getpid(), "addr": addr,
            "backend": getattr(State.backend, "name", None), "error": State.error,
            "updated": time.time()}
    data.update(extra)
    path = client.status_path(addr)
    try:
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    except OSError as e:
        log("status write failed:", e)


class Handler(socketserver.StreamRequestHandler):
    timeout = 60

    def send(self, obj: dict):
        try:
            self.wfile.write((json.dumps(obj, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
            self.wfile.flush()
        except OSError:
            pass

    def handle(self):
        try:
            line = self.rfile.readline(4_000_000)
            req = json.loads(line.decode("utf-8") or "{}")
        except Exception as e:
            self.send({"event": "error", "error": f"bad request: {e}"})
            return
        op = req.get("op", "ping")
        State.requests += 1
        if op == "ping":
            self.send({"event": "pong", "ready": State.ready, "state": State.state,
                       "backend": getattr(State.backend, "name", None),
                       "stub": bool(getattr(State.backend, "is_stub", False)), "pid": os.getpid()})
        elif op == "stats":
            info = {}
            try:
                info = State.backend.device_info() if State.backend else {}
            except Exception as e:
                info = {"error": str(e)}
            self.send({"event": "stats", "device": info, "requests": State.requests,
                       "uptime_s": round(time.time() - State.started, 1)})
        elif op == "solve":
            self._solve(req)
        elif op == "index":
            self._index(req)
        elif op == "generate":
            try:
                text = State.backend.generate(req.get("messages") or [], max_new_tokens=int(req.get("max_new_tokens", 64)),
                                              timeout=float(req.get("timeout", 10)))
                self.send({"event": "generated", "text": text})
            except Exception as e:
                self.send({"event": "error", "error": str(e)})
        else:
            self.send({"event": "error", "error": f"unknown op {op}"})

    def _solve(self, req):
        if State.agent is None:
            self.send({"event": "error", "error": "not ready"})
            return
        task = Task.from_dict(req.get("task") or {})
        budget = float(req.get("budget") or 24.0)
        deadline = time.monotonic() + max(0.5, budget)
        want_trace = bool(req.get("trace"))

        def on_update(r: Result):
            self.send({"event": "provisional", "result": r.to_dict()})

        def trace(kind, **data):
            if want_trace:
                self.send({"event": "trace", "type": kind, "data": data})

        try:
            res = State.agent.run(task, deadline=deadline, on_update=on_update, trace=trace)
        except Exception as e:  # agent.run already guards; belt and braces
            log("solve failed:", e)
            res = Result(method="error")
        self.send({"event": "final", "result": res.to_dict()})

    def _index(self, req):
        from .corpus import load_pages

        root = req.get("root")
        t = time.monotonic()
        try:
            pages = load_pages(root) if root else []
            # warm the passage index too
            if root and State.agent is not None:
                State.agent.run(Task(question="warm up index", corpus=root), deadline=time.monotonic() + 60)
            self.send({"event": "indexed", "pages": len(pages), "seconds": round(time.monotonic() - t, 2)})
        except Exception as e:
            self.send({"event": "error", "error": str(e)})


class ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer if hasattr(socket, "AF_UNIX")
                          else socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


class ThreadingTCP(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(addr: str):
    if addr.startswith("unix:"):
        path = addr[5:]
        if os.path.exists(path):
            if client.ping(timeout=0.5, addr=addr):
                raise RuntimeError(f"another server is already listening on {path}")
            os.remove(path)
        srv = ThreadingUnixServer(path, Handler)
        try:
            os.chmod(path, 0o666)
        except OSError:
            pass
        return srv
    host, port = addr[4:].rsplit(":", 1)
    return ThreadingTCP((host, int(port)), Handler)


def load_backend(name: str):
    """Load the configured backend; on failure fall back to rules so calls still get answers."""
    try:
        be = get_backend(name)
        return be, ""
    except Exception as e:
        log("backend", name, "failed to load:", e)
        traceback.print_exc()
        return RulesBackend(), f"{type(e).__name__}: {e}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wayfinder-server")
    ap.add_argument("--backend", default=os.environ.get("WAYFINDER_BACKEND", "transformers"))
    ap.add_argument("--addr", default=None)
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args(argv)
    addr = args.addr or client.address()
    try:
        srv = make_server(addr)
    except RuntimeError as e:
        log(e)
        return 0
    State.state = "loading"
    write_status(addr)
    # serve pings while the model loads so clients can see "loading"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    log("listening on", addr, "- loading backend", args.backend)
    t0 = time.monotonic()
    be, err = load_backend(args.backend)
    State.backend, State.error = be, err
    State.agent = Agent(be)
    if not args.no_warmup and be.uses_model:
        State.state = "warming"
        write_status(addr)
        try:
            be.warmup()
        except Exception as e:
            log("warm-up failed:", e)
            State.error = State.error or f"warmup: {e}"
    State.ready = True
    State.state = "ready" if not err else "degraded"
    write_status(addr, load_seconds=round(time.monotonic() - t0, 1))
    log("ready in", round(time.monotonic() - t0, 1), "s; backend =", be.name)
    try:
        while th.is_alive():
            th.join(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        if addr.startswith("unix:"):
            try:
                os.remove(addr[5:])
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
