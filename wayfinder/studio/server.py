"""Wayfinder Studio: a local web UI served by the standard library.

    python -m wayfinder.studio --backend stub|rules|openai|transformers|server [--port 8770]

* Live: enter a URL (or a saved-site folder) and a question, watch the agent's
  trace stream in (Server-Sent Events): pages fetched, links chosen, evidence,
  final answer with its source.
* Run report: open a ``reports/*.json`` file written by ``wayfinder bench``.

The bundled fixture site is served at ``/demo-site/`` so the crawler can be
watched over real HTTP without any internet access.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .. import __version__
from ..schema import Task

STATIC = Path(__file__).resolve().parent / "static"
REPO = Path(__file__).resolve().parents[2]


def _fixtures_dir() -> Path:
    env = os.environ.get("WAYFINDER_FIXTURES")
    return Path(env) if env else REPO / "fixtures"


class Studio:
    backend_name = "rules"
    backend = None
    agent = None
    reports_dir = REPO / "reports"
    lock = threading.Lock()

    @classmethod
    def setup(cls, backend: str, reports_dir: str | None):
        cls.backend_name = backend
        if reports_dir:
            cls.reports_dir = Path(reports_dir)
        if backend != "server":
            from ..agent import Agent
            from ..backends import get_backend

            cls.backend = get_backend(backend)
            cls.agent = Agent(cls.backend)

    @classmethod
    def info(cls) -> dict:
        be = cls.backend
        label = {"stub": "TEST STUB", "rules": "RULES · no model", "server": "RESIDENT SERVER"}.get(
            cls.backend_name, cls.backend_name.upper())
        model = ""
        if be is not None and getattr(be, "model", None):
            model = str(be.model)
        elif be is not None and getattr(be, "model_dir", None):
            model = os.path.basename(str(be.model_dir))
        if cls.backend_name == "server":
            try:
                from .. import client

                p = client.ping(timeout=0.5)
                if p:
                    label = f"SERVER · {p.get('backend')}" + (" · TEST STUB" if p.get("stub") else "")
                else:
                    label = "SERVER · offline"
            except Exception:
                pass
        return {"version": __version__, "backend": cls.backend_name, "label": label, "model": model,
                "stub": bool(getattr(be, "is_stub", False)) or "STUB" in label}


class Handler(BaseHTTPRequestHandler):
    server_version = "WayfinderStudio/" + __version__
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if os.environ.get("WAYFINDER_STUDIO_LOG"):
            super().log_message(fmt, *args)

    # ------------------------------------------------------------------ helpers
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _file(self, root: Path, rel: str):
        try:
            p = (root / urllib.parse.unquote(rel)).resolve()
            if root.resolve() not in p.parents and p != root.resolve():
                return self._send(403, b"forbidden", "text/plain")
            if p.is_dir():
                p = p / "index.html"
            if not p.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            if p.suffix == ".md":
                ctype = "text/markdown"
            if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
                ctype += "; charset=utf-8"
            self._send(200, p.read_bytes(), ctype)
        except OSError:
            self._send(500, b"error", "text/plain")

    # ------------------------------------------------------------------ routes
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        path, qs = u.path, urllib.parse.parse_qs(u.query)
        if path in ("/", "/index.html"):
            return self._file(STATIC, "index.html")
        if path.startswith("/static/"):
            return self._file(STATIC, path[len("/static/"):])
        if path.startswith("/demo-site"):
            rel = path[len("/demo-site"):].lstrip("/") or "index.html"
            return self._file(_fixtures_dir() / "site", rel)
        if path == "/api/info":
            info = Studio.info()
            host = self.headers.get("Host") or f"127.0.0.1:{self.server.server_address[1]}"
            info["demo_url"] = f"http://{host}/demo-site/index.html"
            info["examples"] = self._examples(info["demo_url"])
            return self._json(info)
        if path == "/api/reports":
            return self._json({"reports": self._reports()})
        if path == "/api/report":
            name = (qs.get("name") or [""])[0]
            return self._report(name)
        if path == "/api/run":
            return self._run(qs)
        if path == "/healthz":
            return self._json({"ok": True})
        self._send(404, b"not found", "text/plain")

    def _examples(self, demo_url: str) -> list[dict]:
        try:
            spec = json.loads((_fixtures_dir() / "tasks.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        base = demo_url.rsplit("/", 1)[0] + "/"
        out = []
        for t in spec.get("tasks", []):
            out.append({"id": t["id"], "question": t["question"], "url": base + t.get("start", "index.html"),
                        "kind": t.get("kind", "")})
        return out

    def _reports(self) -> list[dict]:
        out = []
        d = Studio.reports_dir
        if d.is_dir():
            for p in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    s = data.get("summary", {})
                    out.append({"name": p.stem, "backend": data.get("backend"), "created": data.get("created"),
                                "pass": s.get("pass"), "scored": s.get("scored"),
                                "model": (data.get("device") or {}).get("model", "")})
                except (OSError, ValueError):
                    continue
        return out

    def _report(self, name: str):
        safe = "".join(c for c in name if c.isalnum() or c in "-_.")
        p = Studio.reports_dir / f"{safe}.json"
        if not safe or not p.is_file():
            return self._json({"error": "not found"}, 404)
        self._send(200, p.read_bytes(), "application/json; charset=utf-8")

    def _run(self, qs):
        question = (qs.get("question") or [""])[0].strip()
        url = (qs.get("url") or [""])[0].strip() or None
        corpus = (qs.get("corpus") or [""])[0].strip() or None
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        t0 = time.monotonic()
        alive = {"ok": True}

        def send(kind: str, data: dict):
            if not alive["ok"]:
                return
            payload = json.dumps({"type": kind, "t": round(time.monotonic() - t0, 3), **data},
                                 ensure_ascii=False, default=str)
            try:
                self.wfile.write(f"event: trace\ndata: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
            except OSError:
                alive["ok"] = False

        task = Task(id="studio", question=question, url=url, corpus=corpus)
        try:
            if Studio.backend_name == "server":
                from ..client import solve_via_server

                res = solve_via_server(task, 25.0, on_trace=lambda k, d: send(k, d))
                if res is None:
                    send("error", {"message": "resident server unavailable"})
            else:
                Studio.agent.run(task, deadline=time.monotonic() + 25.0, trace=lambda k, **d: send(k, d))
        except Exception as e:
            send("error", {"message": f"{type(e).__name__}: {e}"})
        try:
            self.wfile.write(b"event: done\ndata: {}\n\n")
            self.wfile.flush()
        except OSError:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wayfinder studio")
    ap.add_argument("--backend", default=os.environ.get("WAYFINDER_BACKEND", "rules"),
                    help="stub | rules | openai | transformers | server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("WAYFINDER_STUDIO_PORT", "8770")))
    ap.add_argument("--reports", default=None, help="folder of bench reports (default: ./reports)")
    a = ap.parse_args(argv)
    Studio.setup(a.backend, a.reports or (str(Path.cwd() / "reports") if (Path.cwd() / "reports").is_dir() else None))
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"Wayfinder Studio {__version__} on http://{a.host}:{srv.server_address[1]}  (backend: {a.backend})",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
