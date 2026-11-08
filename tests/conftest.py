import functools
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SITE = ROOT / "fixtures" / "site"
TASKS = ROOT / "fixtures" / "tasks.json"
APP = ROOT / "app" / "app.py"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Never talk to a real resident server and keep indexes out of the repo."""
    monkeypatch.setenv("WAYFINDER_ADDR", "tcp:127.0.0.1:9")  # discard port: nothing listens
    monkeypatch.setenv("WAYFINDER_STATUS", str(tmp_path / "no-server.status.json"))
    monkeypatch.setenv("WAYFINDER_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.delenv("WAYFINDER_BACKEND", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(directory: str | os.PathLike, handler=None):
    h = handler or functools.partial(_Quiet, directory=str(directory))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/"


@pytest.fixture(scope="session")
def site_http():
    srv, base = serve(SITE)
    yield base
    srv.shutdown()
