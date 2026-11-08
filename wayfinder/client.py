"""Talk to the resident server over a unix socket (TCP on platforms without AF_UNIX).

Protocol: one newline-delimited JSON request per connection; the server answers
with newline-delimited JSON events and closes. Standard library only.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time

from .schema import Result, Task

DEFAULT_TCP = "tcp:127.0.0.1:47821"


def address() -> str:
    a = os.environ.get("WAYFINDER_ADDR")
    if a:
        return a
    s = os.environ.get("WAYFINDER_SOCKET")
    if s:
        return "unix:" + s
    return "unix:/tmp/wayfinder.sock" if hasattr(socket, "AF_UNIX") else DEFAULT_TCP


def status_path(addr: str | None = None) -> str:
    env = os.environ.get("WAYFINDER_STATUS")
    if env:
        return env
    addr = addr or address()
    if addr.startswith("unix:"):
        return addr[5:] + ".status.json"
    return os.path.join(tempfile.gettempdir(), "wayfinder.status.json")


def read_status(addr: str | None = None) -> dict | None:
    try:
        with open(status_path(addr), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def connect(addr: str | None = None, timeout: float = 1.0) -> socket.socket:
    addr = addr or address()
    if addr.startswith("unix:"):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(addr[5:])
        return s
    host, port = addr[4:].rsplit(":", 1) if addr.startswith("tcp:") else addr.rsplit(":", 1)
    s = socket.create_connection((host, int(port)), timeout=timeout)
    return s


def request(req: dict, timeout: float = 5.0, addr: str | None = None, on_event=None) -> list[dict]:
    """Send one request and collect the event lines until the server closes (or timeout)."""
    t_end = time.monotonic() + timeout
    s = connect(addr, timeout=min(1.0, max(0.1, timeout)))
    events: list[dict] = []
    try:
        s.sendall((json.dumps(req) + "\n").encode("utf-8"))
        buf = b""
        while True:
            left = t_end - time.monotonic()
            if left <= 0:
                raise TimeoutError("server did not finish in time")
            s.settimeout(min(left, 1.0))
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line.decode("utf-8"))
                except ValueError:
                    continue
                events.append(ev)
                if on_event:
                    on_event(ev)
                if ev.get("event") in ("final", "pong", "stats", "error", "indexed", "generated"):
                    return events
    finally:
        try:
            s.close()
        except OSError:
            pass
    return events


def ping(timeout: float = 0.5, addr: str | None = None) -> dict | None:
    try:
        evs = request({"op": "ping"}, timeout=timeout, addr=addr)
        return evs[-1] if evs else None
    except (OSError, TimeoutError, ValueError):
        return None


def wait_ready(max_wait: float, addr: str | None = None, poll: float = 0.25) -> dict | None:
    """Wait for the server while its status says it is starting/loading."""
    t_end = time.monotonic() + max_wait
    while True:
        p = ping(timeout=0.5, addr=addr)
        if p and p.get("ready"):
            return p
        st = read_status(addr)
        starting = st is not None and st.get("state") in ("starting", "loading", "warming")
        if not starting and p is None:
            return None
        if time.monotonic() + poll > t_end:
            return None
        time.sleep(poll)


def solve_via_server(task: Task, budget: float, on_update=None, on_trace=None,
                     addr: str | None = None) -> Result | None:
    """Solve on the resident server. ``None`` means 'server unavailable, use the local fallback'."""
    t0 = time.monotonic()
    # do not burn the whole budget waiting for a server that is still loading weights
    if wait_ready(max_wait=max(0.0, min(budget - 8.0, 12.0)), addr=addr) is None:
        return None
    left = budget - (time.monotonic() - t0)
    if left < 2.0:
        return None
    holder: dict = {}

    def handle(ev):
        kind = ev.get("event")
        if kind == "provisional" and on_update:
            on_update(Result.from_dict(ev.get("result") or {}))
        elif kind == "trace" and on_trace:
            on_trace(ev.get("type"), ev.get("data") or {})
        elif kind == "final":
            holder["final"] = Result.from_dict(ev.get("result") or {})

    try:
        request({"op": "solve", "task": task.to_dict(), "budget": left - 0.6, "trace": on_trace is not None},
                timeout=left - 0.3, addr=addr, on_event=handle)
    except (OSError, TimeoutError, ValueError):
        pass
    return holder.get("final")
