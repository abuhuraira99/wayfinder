"""Supervisor: the container's CMD. Starts the resident server, restarts it on crash,
never exits on its own (the container must stay up for ``docker exec``)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def log(*a):
    print("[wayfinder-supervisor]", *a, file=sys.stderr, flush=True)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([p for p in (here, env.get("PYTHONPATH", "")) if p])
    cmd = [sys.executable, "-m", "wayfinder.server", *argv]
    child: subprocess.Popen | None = None
    stopping = False

    def stop(signum, _frame):
        nonlocal stopping
        stopping = True
        if child and child.poll() is None:
            child.terminate()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, stop)
        except (ValueError, OSError):
            pass

    restarts = 0
    while not stopping:
        started = time.monotonic()
        log("starting:", " ".join(cmd))
        try:
            child = subprocess.Popen(cmd, env=env, cwd=here)
            rc = child.wait()
        except OSError as e:
            log("could not start server:", e)
            rc = -1
        if stopping:
            break
        lived = time.monotonic() - started
        restarts = restarts + 1 if lived < 60 else 0
        delay = min(30.0, 1.0 * (2 ** min(restarts, 5)))
        log(f"server exited rc={rc} after {lived:.0f}s; restarting in {delay:.0f}s")
        time.sleep(delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
