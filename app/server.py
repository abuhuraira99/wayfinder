#!/usr/bin/env python3
"""Wayfinder - container CMD.

Runs the supervisor, which starts the resident model server
(``python -m wayfinder.server``), restarts it if it crashes and never exits,
so the container stays up for ``docker exec ... python3 /app/app.py``.

    python3 /app/server.py            # supervisor (default)
    python3 /app/server.py --child    # the resident server itself, in the foreground
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if os.path.isdir(os.path.join(_p, "wayfinder")) and _p not in sys.path:
        sys.path.insert(0, _p)

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--child" in args:
        args.remove("--child")
        from wayfinder.server import main as server_main

        sys.exit(server_main(args))
    from wayfinder.supervisor import main as supervisor_main

    rc = supervisor_main(args)
    # the supervisor only returns when asked to stop
    sys.exit(rc)
