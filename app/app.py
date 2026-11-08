#!/usr/bin/env python3
"""Wayfinder - container entry point (thin, standard library only).

    python3 /app/app.py --query-id q01 --query "..." [--url https://...] [--corpus /path]
    python3 /app/app.py --input tasks.json
    python3 /app/app.py --index /app/corpus

Writes /app/output/<id>_output.json = {"answer", "citations", "source_url", "confidence"}.
The model lives in the resident server (/app/server.py); this process never loads it.
Always exits 0.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if os.path.isdir(os.path.join(_p, "wayfinder")) and _p not in sys.path:
        sys.path.insert(0, _p)


def _run() -> None:
    argv = sys.argv[1:]
    try:
        from wayfinder.entry import main
    except Exception as e:  # broken install: still honour the output contract
        print(f"[wayfinder] import failed: {e}", file=sys.stderr)
        try:
            from wayfinder.entry import emergency
            emergency(argv)
        except Exception:
            import json
            qid, out = "query", "/app/output"
            for i, t in enumerate(argv):
                if t.lstrip("-").replace("_", "-") in ("query-id", "id", "qid", "task-id") and i + 1 < len(argv):
                    qid = argv[i + 1]
            try:
                os.makedirs(out, exist_ok=True)
                with open(os.path.join(out, f"{qid}_output.json"), "w", encoding="utf-8") as fh:
                    json.dump({"answer": "", "citations": [], "source_url": "", "confidence": 0.0}, fh)
            except Exception:
                pass
        return
    try:
        main(argv)
    except BaseException as e:  # noqa: BLE001
        print(f"[wayfinder] fatal: {e}", file=sys.stderr)


if __name__ == "__main__":
    _run()
    sys.stdout.flush()
    sys.stderr.flush()
    # worker/fetch threads may still be blocked on sockets; the files are final, so leave now
    os._exit(0)
