"""The container contract: liberal argument parsing, task files, atomic output files.

Standard library only; imported by ``/app/app.py`` on every call.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

QUESTION_KEYS = ("query", "question", "task", "prompt", "q", "instruction", "ask", "text")
ID_KEYS = ("query-id", "query_id", "queryid", "id", "qid", "task-id", "task_id", "taskid", "question-id",
           "question_id", "item-id", "item_id", "name")
URL_KEYS = ("url", "start-url", "start_url", "starturl", "website", "site", "base-url", "base_url", "baseurl",
            "start", "homepage", "link", "page", "target", "target-url", "target_url", "domain")
CORPUS_KEYS = ("corpus", "pages", "sites", "web-root", "web_root", "webroot", "root", "site-dir", "site_dir",
               "saved-site", "saved_site", "html-dir", "html_dir", "folder", "dir", "directory", "docs", "snapshot")
INPUT_KEYS = ("input", "tasks", "batch", "input-file", "input_file", "questions", "queries", "task-file", "task_file")
INDEX_KEYS = ("index", "build-index", "build_index")
OUTPUT_KEYS = ("output-dir", "output_dir", "outdir", "out-dir", "out_dir", "output", "out")
BACKEND_KEYS = ("backend",)
BUDGET_KEYS = ("deadline", "timeout", "budget", "time-limit", "time_limit")
FLAG_ONLY = {"verbose", "v", "debug", "quiet", "trace", "no-server"}


def _norm_key(k: str) -> str:
    return k.strip().lstrip("-").lower().replace("_", "-")


def pick(d: dict, keys) -> object:
    """First non-empty value of ``d`` under any spelling of ``keys``."""
    if not isinstance(d, dict):
        return None
    norm = {_norm_key(k): v for k, v in d.items()}
    for k in keys:
        v = norm.get(_norm_key(k))
        if v not in (None, ""):
            return v
    return None


def default_output_dir() -> str:
    env = os.environ.get("WAYFINDER_OUTPUT_DIR")
    if env:
        return env
    if os.path.isdir("/app"):
        return "/app/output"
    return os.path.join(os.getcwd(), "output")


@dataclass
class Args:
    question: str | None = None
    qid: str | None = None
    url: str | None = None
    corpus: str | None = None
    input: str | None = None
    index: str | None = None
    output_dir: str = field(default_factory=default_output_dir)
    backend: str | None = None
    budget: float | None = None
    positional: list[str] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    unknown: list[str] = field(default_factory=list)


_GROUPS = [
    (INDEX_KEYS, "index"), (INPUT_KEYS, "input"), (ID_KEYS, "qid"), (QUESTION_KEYS, "question"),
    (CORPUS_KEYS, "corpus"), (URL_KEYS, "url"), (OUTPUT_KEYS, "output_dir"), (BACKEND_KEYS, "backend"),
    (BUDGET_KEYS, "budget"),
]
_LOOKUP = {}
for keys, attr in _GROUPS:
    for k in keys:
        _LOOKUP.setdefault(_norm_key(k), attr)


def parse_args(argv: list[str]) -> Args:
    """Parse argv without ever raising. Unknown ``--flags`` (and their value) are ignored."""
    a = Args()
    question_flag_seen = False
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("-") and len(tok) > 1 and not re.fullmatch(r"-\d+(\.\d+)?", tok):
            key, eq, val = tok.partition("=")
            nk = _norm_key(key)
            attr = _LOOKUP.get(nk)
            if attr is None and len(nk) == 1 and nk == "q":
                attr = "question"
            if attr is None:
                if nk in FLAG_ONLY:
                    a.flags.add(nk)
                    i += 1
                    continue
                a.unknown.append(tok)
                # skip this unknown option's value too, if it has one
                if not eq and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    a.unknown.append(argv[i + 1])
                    i += 1
                i += 1
                continue
            if not eq:
                if i + 1 < len(argv) and not (argv[i + 1].startswith("--") and len(argv[i + 1]) > 2):
                    val = argv[i + 1]
                    i += 1
                else:
                    val = ""
            if attr == "question":
                question_flag_seen = True
                a.question = val if a.question is None else a.question
            elif attr == "budget":
                try:
                    a.budget = float(val)
                except ValueError:
                    pass
            elif attr == "output_dir":
                if val:
                    a.output_dir = val
            else:
                if getattr(a, attr) in (None, "") and val != "":
                    setattr(a, attr, val)
            i += 1
            continue
        a.positional.append(tok)
        i += 1
    if a.positional and not question_flag_seen:
        joined = " ".join(a.positional).strip()
        obj = None
        if joined.startswith("{"):
            try:
                obj = json.loads(joined)
            except ValueError:
                obj = None
        if isinstance(obj, dict):
            a.question = a.question or (str(pick(obj, QUESTION_KEYS) or "") or None)
            a.qid = a.qid or (str(pick(obj, ID_KEYS)) if pick(obj, ID_KEYS) is not None else None)
            a.url = a.url or pick(obj, URL_KEYS)
            a.corpus = a.corpus or pick(obj, CORPUS_KEYS)
        elif len(a.positional) == 1 and re.search(r"\.(jsonl?|ndjson)$", joined, re.I) and os.path.isfile(joined) \
                and not a.input:
            a.input = joined
        else:
            a.question = joined
    # --input that is not a file is most likely the question itself
    if a.input and not os.path.isfile(a.input) and not re.search(r"\.(jsonl?|ndjson)$", a.input, re.I):
        if a.question is None:
            a.question = a.input
        a.input = None
    # a URL passed as --corpus, or a folder passed as --url, is accepted either way
    if a.corpus and re.match(r"^https?://", a.corpus) and not a.url:
        a.url, a.corpus = a.corpus, None
    return a


def safe_id(qid) -> str:
    s = str(qid if qid not in (None, "") else "query").strip()
    s = re.sub(r"[\\/:*?\"<>|\x00-\x1f\s]+", "_", s)
    s = s.strip("._") or "query"
    return s[:180]


def output_path(output_dir: str, qid) -> str:
    return os.path.join(output_dir, f"{safe_id(qid)}_output.json")


def write_json_atomic(path: str, obj: dict) -> bool:
    """tmp file in the same directory + ``os.replace``; world-readable; never raises."""
    data = json.dumps(obj, ensure_ascii=False)
    d = os.path.dirname(path) or "."
    for attempt in range(2):
        try:
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=d)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(data)
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
                try:
                    os.chmod(tmp, 0o644)
                except OSError:
                    pass
                os.replace(tmp, path)
                return True
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
        except OSError as e:
            print(f"[wayfinder] write failed ({attempt}): {e}", file=sys.stderr)
    return False


EMPTY_OUTPUT = {"answer": "", "citations": [], "source_url": "", "confidence": 0.0}


def load_task_file(path: str) -> list[dict]:
    """A JSON list, an object with a list under tasks/questions/items, a single object, or JSON lines."""
    try:
        with open(path, encoding="utf-8-sig") as fh:
            text = fh.read()
    except OSError as e:
        print(f"[wayfinder] cannot read task file {path}: {e}", file=sys.stderr)
        return []
    items: list = []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for k in ("tasks", "questions", "queries", "items", "data"):
                if isinstance(data.get(k), list):
                    items = data[k]
                    break
            else:
                items = [data]
    except ValueError:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except ValueError:
                continue
    out = []
    for i, it in enumerate(items):
        if isinstance(it, str):
            it = {"question": it}
        if isinstance(it, dict):
            if pick(it, ID_KEYS) is None:
                it = dict(it, id=f"q{i + 1:02d}")
            out.append(it)
    return out
