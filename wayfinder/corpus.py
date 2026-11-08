"""Saved-site corpus: a per-file-fault-tolerant walk and a persisted JSON index."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

from .document import Page, parse_json, parse_markdown, parse_text
from .html_clean import clean_html, decode_html

HTML_EXT = {".html", ".htm", ".xhtml", ".shtml", ".php", ".asp", ".aspx", ".jsp"}
TEXT_EXT = {".txt", ".text", ".log"}
MD_EXT = {".md", ".markdown", ".mdx", ".rst"}
JSON_EXT = {".json", ".jsonl"}
MAX_FILE_BYTES = 3_000_000
MAX_FILES = 6000
INDEX_VERSION = 2

_mem_cache: dict[str, tuple[str, list[Page]]] = {}
_lock = threading.Lock()


def log(*a):
    print("[wayfinder]", *a, file=sys.stderr, flush=True)


def default_index_dir() -> Path:
    env = os.environ.get("WAYFINDER_INDEX_DIR")
    if env:
        return Path(env)
    app_index = Path("/app/index")
    try:
        if app_index.is_dir() and os.access(app_index, os.W_OK) or (
                Path("/app").is_dir() and os.access("/app", os.W_OK)):
            return app_index
    except OSError:
        pass
    return Path(tempfile.gettempdir()) / "wayfinder-index"


def kind_for(path: Path) -> str | None:
    name = path.name.lower()
    ext = path.suffix.lower()
    if ext in HTML_EXT or ".htm" in name:
        return "html"
    if ext in MD_EXT:
        return "md"
    if ext in JSON_EXT:
        return "json"
    if ext in TEXT_EXT:
        return "txt"
    if ext == "":
        return "sniff"
    return None


def list_files(root: Path, max_files: int = MAX_FILES) -> list[tuple[str, Path, os.stat_result]]:
    """Walk ``root`` without ever raising; returns (relpath, path, stat) sorted by relpath."""
    out = []
    try:
        root_real = root.resolve()
    except OSError:
        return out

    def onerror(err):
        log("walk error:", err)

    for dirpath, dirnames, filenames in os.walk(root, onerror=onerror, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith((".git", "__pycache__", "node_modules")))
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            try:
                if kind_for(p) is None:
                    continue
                if p.is_symlink():
                    real = p.resolve()
                    if root_real not in real.parents:
                        continue
                st = p.stat()
                if not os.path.isfile(p):
                    continue
                rel = p.relative_to(root).as_posix()
                out.append((rel, p, st))
            except (OSError, ValueError) as e:
                log("skip", p, e)
            if len(out) >= max_files:
                return sorted(out, key=lambda x: x[0])
    return sorted(out, key=lambda x: x[0])


def signature(files) -> str:
    h = hashlib.sha1(f"v{INDEX_VERSION}".encode())
    for rel, _, st in files:
        h.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\n".encode("utf-8", "replace"))
    return h.hexdigest()


def parse_file(path: Path, rel: str) -> Page | None:
    """Parse one file; returns None (never raises) when unreadable or unknown."""
    try:
        kind = kind_for(path)
        with open(path, "rb") as fh:
            data = fh.read(MAX_FILE_BYTES)
        if not data.strip():
            return None
        if kind == "sniff":
            head = data[:1024].lstrip().lower()
            if head.startswith((b"<!doctype html", b"<html", b"<?xml")) or b"<body" in head:
                kind = "html"
            elif head.startswith((b"{", b"[")):
                kind = "json"
            elif b"\x00" in data[:1024]:
                return None
            else:
                kind = "txt"
        if b"\x00" in data[:2048] and kind != "html":
            return None
        if kind == "html":
            page = clean_html(decode_html(data), ref=rel)
        else:
            text = decode_html(data)
            if kind == "md":
                page = parse_markdown(text, rel)
            elif kind == "json":
                if path.suffix.lower() == ".jsonl":
                    page = parse_text(text, rel)
                else:
                    page = parse_json(text, rel)
            else:
                page = parse_text(text, rel)
        page.ref = rel
        if not page.title:
            page.title = Path(rel).stem.replace("-", " ").replace("_", " ")
        return page
    except Exception as e:  # unreadable, permission denied, bad encoding, ...
        log("unreadable", rel, type(e).__name__, e)
        return None


def index_path(root: Path, index_dir: Path | None = None) -> Path:
    key = hashlib.sha1(str(root.resolve()).encode("utf-8", "replace")).hexdigest()[:16]
    return (index_dir or default_index_dir()) / f"site-{key}.json"


def build_index(root: str | os.PathLike, index_dir: Path | None = None, persist: bool = True,
                deadline: float | None = None) -> list[Page]:
    root = Path(root)
    files = list_files(root)
    sig = signature(files)
    pages: list[Page] = []
    for rel, path, _ in files:
        if deadline is not None and time.monotonic() > deadline:
            log("index build hit its deadline after", len(pages), "pages")
            persist = False
            break
        pg = parse_file(path, rel)
        if pg is not None and (pg.lines or pg.links):
            pages.append(pg)
    if persist:
        try:
            ip = index_path(root, index_dir)
            ip.parent.mkdir(parents=True, exist_ok=True)
            tmp = ip.with_suffix(f".tmp{os.getpid()}")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"version": INDEX_VERSION, "root": str(root.resolve()), "signature": sig,
                           "pages": [p.to_dict() for p in pages]}, fh, ensure_ascii=False)
            os.replace(tmp, ip)
        except OSError as e:
            log("could not persist index:", e)
    with _lock:
        _mem_cache[str(root.resolve())] = (sig, pages)
    return pages


def load_pages(root: str | os.PathLike, index_dir: Path | None = None,
               deadline: float | None = None) -> list[Page]:
    """Load the corpus pages: memory cache -> persisted index -> fresh build."""
    root = Path(root)
    if not root.exists():
        return []
    if root.is_file():
        pg = parse_file(root, root.name)
        return [pg] if pg else []
    files = list_files(root)
    sig = signature(files)
    key = str(root.resolve())
    with _lock:
        hit = _mem_cache.get(key)
    if hit and hit[0] == sig:
        return hit[1]
    ip = index_path(root, index_dir)
    try:
        if ip.is_file():
            with open(ip, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("version") == INDEX_VERSION and data.get("signature") == sig:
                pages = [Page.from_dict(d) for d in data.get("pages", [])]
                with _lock:
                    _mem_cache[key] = (sig, pages)
                return pages
    except (OSError, ValueError, KeyError) as e:
        log("index unreadable, rebuilding:", e)
    return build_index(root, index_dir=index_dir, persist=True, deadline=deadline)
