"""Sites the agent can browse: a live website (stdlib urllib) or a saved copy on disk.

Both expose the same small interface used by the agent:

* ``start_refs()``      -> where to begin
* ``fetch(ref, deadline)`` -> :class:`Page` or ``None`` (never raises)
* ``resolve(page_ref, href)`` -> a same-site ref or ``None``
* ``cite(ref)`` / ``public_url(ref)`` -> citation string / URL for the answer card
"""

from __future__ import annotations

import gzip
import os
import posixpath
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import zlib
from pathlib import Path

from .corpus import load_pages, log
from .document import Link, Page, parse_json, parse_markdown, parse_text
from .html_clean import clean_html, decode_html

USER_AGENT = "WayfinderBot/0.1 (+https://github.com/abuhuraira99/wayfinder; research agent)"
FETCH_TIMEOUT = float(os.environ.get("WAYFINDER_FETCH_TIMEOUT", "4.0"))
MAX_BYTES = int(os.environ.get("WAYFINDER_MAX_BYTES", str(1_500_000)))
SKIP_EXT = re.compile(r"\.(?:pdf|zip|gz|tgz|rar|7z|png|jpe?g|gif|webp|svg|ico|bmp|tiff?|mp[34]|m4a|wav|avi|mov|"
                      r"mkv|webm|exe|dmg|msi|deb|rpm|iso|woff2?|ttf|otf|eot|css|js|mjs|map|xml|rss|atom|csv|"
                      r"xlsx?|docx?|pptx?|apk|bin)$", re.I)
TEXT_TYPES = ("text/html", "application/xhtml", "text/plain", "application/json", "text/markdown",
              "text/x-markdown", "application/ld+json")


def host_key(url: str) -> str:
    h = (urllib.parse.urlsplit(url).hostname or "").lower()
    return h[4:] if h.startswith("www.") else h


def normalize_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "https://" + url
    parts = urllib.parse.urlsplit(url)
    path = parts.path or "/"
    return urllib.parse.urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def is_local_ref(url: str | None) -> bool:
    if not url:
        return False
    if url.startswith("file:"):
        return True
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        return False
    return os.path.exists(url)


class FetchError(Exception):
    pass


# ====================================================================== live web


class HttpSite:
    kind = "http"

    def __init__(self, start_url: str, timeout: float = FETCH_TIMEOUT, max_bytes: int = MAX_BYTES,
                 respect_robots: bool = True, user_agent: str = USER_AGENT):
        self.start_url = normalize_url(start_url)
        self.host = host_key(self.start_url)
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.respect_robots = respect_robots and os.environ.get("WAYFINDER_ROBOTS", "1") != "0"
        self.user_agent = user_agent
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._robots_lock = threading.Lock()
        self._cache: dict[str, Page | None] = {}
        self.offline = False  # set when the network is clearly unreachable

    def start_refs(self) -> list[str]:
        return [self.start_url]

    def same_site(self, url: str) -> bool:
        return host_key(url) == self.host

    def resolve(self, page_ref: str, href: str) -> str | None:
        if not href:
            return None
        href = href.strip()
        low = href.lower()
        if low.startswith(("mailto:", "tel:", "javascript:", "data:", "#", "sms:", "ftp:")):
            return None
        try:
            url = urllib.parse.urljoin(page_ref, href)
            parts = urllib.parse.urlsplit(url)
        except ValueError:
            return None
        if parts.scheme not in ("http", "https"):
            return None
        url = urllib.parse.urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", parts.query, ""))
        if not self.same_site(url) or SKIP_EXT.search(parts.path or ""):
            return None
        return url

    def cite(self, ref: str) -> str:
        return ref

    def public_url(self, ref: str) -> str:
        return ref

    # -- robots -------------------------------------------------------------
    def allowed(self, url: str, deadline: float | None = None) -> bool:
        if not self.respect_robots or url == self.start_url:
            return True
        parts = urllib.parse.urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        with self._robots_lock:
            if origin not in self._robots:
                rp = None
                try:
                    budget = 2.5 if deadline is None else max(0.3, min(2.5, deadline - time.monotonic()))
                    body, _, status = self._get(origin + "/robots.txt", budget, 200_000)
                    if status < 400:
                        rp = urllib.robotparser.RobotFileParser()
                        rp.parse(decode_html(body).splitlines())
                except Exception:
                    rp = None
                self._robots[origin] = rp
            rp = self._robots[origin]
        if rp is None:
            return True
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True

    # -- transport ----------------------------------------------------------
    def _get(self, url: str, timeout: float, max_bytes: int) -> tuple[bytes, dict, int]:
        req = urllib.request.Request(url, headers={
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.8,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
        })
        t_end = time.monotonic() + timeout
        try:
            resp = urllib.request.urlopen(req, timeout=max(0.2, timeout))
        except urllib.error.HTTPError as e:
            return b"", {"final_url": url, "content_type": ""}, e.code
        with resp:
            status = getattr(resp, "status", 200)
            ctype = resp.headers.get("Content-Type", "") or ""
            info = {"final_url": resp.geturl() or url, "content_type": ctype,
                    "charset": resp.headers.get_content_charset()}
            if ctype and not any(t in ctype.lower() for t in TEXT_TYPES):
                return b"", info, 415
            chunks, total = [], 0
            while total < max_bytes:
                if time.monotonic() > t_end:
                    break
                chunk = resp.read(min(65536, max_bytes - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            data = b"".join(chunks)
            enc = (resp.headers.get("Content-Encoding") or "").lower()
        try:
            if "gzip" in enc:
                data = gzip.GzipFile(fileobj=__import__("io").BytesIO(data)).read(max_bytes * 4)
            elif "deflate" in enc:
                data = zlib.decompress(data)
        except (OSError, EOFError, zlib.error):
            pass
        return data, info, status

    def fetch(self, ref: str, deadline: float | None = None) -> Page | None:
        """Fetch and clean one URL; ``None`` on any failure (recorded in ``last_error``)."""
        if ref in self._cache:
            return self._cache[ref]
        page = None
        self.last_error = ""
        try:
            if self.offline:
                raise FetchError("network unreachable")
            budget = self.timeout if deadline is None else min(self.timeout, deadline - time.monotonic())
            if budget <= 0.2:
                raise FetchError("no time left")
            if not self.allowed(ref, deadline):
                raise FetchError("disallowed by robots.txt")
            data, info, status = self._get(ref, budget, self.max_bytes)
            if status >= 400:
                raise FetchError(f"HTTP {status}")
            final = info.get("final_url") or ref
            ctype = (info.get("content_type") or "").lower()
            text = decode_html(data, info.get("charset"))
            if "json" in ctype:
                page = parse_json(text, final)
            elif "markdown" in ctype:
                page = parse_markdown(text, final)
            elif "text/plain" in ctype:
                page = parse_text(text, final)
            else:
                page = clean_html(text, ref=final)
            base = getattr(page, "base_href", "") or final
            base = urllib.parse.urljoin(final, base)
            resolved = []
            seen = set()
            for l in page.links:
                r = self.resolve(base, l.ref)
                if r and r not in seen:
                    seen.add(r)
                    resolved.append(Link(l.text, r))
            page.links = resolved
            page.ref = final
            page.url = final
        except (urllib.error.URLError, socket.gaierror, ConnectionError) as e:
            self.last_error = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
            reason = getattr(e, "reason", e)
            if isinstance(reason, (socket.gaierror, ConnectionRefusedError, OSError)) and ref == self.start_url:
                # DNS or route failure on the very first page: assume no network for this task.
                self.offline = isinstance(reason, socket.gaierror) or "unreachable" in str(reason).lower()
            page = None
        except Exception as e:  # timeouts, malformed URLs, incomplete reads, decode errors...
            self.last_error = f"{type(e).__name__}: {e}"
            page = None
        self._cache[ref] = page
        if page is not None and page.ref != ref:
            self._cache[page.ref] = page
        return page


# ====================================================================== saved site


class LocalSite:
    """A saved website (or any folder of HTML/TXT/MD/JSON) browsed from disk."""

    kind = "local"

    def __init__(self, root: str, start: str | None = None, base_url: str | None = None,
                 deadline: float | None = None):
        p = root[7:] if root.startswith("file://") else root
        if p.startswith("/") and re.match(r"^/[A-Za-z]:", p):
            p = p[1:]
        path = Path(urllib.parse.unquote(p))
        start_rel = None
        if path.is_file():
            start_rel = path.name
            path = path.parent
        self.root = path
        self.base_url = base_url
        self.pages: dict[str, Page] = {}
        self.page_list = load_pages(self.root, deadline=deadline)
        for pg in self.page_list:
            self.pages[pg.ref] = pg
        self._lower = {k.lower(): k for k in self.pages}
        self.start = start_rel
        if start and not start_rel:
            self.start = self.map_url(start)
        if self.start is None:
            for cand in ("index.html", "index.htm", "README.md", "index.md"):
                if cand in self.pages:
                    self.start = cand
                    break
        self._resolved: dict[str, list[Link]] = {}

    # -- mapping --------------------------------------------------------------
    def _lookup(self, rel: str) -> str | None:
        rel = rel.lstrip("/")
        rel = posixpath.normpath(rel) if rel else ""
        if rel in (".", ""):
            rel = ""
        cands = [rel] if rel else []
        base = rel.rstrip("/")
        cands += [base + ".html", base + ".htm", (base + "/" if base else "") + "index.html",
                  (base + "/" if base else "") + "index.htm", base + ".md", base + ".json", base + ".txt"]
        for c in cands:
            if c in self.pages:
                return c
            if c.lower() in self._lower:
                return self._lower[c.lower()]
        return None

    def map_url(self, url: str) -> str | None:
        """Map an absolute URL onto a file of the saved copy (path, or host/path)."""
        if is_local_ref(url):
            try:
                p = Path(url[7:] if url.startswith("file://") else url).resolve()
                return p.relative_to(self.root.resolve()).as_posix()
            except (ValueError, OSError):
                return None
        try:
            parts = urllib.parse.urlsplit(normalize_url(url))
        except ValueError:
            return None
        path = urllib.parse.unquote(parts.path or "/")
        host = (parts.hostname or "").lower()
        alt = host[4:] if host.startswith("www.") else ("www." + host if host else "")
        prefixes = ["", host, alt]
        if self.base_url and host and host_key(url) != host_key(normalize_url(self.base_url)):
            prefixes = [host, alt]  # another domain: only if the saved copy has a folder for it
        for prefix in dict.fromkeys(prefixes):
            hit = self._lookup(posixpath.join(prefix, path.lstrip("/")) if prefix else path)
            if hit:
                return hit
        return None

    def start_refs(self) -> list[str]:
        return [self.start] if self.start else []

    def resolve(self, page_ref: str, href: str) -> str | None:
        if not href:
            return None
        href = href.strip()
        low = href.lower()
        if low.startswith(("mailto:", "tel:", "javascript:", "data:", "#", "sms:")):
            return None
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", href) or href.startswith("//"):
            if href.startswith("//"):
                href = "https:" + href
            return self.map_url(href)
        href = urllib.parse.unquote(href.split("#", 1)[0].split("?", 1)[0])
        if not href:
            return None
        if href.startswith("/"):
            # site-absolute: relative to the site root (or to the host folder if pages live under one)
            hit = self._lookup(href)
            if hit:
                return hit
            first = page_ref.split("/", 1)[0]
            return self._lookup(first + href) if "/" in page_ref else None
        joined = posixpath.join(posixpath.dirname(page_ref), href)
        return self._lookup(joined)

    def links(self, ref: str) -> list[Link]:
        if ref not in self._resolved:
            pg = self.pages.get(ref)
            out, seen = [], set()
            for l in (pg.links if pg else []):
                r = self.resolve(ref, l.ref)
                if r and r != ref and r not in seen:
                    seen.add(r)
                    out.append(Link(l.text, r))
            self._resolved[ref] = out
        return self._resolved[ref]

    def fetch(self, ref: str, deadline: float | None = None) -> Page | None:
        pg = self.pages.get(ref)
        if pg is None:
            return None
        view = Page(ref=pg.ref, url=self.public_url(pg.ref), title=pg.title, lines=pg.lines,
                    links=self.links(ref), tables=pg.tables, description=pg.description)
        return view

    def all_refs(self) -> list[str]:
        return list(self.pages)

    def cite(self, ref: str) -> str:
        return ref

    def public_url(self, ref: str) -> str:
        if not self.base_url:
            return ref
        parts = urllib.parse.urlsplit(normalize_url(self.base_url))
        host = (parts.hostname or "").lower()
        rel = ref
        for prefix in (host + "/", (host[4:] if host.startswith("www.") else "www." + host) + "/"):
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
                break
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc, "/" + rel, "", ""))


def make_site(url: str | None, corpus: str | None, deadline: float | None = None):
    """Pick the site to browse. A saved copy wins over the network when both are given."""
    if corpus:
        try:
            cp = Path(corpus[7:] if corpus.startswith("file://") else corpus)
            if cp.exists():
                return LocalSite(str(cp), start=url, base_url=url if url and not is_local_ref(url) else None,
                                 deadline=deadline)
            log("corpus path does not exist:", corpus)
        except OSError as e:
            log("corpus unusable:", e)
    if url:
        if is_local_ref(url):
            return LocalSite(url, deadline=deadline)
        if os.environ.get("WAYFINDER_OFFLINE", "0") == "1":
            return None
        try:
            return HttpSite(url)
        except ValueError:
            return None
    return None
