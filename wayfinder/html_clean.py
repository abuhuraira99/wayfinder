"""HTML to clean, structured text with the standard library only.

Keeps headings (as ``h1``..``h6`` lines), paragraphs, list items, table rows as
``a | b | c`` and definition lists as ``term: value``. Drops scripts, styles,
navigation, footers and hidden elements from the *text*, but still collects
every link (navigation menus are exactly what a crawler needs).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from .document import Link, Page, Table

SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe", "nav", "footer",
             "button", "select", "textarea", "canvas", "object", "embed", "map"}
BLOCK_TAGS = {"p", "div", "section", "article", "main", "aside", "header", "ul", "ol",
              "blockquote", "pre", "figure", "figcaption", "address", "form", "fieldset",
              "details", "summary", "hr", "caption", "dl", "body", "center", "label"}
VOID_TAGS = {"br", "img", "meta", "link", "input", "hr", "source", "area", "base", "col",
             "embed", "param", "track", "wbr"}
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_SKIP_CLASS = re.compile(r"(?:^|[\s_-])(cookie|consent|breadcrumbs?|skip-link|sr-only|visually-hidden|navbar|site-nav|menu-toggle)(?:$|[\s_-])", re.I)
_WS = re.compile(r"\s+")


def _clean(s: str) -> str:
    return _WS.sub(" ", s.replace("\xa0", " ")).strip()


class _TableState:
    def __init__(self):
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.caption: list[str] = []
        self.in_caption = False

    def close_cell(self):
        if self.cell is not None:
            if self.row is None:
                self.row = []
            self.row.append(_clean("".join(self.cell)))
            self.cell = None

    def close_row(self):
        self.close_cell()
        if self.row is not None:
            if any(c for c in self.row):
                self.rows.append(self.row)
            self.row = None


class _Cleaner(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lines: list[tuple[str, str]] = []
        self.links: list[Link] = []
        self.tables: list[Table] = []
        self.title = ""
        self.description = ""
        self.canonical = ""
        self.base_href = ""
        self._buf: list[str] = []
        self._kind = "p"
        self._skip_tag: str | None = None
        self._skip_depth = 0
        self._in_title = False
        self._title_buf: list[str] = []
        self._link_stack: list[tuple[str, list[str]]] = []
        self._tables: list[_TableState] = []
        self._dt: str | None = None
        self._in_head = False

    # ------------------------------------------------------------------ helpers
    def _flush(self, kind: str | None = None):
        text = _clean("".join(self._buf))
        self._buf = []
        if text:
            k = kind or self._kind
            if self._dt is not None and k == "dd":
                self.lines.append(("kv", f"{self._dt}: {text}"))
                self._dt = None
            elif k == "dt":
                self._dt = text
            else:
                if self._dt is not None:
                    self.lines.append(("p", self._dt))
                    self._dt = None
                self.lines.append((k if k != "dd" else "p", text))
        self._kind = "p"

    def _emit(self, data: str):
        for _, buf in self._link_stack:
            buf.append(data)
        if self._skip_tag is not None or self._in_head and not self._in_title:
            return
        if self._in_title:
            self._title_buf.append(data)
            return
        if self._tables:
            t = self._tables[-1]
            if t.in_caption:
                t.caption.append(data)
                return
            if t.cell is not None:
                t.cell.append(data)
                return
            if t.row is not None:  # stray text inside a row
                return
        self._buf.append(data)

    # ------------------------------------------------------------------ parser hooks
    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if self._skip_tag is not None:
            if tag == self._skip_tag and tag not in VOID_TAGS:
                self._skip_depth += 1
            if tag == "a":
                self._link_stack.append((a.get("href", ""), []))
            elif tag == "img" and a.get("alt"):
                for _, buf in self._link_stack:
                    buf.append(" " + a["alt"] + " ")
            return
        if tag == "head":
            self._in_head = True
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            name = (a.get("name") or a.get("property") or "").lower()
            if name in ("description", "og:description") and not self.description:
                self.description = _clean(a.get("content", ""))
            elif name == "og:title" and not self.title:
                self.title = _clean(a.get("content", ""))
            return
        elif tag == "link":
            if "canonical" in a.get("rel", "").lower():
                self.canonical = a.get("href", "")
            return
        elif tag == "base":
            self.base_href = a.get("href", "")
            return
        hidden = ("hidden" in a or a.get("aria-hidden", "").lower() == "true"
                  or a.get("role", "").lower() in ("navigation", "contentinfo", "search")
                  or "display:none" in a.get("style", "").replace(" ", "").lower()
                  or _SKIP_CLASS.search(a.get("class", "") + " " + a.get("id", "")))
        if (tag in SKIP_TAGS or hidden) and tag not in VOID_TAGS:
            if tag in ("nav", "footer") or hidden:
                self._flush()
            self._skip_tag = tag
            self._skip_depth = 1
            return
        if tag == "a":
            self._link_stack.append((a.get("href", ""), []))
            return
        if tag == "img":
            alt = a.get("alt", "")
            if alt:
                for _, buf in self._link_stack:
                    buf.append(" " + alt + " ")
            return
        if tag == "br":
            if self._tables and self._tables[-1].cell is not None:
                self._tables[-1].cell.append(" ")
            else:
                kind = self._kind
                self._flush(kind)
                self._kind = kind
            return
        if tag == "table":
            self._flush()
            self._tables.append(_TableState())
            return
        if self._tables:
            t = self._tables[-1]
            if tag == "tr":
                t.close_row()
                t.row = []
                return
            if tag in ("td", "th"):
                t.close_cell()
                if t.row is None:
                    t.row = []
                t.cell = []
                return
            if tag == "caption":
                t.in_caption = True
                return
            if t.cell is not None:
                if tag in ("p", "div", "li"):
                    t.cell.append(" ")
                return
        if tag in HEADINGS:
            self._flush()
            self._kind = tag
        elif tag == "li":
            self._flush()
            self._kind = "li"
        elif tag in ("dt", "dd"):
            self._flush()
            self._kind = tag
        elif tag in BLOCK_TAGS:
            self._flush()

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == "a" and self._link_stack:
            href, buf = self._link_stack.pop()
            text = _clean("".join(buf))
            if href:
                self.links.append(Link(text, href.strip()))
            return
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth <= 0:
                    self._skip_tag = None
                    self._buf = []
            return
        if tag == "head":
            self._in_head = False
            return
        if tag == "title":
            self._in_title = False
            if not self.title:
                self.title = _clean("".join(self._title_buf))
            return
        if tag == "table" and self._tables:
            t = self._tables.pop()
            t.close_row()
            if t.rows:
                width = max(len(r) for r in t.rows)
                rows = [r + [""] * (width - len(r)) for r in t.rows]
                cap = _clean("".join(t.caption))
                if cap:
                    self.lines.append(("h6", cap))
                self.tables.append(Table(header=rows[0], rows=rows[1:], caption=cap))
                for r in rows:
                    self.lines.append(("row", " | ".join(r)))
            return
        if self._tables:
            t = self._tables[-1]
            if tag in ("td", "th"):
                t.close_cell()
                return
            if tag == "tr":
                t.close_row()
                return
            if tag == "caption":
                t.in_caption = False
                return
            if t.cell is not None:
                return
        if tag in HEADINGS or tag in ("li", "dt", "dd") or tag in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        self._emit(data)

    def close(self):
        super().close()
        while self._tables:
            self.handle_endtag("table")
        while self._link_stack:
            self.handle_endtag("a")
        self._flush()
        if self._dt is not None:
            self.lines.append(("p", self._dt))
            self._dt = None


def clean_html(html: str, ref: str = "") -> Page:
    """Parse ``html`` into a :class:`Page`. Link refs are left unresolved (raw href)."""
    p = _Cleaner()
    try:
        p.feed(html)
        p.close()
    except Exception:  # malformed markup: keep whatever was parsed
        try:
            p._flush()
        except Exception:
            pass
    page = Page(ref=ref, title=p.title, lines=p.lines, links=p.links, tables=p.tables,
                description=p.description)
    page.canonical = p.canonical  # type: ignore[attr-defined]
    page.base_href = p.base_href  # type: ignore[attr-defined]
    # collapse consecutive duplicates (repeated banners, mobile+desktop copies)
    dedup: list[tuple[str, str]] = []
    for line in page.lines:
        if not dedup or dedup[-1] != line:
            dedup.append(line)
    page.lines = dedup
    return page


_META_CHARSET = re.compile(rb"<meta[^>]+charset=[\"']?([\w-]+)", re.I)


def decode_html(data: bytes, header_charset: str | None = None) -> str:
    """Decode bytes using the HTTP charset, a <meta charset>, then utf-8, then latin-1."""
    cands = []
    if header_charset:
        cands.append(header_charset)
    m = _META_CHARSET.search(data[:4096])
    if m:
        cands.append(m.group(1).decode("ascii", "ignore"))
    cands += ["utf-8", "cp1252", "latin-1"]
    for enc in cands:
        try:
            return data.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", "replace")
