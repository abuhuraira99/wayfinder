"""Page and passage model, plus parsers for the non-HTML file types."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# A line is (kind, text). kinds: "h1".."h6", "p", "li", "row", "kv"
Line = tuple[str, str]


@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]
    caption: str = ""

    def to_dict(self) -> dict:
        return {"header": self.header, "rows": self.rows, "caption": self.caption}

    @classmethod
    def from_dict(cls, d: dict) -> "Table":
        return cls(header=list(d.get("header") or []), rows=[list(r) for r in d.get("rows") or []],
                   caption=d.get("caption") or "")


@dataclass
class Link:
    text: str
    ref: str  # resolved reference (absolute URL, or corpus-relative path)

    def to_dict(self) -> dict:
        return {"text": self.text, "ref": self.ref}


@dataclass
class Page:
    ref: str                     # canonical id: absolute URL (web) or corpus-relative path
    url: str = ""                # public URL when known
    title: str = ""
    lines: list[Line] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    description: str = ""

    @property
    def text(self) -> str:
        return "\n".join(t for _, t in self.lines)

    def to_dict(self) -> dict:
        return {
            "ref": self.ref, "url": self.url, "title": self.title, "description": self.description,
            "lines": [list(x) for x in self.lines],
            "links": [l.to_dict() for l in self.links],
            "tables": [t.to_dict() for t in self.tables],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Page":
        return cls(
            ref=d["ref"], url=d.get("url") or "", title=d.get("title") or "",
            description=d.get("description") or "",
            lines=[(str(k), str(t)) for k, t in d.get("lines") or []],
            links=[Link(l.get("text", ""), l.get("ref", "")) for l in d.get("links") or []],
            tables=[Table.from_dict(t) for t in d.get("tables") or []],
        )


@dataclass
class Passage:
    pid: str
    page_ref: str
    text: str
    heading: str = ""
    kind: str = "text"  # "text" | "table"
    order: int = 0


MAX_PASSAGE_CHARS = 700


def chunk_page(page: Page, start_index: int = 0) -> list[Passage]:
    """Split a page into passages that keep their heading context.

    Table rows are grouped with their header row so a row like
    ``Team | $49 | 50`` is never separated from ``Plan | Price | Sensors``.
    """
    out: list[Passage] = []
    heads: dict[int, str] = {}
    buf: list[str] = []
    buf_kind = "text"
    title = page.title.strip()

    def heading_ctx() -> str:
        return " > ".join(heads[k] for k in sorted(heads))

    def flush():
        nonlocal buf, buf_kind
        if buf:
            body = "\n".join(buf).strip()
            if body:
                ctx = heading_ctx()
                out.append(Passage(pid="", page_ref=page.ref, text=body, heading=ctx or title,
                                   kind=buf_kind, order=len(out)))
        buf = []
        buf_kind = "text"

    size = 0
    for kind, text in page.lines:
        if kind.startswith("h") and kind[1:].isdigit():
            flush()
            size = 0
            lvl = int(kind[1:])
            for k in [k for k in heads if k >= lvl]:
                del heads[k]
            heads[lvl] = text.strip()
            continue
        if kind == "row" and buf_kind != "table":
            flush()
            size = 0
            buf_kind = "table"
        elif kind != "row" and buf_kind == "table":
            flush()
            size = 0
        if size + len(text) > MAX_PASSAGE_CHARS and buf:
            header = buf[0] if buf_kind == "table" else None
            kind_keep = buf_kind
            flush()
            size = 0
            if header is not None:
                buf = [header]
                buf_kind = kind_keep
                size = len(header)
        buf.append(text)
        size += len(text) + 1
    flush()
    if not out and title:
        out.append(Passage(pid="", page_ref=page.ref, text=title, heading=title, order=0))
    for i, p in enumerate(out):
        p.pid = f"{page.ref}#{start_index + i}"
    return out


# ---------------------------------------------------------------- non-HTML parsers

_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def parse_markdown(text: str, ref: str) -> Page:
    page = Page(ref=ref)
    in_code = False
    table: list[list[str]] = []

    def end_table():
        nonlocal table
        if table:
            page.tables.append(Table(header=table[0], rows=table[1:]))
        table = []

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        for m in _MD_LINK.finditer(line):
            page.links.append(Link(m.group(1).strip(), m.group(2).strip()))
        plain = _MD_LINK.sub(lambda m: m.group(1), line)
        plain = re.sub(r"[*_`]{1,3}([^*_`]+)[*_`]{1,3}", r"\1", plain)
        s = plain.strip()
        if not s:
            end_table()
            continue
        if in_code:
            page.lines.append(("p", s))
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            end_table()
            lvl = len(m.group(1))
            page.lines.append((f"h{lvl}", m.group(2).strip()))
            if lvl == 1 and not page.title:
                page.title = m.group(2).strip()
            continue
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            table.append(cells)
            page.lines.append(("row", " | ".join(cells)))
            continue
        end_table()
        m = re.match(r"^(?:[-*+]|\d+[.)])\s+(.*)$", s)
        if m:
            page.lines.append(("li", m.group(1).strip()))
        else:
            page.lines.append(("p", s))
    end_table()
    return page


def parse_text(text: str, ref: str) -> Page:
    page = Page(ref=ref)
    for raw in text.splitlines():
        s = raw.strip()
        if s:
            page.lines.append(("p", s))
    if page.lines:
        page.title = page.lines[0][1][:120]
    return page


def parse_json(text: str, ref: str) -> Page:
    """Flatten JSON into ``path: value`` lines (and tables for lists of flat objects)."""
    page = Page(ref=ref)
    try:
        data = json.loads(text)
    except ValueError:
        return parse_text(text, ref)

    def walk(node, path: str, depth: int = 0):
        if depth > 12 or len(page.lines) > 20000:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k), depth + 1)
        elif isinstance(node, list):
            if node and all(isinstance(x, dict) for x in node):
                keys: list[str] = []
                for x in node:
                    for k in x:
                        if k not in keys and not isinstance(x[k], (dict, list)):
                            keys.append(k)
                if keys:
                    rows = [[_scalar(x.get(k)) for k in keys] for x in node]
                    page.tables.append(Table(header=keys, rows=rows, caption=path))
                    if path:
                        page.lines.append(("h3", path))
                    page.lines.append(("row", " | ".join(keys)))
                    for r in rows:
                        page.lines.append(("row", " | ".join(r)))
                for i, x in enumerate(node):
                    for k, v in x.items():
                        if isinstance(v, (dict, list)):
                            walk(v, f"{path}[{i}].{k}", depth + 1)
            else:
                for i, x in enumerate(node):
                    walk(x, f"{path}[{i}]", depth + 1)
        else:
            page.lines.append(("kv", f"{path}: {_scalar(node)}" if path else _scalar(node)))

    walk(data, "")
    if isinstance(data, dict):
        for key in ("title", "name", "headline"):
            if isinstance(data.get(key), str):
                page.title = data[key]
                break
    return page


def _scalar(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)
