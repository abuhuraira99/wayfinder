"""Prompt construction and decision parsing for the browse-or-answer loop."""

from __future__ import annotations

import json
import re

SYSTEM = """You are Wayfinder, a careful web research agent. You answer a question using ONLY the numbered evidence passages, which were extracted from the website being browsed. Never use outside knowledge: the site may describe things you have never heard of.

Reply with exactly one JSON object and nothing else, in one of these forms:
{"action": "answer", "answer": "<value>", "sources": ["P1"]}
{"action": "follow", "links": ["L2"]}
{"action": "none"}

Rules:
- "answer": the value only, copied exactly as it is written in the evidence: a number with its unit or currency, a date, a version, a name, an email address, a short phrase. No sentence, no explanation, no restating the question. Keep qualifiers that are part of the value (for example Q3 FY27, REV-C2, v2.4.1, 18 months). Give exactly what was asked and no more: "how many" -> the number alone; "which city" -> the city alone, without the country.
- "sources": the passage ids you actually needed. Usually one. If one passage identifies the item (for example which product won an award) and another passage gives the value for it, cite both.
- "follow": when the evidence does not contain the answer but one or two of the listed links probably lead to it. Pick at most two link ids.
- "none": when neither the evidence nor the links can answer the question. If the question asks about something the site does not mention, reply none rather than guessing."""


def _clip(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def build_messages(question: str, evidence: list[dict], links: list[dict], *, final: bool,
                   site: str = "", note: str = "", hints: list[dict] | None = None,
                   passage_chars: int = 900) -> list[dict]:
    """``evidence`` items: {id, page, title, heading, text}; ``links``: {id, text, ref}."""
    parts = [f"Question: {question.strip()}"]
    if site:
        parts.append(f"Website: {site}")
    parts.append("\nEvidence:")
    if not evidence:
        parts.append("(no passages yet)")
    for e in evidence:
        head = e.get("heading") or ""
        title = e.get("title") or ""
        label = f"[{e['id']}] page: {e['page']}"
        if title:
            label += f" | {title}"
        if head and head != title:
            label += f" | section: {head}"
        parts.append(label + "\n" + _clip(e["text"], passage_chars))
    if hints:
        parts.append("\nPattern matches found in the evidence (unverified, may be irrelevant):")
        for h in hints:
            parts.append(f"- {h['value']} (in {h['id']})")
    if links and not final:
        parts.append("\nLinks not opened yet:")
        for l in links:
            parts.append(f"[{l['id']}] \"{_clip(l['text'] or '(no text)', 80)}\" -> {l['ref']}")
    if note:
        parts.append("\nNote: " + note)
    if final:
        parts.append("\nThis is the last step: reply with an answer or with none. Do not follow links.")
    parts.append("\nReply with one JSON object.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(parts)}]


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _first_json_object(text: str) -> dict | None:
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        obj = json.loads(chunk)
                        if isinstance(obj, dict):
                            return obj
                    except ValueError:
                        try:
                            obj = json.loads(chunk.replace("'", '"'))
                            if isinstance(obj, dict):
                                return obj
                        except ValueError:
                            pass
                    break
        start = text.find("{", start + 1)
    return None


def parse_decision(raw: str) -> dict | None:
    """Normalise the model's reply into {action, answer, sources, links} or None."""
    if not raw:
        return None
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    obj = _first_json_object(text)
    if obj is None:
        return None
    action = str(obj.get("action") or obj.get("decision") or "").lower().strip()
    ans = obj.get("answer", obj.get("value"))
    sources = obj.get("sources") or obj.get("source") or obj.get("evidence") or obj.get("citations") or []
    links = obj.get("links") or obj.get("link") or obj.get("follow") or []
    if isinstance(sources, str):
        sources = [sources]
    if isinstance(links, str):
        links = [links]
    sources = [str(s).strip().strip("[]") for s in sources if s is not None]
    links = [str(s).strip().strip("[]") for s in links if s is not None]
    if not action:
        action = "answer" if ans not in (None, "") else ("follow" if links else "none")
    if action in ("answer", "final", "final_answer"):
        if ans is None or (isinstance(ans, str) and not ans.strip()):
            return {"action": "none", "answer": "", "sources": [], "links": []}
        if isinstance(ans, (list, tuple)):
            ans = ", ".join(str(a) for a in ans)
        return {"action": "answer", "answer": str(ans), "sources": sources, "links": []}
    if action in ("follow", "open", "click", "navigate", "browse"):
        return {"action": "follow", "answer": "", "sources": [], "links": links}
    if action in ("none", "refuse", "unanswerable", "not_found", "no_answer"):
        return {"action": "none", "answer": "", "sources": [], "links": []}
    return None
