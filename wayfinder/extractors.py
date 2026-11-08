"""Deterministic answer extractors (the rules path).

Given ranked passages they produce typed candidates: prices, dates, versions,
emails, phones, URLs, counts, percentages, durations, years, people's names,
``label: value`` lines and table cells located by row and column label. The
model path uses them as a fallback and as hints; with no model they are the
answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .document import Page, Passage, Table
from .ranker import expand_query
from .textutil import INTENT_STEMS as INTENT_WORDS, STOPWORDS, fold, grade_normalize, special_terms, tokens

MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december|"
          "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")

PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+"),
    "url": re.compile(r"https?://[^\s<>\"')\]]+"),
    "phone": re.compile(r"(?<![\w.])\+?\(?\d{1,4}\)?(?:[\s.\-]?\(?\d{1,4}\)?){2,6}(?![\w.])"),
    "price": re.compile(
        r"(?:(?:US|CA|AU|NZ)?[$€£¥₹]|USD|EUR|GBP|NOK|SEK|CHF)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|K|M|bn))?"
        r"|\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|NOK|SEK|CHF|dollars|euros|kr)\b"
        r"|\b(?:free|Free|FREE)\b"),
    "percent": re.compile(r"\d+(?:[.,]\d+)?\s?%"),
    "version": re.compile(r"(?<![\w.])(?:v|V|version\s)?\d+\.\d+(?:\.\d+){0,3}(?:[-+][0-9A-Za-z.]+)?(?![\w])"),
    "date": re.compile(
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s(?:{MONTHS})\.?,?\s\d{{4}}\b"
        rf"|\b(?:{MONTHS})\.?\s\d{{1,2}}(?:st|nd|rd|th)?,?\s\d{{4}}\b"
        r"|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b"
        r"|\bQ[1-4]\s?(?:FY)?\s?'?\d{2,4}\b"
        rf"|\b(?:{MONTHS})\.?\s\d{{4}}\b", re.I),
    "year": re.compile(r"(?<![\w.\-])(?:1[6-9]\d\d|20\d\d)(?![\w.\-])"),
    "duration": re.compile(
        r"\b\d+(?:[.,]\d+)?\+?\s?(?:-\s?\d+\s?)?(?:ms|milliseconds?|seconds?|secs?|minutes?|mins?|hours?|hrs?|"
        r"business hours?|days?|business days?|weeks?|months?|years?|yrs?)\b", re.I),
    "count": re.compile(r"(?<![\w.$€£\-/])\d{1,3}(?:,\d{3})+(?![\w.%])|(?<![\w.$€£\-/])\d+\+?(?![\w.%/])"),
}

INTENT_RULES: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"\b(e-?mail|email address|mail address)\b", re.I)),
    ("url", re.compile(r"\b(url|web ?address|link to|homepage|website address)\b", re.I)),
    ("phone", re.compile(r"\b(phone|telephone|call|hotline|fax|mobile number)\b", re.I)),
    ("percent", re.compile(r"(%|\bpercent|\bpercentage|\buptime\b|\bdiscount\b|\bavailability\b|\bsuccess rate\b)", re.I)),
    ("price", re.compile(r"\b(price|prices|priced|cost|costs|how much|fee|fees|charge|charged|pay)\b|\$", re.I)),
    ("version", re.compile(r"\b(version|firmware|release number|build number|semver)\b", re.I)),
    ("date", re.compile(r"\b(when|what date|which date|what day|on which day|date|deadline)\b", re.I)),
    ("year", re.compile(r"\b(what year|which year|in what year|year was|founded|established|incorporated)\b", re.I)),
    ("duration", re.compile(r"\b(how quickly|how fast|how soon|how long|duration|response time|retention|lead time|turnaround|battery life|"
                            r"warranty|trial)\b", re.I)),
    ("count", re.compile(r"\b(how many|number of|count of|total number)\b", re.I)),
    ("name", re.compile(r"^\s*(who|whom)\b|\b(name of the|who is|who was|who leads|who runs)\b", re.I)),
]


def question_intent(question: str) -> str | None:
    q = question.strip()
    found = [name for name, rx in INTENT_RULES if rx.search(q)]
    if not found:
        return None
    # "When was X founded" is a year question, "how many days" a duration question
    if "year" in found and "date" in found:
        return "year"
    if "date" in found and re.match(r"\s*(when|on what date|what date)\b", q, re.I):
        return "date"
    if "count" in found and "duration" in found:
        return "duration"
    if "count" in found and re.search(r"how many (hours|days|weeks|months|minutes|seconds|years)", q, re.I):
        return "duration"
    if "percent" in found and "price" in found and "%" not in q and "percent" not in q.lower():
        return "price"
    return found[0]


@dataclass
class Candidate:
    value: str
    page_ref: str
    pid: str = ""
    score: float = 0.0
    kind: str = ""
    line: str = ""
    support: list[str] = field(default_factory=list)  # extra pages needed (bridges)

    def to_dict(self) -> dict:
        return {"value": self.value, "page": self.page_ref, "pid": self.pid, "score": round(self.score, 3),
                "kind": self.kind, "line": self.line[:300]}


def _focus(question: str) -> tuple[dict[str, float], set[str]]:
    special = set(special_terms(question))
    w: dict[str, float] = {}
    for t in tokens(question):
        if t in INTENT_WORDS:
            continue
        w[t] = 2.0 if t in special else 1.0
    return w, special


def _overlap(text: str, weights: dict[str, float]) -> float:
    ts = set(tokens(text))
    return sum(v for t, v in weights.items() if t in ts)


def _all_overlap(text: str, question_toks: set[str]) -> int:
    return len(set(tokens(text)) & question_toks)


def _version_key(v: str) -> tuple:
    nums = re.findall(r"\d+", v)
    return tuple(int(n) for n in nums[:4])


def _clean_value(v: str) -> str:
    v = v.strip().strip(",;:()[]").strip()
    v = re.sub(r"^version\s+", "", v, flags=re.I)
    return v


def _plausible_phone(v: str) -> bool:
    digits = re.sub(r"\D", "", v)
    return 7 <= len(digits) <= 15 and not re.fullmatch(r"(19|20)\d\d", digits)


def _typed_matches(intent: str, text: str) -> list[re.Match]:
    rx = PATTERNS.get(intent)
    if rx is None:
        return []
    ms = list(rx.finditer(text))
    if intent == "phone":
        ms = [m for m in ms if _plausible_phone(m.group(0))]
    if intent == "count":
        # drop numbers that are parts of dates, versions, prices or codes
        ms = [m for m in ms if not re.fullmatch(r"(19|20)\d\d", m.group(0))]
    return ms


_KV = re.compile(r"^\s*(?:[-*•]\s+)?([^:|=]{2,60}?)\s*(?:[:：]\s+|=\s*)(.{1,200})$")
_UNITS = (r"(?:[CFK]|ms|ns|us|µs|s|secs?|seconds?|mins?|minutes?|h|hrs?|hours?|days?|weeks?|months?|years?|"
          r"mm|cm|m|km|nm|µm|um|mg|g|kg|t|lbs?|oz|mV|V|kV|mW|W|kW|MW|mA|A|Hz|kHz|MHz|GHz|dB|dBm|rpm|psi|bar|Pa|kPa|"
          r"B|KB|MB|GB|TB|Kb|Mb|Gb|bps|Mbps|Gbps|req/min|req/s|x)")
# "... is 94 C", "... was REV-C2", "... equals 3,000 req/min": a number (with unit) or a code-like token
_COPULA = re.compile(
    r"\b(?:is|are|was|were|equals|reaches|of|at|to)\s+(?:about\s+|approximately\s+|up to\s+|set to\s+)?"
    r"(?P<v>[-+]?[$€£]?\d[\d,]*(?:\.\d+)?(?:\s?(?:°\s?[CF]|%|" + _UNITS + r")(?![\w]))?"
    r"|[A-Z][A-Z0-9]*(?:[-_.][A-Z0-9]+)+\b|[a-z]+(?:-[a-z0-9]+){1,4}\b)")
IDENT = re.compile(r"\b[A-Z][A-Z0-9]{1,9}[-_]\d[\w\-]*\b|\b[A-Z]{2,}\d{2,}\w*\b")
_NAME = re.compile(r"\b([A-Z][a-zà-ÿ]+(?:[\s\-][A-Z][a-zà-ÿ]+){1,3})\b")


def _lines(p: Passage) -> list[str]:
    out = []
    if p.heading:
        out.append(p.heading.split(" > ")[-1])
    out += [l for l in p.text.splitlines() if l.strip()]
    return out


def extract_candidates(question: str, ranked: list[tuple[Passage, float]], pages: dict[str, Page],
                       intent: str | None = None) -> list[Candidate]:
    intent = intent if intent is not None else question_intent(question)
    weights, special = _focus(question)
    qtoks = set(tokens(question)) | {t for t, w in expand_query(question)[0].items() if w < 1}
    cands: list[Candidate] = []
    if not ranked:
        return cands
    top = max(s for _, s in ranked) or 1.0
    qf = fold(question)

    for rank_i, (p, s) in enumerate(ranked):
        pnorm = max(0.0, s) / top
        head_ov = _overlap(p.heading, weights)
        lines = _lines(p)
        header_row = lines[1] if p.kind == "table" and len(lines) > 1 else ""
        for li, line in enumerate(lines):
            ov = _overlap(line, weights)
            ctx_ov = ov + 0.35 * head_ov
            if p.kind == "table" and header_row and line != header_row:
                ctx_ov += 0.25 * _overlap(header_row, weights)
            # ---- typed values
            if intent and intent in PATTERNS:
                for m in _typed_matches(intent, line):
                    val = _clean_value(m.group(0))
                    if not val or fold(val) in qf and intent not in ("email", "url"):
                        continue  # never answer with a value copied from the question
                    near = _near_focus(line, m.start(), m.end(), weights)
                    sc = 1.0 * pnorm + 1.1 * ctx_ov + 0.6 * near - 0.05 * rank_i
                    cands.append(Candidate(val, p.page_ref, p.pid, sc, intent, line))
            # ---- label: value
            m = _KV.match(line)
            if m and " | " not in line:
                label, value = m.group(1), m.group(2)
                lov = _all_overlap(label, qtoks)
                if lov:
                    val = _narrow(question, _pick_from_value(value, intent))
                    if val:
                        sc = 0.8 * pnorm + 1.3 * lov + 0.3 * head_ov + (0.5 if intent else 0.2)
                        cands.append(Candidate(val, p.page_ref, p.pid, sc, "label", line))
            # ---- "<focus> is <value>" sentences (prose in .txt/.md/HTML) when no typed pattern applies
            if (intent is None or intent not in PATTERNS) and ov >= 1:
                for cm in _COPULA.finditer(line):
                    val = cm.group("v").strip().rstrip(".,;:")
                    before = line[: cm.start()]
                    b_ov = _overlap(before, weights)
                    if not val or b_ov < 1 or fold(val) in qf:
                        continue
                    sc = 0.9 * pnorm + 1.2 * b_ov + 0.3 * head_ov + 0.3 - 0.05 * rank_i
                    cands.append(Candidate(val, p.page_ref, p.pid, sc, "copula", line))
            # ---- names
            if intent == "name" and ov > 0:
                for nm in _NAME.finditer(line):
                    val = nm.group(1)
                    if fold(val) in qf or all(t in qtoks for t in tokens(val)):
                        continue
                    near = _near_focus(line, nm.start(), nm.end(), weights)
                    cands.append(Candidate(val, p.page_ref, p.pid, 0.8 * pnorm + ov + 0.6 * near, "name", line))

    # ---- tables located by row and column label
    seen_pages = []
    for p, _ in ranked:
        if p.page_ref not in seen_pages:
            seen_pages.append(p.page_ref)
    for rank_i, ref in enumerate(seen_pages[:5]):
        pg = pages.get(ref)
        if not pg:
            continue
        for t in pg.tables:
            cands += _table_candidates(question, t, ref, qtoks, intent, weights, rank_i)

    # merge duplicates: same normalised value -> keep best, small bonus for agreement
    merged: dict[tuple[str, str], Candidate] = {}
    for c in cands:
        key = (grade_normalize(c.value), c.page_ref)
        if key in merged:
            if c.score > merged[key].score:
                c.score += 0.1
                merged[key] = c
            else:
                merged[key].score += 0.1
        else:
            merged[key] = c
    out = sorted(merged.values(), key=lambda c: -c.score)
    if intent == "version" and re.search(r"\b(latest|newest|current|most recent|last)\b", question, re.I) and out:
        best = out[0].score
        close = [c for c in out if c.score >= best - 0.9 and c.kind in ("version", "label", "table")]
        if len(close) > 1:
            newest = max(close, key=lambda c: _version_key(c.value))
            newest.score = best + 0.01
            out.sort(key=lambda c: -c.score)
    return out


def _near_focus(line: str, start: int, end: int, weights: dict[str, float]) -> float:
    """1.0 when a focus term sits within ~40 characters of the value, else decays."""
    low = fold(line)
    best = 0.0
    for t in weights:
        for m in re.finditer(re.escape(t), low):
            d = max(0, start - m.end(), m.start() - end)
            best = max(best, 1.0 if d <= 40 else 0.5 if d <= 100 else 0.1)
    return best


def _pick_from_value(value: str, intent: str | None) -> str:
    value = value.strip()
    if intent and intent in PATTERNS:
        ms = _typed_matches(intent, value)
        return _clean_value(ms[0].group(0)) if ms else ""
    if len(value) > 90:
        return ""
    return _clean_value(re.split(r"\s+[—–]\s+|\s\(\s?|;\s", value)[0])


def _narrow(question: str, value: str) -> str:
    """'Which city ...' + 'Tromsø, Norway' -> 'Tromsø'."""
    if re.search(r"\b(which|what)\s+(city|town|country|state|region)\b", question, re.I) and "," in value:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if re.search(r"\bcountry\b", question, re.I):
            return parts[-1]
        return parts[0]
    return value


def _cell_value(cell: str, intent: str | None) -> str:
    if intent and intent in PATTERNS:
        ms = _typed_matches(intent, cell)
        if ms:
            return _clean_value(ms[0].group(0))
    return cell.strip()


def _table_candidates(question: str, t: Table, ref: str, qtoks: set[str], intent: str | None,
                      weights: dict[str, float], rank_i: int) -> list[Candidate]:
    out = []
    header = t.header
    if not header or not t.rows:
        return out
    width = len(header)
    col_ov = [_all_overlap(h, qtoks) for h in header]
    cc = _constrained_cell(question, t, qtoks, intent)
    if cc is not None:
        val, line, sc = cc
        out.append(Candidate(val, ref, "", sc - 0.1 * rank_i, "table-constraint", line))
    for r in t.rows:
        if not r:
            continue
        row_label = r[0]
        r_ov = _all_overlap(row_label, qtoks)
        r_w = _overlap(row_label, weights)
        if width == 2 or (width > 2 and not any(col_ov[1:]) and r_ov):
            # key/value table: "Battery life | 18 months"
            if r_ov and len(r) > 1 and r[1].strip() and not _NEG_CELL.match(r[1]):
                val = _cell_value(r[1], intent)
                if val and fold(val) not in fold(question):
                    typed_ok = not intent or intent not in PATTERNS or bool(_typed_matches(intent, r[1]))
                    sc = 1.2 * r_ov + 0.4 * r_w + (0.6 if typed_ok else -0.8) + 0.8 - 0.1 * rank_i
                    if width > 2:
                        sc -= 1.2  # first option of a comparison table is only a weak guess
                    out.append(Candidate(val, ref, "", sc, "table", " | ".join(r)))
            continue
        for j in range(1, min(width, len(r))):
            cell = r[j].strip()
            if not cell or cell in ("—", "-", "–"):
                continue
            c_ov = col_ov[j]
            if r_ov == 0 or c_ov == 0:
                continue
            union = set(tokens(row_label + " " + header[j])) & qtoks
            val = _cell_value(cell, intent)
            if fold(val) in fold(question):
                continue
            typed_ok = not intent or intent not in PATTERNS or bool(_typed_matches(intent, cell))
            sc = 1.0 * len(union) + 0.5 * (r_w + _overlap(header[j], weights)) + (0.7 if typed_ok else -0.9) \
                + 0.6 - 0.1 * rank_i
            out.append(Candidate(val, ref, "", sc, "table", f"{header[j]} / {row_label}: {cell}"))
    return out


def gate(question: str, cand: Candidate, pages: dict[str, Page], min_score: float = 1.2) -> bool:
    """Is a rules candidate trustworthy enough to return (vs. refusing)?"""
    if cand.score < min_score:
        return False
    special = special_terms(question)
    text = " ".join(pages[r].text + " " + pages[r].title for r in [cand.page_ref, *cand.support] if r in pages)
    toks = set(tokens(text))
    ftext = fold(text)
    missing = [t for t in special if t not in toks and t not in ftext and not acronym_in(text, t)]
    return not missing


def acronym_in(text: str, acr: str) -> bool:
    """'sla' matches 'Service Level Agreement' (initials of consecutive words)."""
    if not (2 <= len(acr) <= 6) or not acr.isalpha():
        return False
    words = [w for w in re.findall(r"[A-Za-z]+", text) if w.lower() not in STOPWORDS]
    n = len(acr)
    initials = "".join(w[0].lower() for w in words)
    return acr.lower() in initials and any(
        initials[i:i + n] == acr.lower() and all(len(w) > 1 for w in words[i:i + n])
        for i in range(len(initials) - n + 1))


def best_rule_answer(question: str, ranked: list[tuple[Passage, float]], pages: dict[str, Page]
                     ) -> Candidate | None:
    for c in extract_candidates(question, ranked, pages)[:6]:
        if gate(question, c, pages):
            return c
    return None


def grounded(answer: str, text: str) -> bool:
    """Does ``answer`` appear in ``text`` (after grading normalisation, or token-wise)?"""
    if not answer:
        return False
    a = grade_normalize(fold(answer))
    t = grade_normalize(fold(text))
    if a and a in t:
        return True
    a2 = grade_normalize(fold(answer).replace(",", ""))
    if a2 and a2 in t.replace(",", ""):
        return True
    at = [x for x in tokens(answer, keep_stop=True) if x not in STOPWORDS]
    if not at or len(at) > 8:
        return False
    tt = set(tokens(text, keep_stop=True))
    return all(x in tt for x in at)


_NEG_CELL = re.compile(r"^\s*(?:—|–|-|no|none|n/?a|✗|✕|×|x|not included|unavailable)?\s*$", re.I)
_CHEAP = re.compile(r"\b(cheapest|least expensive|lowest[- ]priced|lowest|smallest|minimum|entry[- ]level)\b", re.I)
_DEAR = re.compile(r"\b(most expensive|priciest|highest|largest|biggest|maximum|top[- ]tier)\b", re.I)


def _num(cell: str) -> float | None:
    m = re.search(r"\d[\d,]*(?:\.\d+)?", cell)
    if m:
        return float(m.group(0).replace(",", ""))
    if re.search(r"\bfree\b", cell, re.I):
        return 0.0
    return None


def _constrained_cell(question: str, t: Table, qtoks: set[str], intent: str | None):
    """Plan-comparison reasoning: 'X on the cheapest plan that includes SSO'.

    Columns are options (plans); one row is the constraint (has SSO), one row is
    the asked attribute (data retention). Keep options whose constraint cell is
    positive, order them by a price row (or left-to-right), take the extreme.
    """
    m = re.search(r"\b(?:that|which|with|where)\b\s+(?:includes?|has|have|offers?|supports?|comes? with|"
                  r"provides?)?\s*(.+?)[?.!]*$", question, re.I)
    if not m or len(t.header) < 3 or len(t.rows) < 2:
        return None
    ctoks = set(tokens(m.group(1)))
    main_toks = set(tokens(question[: m.start()])) & qtoks
    if not ctoks:
        return None

    def ov(label, toks):
        return len(set(tokens(label)) & toks)

    con_row = max(t.rows, key=lambda r: ov(r[0], ctoks), default=None)
    if con_row is None or ov(con_row[0], ctoks) == 0:
        return None
    target = max((r for r in t.rows if r is not con_row), key=lambda r: ov(r[0], main_toks), default=None)
    if target is None or ov(target[0], main_toks) == 0:
        return None
    cols = [j for j in range(1, min(len(t.header), len(con_row), len(target)))
            if not _NEG_CELL.match(con_row[j] or "")]
    if not cols:
        return None
    price_row = next((r for r in t.rows if re.search(r"\b(price|cost|fee|monthly|per month)\b", r[0], re.I)), None)

    def key(j):
        if price_row is not None and j < len(price_row):
            n = _num(price_row[j])
            if n is not None:
                return (0, n, j)
        return (1, 0.0, j)  # "Contact sales" etc. sorts after priced options

    if _CHEAP.search(question):
        j = min(cols, key=key)
    elif _DEAR.search(question):
        priced = [c for c in cols if key(c)[0] == 0]
        j = max(priced or cols, key=lambda c: (key(c)[1], c))
    elif len(cols) == 1:
        j = cols[0]
    else:
        return None
    cell = target[j].strip()
    if not cell or _NEG_CELL.match(cell):
        return None
    val = _cell_value(cell, intent)
    return val, f"{t.header[j]} / {target[0]}: {cell} (where {con_row[0]} = {con_row[j]})", 6.0


def shape_value(question: str, intent: str | None, value: str) -> str:
    """Trim a correct-but-wordy value to what was asked (value-only convention).

    'How many nodes ...' + '3 nodes' -> '3';  'Which city ...' + 'Tromsø, Norway' -> 'Tromsø'.
    Units that are part of the value ('18 months', '95 g') are kept.
    """
    v = (value or "").strip()
    if not v:
        return v
    v = _narrow(question, v)
    if intent == "count":
        m = re.fullmatch(r"(\d[\d,]*\+?)\s+([A-Za-z][\w\- ]{0,40})", v)
        if m:
            noun = set(tokens(m.group(2)))
            if noun and noun <= set(tokens(question)):
                v = m.group(1)
    return v
