"""Tokenisation, stop words and the grading normalisation."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = frozenset(
    """a an the of to in on at for from by with and or but is are was were be been being
    do does did what which who whom whose when where why how this that these those it its
    as into than then there their they them i you your we our us he she his her my me
    can could should would will shall may might must about over under per any all each
    tell give find show list please name s get has have had not no yes if so up out
    much many also only just there here one value""".split()
)

# Words that describe the *kind* of answer rather than the entity it is about.
INTENT_WORDS = frozenset(
    """price cost costs priced pricing fee fees charge how much monthly month annual annually year
    yearly email e-mail address phone telephone number call version latest current release
    released date day when founded established percent percentage rate many count total
    long duration time default maximum minimum max min""".split()
)

INTENT_STEMS: frozenset = frozenset()  # filled below, after stem() is defined

_TOKEN_RE = re.compile(r"[0-9a-zA-ZÀ-ɏͰ-ϿЀ-ӿ一-鿿]+(?:[.\-][0-9a-zA-Z]+)*")


def fold(text: str) -> str:
    """Lowercase and strip accents (Tromsø -> tromso)."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace("ø", "o").replace("Ø", "o").replace("ß", "ss").replace("æ", "ae").lower()


_IRREGULAR = {"monthly": "month", "yearly": "year", "annually": "annual", "weekly": "week", "daily": "day",
              "hourly": "hour", "quickly": "quick", "cheapest": "cheap", "cheaper": "cheap", "largest": "larg",
              "biggest": "big", "newest": "new", "latest": "latest", "weighs": "weight", "weigh": "weight",
              "costs": "cost", "priced": "price", "pricing": "price", "prices": "price", "children": "child",
              "people": "people", "people's": "people", "headquartered": "headquarter", "hq": "headquarter",
              "owns": "own", "owner": "own", "owners": "own", "owned": "own", "ownership": "own",
              "maintainer": "maintain", "maintained": "maintain", "maintains": "maintain",
              "assignee": "assign", "assigned": "assign"}


def stem(tok: str) -> str:
    """A deliberately light, consistent stemmer (plural, -ed, -ing, trailing e)."""
    if tok in _IRREGULAR:
        tok = _IRREGULAR[tok]
    if len(tok) <= 3 or any(c.isdigit() for c in tok) or "." in tok or "-" in tok:
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        tok = tok[:-3] + "y"
    elif tok.endswith(("sses", "ches", "shes", "xes", "zes")):
        tok = tok[:-2]
    elif tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        tok = tok[:-1]
    if tok.endswith("ing") and len(tok) >= 6:
        tok = tok[:-3]
    elif tok.endswith("ed") and len(tok) >= 6:
        tok = tok[:-2]
    if tok.endswith("e") and len(tok) >= 5:
        tok = tok[:-1]
    return tok


def raw_tokens(text: str) -> list[str]:
    """Folded tokens, keeping dotted/hyphenated codes like 4.2.1 or rev-c2 whole."""
    return _TOKEN_RE.findall(fold(text))


def tokens(text: str, keep_stop: bool = False) -> list[str]:
    """Search tokens: folded, stemmed, stop words removed; codes also split into parts."""
    out: list[str] = []
    for t in raw_tokens(text):
        if not keep_stop and t in STOPWORDS:
            continue
        out.append(stem(t))
        if ("." in t or "-" in t) and len(t) > 2:
            for part in re.split(r"[.\-]", t):
                if part and part not in STOPWORDS:
                    out.append(stem(part))
    return out


def special_terms(question: str) -> list[str]:
    """Terms that must be present in the source: codes, numbers and proper names.

    "What is the price of the Aurora X9?" -> ["aurora", "x9"]. A page that never
    mentions X9 cannot be the answer source, which is how unanswerable questions
    get refused on the rules path.
    """
    out: list[str] = []
    words = re.findall(r"[\wÀ-ɏ][\wÀ-ɏ.\-']*", question)
    for i, w in enumerate(words):
        w = w.strip(".'-")
        if not w:
            continue
        lw = fold(w)
        if lw in STOPWORDS or lw in INTENT_WORDS:
            continue
        has_digit = any(c.isdigit() for c in w)
        is_cap = w[0].isupper() and i > 0 and not w.isupper() or (w.isupper() and len(w) >= 2)
        if has_digit or is_cap:
            # possessive / plural forms
            lw = re.sub(r"'s$", "", lw)
            for t in tokens(lw):
                if t not in out and len(t) >= 2:
                    out.append(t)
    return out


def grade_normalize(text: str) -> str:
    """The organisers' normalisation: uppercase, no whitespace, no ``- . · _``."""
    if text is None:
        return ""
    t = str(text).upper()
    t = re.sub(r"\s+", "", t)
    return re.sub(r"[\-\.·_]", "", t)


def loose_normalize(text: str) -> str:
    """Grading normalisation plus units/currency/punctuation removed (diagnostic only)."""
    t = fold(str(text or ""))
    t = re.sub(r"[^0-9a-z@%]+", "", t)
    return t.upper()


def clean_answer(ans: str) -> str:
    """Turn model or extractor output into a value-only answer."""
    if ans is None:
        return ""
    a = str(ans).strip()
    a = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[:\-]?\s*", "", a, flags=re.I)
    a = re.sub(r"\s+", " ", a)
    for _ in range(3):  # 'Answer: "18 months".' -> 18 months
        before = a
        a = a.strip().strip("\"'`“”‘’").strip()
        if a.endswith(".") and not re.search(r"\b(?:inc|ltd|co|corp|jr|sr|no|[A-Z]\.[A-Z])\.$", a, re.I):
            a = a[:-1].rstrip()
        if a == before:
            break
    if a.lower() in {"none", "null", "n/a", "na", "unknown", "not found", "not available",
                     "i don't know", "i do not know", "no answer", "-", "?"}:
        return ""
    return a


INTENT_STEMS = frozenset(stem(w) for w in INTENT_WORDS) | INTENT_WORDS
