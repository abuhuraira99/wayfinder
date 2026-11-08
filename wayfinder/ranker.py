"""Passage and link ranking: BM25 with exact-token, phrase and heading boosts."""

from __future__ import annotations

import math
import re
from collections import Counter

from .document import Link, Passage
from .textutil import INTENT_STEMS as INTENT_WORDS, fold, special_terms, stem, tokens

# intent -> words that tend to appear in the URL / anchor of the page holding the answer
INTENT_LINK_HINTS = {
    "price": "pricing plans plan price prices cost buy store shop subscription billing tiers",
    "email": "contact support help about team impressum",
    "phone": "contact support about offices locations",
    "version": "changelog releases release notes news download updates whats-new history firmware",
    "date": "news blog press events changelog releases announcements history",
    "year": "about company history story",
    "count": "about company careers jobs team",
    "percent": "sla pricing status uptime terms",
    "duration": "sla support terms shipping returns faq",
    "name": "about team leadership people company",
    "url": "docs documentation api developers",
}

# a small synonym table used to expand questions (applied with a lower weight)
SYNONYMS = {
    "price": ["pricing", "cost", "plan", "month", "mo"],
    "cost": ["price", "pricing"],
    "email": ["e-mail", "mail", "contact"],
    "phone": ["tel", "telephone", "call"],
    "founded": ["established", "since", "history"],
    "headquarter": ["hq", "headquarters", "office", "based"],
    "uptime": ["availability", "sla"],
    "latest": ["current", "newest", "recent"],
    "version": ["release", "v"],
    "ceo": ["chief", "executive", "founder"],
    "weight": ["weighs", "mass"],
    "battery": ["power"],
    "employee": ["staff", "people", "team"],
    "port": ["listen", "listens"],
}


def expand_query(question: str) -> tuple[Counter, set[str]]:
    """Weighted query terms and the set of 'special' (must-match-ish) terms."""
    q = Counter()
    for t in tokens(question):
        q[t] += 1.0
    for t in list(q):
        for s in SYNONYMS.get(t, []):
            for st in tokens(s):
                if st not in q:
                    q[st] += 0.35
    return q, set(special_terms(question))


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.6):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / self.n) if self.n else 1.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, i: int, q: Counter) -> float:
        tf, dl, s = self.tf[i], self.len[i], 0.0
        for t, w in q.items():
            f = tf.get(t)
            if not f:
                continue
            idf = self.idf.get(t, 0.0)
            s += w * idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return s


def _bigrams(toks: list[str]) -> set[tuple[str, str]]:
    return set(zip(toks, toks[1:]))


class PassageIndex:
    """Tokenised passages + BM25 statistics, built once and queried many times."""

    def __init__(self, passages: list[Passage], titles: dict[str, str] | None = None):
        self.passages = passages
        self.titles = titles or {}
        self.docs: list[list[str]] = []
        self.ctx: list[set[str]] = []
        for p in passages:
            ctx = f"{p.heading} {self.titles.get(p.page_ref, '')}"
            ct = tokens(ctx)
            self.docs.append(tokens(p.text) + ct)
            self.ctx.append(set(ct))
        self.sets = [set(d) for d in self.docs]
        self.bm = BM25(self.docs)
        self.postings: dict[str, list[int]] = {}
        for i, d in enumerate(self.sets):
            for t in d:
                self.postings.setdefault(t, []).append(i)

    def search(self, question: str, top_k: int = 8, per_page: int = 3,
               boost_pages: dict[str, float] | None = None, only_pages: set[str] | None = None
               ) -> list[tuple[Passage, float]]:
        q, special = expand_query(question)
        qbi = _bigrams(tokens(question))
        cand: set[int] = set()
        for t in q:
            cand.update(self.postings.get(t, ()))
        if boost_pages:
            for i, p in enumerate(self.passages):
                if p.page_ref in boost_pages:
                    cand.add(i)
        scored = []
        for i in cand:
            p = self.passages[i]
            if only_pages is not None and p.page_ref not in only_pages:
                continue
            s = self.bm.score(i, q)
            dset = self.sets[i]
            if s > 0:
                # exact special terms (codes, names, numbers) matter more than generic words
                s += 1.6 * sum(1 for t in special if t in dset)
                s -= 0.6 * sum(1 for t in special if t not in dset)
                s += 0.7 * len(qbi & _bigrams(self.docs[i]))
                s += 0.4 * sum(1 for t in q if t in self.ctx[i] and t not in INTENT_WORDS)
            if boost_pages:
                s += boost_pages.get(p.page_ref, 0.0)
            if s > 0:
                scored.append((p, s))
        scored.sort(key=lambda x: (-x[1], x[0].page_ref, x[0].order))
        out, per = [], Counter()
        for p, s in scored:
            if per[p.page_ref] >= per_page:
                continue
            per[p.page_ref] += 1
            out.append((p, s))
            if len(out) >= top_k:
                break
        return out


def rank_passages(question: str, passages: list[Passage], titles: dict[str, str] | None = None,
                  top_k: int = 8, per_page: int = 3, boost_pages: dict[str, float] | None = None
                  ) -> list[tuple[Passage, float]]:
    """Return the ``top_k`` passages with scores, at most ``per_page`` from one page."""
    if not passages:
        return []
    return PassageIndex(passages, titles).search(question, top_k, per_page, boost_pages)


_PATH_SPLIT = re.compile(r"[/_\-.?=&#]+")


def link_tokens(link: Link) -> list[str]:
    path = re.sub(r"^[a-z]+://[^/]+", "", link.ref.lower())
    path = re.sub(r"\.(html?|php|aspx?|md)$", "", path)
    return tokens(link.text) + tokens(" ".join(_PATH_SPLIT.split(path)))


def score_link(question: str, link: Link, intent: str | None = None, q: Counter | None = None,
               special: set[str] | None = None) -> float:
    if q is None:
        q, special = expand_query(question)
    special = special or set()
    lt = link_tokens(link)
    if not lt:
        return 0.0
    lset = set(lt)
    s = sum(w for t, w in q.items() if t in lset and t not in INTENT_WORDS)
    s += 0.6 * sum(w for t, w in q.items() if t in lset and t in INTENT_WORDS)
    s += 1.2 * sum(1 for t in special if t in lset)
    if intent and intent in INTENT_LINK_HINTS:
        hints = set(stem(h) for h in INTENT_LINK_HINTS[intent].split())
        if lset & hints:
            s += 1.2
    # prefer shallow, content-like pages a little; penalise obvious utility pages
    if re.search(r"(login|signin|sign-in|register|signup|cart|checkout|privacy|cookie|terms-of|legal|"
                 r"/tag/|/tags/|/author/|/page/\d|lang=|/feed)", link.ref.lower()):
        s -= 1.5
    return s


def rank_links(question: str, links: list[Link], intent: str | None = None, exclude: set[str] | None = None,
               top_k: int = 10) -> list[tuple[Link, float]]:
    exclude = exclude or set()
    q, special = expand_query(question)
    seen, out = set(), []
    for l in links:
        if l.ref in exclude or l.ref in seen:
            continue
        seen.add(l.ref)
        out.append((l, score_link(question, l, intent, q, special)))
    out.sort(key=lambda x: -x[1])
    return out[:top_k]


def page_text_tokens(text: str) -> set[str]:
    return set(tokens(text))


def contains_terms(text: str, terms) -> bool:
    toks = set(tokens(text))
    folded = fold(text)
    return all(t in toks or t in folded for t in terms)
