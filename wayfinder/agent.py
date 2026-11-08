"""The Wayfinder agent: crawl -> clean -> rank -> (model decides answer | follow) -> verify -> cite.

Every step reports to an optional ``trace`` callback (the Studio renders it
live) and the best answer known so far is pushed to ``on_update`` so the
caller can persist a provisional result before the deadline.
"""

from __future__ import annotations

import heapq
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from .backends.base import Backend, RulesBackend
from .document import Link, Page, Passage, chunk_page
from .extractors import IDENT, Candidate, acronym_in, shape_value, best_rule_answer, extract_candidates, gate, grounded, question_intent
from .fetcher import HttpSite, LocalSite, make_site
from .prompts import build_messages, parse_decision
from .ranker import PassageIndex, rank_links, score_link
from .schema import Result, Task
from .textutil import clean_answer, fold, special_terms, tokens

HEADING_Q = re.compile(r"\b(main heading|page heading|headline|page title|title of (?:this|the) page|"
                       r"heading of (?:this|the) page)\b", re.I)
_URL_IN_TEXT = re.compile(r"https?://[^\s\"'<>]+")
_local_index_cache: dict[tuple[str, int], tuple[list, PassageIndex]] = {}
_cache_lock = threading.Lock()


def _overlap_count(text: str, question: str) -> int:
    return len(set(tokens(text)) & set(tokens(question)))


def _noop(*_a, **_k):
    return None


class Agent:
    def __init__(self, backend: Backend | None = None, max_pages: int = 12, max_hops: int = 3,
                 top_k: int = 8, workers: int = 4):
        self.backend = backend or RulesBackend()
        self.max_pages = max_pages
        self.max_hops = max_hops
        self.top_k = top_k
        self.workers = workers

    # ================================================================== public
    def run(self, task: Task, deadline: float | None = None, on_update=None, trace=None) -> Result:
        t0 = time.monotonic()
        deadline = deadline or (t0 + 25.0)
        emit = trace or _noop
        update = on_update or _noop
        try:
            return self._run(task, t0, deadline, emit, update)
        except Exception as e:  # the agent must never take the caller down
            emit("error", message=f"{type(e).__name__}: {e}")
            return Result(method="error")

    # ================================================================== internals
    def _run(self, task: Task, t0: float, deadline: float, emit, update) -> Result:
        question = (task.question or "").strip()
        url = task.url
        if not url and not task.corpus:
            m = _URL_IN_TEXT.search(question)
            if m:
                url = m.group(0).rstrip(".,;)")
                rest = (question[: m.start()] + question[m.end():]).strip()
                rest = re.sub(r"^(?:on|at|from|in|using|see|visit)\s*[,:]?\s*", "", rest, flags=re.I).lstrip(",:; ")
                question = rest[:1].upper() + rest[1:] if rest else question
        emit("start", question=question, url=url or "", corpus=task.corpus or "", backend=self.backend.name,
             stub=self.backend.is_stub)
        if not question:
            emit("final", result=Result(method="no-question").to_dict(), elapsed=0.0)
            return Result(method="no-question")
        model_reserve = 9.0 if self.backend.uses_model else 1.0
        site = self._open_site(url, task.corpus, deadline)
        if site is None:
            res = Result(method="no-source")
            emit("final", result=res.to_dict(), elapsed=round(time.monotonic() - t0, 3),
                 note="no reachable website or saved copy; refusing rather than answering from memory")
            return res
        intent = question_intent(question)
        emit("plan", intent=intent or "lookup", mode=site.kind, special=special_terms(question))

        state = _State(site=site, question=question, intent=intent)
        if isinstance(site, LocalSite):
            self._gather_local(state, emit)
        else:
            crawl_deadline = min(deadline - model_reserve, t0 + 14.0)
            self._crawl(state, crawl_deadline, emit)
        emit("crawl_done", pages=len(state.pool), elapsed=round(time.monotonic() - t0, 3))
        if not state.pool:
            res = Result(method="unreachable")
            emit("final", result=res.to_dict(), elapsed=round(time.monotonic() - t0, 3),
                 note=getattr(site, "last_error", "") or "nothing could be fetched")
            return res

        ranked = state.rank(self.top_k)
        self._emit_rank(emit, state, ranked)

        # ---- deterministic answer first (persisted as a provisional result)
        rule = self._rules_answer(state, ranked, deadline, emit)
        best = self._result_from_candidate(state, rule) if rule else Result(method="rules-refuse")
        best.pages_visited = len(state.visited)
        update(best)

        if self.backend.uses_model and time.monotonic() < deadline - 2.0:
            model_res = self._model_loop(state, ranked, rule, deadline, emit)
            if model_res is not None:
                best = model_res
        best.pages_visited = len(state.visited)
        best.answer = clean_answer(best.answer)
        if not best.answer:
            best.citations, best.source_url, best.confidence = [], "", 0.0
        emit("final", result=best.to_dict(), elapsed=round(time.monotonic() - t0, 3),
             evidence=best.evidence, visited=list(state.visited))
        return best

    # ------------------------------------------------------------------ sites
    def _open_site(self, url, corpus, deadline):
        site = make_site(url, corpus, deadline=deadline - 2.0)
        return site

    def _gather_local(self, state: "_State", emit):
        site: LocalSite = state.site  # type: ignore[assignment]
        key = (str(site.root.resolve()), id(site.page_list))
        idx = None
        with _cache_lock:
            hit = _local_index_cache.get(key)
            if hit and hit[0] is site.page_list:
                idx = hit[1]
        for ref in site.all_refs():
            pg = site.fetch(ref)
            if pg is not None:
                state.pool[ref] = pg
        if idx is None:
            passages = []
            for ref, pg in state.pool.items():
                passages += chunk_page(pg)
            idx = PassageIndex(passages, {r: p.title for r, p in state.pool.items()})
            with _cache_lock:
                if len(_local_index_cache) > 8:
                    _local_index_cache.clear()
                _local_index_cache[key] = (site.page_list, idx)
        state.index = idx
        start = site.start_refs()
        emit("fetch", ref=start[0] if start else "", url=site.public_url(start[0]) if start else "",
             title=state.pool[start[0]].title if start and start[0] in state.pool else "",
             status="ok", source="saved copy", pages=len(state.pool), ms=0)
        if start:
            state.visited.append(start[0])

    def _crawl(self, state: "_State", crawl_deadline: float, emit):
        site: HttpSite = state.site  # type: ignore[assignment]
        q = state.question
        heap: list[tuple[float, int, int, str, str, str]] = []
        counter = 0
        seen: set[str] = set()

        def push(links: list[Link], depth: int, parent_rel: float, parent: str):
            nonlocal counter
            for l in links:
                if l.ref in seen:
                    continue
                s = score_link(q, l, state.intent) + 0.25 * parent_rel - 0.35 * depth
                counter += 1
                heapq.heappush(heap, (-s, depth, counter, l.ref, l.text, parent))

        def do_fetch(ref):
            t = time.monotonic()
            pg = site.fetch(ref, deadline=crawl_deadline)
            return ref, pg, int((time.monotonic() - t) * 1000)

        start = site.start_refs()[0]
        seen.add(start)
        ref, pg, ms = do_fetch(start)
        emit("fetch", ref=start, url=start, title=pg.title if pg else "", status="ok" if pg else "error",
             error="" if pg else getattr(site, "last_error", ""), ms=ms, depth=0,
             links=len(pg.links) if pg else 0)
        if pg is None:
            return
        state.add_page(pg)
        state.visited.append(pg.ref)
        seen.add(pg.ref)
        push(pg.links, 1, state.page_relevance(pg), pg.ref)
        if heap:
            top = sorted(heap)[:6]
            emit("links", from_ref=pg.ref, ranked=[{"text": t[4], "ref": t[3], "score": round(-t[0], 2)}
                                                   for t in top])
        ex = ThreadPoolExecutor(max_workers=self.workers)
        try:
            inflight = {}
            while (heap or inflight) and len(state.pool) + len(inflight) <= self.max_pages:
                now = time.monotonic()
                if now >= crawl_deadline:
                    break
                while heap and len(inflight) < self.workers and len(state.pool) + len(inflight) < self.max_pages:
                    negs, depth, _, ref, text, parent = heapq.heappop(heap)
                    if ref in seen or depth > 3:
                        continue
                    if -negs <= -0.5 and len(state.pool) >= 6:
                        heap.clear()
                        break
                    seen.add(ref)
                    fut = ex.submit(do_fetch, ref)
                    inflight[fut] = (ref, depth, text, -negs, parent)
                if not inflight:
                    break
                done, _ = wait(list(inflight), timeout=max(0.05, crawl_deadline - time.monotonic()),
                               return_when=FIRST_COMPLETED)
                if not done:
                    break
                for fut in done:
                    ref, depth, text, score, parent = inflight.pop(fut)
                    try:
                        _, pg, ms = fut.result()
                    except Exception:
                        pg, ms = None, 0
                    emit("fetch", ref=ref, url=ref, title=pg.title if pg else "", status="ok" if pg else "error",
                         error="" if pg else "fetch failed", ms=ms, depth=depth, via=text, score=round(score, 2),
                         parent=parent, links=len(pg.links) if pg else 0)
                    if pg is None or pg.ref in state.pool:
                        continue
                    state.add_page(pg)
                    state.visited.append(pg.ref)
                    seen.add(pg.ref)
                    push(pg.links, depth + 1, state.page_relevance(pg), pg.ref)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    def _fetch_more(self, state: "_State", refs: list[str], deadline: float, emit, via: str = "") -> list[str]:
        """Open specific refs (model 'follow' / rules bridge). Returns refs that are now in the pool."""
        got = []
        site = state.site
        if isinstance(site, LocalSite):
            for r in refs:
                if r in state.pool:
                    got.append(r)
                    if r not in state.visited:
                        state.visited.append(r)
                    emit("fetch", ref=r, url=site.public_url(r), title=state.pool[r].title, status="ok",
                         source="saved copy", ms=0, via=via)
            return got
        todo = [r for r in refs if r not in state.pool]
        got = [r for r in refs if r in state.pool]
        if not todo:
            return got
        ex = ThreadPoolExecutor(max_workers=max(1, len(todo)))
        try:
            futs = {ex.submit(site.fetch, r, deadline): r for r in todo}
            done, _ = wait(list(futs), timeout=max(0.1, deadline - time.monotonic()))
            for f in done:
                r = futs[f]
                try:
                    pg = f.result()
                except Exception:
                    pg = None
                emit("fetch", ref=r, url=r, title=pg.title if pg else "", status="ok" if pg else "error", ms=0, via=via)
                if pg is not None:
                    state.add_page(pg)
                    state.visited.append(pg.ref)
                    got.append(pg.ref)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        return got

    # ------------------------------------------------------------------ rules
    def _rules_answer(self, state: "_State", ranked, deadline, emit) -> Candidate | None:
        q = state.question
        if HEADING_Q.search(q) and state.visited:
            pg = state.pool.get(state.visited[0])
            if pg is not None:
                h1 = next((t for k, t in pg.lines if k == "h1"), "") or pg.title
                if h1:
                    c = Candidate(h1, pg.ref, "", 5.0, "heading", h1)
                    emit("rules_pick", candidate=c.to_dict())
                    return c
        cands = extract_candidates(q, ranked, state.pool, state.intent)
        emit("rules", candidates=[c.to_dict() for c in cands[:5]])
        for c in cands[:6]:
            if gate(q, c, state.pool):
                emit("rules_pick", candidate=c.to_dict())
                return c
        bridged = self._rules_bridge(state, ranked, deadline, emit) or self._ident_bridge(state, ranked, emit)
        if bridged:
            emit("rules_pick", candidate=bridged.to_dict(), bridge=bridged.support)
        return bridged

    def _rules_bridge(self, state: "_State", ranked, deadline, emit) -> Candidate | None:
        """Two-hop rules: find the page that matches the question's constraint, take the
        entity it links to, and look the asked-for value up for that entity."""
        q = state.question
        m = re.search(r"\b(that|which|whose|who)\b\s+(.*)$", q, re.I)
        if not m or m.start() < 8:
            return None
        main = q[: m.start()].strip()
        clause = m.group(2).rstrip("?.! ")
        clause_special = set(special_terms(clause)) or set(tokens(clause))
        best_bridge = None
        for p, s in ranked[:6]:
            ptoks = set(tokens(p.text + " " + p.heading))
            cover = len(clause_special & ptoks) / max(1, len(clause_special))
            if cover >= 0.6 and (best_bridge is None or cover > best_bridge[1]):
                best_bridge = (p, cover)
        if not best_bridge:
            return None
        bp = best_bridge[0]
        page = state.pool.get(bp.page_ref)
        if page is None:
            return None
        # entity = a link in/near the bridge passage whose anchor text appears in it
        ent_links = [l for l in page.links if l.text and len(l.text) > 2 and fold(l.text) in fold(bp.text)
                     and l.ref != page.ref]
        ent_links.sort(key=lambda l: -score_link(main, l, state.intent))
        for l in ent_links[:2]:
            if time.monotonic() > deadline - 1.0:
                break
            got = self._fetch_more(state, [l.ref], min(deadline - 1.0, time.monotonic() + 4.0), emit, via=l.text)
            if not got:
                continue
            target = got[0]
            q2 = f"{main} {l.text}"
            ranked2 = state.rank(self.top_k, question=q2, only_pages={target})
            cands = extract_candidates(q2, ranked2, state.pool, state.intent)
            for c in cands[:4]:
                c.support = [bp.page_ref] if bp.page_ref != c.page_ref else []
                if gate(q2, c, state.pool) and gate(q, c, state.pool, min_score=0.0):
                    c.kind = c.kind + "+bridge"
                    return c
        return None

    def _ident_bridge(self, state: "_State", ranked, emit) -> Candidate | None:
        """Two-hop by identifier: page A (matching the question) mentions an id such as BUG-4471
        or a part number; page B holds the asked-for value for that id. Cite both."""
        q = state.question
        qf = fold(q)
        seen: set[str] = set()
        for p, _ in ranked[:4]:
            for line in p.text.splitlines():
                if _overlap_count(line, q) < 1:
                    continue
                for m in IDENT.finditer(line):
                    ident = m.group(0)
                    if fold(ident) in qf or ident in seen:
                        continue
                    seen.add(ident)
                    others = {r for r, pg in state.pool.items()
                              if r != p.page_ref and ident.lower() in (pg.text + " " + pg.title).lower()}
                    if not others:
                        continue
                    q2 = f"{q} {ident}"
                    ranked2 = state.rank(self.top_k, question=q2, only_pages=others)
                    for c in extract_candidates(q2, ranked2, state.pool, state.intent)[:4]:
                        if fold(ident) not in fold(c.line) or fold(c.value) == fold(ident):
                            continue
                        c.support = [p.page_ref]
                        if gate(q, c, state.pool, min_score=1.0):
                            c.kind = c.kind + "+id-bridge"
                            return c
                    if len(seen) >= 6:
                        return None
        return None

    def _result_from_candidate(self, state: "_State", c: Candidate) -> Result:
        cites = [c.page_ref] + [s for s in c.support if s != c.page_ref]
        conf = 0.45 + min(0.4, c.score / 10.0)
        return Result(answer=clean_answer(c.value), citations=[state.site.cite(r) for r in cites],
                      source_url=state.site.public_url(c.page_ref), confidence=round(conf, 3),
                      method=f"rules:{c.kind}", evidence=c.line)

    # ------------------------------------------------------------------ model loop
    def _model_loop(self, state: "_State", ranked, rule: Candidate | None, deadline, emit) -> Result | None:
        q = state.question
        boosts: dict[str, float] = {}
        if rule is not None and rule.support:
            # a two-hop rules answer: make sure both hops are in front of the model
            boosts[rule.page_ref] = 3.0
            for r in rule.support:
                boosts[r] = boosts.get(r, 0.0) + 1.0
        note = ""
        followed_from: dict[str, str] = {}
        for hop in range(self.max_hops):
            remaining = deadline - time.monotonic() - 0.8
            if remaining < 1.5:
                emit("decision", hop=hop, action="timeout", raw="")
                return None
            ranked = state.rank(self.top_k, boosts=boosts)
            labels: dict[str, Passage] = {}
            evidence = []
            for i, (p, s) in enumerate(ranked):
                lab = f"P{i + 1}"
                labels[lab] = p
                pg = state.pool.get(p.page_ref)
                evidence.append({"id": lab, "page": state.site.cite(p.page_ref), "title": pg.title if pg else "",
                                 "heading": p.heading, "text": p.text})
            ev_pages = []
            for p, _ in ranked:
                if p.page_ref not in ev_pages:
                    ev_pages.append(p.page_ref)
            links = self._candidate_links(state, ev_pages)
            final = hop == self.max_hops - 1 or remaining < 5.0 or not links
            link_items = [{"id": f"L{i + 1}", "text": l.text, "ref": state.site.cite(l.ref), "_ref": l.ref}
                          for i, (l, _) in enumerate(links)]
            hints = []
            for c in extract_candidates(q, ranked, state.pool, state.intent)[:3]:
                lab = next((k for k, v in labels.items() if v.pid == c.pid or (not c.pid and v.page_ref == c.page_ref)), None)
                if lab:
                    hints.append({"value": c.value, "id": lab})
            msgs = build_messages(q, evidence, link_items, final=final, note=note, hints=hints,
                                  site=getattr(state.site, "start_url", "") or "")
            meta = {"question": q, "evidence": evidence, "links": link_items, "final": final}
            if rule:
                lab = next((k for k, v in labels.items() if v.page_ref == rule.page_ref and (not rule.pid or v.pid == rule.pid)),
                           None) or next((k for k, v in labels.items() if v.page_ref == rule.page_ref), None)
                meta["rule_candidate"] = {"value": rule.value, "pid_label": lab}
            emit("think", hop=hop, passages=len(evidence), links=len(link_items), final=final,
                 prompt_chars=sum(len(m["content"]) for m in msgs))
            t = time.monotonic()
            try:
                raw = self.backend.generate(msgs, max_new_tokens=96, timeout=max(1.0, remaining - 0.5), meta=meta)
            except Exception as e:
                emit("decision", hop=hop, action="error", raw=f"{type(e).__name__}: {e}")
                return None
            gen_ms = int((time.monotonic() - t) * 1000)
            if time.monotonic() > deadline:
                emit("decision", hop=hop, action="timeout", raw=raw[:300], ms=gen_ms)
                return None
            d = parse_decision(raw)
            if d is None:
                emit("decision", hop=hop, action="unparsed", raw=raw[:300], ms=gen_ms)
                return None
            emit("decision", hop=hop, action=d["action"], answer=d.get("answer", ""), sources=d.get("sources", []),
                 links=[next((li["ref"] for li in link_items if li["id"] == x), x) for x in d.get("links", [])],
                 raw=raw[:300], ms=gen_ms)
            if d["action"] == "none":
                if final or not links:
                    return Result(method="model:none", hops=hop + 1)
                note = "If the answer may be behind one of the links, follow it; otherwise reply none."
                # one more chance with an explicit nudge, then accept the refusal
                if hop >= 1:
                    return Result(method="model:none", hops=hop + 1)
                continue
            if d["action"] == "follow":
                refs = [li["_ref"] for li in link_items if li["id"] in d["links"]][:2]
                if not refs:
                    note = "Use only link ids from the list."
                    continue
                emit("follow", hop=hop, refs=[state.site.cite(r) for r in refs])
                got = self._fetch_more(state, refs, min(deadline - 2.5, time.monotonic() + 5.0), emit, via="model")
                for r in got:
                    boosts[r] = boosts.get(r, 0.0) + 4.0
                    for p_ref in ev_pages:
                        if any(l.ref == r for l in state.pool[p_ref].links):
                            followed_from[r] = p_ref
                            break
                continue
            # ---- answer
            ans = shape_value(q, state.intent, clean_answer(d["answer"]))
            res = self._verify(state, ans, [labels[s] for s in d["sources"] if s in labels], ranked,
                               followed_from, emit, rule)
            if res is not None:
                res.hops = hop + 1
                return res
            note = (f"Your previous answer \"{ans}\" does not appear in the evidence. Copy the value exactly as "
                    f"written, or reply none.")
        return None

    def _candidate_links(self, state: "_State", ev_pages: list[str]) -> list[tuple[Link, float]]:
        pool_links: list[Link] = []
        site = state.site
        shown = set(ev_pages)
        sources = ev_pages[:5] + [r for r in state.visited[:1] if r not in ev_pages]
        for ref in sources:
            pg = state.pool.get(ref)
            if pg:
                pool_links += pg.links
        if isinstance(site, LocalSite):
            exclude = shown
        else:
            exclude = set(state.pool)
        ranked = rank_links(state.question, pool_links, state.intent, exclude=exclude, top_k=8)
        return [(l, s) for l, s in ranked]

    def _verify(self, state: "_State", ans: str, cited: list[Passage], ranked, followed_from, emit,
                rule: Candidate | None = None) -> Result | None:
        if not ans:
            return None
        answer_page = None
        for p in cited:
            if grounded(ans, p.heading + "\n" + p.text):
                answer_page = p.page_ref
                break
        if answer_page is None:
            for p in cited:
                pg = state.pool.get(p.page_ref)
                if pg and grounded(ans, pg.title + "\n" + pg.text):
                    answer_page = p.page_ref
                    break
        if answer_page is None:
            for p, _ in ranked:
                if grounded(ans, p.heading + "\n" + p.text):
                    answer_page = p.page_ref
                    break
        if answer_page is None:
            emit("verify", answer=ans, grounded=False)
            return None
        cites = [answer_page]
        q = state.question
        special = set(special_terms(q))
        content = set(t for t in tokens(q))
        ans_pg = state.pool[answer_page]
        ans_toks = set(tokens(ans_pg.title + " " + ans_pg.text))
        ans_text = ans_pg.title + " " + ans_pg.text
        missing_special = {t for t in special if t not in ans_toks and t not in fold(ans_text)
                           and not acronym_in(ans_text, t)}
        missing_content = {t for t in content if t not in ans_toks}
        bridge_cands = [p.page_ref for p in cited if p.page_ref != answer_page]
        if answer_page in followed_from:
            bridge_cands.append(followed_from[answer_page])
        if rule is not None and rule.page_ref == answer_page and grounded(ans, rule.value):
            bridge_cands += rule.support
        if missing_special:
            # necessity: the answer page alone does not mention what the question is about;
            # the evidence page that supplies the most missing key terms is part of the chain
            best, best_n = None, 0
            for p, _ in ranked:
                if p.page_ref == answer_page or p.page_ref not in state.pool:
                    continue
                pt = set(tokens(state.pool[p.page_ref].title + " " + p.heading + " " + p.text))
                n = len(missing_special & pt)
                if n > best_n:
                    best, best_n = p.page_ref, n
            if best and best_n >= max(1, len(missing_special) // 2):
                bridge_cands.append(best)
        for ref in dict.fromkeys(bridge_cands):
            if ref == answer_page or ref not in state.pool:
                continue
            pg = state.pool[ref]
            bt = set(tokens(pg.title + " " + pg.text))
            if (missing_special & bt) or len(missing_content & bt) >= 2:
                cites.append(ref)
                missing_special -= bt
                missing_content -= bt
            if len(cites) >= 3:
                break
        line = ""
        for p in cited + [p for p, _ in ranked]:
            full = (p.heading.split(" > ")[-1] + "\n" if p.heading else "") + p.text
            if p.page_ref == answer_page and grounded(ans, full):
                line = next((l for l in full.splitlines() if grounded(ans, l)), full[:300])
                break
        emit("verify", answer=ans, grounded=True, page=state.site.cite(answer_page),
             citations=[state.site.cite(c) for c in cites])
        return Result(answer=ans, citations=[state.site.cite(c) for c in cites],
                      source_url=state.site.public_url(answer_page), confidence=0.85 if len(cites) == 1 else 0.75,
                      method="model", evidence=line)

    def _emit_rank(self, emit, state, ranked):
        emit("rank", passages=[{"pid": p.pid, "page": state.site.cite(p.page_ref),
                                "title": state.pool[p.page_ref].title if p.page_ref in state.pool else "",
                                "heading": p.heading, "score": round(s, 2), "text": p.text[:400]}
                               for p, s in ranked])


class _State:
    def __init__(self, site, question: str, intent: str | None):
        self.site = site
        self.question = question
        self.intent = intent
        self.pool: dict[str, Page] = {}
        self.visited: list[str] = []
        self.passages: list[Passage] = []
        self.index: PassageIndex | None = None
        self._dirty = True

    def add_page(self, pg: Page):
        if pg.ref in self.pool:
            return
        self.pool[pg.ref] = pg
        self.passages += chunk_page(pg)
        self._dirty = True

    def rank(self, top_k: int, question: str | None = None, boosts: dict | None = None,
             only_pages: set[str] | None = None):
        if isinstance(self.site, HttpSite) and (self._dirty or self.index is None):
            self.index = PassageIndex(self.passages, {r: p.title for r, p in self.pool.items()})
            self._dirty = False
        if self.index is None:
            return []
        return self.index.search(question or self.question, top_k=top_k, per_page=3, boost_pages=boosts,
                                 only_pages=only_pages)

    def page_relevance(self, pg: Page) -> float:
        qt = set(tokens(self.question))
        if not qt:
            return 0.0
        pt = set(tokens(pg.title + " " + pg.text[:4000]))
        return len(qt & pt) / len(qt)
