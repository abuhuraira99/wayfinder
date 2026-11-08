/* Wayfinder Studio — vanilla JS. All page-derived text goes through textContent. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  };
  const fmtS = (s) => (s == null || isNaN(s) ? "–" : `${Number(s).toFixed(2)} s`);
  let INFO = {};

  /* ------------------------------------------------------------ theme */
  const root = document.documentElement;
  const saved = localStorage.getItem("wf-theme");
  if (saved) root.dataset.theme = saved;
  else if (matchMedia("(prefers-color-scheme: light)").matches) root.dataset.theme = "light";
  $("#theme").addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    localStorage.setItem("wf-theme", root.dataset.theme);
  });

  /* ------------------------------------------------------------ mode switch */
  const ink = $(".seg-ink");
  function placeInk() {
    const a = $(".seg-btn.active");
    if (!a) return;
    ink.style.width = a.offsetWidth + "px";
    ink.style.transform = `translateX(${a.offsetLeft - 4}px)`;
  }
  $$(".seg-btn").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  function setMode(mode) {
    $$(".seg-btn").forEach((x) => {
      const on = x.dataset.mode === mode;
      x.classList.toggle("active", on);
      x.setAttribute("aria-selected", on);
    });
    $("#view-live").hidden = mode !== "live";
    $("#view-report").hidden = mode !== "report";
    placeInk();
    history.replaceState(null, "", mode === "report" ? "#report" : "#");
    if (mode === "report") loadReportList();
  }
  addEventListener("resize", placeInk);

  /* ------------------------------------------------------------ info */
  fetch("/api/info").then((r) => r.json()).then((info) => {
    INFO = info;
    const pill = $("#backend-pill");
    pill.textContent = info.label + (info.model ? ` · ${info.model}` : "");
    pill.classList.toggle("stub", !!info.stub);
    pill.classList.toggle("rules", info.backend === "rules");
    $("#stub-banner").hidden = !info.stub;
    $("#url").value = info.demo_url;
    const box = $("#examples");
    (info.examples || []).slice(0, 9).forEach((ex) => {
      const c = el("button", "chip", ex.question);
      c.type = "button";
      c.title = `${ex.kind || ""} — start: ${ex.url}`;
      c.addEventListener("click", () => {
        $("#question").value = ex.question;
        $("#url").value = ex.url;
        $$(".chip", box).forEach((x) => x.classList.remove("active"));
        c.classList.add("active");
        $("#question").focus();
      });
      box.appendChild(c);
    });
    placeInk();
  }).catch(() => { $("#backend-pill").textContent = "offline"; });
  $("#use-demo").addEventListener("click", () => { if (INFO.demo_url) $("#url").value = INFO.demo_url; });
  if (location.hash === "#report") setMode("report"); else requestAnimationFrame(placeInk);

  /* ------------------------------------------------------------ live run */
  let es = null;
  let state = {};
  $("#ask").addEventListener("submit", (e) => { e.preventDefault(); run(); });
  $("#question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); }
  });

  function resetLive() {
    state = { pages: [], steps: 0, t: 0 };
    $("#timeline").replaceChildren();
    $("#pages").replaceChildren(el("li", "muted", "—"));
    $("#passages").replaceChildren(el("li", "muted", "—"));
    const card = $("#answer-card");
    card.className = "card answer idle";
    $("#answer").textContent = "Searching…";
    $("#source").hidden = true;
    $("#citations").replaceChildren(el("li", "muted", "—"));
    $("#evidence").className = "evidence muted";
    $("#evidence").textContent = "…";
    $("#method").textContent = "";
    setRing(null);
    stats();
  }

  function stats() {
    $("#st-pages").textContent = `${state.pages.length} page${state.pages.length === 1 ? "" : "s"}`;
    $("#st-steps").textContent = `${state.steps} model step${state.steps === 1 ? "" : "s"}`;
    $("#st-time").textContent = `${(state.t || 0).toFixed(1)} s`;
  }

  function run() {
    const question = $("#question").value.trim();
    if (!question) { $("#question").focus(); return; }
    if (es) es.close();
    resetLive();
    const btn = $("#run");
    btn.classList.add("busy");
    $(".run-label", btn).textContent = "Browsing…";
    const qs = new URLSearchParams({ question, url: $("#url").value.trim(), corpus: $("#corpus").value.trim() });
    es = new EventSource(`/api/run?${qs}`);
    es.addEventListener("trace", (m) => { try { onTrace(JSON.parse(m.data)); } catch (err) { console.error(err); } });
    const end = () => { btn.classList.remove("busy"); $(".run-label", btn).textContent = "Find answer"; if (es) es.close(); es = null; };
    es.addEventListener("done", end);
    es.onerror = end;
  }

  function step(kind, title, cls, t) {
    const li = el("li", `step ${cls || ""}`);
    li.appendChild(el("span", "dot"));
    const head = el("div", "step-title");
    head.appendChild(el("span", "step-kind", kind));
    head.appendChild(el("span", "", title));
    head.appendChild(el("span", "step-time", t != null ? `+${Number(t).toFixed(2)}s` : ""));
    li.appendChild(head);
    const body = el("div", "step-body");
    li.appendChild(body);
    $("#timeline").appendChild(li);
    li.scrollIntoView({ block: "nearest", behavior: "smooth" });
    return body;
  }

  function addPage(title, url, via) {
    if (state.pages.some((p) => p.url === url)) return;
    state.pages.push({ title, url });
    const list = $("#pages");
    if (list.firstChild && list.firstChild.classList.contains("muted")) list.replaceChildren();
    const li = el("li");
    const d = el("div");
    d.appendChild(el("div", "pt", title || "(untitled)"));
    d.appendChild(el("div", "pu", url + (via ? `  ← "${via}"` : "")));
    li.appendChild(d);
    list.appendChild(li);
  }

  function onTrace(ev) {
    state.t = ev.t || state.t;
    switch (ev.type) {
      case "start": {
        const b = step("start", ev.question, "", ev.t);
        b.appendChild(el("div", "url", ev.corpus ? `saved copy: ${ev.corpus}` : ev.url || "(no URL)"));
        if (ev.stub) b.appendChild(el("span", "badge bad", "TEST STUB"));
        break;
      }
      case "plan": {
        const b = step("plan", `Looking for a ${ev.intent} value`, "", ev.t);
        b.textContent = `${ev.mode === "local" ? "Reading a saved copy of the site" : "Crawling the live site"}` +
          (ev.special && ev.special.length ? ` · key terms: ${ev.special.join(", ")}` : "");
        break;
      }
      case "fetch": {
        const ok = ev.status === "ok";
        const b = step("open", ev.title || ev.ref, ok ? "ok" : "err", ev.t);
        b.appendChild(el("div", "url", ev.url || ev.ref));
        const meta = [];
        if (ev.via) meta.push(`via “${ev.via}”`);
        if (ev.ms) meta.push(`${ev.ms} ms`);
        if (ev.links) meta.push(`${ev.links} links`);
        if (ev.pages) meta.push(`${ev.pages} pages in the saved copy`);
        if (!ok && ev.error) meta.push(ev.error);
        if (meta.length) b.appendChild(el("div", "", meta.join(" · ")));
        if (ok) addPage(ev.title, ev.url || ev.ref, ev.via);
        break;
      }
      case "links": {
        const b = step("links", "Ranked next links", "", ev.t);
        const box = el("div", "box");
        const max = Math.max(1, ...ev.ranked.map((l) => l.score));
        ev.ranked.forEach((l) => {
          const r = el("div", "linkrow");
          const t = el("div", "lt", l.text || l.ref);
          t.title = l.ref;
          r.appendChild(t);
          const bar = el("div", "bar"); const i = el("i"); i.style.width = `${Math.max(4, (100 * Math.max(0, l.score)) / max)}%`;
          bar.appendChild(i); r.appendChild(bar); box.appendChild(r);
        });
        b.appendChild(box);
        break;
      }
      case "crawl_done":
        step("read", `${ev.pages} page${ev.pages === 1 ? "" : "s"} cleaned and split into passages`, "ok", ev.t);
        break;
      case "rank": {
        const list = $("#passages");
        list.replaceChildren();
        (ev.passages || []).slice(0, 6).forEach((p, i) => {
          const li = el("li");
          const h = el("div", "ph");
          h.appendChild(el("span", "", `P${i + 1} · ${p.page}`));
          h.appendChild(el("span", "", p.score));
          li.appendChild(h);
          li.appendChild(el("div", "", p.text.slice(0, 220) + (p.text.length > 220 ? "…" : "")));
          list.appendChild(li);
        });
        if (!list.children.length) list.appendChild(el("li", "muted", "No passage matched the question"));
        const b = step("rank", `Top ${ev.passages.length} passages by BM25 + exact-term boosts`, "", ev.t);
        if (ev.passages[0]) b.textContent = `best: ${ev.passages[0].page} — ${ev.passages[0].heading || ev.passages[0].title || ""}`;
        break;
      }
      case "rules": {
        if (!ev.candidates || !ev.candidates.length) break;
        const b = step("extract", "Pattern and table candidates", "", ev.t);
        const box = el("div", "box");
        ev.candidates.slice(0, 4).forEach((c) => {
          const r = el("div", "linkrow");
          r.appendChild(el("div", "lt", `${c.value}  ·  ${c.page}  ·  ${c.kind}`));
          r.appendChild(el("span", "badge", c.score));
          box.appendChild(r);
        });
        b.appendChild(box);
        break;
      }
      case "rules_pick":
        step("fallback", `Rules answer ready: ${ev.candidate.value}`, "ok", ev.t)
          .textContent = "Saved as the provisional answer in case the model runs out of time.";
        break;
      case "think": {
        state.steps += 1;
        const b = step("model", `Step ${ev.hop + 1}: reading ${ev.passages} passages, ${ev.links} links${ev.final ? " (last step)" : ""}`, "model", ev.t);
        b.textContent = `${ev.prompt_chars.toLocaleString()} prompt characters`;
        break;
      }
      case "decision": {
        const label = { answer: `Answer → ${ev.answer}`, follow: "Follow links", none: "Not on this site", timeout: "Out of time", unparsed: "Unreadable reply", error: "Model error" }[ev.action] || ev.action;
        const b = step("decide", label, ev.action === "answer" ? "model" : ev.action === "follow" ? "model" : "err", ev.t);
        const box = el("div", "box bubble");
        if (ev.action === "follow") (ev.links || []).forEach((l) => box.appendChild(el("div", "url", "→ " + l)));
        if (ev.action === "answer") box.appendChild(el("div", "", `sources: ${(ev.sources || []).join(", ") || "—"}`));
        if (ev.raw) box.appendChild(el("div", "raw", ev.raw));
        if (ev.ms) box.appendChild(el("div", "raw", `${ev.ms} ms`));
        b.appendChild(box);
        break;
      }
      case "follow":
        break;
      case "verify": {
        const b = step("verify", ev.grounded ? "Answer found verbatim in the cited page" : `“${ev.answer}” is not in the evidence — rejected`, ev.grounded ? "ok" : "err", ev.t);
        if (ev.grounded) b.textContent = `cites: ${(ev.citations || []).join(", ")}`;
        break;
      }
      case "error":
        step("error", ev.message, "err", ev.t);
        break;
      case "final":
        finalCard(ev);
        break;
    }
    stats();
  }

  function setRing(v) {
    const fg = $("#ring-fg");
    const C = 119.4;
    fg.style.strokeDashoffset = v == null ? C : C * (1 - Math.max(0, Math.min(1, v)));
    $("#conf").textContent = v == null ? "–" : `${Math.round(v * 100)}`;
  }

  function highlight(target, text, needle) {
    target.replaceChildren();
    if (!text) { target.textContent = "—"; return; }
    const i = needle ? text.toLowerCase().indexOf(needle.toLowerCase()) : -1;
    if (i < 0) { target.textContent = text; return; }
    target.append(text.slice(0, i), el("mark", "", text.slice(i, i + needle.length)), text.slice(i + needle.length));
  }

  function finalCard(ev) {
    const r = ev.result || {};
    const card = $("#answer-card");
    const has = !!r.answer;
    card.className = `card answer ${has ? "done" : "empty"}`;
    $("#answer").textContent = has ? r.answer : "No answer on this site";
    $("#method").textContent = r.method || "";
    setRing(has ? r.confidence : 0);
    const src = $("#source");
    if (has && r.source_url) {
      src.hidden = false;
      src.textContent = r.source_url;
      src.href = /^https?:/.test(r.source_url) ? r.source_url : (INFO.demo_url ? INFO.demo_url.replace(/index\.html$/, "") + r.source_url : "#");
    }
    const cl = $("#citations");
    cl.replaceChildren();
    (r.citations || []).forEach((c) => cl.appendChild(el("li", "", c)));
    if (!cl.children.length) cl.appendChild(el("li", "muted", has ? "—" : "Empty citations: the site did not support an answer"));
    const evd = $("#evidence");
    evd.className = "evidence";
    if (has) highlight(evd, r.evidence || "", r.answer);
    else evd.textContent = ev.note || "Wayfinder refuses rather than guessing or answering from memory.";
    const b = step("done", has ? `Answer: ${r.answer}` : "Refused: not supported by the site", "final", ev.elapsed);
    b.textContent = `${(r.citations || []).length} citation(s) · ${ev.elapsed.toFixed(2)} s total`;
    state.t = ev.elapsed;
  }

  /* ------------------------------------------------------------ run report */
  const drop = $("#drop");
  const file = $("#file");
  drop.addEventListener("click", () => file.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) readFile(f); });
  file.addEventListener("change", () => { if (file.files[0]) readFile(file.files[0]); });

  function readFile(f) {
    const rd = new FileReader();
    rd.onload = () => { try { renderReport(JSON.parse(rd.result), f.name); } catch (e) { alert("Not a valid report JSON: " + e.message); } };
    rd.readAsText(f);
  }

  function loadReportList() {
    fetch("/api/reports").then((r) => r.json()).then(({ reports }) => {
      const ul = $("#report-list");
      ul.replaceChildren();
      if (!reports.length) { ul.appendChild(el("li", "muted", "No reports found in ./reports")); return; }
      reports.forEach((rp) => {
        const li = el("li");
        const b = el("button");
        b.type = "button";
        b.appendChild(el("span", "", rp.name));
        b.appendChild(el("span", "muted", `${rp.pass ?? "?"}/${rp.scored ?? "?"} · ${rp.backend}${rp.model ? " · " + rp.model : ""}`));
        b.addEventListener("click", () => fetch(`/api/report?name=${encodeURIComponent(rp.name)}`).then((r) => r.json()).then((d) => renderReport(d, rp.name)));
        li.appendChild(b);
        ul.appendChild(li);
      });
      if (!$("#report").dataset.loaded && reports[0]) ul.querySelector("button").click();
    }).catch(() => {});
  }

  let REPORT = null;
  let FILTER = "all";
  $$(".filters .chip").forEach((c) => c.addEventListener("click", () => {
    FILTER = c.dataset.f;
    $$(".filters .chip").forEach((x) => x.classList.toggle("active", x === c));
    renderItems();
  }));

  function kpi(k, v, s, hero) {
    const d = el("div", `kpi ${hero ? "hero" : ""}`);
    d.appendChild(el("div", "k", k));
    d.appendChild(el("div", "v", v));
    if (s) d.appendChild(el("div", "s", s));
    return d;
  }

  function renderReport(rep, label) {
    REPORT = rep;
    const sec = $("#report");
    sec.hidden = false;
    sec.dataset.loaded = "1";
    const s = rep.summary || {};
    const dev = rep.device || {};
    $("#r-name").textContent = rep.name || label;
    $("#r-meta").textContent = [rep.created, `backend: ${rep.backend}`, dev.model ? `model: ${dev.model}` : "", `transport: ${rep.transport}`, `via: ${rep.via}`].filter(Boolean).join(" · ");
    const ban = $("#report-banner");
    ban.className = "banner";
    if (rep.backend_is_stub || rep.backend === "stub") {
      ban.className = "banner warn";
      ban.replaceChildren(el("strong", "", "TEST STUB report. "), document.createTextNode("These numbers test the plumbing, not a model."));
    } else if (rep.backend === "rules") {
      ban.textContent = "Rules path: deterministic crawl and extractors on the CPU, no model involved.";
    } else if (rep.backend === "openai") {
      ban.textContent = `Remote model through an OpenAI-compatible endpoint${dev.model ? ` (${dev.model})` : ""}. Not the shipped on-device model.`;
    } else {
      ban.textContent = `On-device model run${dev.gpu ? ` on ${dev.gpu}` : ""}.`;
    }
    if (rep.note) ban.appendChild(el("span", "note", rep.note));
    const k = $("#kpis");
    k.replaceChildren(
      kpi("Pass", s.scored ? `${s.pass}/${s.scored}` : "–", s.accuracy != null ? `${(s.accuracy * 100).toFixed(1)}% exact answer + citations` : "", true),
      kpi("Answer match", s.scored ? `${s.answer_pass}/${s.scored}` : "–", "after grading normalisation"),
      kpi("Citation match", s.scored ? `${s.citation_pass}/${s.scored}` : "–", "exact set"),
      kpi("Mean latency", fmtS(s.mean_latency_s), `p95 ${fmtS(s.p95_latency_s)}`),
      kpi("Max latency", fmtS(s.max_latency_s), `budget ${rep.budget_s ?? 25} s`),
      kpi("Peak VRAM", dev.peak_vram_gib != null ? `${dev.peak_vram_gib} GiB` : "–", dev.gpu || "no GPU in this run"),
    );
    const dl = $("#device");
    dl.replaceChildren();
    Object.entries(dev).forEach(([kk, vv]) => { dl.appendChild(el("dt", "", kk)); dl.appendChild(el("dd", "", typeof vv === "object" ? JSON.stringify(vv) : vv)); });
    renderItems();
  }

  function renderItems() {
    if (!REPORT) return;
    const tb = $("#items");
    tb.replaceChildren();
    const items = REPORT.items || [];
    const maxLat = Math.max(1, ...items.map((i) => i.latency_s || 0));
    items.filter((it) => FILTER === "all" || (FILTER === "pass" ? it.pass : it.pass === false)).forEach((it) => {
      const tr = el("tr");
      const st = el("td"); st.appendChild(el("span", `status ${it.pass ? "ok" : it.pass === false ? "bad" : ""}`)); tr.appendChild(st);
      tr.appendChild(el("td", "mono", it.id));
      const q = el("td"); q.appendChild(el("div", "", it.question)); if (it.kind) q.appendChild(el("div", "muted", it.kind)); tr.appendChild(q);
      const ex = Array.isArray(it.expected) ? it.expected.join(" | ") : it.expected;
      tr.appendChild(el("td", "mono", ex === "" ? "(refuse)" : ex ?? "(unscored)"));
      const got = el("td", "mono", it.got === "" ? "(refused)" : it.got);
      if (it.answer_pass === false) got.style.color = "var(--bad)";
      tr.appendChild(got);
      const c = el("td", "mono", (it.citations || []).join(", ") || "—");
      if (it.citation_pass === false) c.style.color = "var(--bad)";
      tr.appendChild(c);
      const l = el("td", "num", fmtS(it.latency_s));
      const bar = el("span", "lat"); const i = el("i"); i.style.width = `${(100 * (it.latency_s || 0)) / maxLat}%`; bar.appendChild(i); l.appendChild(bar);
      tr.appendChild(l);
      tb.appendChild(tr);
      tr.addEventListener("click", () => {
        const nx = tr.nextElementSibling;
        if (nx && nx.classList.contains("detail")) { nx.remove(); return; }
        const d = el("tr", "detail");
        const td = el("td"); td.colSpan = 7;
        td.textContent = `method: ${it.method} · pages fetched: ${it.pages_fetched} · model steps: ${it.model_steps} · confidence: ${it.confidence}` +
          ` · expected citations: ${(it.expected_citations || []).join(", ") || "none"}` + (it.source_url ? ` · source: ${it.source_url}` : "");
        d.appendChild(td);
        tr.after(d);
      });
    });
    if (!tb.children.length) { const tr = el("tr"); const td = el("td", "muted", "Nothing to show"); td.colSpan = 7; tr.appendChild(td); tb.appendChild(tr); }
  }
})();
