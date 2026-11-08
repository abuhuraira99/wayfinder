"""Resident server protocol, client fallback, watchdog and the Studio."""

import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from conftest import SITE, TASKS
from wayfinder import client, server
from wayfinder.agent import Agent
from wayfinder.backends.stub import StubBackend
from wayfinder.entry import run_one
from wayfinder.prompts import build_messages, parse_decision
from wayfinder.schema import Task


@pytest.fixture
def stub_server(monkeypatch, tmp_path):
    srv = server.make_server("tcp:127.0.0.1:0")
    addr = f"tcp:127.0.0.1:{srv.server_address[1]}"
    server.State.backend = StubBackend()
    server.State.agent = Agent(server.State.backend)
    server.State.ready, server.State.state = True, "ready"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("WAYFINDER_ADDR", addr)
    yield addr
    srv.shutdown()
    srv.server_close()


def test_ping_stats_and_solve_with_provisional(stub_server):
    p = client.ping()
    assert p["ready"] and p["backend"] == "stub" and p["stub"]
    assert client.request({"op": "stats"})[-1]["device"]["backend"] == "stub"
    updates, traces = [], []
    r = client.solve_via_server(Task(id="s1", question="What is the battery life of the Aurora X2?", corpus=str(SITE)),
                                budget=20, on_update=updates.append, on_trace=lambda k, d: traces.append(k))
    assert r.answer == "18 months" and r.method == "model"
    assert updates and updates[0].answer == "18 months"
    assert "final" in traces


def test_run_one_uses_server_and_writes_file(stub_server, tmp_path):
    out = run_one(Task(id="r1", question="What is the support email address?", corpus=str(SITE)), str(tmp_path), 20, None)
    assert out["answer"] == "support@halcyon-instruments.example"
    assert json.loads((tmp_path / "r1_output.json").read_text())["answer"] == out["answer"]


def test_client_returns_none_without_server():
    assert client.ping(timeout=0.3) is None
    assert client.solve_via_server(Task(question="x", corpus=str(SITE)), budget=3) is None


def test_watchdog_finalises_with_provisional_answer(tmp_path, monkeypatch):
    # a server that is "loading" forever must not eat the budget, and a slow model must not overrun it
    monkeypatch.setenv("WAYFINDER_STUB_DELAY", "30")
    t = time.monotonic()
    out = run_one(Task(id="w1", question="What is the battery life of the Aurora X2?", corpus=str(SITE)),
                  str(tmp_path), 3.0, "stub")
    assert time.monotonic() - t < 4.5
    assert out["answer"] == "18 months"  # rules answer persisted before the model call
    assert json.loads((tmp_path / "w1_output.json").read_text())["answer"] == "18 months"


def test_prompt_and_decision_parsing():
    msgs = build_messages("Q?", [{"id": "P1", "page": "a.html", "title": "A", "heading": "H", "text": "t"}],
                          [{"id": "L1", "text": "Pricing", "ref": "/p"}], final=False)
    assert msgs[0]["role"] == "system" and "[P1]" in msgs[1]["content"] and "[L1]" in msgs[1]["content"]
    assert parse_decision('```json\n{"action": "answer", "answer": "$5", "sources": ["P2"]}\n```') == \
        {"action": "answer", "answer": "$5", "sources": ["P2"], "links": []}
    assert parse_decision('sure! {"action":"follow","links":"L3"} ok')["links"] == ["L3"]
    assert parse_decision('{"answer": ""}')["action"] == "none"
    assert parse_decision("<think>hmm {x}</think>{'action': 'none'}")["action"] == "none"
    assert parse_decision("no json here") is None


def test_bench_report_shape(tmp_path):
    from wayfinder.bench import run_bench

    rep = run_bench(str(TASKS), backend="rules", out_dir=str(tmp_path), limit=4, quiet=True, name="t")
    assert rep["summary"]["scored"] == 4 and rep["summary"]["pass"] == 4
    item = rep["items"][0]
    for k in ("id", "expected", "got", "pass", "latency_s", "citations", "expected_citations"):
        assert k in item
    assert "python" in rep["device"]


def test_studio_serves_ui_info_and_trace_stream():
    from wayfinder.studio.server import Handler, Studio

    Studio.setup("stub", None)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        html = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "Wayfinder" in html and "Run report" in html
        info = json.loads(urllib.request.urlopen(base + "/api/info", timeout=5).read())
        assert info["stub"] is True and info["label"] == "TEST STUB" and info["examples"]
        demo = urllib.request.urlopen(base + "/demo-site/pricing.html", timeout=5).read().decode()
        assert "$49" in demo
        q = urllib.parse.urlencode({"question": "What is the battery life of the Aurora X2?",
                                    "url": base + "/demo-site/index.html"})
        stream = urllib.request.urlopen(base + "/api/run?" + q, timeout=30).read().decode()
        events = [json.loads(l[6:]) for l in stream.splitlines() if l.startswith("data: ")]
        finals = [e for e in events if e.get("type") == "final"]
        assert finals and finals[0]["result"]["answer"] == "18 months"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/static/../../../pyproject.toml", timeout=5)
    finally:
        srv.shutdown()


import urllib.error  # noqa: E402
import urllib.parse  # noqa: E402
