"""Agent behaviour on the fixture site, on every path that runs without a GPU."""

import json
import time

import pytest

from conftest import SITE, TASKS
from wayfinder.agent import Agent
from wayfinder.backends import get_backend
from wayfinder.backends.stub import StubBackend
from wayfinder.schema import Task
from wayfinder.textutil import grade_normalize

SPEC = json.loads(TASKS.read_text(encoding="utf-8"))


def check(res, t, base=""):
    cites = sorted(c[len(base):] if base and c.startswith(base) else c for c in res.citations)
    return grade_normalize(res.answer) == grade_normalize(t["expected"]) and cites == sorted(t["citations"])


@pytest.mark.parametrize("backend", ["rules", "stub"])
def test_fixture_offline(backend):
    agent = Agent(get_backend(backend))
    failed = []
    for t in SPEC["tasks"]:
        r = agent.run(Task(id=t["id"], question=t["question"], corpus=str(SITE)), deadline=time.monotonic() + 25)
        if not check(r, t):
            failed.append((t["id"], r.answer, r.citations))
    assert not failed


@pytest.mark.parametrize("backend", ["rules", "stub"])
def test_fixture_over_http(backend, site_http):
    agent = Agent(get_backend(backend))
    failed = []
    for t in SPEC["tasks"]:
        r = agent.run(Task(id=t["id"], question=t["question"], url=site_http + t["start"]),
                      deadline=time.monotonic() + 25)
        if not check(r, t, site_http):
            failed.append((t["id"], r.answer, r.citations))
    assert not failed


def test_trace_events_and_provisional_update():
    events, updates = [], []
    Agent(StubBackend()).run(Task(question="What is the battery life of the Aurora X2?", corpus=str(SITE)),
                             on_update=updates.append, trace=lambda k, **d: events.append(k))
    for k in ("start", "plan", "fetch", "rank", "rules", "think", "decision", "verify", "final"):
        assert k in events
    assert updates and updates[0].answer == "18 months"


def test_invented_model_answer_is_rejected_and_rules_fallback_kept():
    r = Agent(StubBackend(mode="invent")).run(Task(question="What is the battery life of the Aurora X2?",
                                                   corpus=str(SITE)))
    assert r.answer == "18 months" and r.method.startswith("rules")


def test_unparseable_model_reply_falls_back_to_rules():
    r = Agent(StubBackend(mode="garbage")).run(Task(question="In which year was Halcyon Instruments founded?",
                                                    corpus=str(SITE)))
    assert r.answer == "2017"


def test_model_refusal_wins():
    r = Agent(StubBackend(mode="none")).run(Task(question="In which year was Halcyon Instruments founded?",
                                                 corpus=str(SITE)))
    assert r.answer == "" and r.citations == []


def test_slow_model_respects_deadline():
    t = time.monotonic()
    r = Agent(StubBackend(delay=3.0)).run(Task(question="What is the support email address?", corpus=str(SITE)),
                                          deadline=time.monotonic() + 1.5)
    # the stub ignores its timeout, so the agent must notice the overrun and keep the rules answer
    assert time.monotonic() - t < 5
    assert r.answer in ("support@halcyon-instruments.example", "")


def test_model_follow_loop_is_bounded():
    stub = StubBackend(mode="follow")
    Agent(stub, max_hops=3).run(Task(question="What is the weight of the Beacon Mini?", corpus=str(SITE)))
    assert stub.calls <= 3


def test_no_source_and_no_question_refuse():
    assert Agent().run(Task(question="What is the price?")).answer == ""
    assert Agent().run(Task(question="", corpus=str(SITE))).answer == ""


def test_url_in_question_text_is_used(site_http):
    r = Agent().run(Task(question=f"On {site_http}index.html, what is the battery life of the Aurora X2?"))
    assert r.answer == "18 months"


def test_corpus_with_unreadable_and_odd_files(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "a.html").write_text("<h1>Widget</h1><p>The Widget weighs 12 kg.</p>", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "noext").write_bytes(b"\x00\x00binary")
    (tmp_path / "z.md").write_text("# Gadget\n\nGadget weight: 3 kg\n", encoding="utf-8")
    r = Agent().run(Task(question="What is the weight of the Gadget?", corpus=str(tmp_path)))
    assert r.answer == "3 kg" and r.citations == ["z.md"]


def test_main_heading_question(site_http):
    r = Agent().run(Task(question="What is the main heading of this page?", url=site_http + "index.html"))
    assert r.answer == "Environmental sensing for teams that cannot afford guesswork"
    assert r.citations == [site_http + "index.html"]
