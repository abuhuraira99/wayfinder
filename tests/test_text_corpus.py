"""Plain-text and Markdown corpora are first-class (M3-style saved folders)."""

import json
import os
import subprocess
import sys

import pytest

from conftest import APP
from wayfinder.agent import Agent
from wayfinder.backends.openai_backend import OpenAIBackend
from wayfinder.schema import Task


@pytest.fixture
def txt_corpus(tmp_path):
    root = tmp_path / "corpus"
    (root / "notes").mkdir(parents=True)
    (root / "specs").mkdir()
    (root / "empty").mkdir()
    (root / "notes" / "release_notes.txt").write_text(
        "Release notes - TQ-40 controller\n\n"
        "Firmware REV-C2 was released on 12 March 2026.\n"
        "The maximum junction temperature is 94 C.\n"
        "Known issue tracked as BUG-4471.\n", encoding="utf-8")
    (root / "specs" / "bugs.md").write_text(
        "# Bug database\n\n| Ticket | Owner | Status |\n|---|---|---|\n"
        "| BUG-4471 | Priya Raman | Open |\n| BUG-4402 | Tom Ek | Closed |\n", encoding="utf-8")
    (root / "specs" / "settings.txt").write_text(
        "# defaults\nlisten_port = 7781\nlog level: debug\n", encoding="utf-8")
    (root / "specs" / "blob.dat").write_bytes(b"\x00\x01")
    return root


@pytest.mark.parametrize("question,answer,cites", [
    ("What is the maximum junction temperature of the TQ-40?", "94 C", ["notes/release_notes.txt"]),
    ("When was firmware REV-C2 released?", "12 March 2026", ["notes/release_notes.txt"]),
    ("What is the default listen port?", "7781", ["specs/settings.txt"]),
    # multi-hop: the release notes name the ticket, the bug table names its owner
    ("Who owns the known issue in the TQ-40 release notes?", "Priya Raman",
     ["notes/release_notes.txt", "specs/bugs.md"]),
    # unanswerable
    ("What is the price of the TQ-90?", "", []),
])
def test_txt_md_corpus_rules(txt_corpus, question, answer, cites):
    r = Agent().run(Task(question=question, corpus=str(txt_corpus)))
    assert r.answer == answer
    assert sorted(r.citations) == sorted(cites)


def test_index_then_query_on_text_corpus(txt_corpus, tmp_path):
    out = tmp_path / "out"
    env = dict(os.environ, WAYFINDER_INDEX_WAIT="0.5")
    p = subprocess.run([sys.executable, str(APP), "--index", str(txt_corpus)], capture_output=True, text=True,
                       timeout=60, env=env)
    assert p.returncode == 0
    p = subprocess.run([sys.executable, str(APP), "--corpus", str(txt_corpus), "--query-id", "query_01",
                        "--query", "What is the maximum junction temperature of the TQ-40?", "--output-dir", str(out)],
                       capture_output=True, text=True, timeout=60, env=env)
    assert p.returncode == 0
    d = json.loads((out / "query_01_output.json").read_text(encoding="utf-8"))
    assert d["answer"] == "94 C" and d["citations"] == ["notes/release_notes.txt"]


def test_unreachable_url_and_missing_corpus_still_valid_json(tmp_path):
    p = subprocess.run([sys.executable, str(APP), "--query-id", "x1", "--query", "What is the price?",
                        "--url", "http://127.0.0.1:9/start", "--corpus", str(tmp_path / "does-not-exist"),
                        "--output-dir", str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0
    d = json.loads((tmp_path / "x1_output.json").read_text(encoding="utf-8"))
    assert d == {"answer": "", "citations": [], "source_url": "", "confidence": 0.0}


def test_openai_backend_never_records_endpoint():
    info = OpenAIBackend(base_url="https://secret-host.example/v1", api_key="sk-test").device_info()
    blob = json.dumps(info)
    assert "secret-host" not in blob and "sk-test" not in blob
    assert info["endpoint"].startswith("remote OpenAI-compatible endpoint")
