"""The container contract, end to end: real ``app.py`` subprocesses."""

import json
import os
import subprocess
import sys
import time

from conftest import APP, SITE


def run_app(args, tmp_path, env_extra=None, timeout=60):
    env = dict(os.environ)
    env.update(env_extra or {})
    t = time.monotonic()
    p = subprocess.run([sys.executable, str(APP), *args, "--output-dir", str(tmp_path)], capture_output=True,
                       text=True, encoding="utf-8", timeout=timeout, env=env)
    return p, time.monotonic() - t


def read(tmp_path, qid):
    return json.loads((tmp_path / f"{qid}_output.json").read_text(encoding="utf-8"))


def test_single_question_offline_corpus(tmp_path):
    p, dt = run_app(["--query-id", "q01", "--query", "How much does the Team plan cost per month?",
                     "--corpus", str(SITE)], tmp_path)
    assert p.returncode == 0
    out = read(tmp_path, "q01")
    assert out == {"answer": "$49", "citations": ["pricing.html"], "source_url": "pricing.html",
                   "confidence": out["confidence"]}
    assert 0 < out["confidence"] <= 1
    assert dt < 25


def test_url_over_http_cites_urls(tmp_path, site_http):
    p, _ = run_app(["--id", "h1", "--question", "What uptime does the SLA guarantee for the Enterprise plan?",
                    "--start-url", site_http + "pricing.html", "--unknown-flag", "zzz"], tmp_path)
    assert p.returncode == 0
    out = read(tmp_path, "h1")
    assert out["answer"] == "99.99%"
    assert out["citations"] == [site_http + "support/sla.html"]
    assert out["source_url"] == site_http + "support/sla.html"


def test_unanswerable_is_empty_with_empty_citations(tmp_path):
    run_app(["--qid", "u1", "--prompt", "What is the unit price of the Aurora X9?", "--pages", str(SITE)], tmp_path)
    assert read(tmp_path, "u1") == {"answer": "", "citations": [], "source_url": "", "confidence": 0.0}


def test_batch_input_jsonl(tmp_path):
    f = tmp_path / "tasks.jsonl"
    f.write_text('{"id": "b1", "question": "In which year was Halcyon Instruments founded?"}\n'
                 '{"task_id": "b2", "query": "What is the support email address?"}\n', encoding="utf-8")
    out_dir = tmp_path / "out"
    p, _ = run_app(["--input", str(f), "--corpus", str(SITE)], out_dir)
    assert p.returncode == 0
    assert read(out_dir, "b1")["answer"] == "2017"
    assert read(out_dir, "b2")["answer"] == "support@halcyon-instruments.example"
    lines = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    assert [l["id"] for l in lines] == ["b1", "b2"]


def test_no_arguments_still_writes_valid_placeholder(tmp_path):
    p, _ = run_app([], tmp_path)
    assert p.returncode == 0
    assert read(tmp_path, "query") == {"answer": "", "citations": [], "source_url": "", "confidence": 0.0}


def test_unreachable_url_refuses_quickly(tmp_path):
    p, dt = run_app(["--query-id", "n1", "--query", "What is the price?", "--url", "http://127.0.0.1:9/"], tmp_path)
    assert p.returncode == 0
    assert read(tmp_path, "n1")["answer"] == ""
    assert dt < 15


def test_missing_corpus_and_no_url_refuses(tmp_path):
    p, _ = run_app(["--query-id", "m1", "--query", "What is the price?", "--corpus", str(tmp_path / "nope")], tmp_path)
    assert p.returncode == 0 and read(tmp_path, "m1")["answer"] == ""


def test_index_then_query(tmp_path):
    p, _ = run_app(["--index", str(SITE)], tmp_path, env_extra={"WAYFINDER_INDEX_WAIT": "0.5"})
    assert p.returncode == 0
    assert json.loads(p.stdout.strip().splitlines()[-1])["indexed"] >= 15
    run_app(["--query-id", "i1", "--query", "What port does the Halcyon Edge agent listen on by default?",
             "--corpus", str(SITE)], tmp_path)
    assert read(tmp_path, "i1")["citations"] == ["docs/getting-started.html"]


def test_app_never_imports_torch(tmp_path):
    code = (f"import sys; sys.argv=['app.py','--query','x','--corpus',{str(SITE)!r},'--output-dir',{str(tmp_path)!r}];"
            f"sys.path.insert(0, {str(APP.parent.parent)!r});"
            "from wayfinder.entry import main; main(sys.argv[1:]);"
            "bad=[m for m in ('torch','transformers','numpy') if m in sys.modules]; print('HEAVY', bad)")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert "HEAVY []" in p.stdout
