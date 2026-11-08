import json
import os

import pytest

from wayfinder.contract import load_task_file, output_path, parse_args, safe_id, write_json_atomic
from wayfinder.schema import Result


@pytest.mark.parametrize("flag", ["--query", "--question", "--task", "--prompt", "-q", "--q"])
def test_question_aliases(flag):
    a = parse_args([flag, "What is X?", "--query-id", "q01"])
    assert a.question == "What is X?" and a.qid == "q01"


@pytest.mark.parametrize("flag", ["--query-id", "--query_id", "--id", "--qid", "--task-id", "--task_id"])
def test_id_aliases(flag):
    assert parse_args([flag, "abc", "--query", "x"]).qid == "abc"


@pytest.mark.parametrize("flag", ["--url", "--start-url", "--start_url", "--website", "--site", "--base-url", "--base_url"])
def test_url_aliases(flag):
    assert parse_args([flag, "https://a.example/", "--query", "x"]).url == "https://a.example/"


@pytest.mark.parametrize("flag", ["--corpus", "--pages", "--sites", "--web-root", "--web_root", "--root"])
def test_corpus_aliases(flag):
    assert parse_args([flag, "/data/site", "--query", "x"]).corpus == "/data/site"


def test_equals_form_and_unknown_flags_ignored():
    a = parse_args(["--mystery", "value", "--query=hello there", "--id=q7", "--verbose", "--other-flag"])
    assert a.question == "hello there" and a.qid == "q7"
    assert "--mystery" in a.unknown and "value" in a.unknown


def test_positional_question_and_json():
    assert parse_args(["What", "is", "it?"]).question == "What is it?"
    a = parse_args(['{"id": "j1", "question": "Q?", "url": "https://x.example"}'])
    assert (a.qid, a.question, a.url) == ("j1", "Q?", "https://x.example")


def test_question_flag_suppresses_positional():
    a = parse_args(["--query", "", "stray"])
    assert a.question == ""


def test_url_given_as_corpus_is_moved():
    a = parse_args(["--corpus", "https://site.example", "--query", "x"])
    assert a.url == "https://site.example" and a.corpus is None


def test_missing_id_defaults_to_query(tmp_path):
    assert safe_id(None) == "query"
    assert safe_id("") == "query"
    assert safe_id("../etc/pass wd") == "etc_pass_wd"
    assert output_path(str(tmp_path), None).endswith("query_output.json")
    assert output_path(str(tmp_path), "q01").endswith("q01_output.json")


def test_atomic_write_leaves_no_temp_files(tmp_path):
    p = tmp_path / "out" / "q01_output.json"
    assert write_json_atomic(str(p), {"answer": "1", "citations": []})
    assert json.loads(p.read_text(encoding="utf-8"))["answer"] == "1"
    assert [f.name for f in p.parent.iterdir()] == ["q01_output.json"]


def test_output_shape_and_empty_answer_has_no_citations():
    r = Result(answer="", citations=["x.html"], source_url="x", confidence=0.9)
    out = r.to_output()
    assert set(out) == {"answer", "citations", "source_url", "confidence"}
    assert out == {"answer": "", "citations": [], "source_url": "", "confidence": 0.0}
    out = Result(answer="42", citations=["a.html"], source_url="a.html", confidence=0.71234).to_output()
    assert out["citations"] == ["a.html"] and out["confidence"] == 0.712


def test_task_files_json_list_object_and_jsonl(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps([{"query_id": "x1", "query": "Q1"}, "bare question"]), encoding="utf-8")
    items = load_task_file(str(a))
    assert items[0]["query_id"] == "x1" and items[1]["id"] == "q02"
    b = tmp_path / "b.json"
    b.write_text(json.dumps({"tasks": [{"id": "t", "question": "Q"}]}), encoding="utf-8")
    assert load_task_file(str(b))[0]["id"] == "t"
    c = tmp_path / "c.jsonl"
    c.write_text('{"id": "l1", "prompt": "P"}\n\nnot json\n{"qid": "l2", "task": "T"}\n', encoding="utf-8")
    assert [i.get("id") or i.get("qid") for i in load_task_file(str(c))] == ["l1", "l2"]
    assert load_task_file(str(tmp_path / "missing.json")) == []


def test_input_flag_with_plain_text_becomes_question():
    a = parse_args(["--input", "What is the price?"])
    assert a.input is None and a.question == "What is the price?"
