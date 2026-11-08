import pytest

from wayfinder.document import Page, Passage, Table
from wayfinder.extractors import (acronym_in, best_rule_answer, extract_candidates, grounded, question_intent,
                                  shape_value)
from wayfinder.textutil import clean_answer, grade_normalize, special_terms, stem


@pytest.mark.parametrize("q,intent", [
    ("How much does the Pro plan cost?", "price"),
    ("What is the support email address?", "email"),
    ("What phone number should I call?", "phone"),
    ("What is the latest firmware version?", "version"),
    ("When was 3.8 released?", "date"),
    ("In which year was the company founded?", "year"),
    ("How many sensors can a hub manage?", "count"),
    ("How long is the free trial?", "duration"),
    ("What uptime does the SLA guarantee?", "percent"),
    ("Who is the CEO?", "name"),
    ("What is the capital?", None),
])
def test_intents(q, intent):
    assert question_intent(q) == intent


def _ranked(text, ref="p.html", heading=""):
    return [(Passage(pid=f"{ref}#0", page_ref=ref, text=text, heading=heading), 5.0)]


def test_typed_value_near_focus_wins():
    text = "Starter costs $0.\nThe Pro plan costs $20 per month.\nEnterprise: contact us"
    pages = {"p.html": Page(ref="p.html", title="Pricing", lines=[("p", l) for l in text.splitlines()])}
    c = best_rule_answer("How much does the Pro plan cost?", _ranked(text), pages)
    assert c.value == "$20"


def test_table_cell_by_row_and_column_both_orientations():
    t1 = Table(header=["Feature", "Free", "Pro"], rows=[["Monthly price", "$0", "$20"], ["Seats", "1", "10"]])
    t2 = Table(header=["Plan", "Monthly price", "Seats"], rows=[["Free", "$0", "1"], ["Pro", "$20", "10"]])
    for t in (t1, t2):
        lines = [("row", " | ".join(t.header))] + [("row", " | ".join(r)) for r in t.rows]
        pg = Page(ref="t.html", title="Plans", lines=lines, tables=[t])
        c = best_rule_answer("What is the monthly price of the Pro plan?",
                             _ranked("\n".join(x for _, x in lines), "t.html"), {"t.html": pg})
        assert c.value == "$20", t


def test_constraint_reasoning_on_plan_table():
    t = Table(header=["Feature", "Basic", "Plus", "Max"],
              rows=[["Price", "$5", "$15", "$40"], ["Storage", "10 GB", "100 GB", "1 TB"],
                    ["SSO", "—", "Yes", "Yes"]])
    lines = [("row", " | ".join(t.header))] + [("row", " | ".join(r)) for r in t.rows]
    pg = Page(ref="t.html", title="Plans", lines=lines, tables=[t])
    c = best_rule_answer("How much storage comes with the cheapest plan that includes SSO?",
                         _ranked("\n".join(x for _, x in lines), "t.html"), {"t.html": pg})
    assert c.value == "100 GB"


def test_gate_refuses_when_key_term_missing():
    text = "Aurora X2 unit price: $349"
    pages = {"p.html": Page(ref="p.html", title="Aurora X2", lines=[("p", text)])}
    assert best_rule_answer("What is the unit price of the Aurora X9?", _ranked(text), pages) is None
    assert best_rule_answer("What is the unit price of the Aurora X2?", _ranked(text), pages).value == "$349"


def test_latest_version_picks_highest():
    text = "4.1.0 - January\n4.2.1 - October\n4.2.0 - June"
    pages = {"c.html": Page(ref="c.html", title="Changelog", lines=[("p", l) for l in text.splitlines()])}
    cands = extract_candidates("What is the latest firmware version?", _ranked(text, "c.html", "Firmware"), pages)
    assert cands[0].value == "4.2.1"


def test_never_answers_with_a_value_from_the_question():
    text = "Version 3.8.0 was released on 14 September 2026."
    pages = {"c.html": Page(ref="c.html", title="Changelog", lines=[("p", text)])}
    c = best_rule_answer("When was version 3.8.0 released?", _ranked(text, "c.html"), pages)
    assert c.value == "14 September 2026"


def test_grounding_and_normalisation():
    assert grounded("$49", "Team | $49 | 50")
    assert grounded("3000", "rate 3,000 req/min")
    assert grounded("99.99 %", "uptime 99.99% guaranteed")
    assert not grounded("Invented Value 123", "nothing like that here")
    assert grade_normalize("7-ABC 123.") == "7ABC123"
    assert grade_normalize("Q3 FY27") == "Q3FY27"


def test_clean_answer_and_shape_value():
    assert clean_answer('Answer: "18 months".') == "18 months"
    assert clean_answer("Not found") == ""
    assert clean_answer("Acme Inc.") == "Acme Inc."
    assert shape_value("How many nodes are required?", "count", "3 nodes") == "3"
    assert shape_value("Which city is it in?", None, "Tromsø, Norway") == "Tromsø"
    assert shape_value("What is the battery life?", "duration", "18 months") == "18 months"


def test_special_terms_and_acronyms():
    assert special_terms("What is the price of the Aurora X9?") == ["aurora", "x9"]
    assert acronym_in("our Service Level Agreement covers it", "sla")
    assert not acronym_in("something else entirely", "sla")
    assert stem("headquartered") == stem("headquarters")
    assert stem("monthly") == stem("month")
