from wayfinder.document import chunk_page, parse_json, parse_markdown
from wayfinder.html_clean import clean_html, decode_html

HTML = """<!doctype html><html><head><title>Plans | Acme</title>
<meta name="description" content="Acme plans"><script>var secret = "$999";</script><style>.x{}</style></head>
<body>
<div class="cookie-banner">We use cookies <a href="/privacy">privacy</a></div>
<nav><a href="/pricing">Pricing</a> <a href="/contact">Contact us</a></nav>
<main>
  <h1>Plans</h1>
  <p>Simple pricing &amp; no surprises.</p>
  <table><caption>Monthly</caption>
    <tr><th>Plan</th><th>Price</th></tr>
    <tr><td>Pro</td><td>$20</td></tr>
  </table>
  <dl><dt>Support</dt><dd>help@acme.example</dd></dl>
  <ul><li>First</li><li>Second <a href="docs.html">docs</a></li></ul>
  <div hidden>Hidden text 123</div>
  <p>Line one<br>Line two</p>
</main>
<footer>Call +1 555 0100 <a href="/legal">Legal</a></footer>
</body></html>"""


def test_drops_boilerplate_keeps_structure():
    p = clean_html(HTML, "pricing.html")
    text = p.text
    assert p.title == "Plans | Acme"
    assert "$999" not in text and "cookies" not in text and "555 0100" not in text
    assert "Hidden text" not in text and "Contact us" not in text
    assert ("h1", "Plans") in p.lines
    assert ("row", "Plan | Price") in p.lines and ("row", "Pro | $20") in p.lines
    assert ("kv", "Support: help@acme.example") in p.lines
    assert ("li", "Second docs") in p.lines
    assert "Simple pricing & no surprises." in text
    assert ("p", "Line one") in p.lines and ("p", "Line two") in p.lines
    assert p.tables[0].header == ["Plan", "Price"] and p.tables[0].rows == [["Pro", "$20"]]


def test_links_are_collected_even_from_nav_and_footer():
    p = clean_html(HTML)
    hrefs = {l.ref: l.text for l in p.links}
    assert hrefs["/pricing"] == "Pricing" and hrefs["/contact"] == "Contact us"
    assert "/legal" in hrefs and "docs.html" in hrefs


def test_malformed_html_does_not_raise():
    p = clean_html("<html><body><table><tr><td>a<td>b<p>unclosed <b>bold")
    assert "a | b" in p.text


def test_decode_charsets():
    assert decode_html("café".encode("cp1252"), None) == "café"
    assert decode_html('<meta charset="utf-8">é'.encode("utf-8")).endswith("é")


def test_chunks_keep_heading_and_table_header():
    lines = [("h1", "Doc"), ("h2", "Limits")] + [("row", "Name | Value")] + [("row", f"r{i} | {i}" + "x" * 60)
                                                                              for i in range(20)]
    from wayfinder.document import Page

    ps = chunk_page(Page(ref="d.html", title="Doc", lines=lines))
    tables = [p for p in ps if p.kind == "table"]
    assert len(tables) >= 2 and all(p.text.startswith("Name | Value") for p in tables)
    assert all(p.heading == "Doc > Limits" for p in tables)


def test_markdown_and_json_parsers():
    md = parse_markdown("# T\n\nSee [the docs](docs/a.md).\n\n| a | b |\n|---|---|\n| 1 | 2 |\n", "x.md")
    assert md.title == "T" and md.links[0].ref == "docs/a.md" and md.tables[0].rows == [["1", "2"]]
    js = parse_json('{"name": "Widget", "price": 5, "tags": ["a"], "plans": [{"n": "x", "p": 1}]}', "w.json")
    assert ("kv", "price: 5") in js.lines and js.title == "Widget" and js.tables[0].header == ["n", "p"]
