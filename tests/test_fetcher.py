import gzip
import time
from http.server import BaseHTTPRequestHandler

from conftest import SITE, serve
from wayfinder.fetcher import HttpSite, LocalSite, make_site, normalize_url


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        routes = {
            "/robots.txt": (b"User-agent: *\nDisallow: /private/\n", "text/plain", None),
            "/": (b'<html><title>Home</title><a href="/a">A page</a><a href="https://elsewhere.example/x">ext</a>'
                  b'<a href="/file.pdf">pdf</a><a href="mailto:x@y.z">mail</a><a href="/private/p">p</a></html>',
                  "text/html", None),
            "/a": (gzip.compress(b"<h1>Alpha</h1><p>compressed body</p>"), "text/html; charset=utf-8", "gzip"),
            "/private/p": (b"<p>secret</p>", "text/html", None),
            "/img": (b"\x89PNG....", "image/png", None),
            "/big": (b"<p>" + b"x" * 3_000_000 + b"</p>", "text/html", None),
        }
        if self.path not in routes:
            self.send_response(404)
            self.end_headers()
            return
        body, ctype, enc = routes[self.path]
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        if enc:
            self.send_header("Content-Encoding", enc)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_http_site_fetch_links_robots_gzip_caps():
    srv, base = serve(None, Handler)
    try:
        site = HttpSite(base)
        home = site.fetch(base)
        refs = [l.ref for l in home.links]
        assert base + "a" in refs
        assert not any("elsewhere" in r or r.endswith(".pdf") or r.startswith("mailto") for r in refs)
        assert "compressed body" in site.fetch(base + "a").text
        assert site.fetch(base + "private/p") is None and "robots" in site.last_error
        assert site.fetch(base + "img") is None
        big = HttpSite(base, max_bytes=100_000).fetch(base + "big")
        assert big is None or len(big.text) <= 100_000
        assert site.fetch(base + "missing") is None
    finally:
        srv.shutdown()


def test_unreachable_host_fails_fast():
    t = time.monotonic()
    assert HttpSite("http://127.0.0.1:9/").fetch("http://127.0.0.1:9/") is None
    assert time.monotonic() - t < 6


def test_local_site_resolution_and_url_mapping():
    s = LocalSite(str(SITE), base_url="https://www.halcyon.example/")
    assert s.start_refs() == ["index.html"]
    assert s.resolve("products/index.html", "aurora-x2.html") == "products/aurora-x2.html"
    assert s.resolve("products/aurora-x2.html", "../pricing.html#plans") == "pricing.html"
    assert s.resolve("index.html", "/docs/") == "docs/index.html"
    assert s.resolve("index.html", "https://www.halcyon.example/contact.html") == "contact.html"
    assert s.resolve("index.html", "https://other.example/contact.html") is None
    assert s.map_url("https://halcyon.example/pricing") == "pricing.html"
    assert s.public_url("support/sla.html") == "https://www.halcyon.example/support/sla.html"
    links = s.fetch("index.html").links
    assert any(l.ref == "products/beacon-mini.html" for l in links)


def test_make_site_prefers_saved_copy(tmp_path):
    assert isinstance(make_site("https://example.com", str(SITE)), LocalSite)
    assert isinstance(make_site(str(SITE / "index.html"), None), LocalSite)
    assert make_site(None, None) is None
    assert isinstance(make_site("example.com", None), HttpSite)
    assert normalize_url("Example.com") == "https://example.com/"
