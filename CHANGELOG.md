# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses semantic versioning.

## [0.1.0] - 2026-11-08

### Added
- Agent loop: crawl, clean, rank (BM25 with exact-term, phrase and heading boosts), model decides
  `answer` / `follow` / `none` for at most three steps, grounding check, minimal citations.
- Live web fetcher on the standard library: same-site crawl ranked by link text and URL overlap
  with the question, robots.txt, gzip, size and time caps, parallel fetches.
- Saved-site mode (`--corpus` / `--index`): HTML, HTM, TXT, MD and JSON parsed per file without
  ever stopping on a bad file; persisted JSON index.
- Deterministic extractors used as the fallback answer: prices, dates, versions, emails, phones,
  URLs, counts, percentages, durations, years, names, `label: value` lines, table cells by row and
  column label, and plan-table constraints ("cheapest plan that includes SSO").
- Backends: `transformers` (Qwen3-VL-8B-Instruct on the GPU), `openai` (any compatible endpoint,
  with a client-side rate limit), `rules` and a `stub` for tests.
- Container contract: `app.py` thin client with argument aliases, batch input, placeholder and
  atomic writes, watchdog; supervisor + resident server over a unix socket.
- `bench` command with offline and HTTP transports and an `app.py` subprocess mode.
- Wayfinder Studio: live trace view and run-report view.
- Fixture website (fictional sensor company) with 19 tasks including multi-hop and unanswerable.
