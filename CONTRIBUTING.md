# Contributing

Thanks for helping. Wayfinder is small on purpose; keep it that way.

## Ground rules

- The agent, `app.py`, the CLI and the Studio stay **standard-library only**. Heavy imports
  (torch, transformers) live only in `wayfinder/backends/transformers_backend.py`, which only the
  resident server loads.
- Never answer from model memory. Every answer must be found verbatim (after normalisation) in a
  fetched page, and citations must be the pages the answer needs: no more, no fewer.
- No hardcoded answers, no site-specific rules. The fixture site exists to test general behaviour;
  a change that only helps one fixture task is not a fix.
- The output contract never breaks: a valid JSON file is written for every call, and `app.py`
  always exits 0.

## Development

```bash
uv run --python 3.14 --with pytest python -m pytest -q          # CPU tests (stub + rules)
uv run --python 3.14 python -m wayfinder bench --backend rules  # fixture bench
uv run --python 3.14 python -m wayfinder.studio --backend stub  # UI on http://127.0.0.1:8770
```

Add a test for every bug you fix. If you add a fixture task, add the page that answers it to
`fixtures/site/` and make sure the expected citations are the only pages that hold the value.

## Pull requests

Describe the behaviour change, include bench numbers before and after (`reports/*.json`), and say
which backend produced them. Results from the `stub` backend are never results.
