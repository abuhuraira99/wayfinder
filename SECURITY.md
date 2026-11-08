# Security policy

## Reporting a vulnerability

Please open a private security advisory on the GitHub repository (Security tab, "Report a
vulnerability") rather than a public issue. You will get a reply within a week.

## What Wayfinder does on the network

- It fetches only the start URL you give it and same-site links from there, with a short timeout,
  a size cap per page and a polite, identifiable User-Agent. It reads `robots.txt` and skips
  disallowed pages (the start URL itself is always fetched, because you asked for it).
- It never submits forms, never executes page JavaScript and never sends cookies or credentials.
- Page text is untrusted input. It is shown to the model as evidence, never as instructions, and the
  Studio renders it with `textContent` only.

## Secrets

- No keys, tokens or `.env` files belong in this repository or in the image. The `openai` backend
  reads `OPENAI_API_KEY` from the environment at run time only, and bench reports record just the
  endpoint host name.
- The resident server listens on a local unix socket (or `127.0.0.1` where unix sockets are not
  available) and the Studio binds to `127.0.0.1` by default. Do not expose either to a network.
