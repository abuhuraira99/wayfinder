"""Any OpenAI-compatible chat endpoint (vLLM on AMD Developer Cloud, llama.cpp server, ...).

Configured with ``OPENAI_BASE_URL``, ``OPENAI_MODEL`` and ``OPENAI_API_KEY``.
Standard library only.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .base import Backend


class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1").rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL") or "Qwen/Qwen3-VL-8B-Instruct"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        # client-side rate limit (requests per minute); 0 disables
        self.rpm = float(os.environ.get("WAYFINDER_OPENAI_RPM", "0") or 0)
        self._last = 0.0
        self._lock = threading.Lock()
        self.calls = 0

    def _throttle(self, timeout: float) -> float:
        if self.rpm <= 0:
            return timeout
        gap = 60.0 / self.rpm
        with self._lock:
            wait = self._last + gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
                timeout -= wait
            self._last = time.monotonic()
        return timeout

    def generate(self, messages, *, max_new_tokens=96, timeout=10.0, meta=None) -> str:
        timeout = self._throttle(timeout)
        self.calls += 1
        payload = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_new_tokens}
        if os.environ.get("WAYFINDER_OPENAI_NO_THINK", "1") == "1":
            # hybrid reasoning models: ask for a direct answer (ignored by servers that do not know it)
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url + "/chat/completions", data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=max(0.5, timeout)) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"] or ""

    def device_info(self) -> dict:
        d = super().device_info()
        # never record the endpoint (host, path, key) in reports: a generic label only
        scheme = urllib.parse.urlsplit(self.base_url).scheme or "http"
        d.update({"endpoint": f"remote OpenAI-compatible endpoint ({scheme}://redacted)", "model": self.model,
                  "calls": self.calls})
        return d
