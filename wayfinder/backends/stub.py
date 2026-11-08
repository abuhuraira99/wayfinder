"""TEST STUB backend: a deterministic stand-in for the model.

It speaks the same JSON protocol as the real model so the whole agent loop
(answer / follow / none, source ids, grounding) is exercised on a CPU. It is
never a source of real results and the Studio labels it "TEST STUB".
"""

from __future__ import annotations

import json
import os
import time

from .base import Backend


class StubBackend(Backend):
    name = "stub"
    is_stub = True
    uses_model = True

    def __init__(self, delay: float | None = None, mode: str | None = None):
        self.delay = float(os.environ.get("WAYFINDER_STUB_DELAY", "0") if delay is None else delay)
        # mode: "smart" (default), "follow" (always follow first), "none", "garbage", "invent"
        self.mode = mode or os.environ.get("WAYFINDER_STUB_MODE", "smart")
        self.calls = 0

    def generate(self, messages, *, max_new_tokens=96, timeout=10.0, meta=None) -> str:
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)  # deliberately ignores ``timeout`` to test deadlines
        meta = meta or {}
        if self.mode == "garbage":
            return "I think the answer might be on the pricing page somewhere."
        if self.mode == "none":
            return '{"action": "none"}'
        if self.mode == "invent":
            return json.dumps({"action": "answer", "answer": "Invented Value 123", "sources": ["P1"]})
        links = meta.get("links") or []
        if self.mode == "follow" and links and not meta.get("final"):
            return json.dumps({"action": "follow", "links": [links[0]["id"]]})
        cand = meta.get("rule_candidate")
        if cand and cand.get("pid_label"):
            return "```json\n" + json.dumps({"action": "answer", "answer": cand["value"],
                                             "sources": [cand["pid_label"]]}) + "\n```"
        if links and not meta.get("final"):
            return json.dumps({"action": "follow", "links": [links[0]["id"]]})
        return '{"action": "none"}'

    def device_info(self) -> dict:
        d = super().device_info()
        d["note"] = "TEST STUB - not a model"
        return d
