"""Model backends behind one small interface.

* ``transformers`` - Qwen3-VL-8B-Instruct on the GPU (used in the image)
* ``openai``       - any OpenAI-compatible endpoint (vLLM, etc.)
* ``rules``        - no model: deterministic crawl + extractors
* ``stub``         - deterministic fake model for tests (labelled TEST STUB)
"""

from __future__ import annotations

import os

from .base import Backend, RulesBackend


def get_backend(name: str | None = None, **kw) -> Backend:
    name = (name or os.environ.get("WAYFINDER_BACKEND") or "rules").lower()
    if name in ("rules", "none", "regex"):
        return RulesBackend()
    if name == "stub":
        from .stub import StubBackend

        return StubBackend(**kw)
    if name in ("openai", "vllm", "oai"):
        from .openai_backend import OpenAIBackend

        return OpenAIBackend(**kw)
    if name in ("transformers", "hf", "gpu"):
        from .transformers_backend import TransformersBackend

        return TransformersBackend(**kw)
    raise ValueError(f"unknown backend: {name}")


__all__ = ["Backend", "RulesBackend", "get_backend"]
