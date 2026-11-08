"""Backend interface."""

from __future__ import annotations

import platform


class Backend:
    """A text generator. ``uses_model`` False means the agent decides with rules alone."""

    name = "base"
    is_stub = False
    uses_model = True

    def generate(self, messages: list[dict], *, max_new_tokens: int = 96, timeout: float = 10.0,
                 meta: dict | None = None) -> str:
        raise NotImplementedError

    def warmup(self) -> None:
        return None

    def device_info(self) -> dict:
        return {"backend": self.name, "python": platform.python_version(), "platform": platform.platform()}


class RulesBackend(Backend):
    name = "rules"
    uses_model = False

    def generate(self, messages, *, max_new_tokens=96, timeout=10.0, meta=None) -> str:
        return ""
