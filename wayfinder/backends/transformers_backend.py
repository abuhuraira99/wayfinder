"""Qwen3-VL-8B-Instruct on the GPU via transformers (text-only prompts).

Only the resident server imports this module, so ``app.py`` never touches torch.
On ROCm, torch exposes the AMD GPU as ``cuda``.
"""

from __future__ import annotations

import os
import threading
import time

from .base import Backend

DEFAULT_MODEL_DIR = "/models/Qwen3-VL-8B-Instruct"


class TransformersBackend(Backend):
    name = "transformers"

    def __init__(self, model_dir: str | None = None, device: str | None = None):
        import torch  # noqa: F401  (heavy import, server process only)

        self.model_dir = model_dir or os.environ.get("MODEL_DIR") or DEFAULT_MODEL_DIR
        self.device = device or os.environ.get("WAYFINDER_DEVICE", "cuda")
        self._lock = threading.Lock()
        self.load_seconds = 0.0
        self._load()

    def _load(self):
        import torch
        import transformers

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("GPU not available (torch.cuda.is_available() is False)")
        t0 = time.monotonic()
        self.processor = transformers.AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None) or transformers.AutoModelForImageTextToText
        try:
            model = cls.from_pretrained(self.model_dir, dtype=torch.bfloat16, local_files_only=True)
        except TypeError:
            model = cls.from_pretrained(self.model_dir, torch_dtype=torch.bfloat16, local_files_only=True)
        self.model = model.to(self.device).eval()
        gc = self.model.generation_config
        for k in ("temperature", "top_p", "top_k"):
            if hasattr(gc, k):
                setattr(gc, k, None)
        gc.do_sample = False
        self.load_seconds = time.monotonic() - t0

    def warmup(self) -> None:
        msgs = [{"role": "user", "content": "Reply with the JSON object {\"ok\": true}."}]
        self.generate(msgs, max_new_tokens=8, timeout=60)
        long = [{"role": "user", "content": ("Evidence passage. " * 400) + "\nReply with {\"ok\": true}."}]
        self.generate(long, max_new_tokens=8, timeout=60)

    def generate(self, messages, *, max_new_tokens=96, timeout=10.0, meta=None) -> str:
        import torch

        msgs = []
        for m in messages:
            c = m["content"]
            msgs.append({"role": m["role"], "content": [{"type": "text", "text": c}] if isinstance(c, str) else c})
        with self._lock:
            inputs = self.processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            n_in = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                          max_time=max(0.5, float(timeout)))
            text = self.processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
        return text.strip()

    def device_info(self) -> dict:
        import torch

        d = super().device_info()
        d.update({"model_dir": self.model_dir, "torch": torch.__version__, "load_seconds": round(self.load_seconds, 1)})
        try:
            import transformers

            d["transformers"] = transformers.__version__
        except Exception:
            pass
        if torch.cuda.is_available():
            d["gpu"] = torch.cuda.get_device_name(0)
            d["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            d["hip"] = getattr(torch.version, "hip", None)
        return d
