"""Plain data types shared by every layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Task:
    """One question to answer."""

    id: str = "query"
    question: str = ""
    url: str | None = None
    corpus: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        from .contract import pick, URL_KEYS, CORPUS_KEYS, ID_KEYS, QUESTION_KEYS

        return cls(
            id=str(pick(d, ID_KEYS) or "query"),
            question=str(pick(d, QUESTION_KEYS) or ""),
            url=pick(d, URL_KEYS) or None,
            corpus=pick(d, CORPUS_KEYS) or None,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Result:
    """The graded payload plus optional diagnostics (never written to the output file)."""

    answer: str = ""
    citations: list[str] = field(default_factory=list)
    source_url: str = ""
    confidence: float = 0.0
    # diagnostics, not part of the contract file
    method: str = "empty"
    evidence: str = ""
    pages_visited: int = 0
    hops: int = 0

    def to_output(self) -> dict:
        """The exact JSON object written to ``<id>_output.json``."""
        return {
            "answer": self.answer or "",
            "citations": list(self.citations or []) if self.answer else [],
            "source_url": (self.source_url or "") if self.answer else "",
            "confidence": round(float(self.confidence or 0.0), 3) if self.answer else 0.0,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(self.to_output())
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Result":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        r = cls(**known)
        if not isinstance(r.citations, list):
            r.citations = [str(r.citations)] if r.citations else []
        r.citations = [str(c) for c in r.citations]
        try:
            r.confidence = float(r.confidence)
        except (TypeError, ValueError):
            r.confidence = 0.0
        r.answer = "" if r.answer is None else str(r.answer)
        return r
