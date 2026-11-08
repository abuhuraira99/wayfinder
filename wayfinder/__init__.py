"""Wayfinder: a grounded web information retrieval agent.

Give it a start URL (or a saved copy of a site) and a natural-language question;
it crawls, reads, ranks, follows links and returns the value that answers the
question together with the page(s) it came from.

The core package is standard-library only. Model backends import their heavy
dependencies lazily, so ``import wayfinder`` never pulls in torch.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .schema import Result, Task  # noqa: E402

__all__ = ["Result", "Task", "__version__", "answer"]


def answer(question: str, url: str | None = None, corpus: str | None = None,
           backend: str = "rules", deadline_s: float = 25.0) -> Result:
    """Convenience API: answer one question and return a :class:`Result`."""
    import time

    from .agent import Agent
    from .backends import get_backend

    task = Task(id="query", question=question, url=url, corpus=corpus)
    agent = Agent(get_backend(backend))
    return agent.run(task, deadline=time.monotonic() + deadline_s)
