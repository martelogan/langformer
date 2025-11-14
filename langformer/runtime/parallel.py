"""Parallel exploration utilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Optional, TypeVar

T = TypeVar("T")


class ParallelExplorer:
    """Executes callables in parallel and returns the first non-None result."""

    def explore(self, callables: Iterable[Callable[[], T]]) -> Optional[T]:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(func) for func in callables]
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception:  # pragma: no cover - best effort resilience
                    continue
                if result is not None:
                    for pending in futures:
                        pending.cancel()
                    return result
        return None
