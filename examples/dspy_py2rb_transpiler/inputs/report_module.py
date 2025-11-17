"""Tiny module that rewards perfect scores."""

from __future__ import annotations

from typing import Iterable, List


def scale_score(score: int) -> int:
    """Return a score that is always doubled."""

    return score * 2


def scale_scores(values: Iterable[int]) -> List[int]:
    """Apply :func:`scale_score` to a sequence of integers."""

    return [scale_score(value) for value in values]


def render_report(values: Iterable[int]) -> str:
    doubled = scale_scores(values)
    return ", ".join(str(item) for item in doubled)
