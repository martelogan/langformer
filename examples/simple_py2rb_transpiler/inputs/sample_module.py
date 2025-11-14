"""Reference Python module used by the simple Python→Ruby example.

Expectations for the Ruby translation:
- Define a module named `Report` exposing `normalize_scores` and `generate_report`
  as module methods (e.g., `def self.normalize_scores`).
- Preserve keyword-only arguments and default values.
- Keep the functions pure: no I/O or shared mutable state beyond the inputs.
"""

from __future__ import annotations

from typing import Sequence


def normalize_scores(
    values: Sequence[int | float], *, min_size: int = 3
) -> list[int]:
    """Normalize a sequence by shifting values relative to the first element."""

    cleaned = [int(v) for v in values if v is not None]
    if len(cleaned) < min_size:
        return [0] * min_size
    base = cleaned[0]
    return [value - base for value in cleaned]


def generate_report(
    metrics: dict[str, int],
    *,
    threshold: int = 3,
    fmt: str = "plain",
) -> str:
    """Return a textual (or JSON) report of winners over the threshold."""

    winners = sorted(
        name for name, score in metrics.items() if score >= threshold
    )
    if fmt == "json":
        import json

        return json.dumps({"winners": winners, "count": len(winners)})

    body = ", ".join(winners) if winners else "n/a"
    return f"Winners ({len(winners)}): {body}"
