"""Persistent state manager for long-running transpilation sessions."""

from __future__ import annotations

import json

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class TranspilationState:
    """Serializable state useful for resuming transpilation runs."""

    checkpoint_dir: Path
    completed_units: List[str] = field(default_factory=list)
    pending_units: List[str] = field(default_factory=list)
    verification_results: Dict[str, Dict[str, str]] = field(
        default_factory=dict
    )

    def save(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "state.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def load(cls, checkpoint_dir: Path) -> "TranspilationState":
        path = checkpoint_dir / "state.json"
        if not path.exists():
            return cls(checkpoint_dir=checkpoint_dir)
        data = json.loads(path.read_text())
        data["checkpoint_dir"] = checkpoint_dir
        return cls(**data)
