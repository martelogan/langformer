# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications copyright (c) 2025 Logan Martel.
# Adapted from https://github.com/meta-pytorch/KernelAgent (Apache-2.0).

"""Run configuration/state helpers inspired by Fuser/config.py."""

from __future__ import annotations

import json
import threading
import time
import uuid

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from langformer.runtime.paths import RunDirectories, make_run_dirs


@dataclass
class OrchestratorConfig:
    run_root: Path
    model: str
    workers: int = 1
    max_iters: int = 5
    llm_timeout_s: int = 120
    run_timeout_s: int = 180
    store_responses: bool = False

    def to_json(self) -> str:
        data = asdict(self)
        data["run_root"] = str(self.run_root)
        return json.dumps(data, indent=2)


@dataclass
class OrchestratorState:
    run_id: str
    run_dir: Path
    started_ts: float


@dataclass
class WorkerConfig:
    run_id: str
    worker_id: str
    variant_index: int
    model: str
    max_iters: int
    llm_timeout_s: int
    run_timeout_s: int
    store_responses: bool
    stream_dir: Path
    workspace_dir: Path


@dataclass
class UnitRecord:
    unit_id: str
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    started_ts: Optional[float] = None
    completed_ts: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "status": self.status,
            "metadata": self.metadata,
            "result": self.result,
            "started_ts": self.started_ts,
            "completed_ts": self.completed_ts,
        }

    @classmethod
    def from_path(cls, path: Path, unit_id: str) -> "UnitRecord":
        if not path.exists():
            return cls(unit_id=unit_id)
        data = json.loads(path.read_text())
        return cls(
            unit_id=str(data.get("unit_id", unit_id)),
            status=str(data.get("status", "pending")),
            metadata=dict(data.get("metadata", {})),
            result=dict(data.get("result", {})),
            started_ts=data.get("started_ts"),
            completed_ts=data.get("completed_ts"),
        )

    def mark_started(self, metadata: Dict[str, Any]) -> "UnitRecord":
        self.status = "in_progress"
        self.metadata = metadata
        self.started_ts = time.time()
        return self

    def mark_completed(
        self,
        status: str,
        result: Dict[str, Any],
    ) -> "UnitRecord":
        self.status = status
        self.result = result
        self.completed_ts = time.time()
        return self


@dataclass
class EventRecord:
    kind: str
    data: Dict[str, Any]
    ts: float = field(default_factory=time.time)

    def to_json_line(self) -> str:
        payload = {"ts": self.ts, "kind": self.kind, "data": self.data}
        return json.dumps(payload, default=str)


@dataclass
class ResultSummary:
    run_id: str
    winner_worker_id: Optional[str]
    artifact_path: Optional[str]
    reason: str


def new_run_id() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"run_{ts}_{short}"


class RunSession:
    """Helper that materializes run directories and metadata files."""

    def __init__(
        self,
        run_root: Path,
        run_id: Optional[str] = None,
        *,
        resume: bool = False,
    ) -> None:
        self.run_root = run_root
        self.resume = resume
        if run_id:
            self.run_id = run_id
            run_dir = run_root / run_id
            if not run_dir.exists() and resume:
                raise FileNotFoundError(
                    f"Run {run_id} not found under {run_root}"
                )
            self.dirs: RunDirectories = make_run_dirs(
                run_root,
                run_id,
                exist_ok=True,
            )
        else:
            self.run_id = new_run_id()
            self.dirs = make_run_dirs(run_root, self.run_id)
        self.units_dir = self.dirs.run_dir / "units"
        self.units_dir.mkdir(exist_ok=True)
        self.events_path = self.dirs.run_dir / "events.jsonl"
        self.session_path = self.dirs.run_dir / "session.json"
        self._lock = threading.Lock()
        if not self.session_path.exists() or not resume:
            self.session_path.write_text(
                json.dumps(
                    {
                        "run_id": self.run_id,
                        "created_ts": time.time(),
                        "run_dir": str(self.dirs.run_dir),
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

    def write_metadata(self, name: str, data: dict) -> Path:
        path = self.dirs.run_dir / f"{name}.json"
        path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
        return path

    def log_event(self, kind: str, data: Dict[str, Any]) -> None:
        record = EventRecord(kind=kind, data=data)
        line = record.to_json_line()
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def mark_unit_started(
        self, unit_id: str, metadata: Dict[str, Any]
    ) -> Path:
        record = self._load_unit_entry(unit_id)
        record.mark_started(metadata)
        return self._write_unit_entry(unit_id, record)

    def mark_unit_completed(
        self, unit_id: str, status: str, result: Dict[str, Any]
    ) -> Path:
        record = self._load_unit_entry(unit_id)
        record.mark_completed(status, result)
        return self._write_unit_entry(unit_id, record)

    def persist_files(
        self, unit_id: str, files: Dict[Path, str]
    ) -> List[Dict[str, str]]:
        artifact_dir = self.units_dir / unit_id / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest: List[Dict[str, str]] = []
        for orig_path, contents in files.items():
            original_str = str(orig_path)
            relative = Path(original_str)
            if relative.is_absolute():
                relative = Path(relative.name)
            dest = artifact_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(contents, encoding="utf-8")
            manifest.append(
                {
                    "original_path": original_str,
                    "artifact": str(dest.relative_to(artifact_dir)),
                }
            )
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest

    def load_files(self, unit_id: str) -> Dict[Path, str]:
        artifact_dir = self.units_dir / unit_id / "artifacts"
        manifest_path = artifact_dir / "manifest.json"
        if not manifest_path.exists():
            return {}
        manifest = json.loads(manifest_path.read_text())
        files: Dict[Path, str] = {}
        for entry in manifest:
            artifact = artifact_dir / entry["artifact"]
            files[Path(entry["original_path"])] = artifact.read_text()
        return files

    def list_units(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        if not self.units_dir.exists():
            return entries
        for unit_file in sorted(self.units_dir.glob("*.json")):
            data = json.loads(unit_file.read_text(encoding="utf-8"))
            if status and data.get("status") != status:
                continue
            entries.append(data)
        return entries

    def load_unit_entry(self, unit_id: str) -> Optional[Dict[str, Any]]:
        record = self._load_unit_entry(unit_id)
        if (
            record.status == "pending"
            and not (self.units_dir / f"{unit_id}.json").exists()
        ):
            return None
        return record.to_dict()

    def write_summary(self, extra: Optional[Dict[str, Any]] = None) -> Path:
        summary = {
            "run_id": self.run_id,
            "written_ts": time.time(),
            "units": self.list_units(),
        }
        if extra:
            summary.update(extra)
        return self.write_metadata("summary", summary)

    def _load_unit_entry(self, unit_id: str) -> UnitRecord:
        path = self.units_dir / f"{unit_id}.json"
        return UnitRecord.from_path(path, unit_id)

    def _write_unit_entry(
        self,
        unit_id: str,
        record: UnitRecord,
    ) -> Path:
        path = self.units_dir / f"{unit_id}.json"
        with self._lock:
            data = json.dumps(record.to_dict(), indent=2, default=str)
            path.write_text(data, encoding="utf-8")
        return path
