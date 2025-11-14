from __future__ import annotations

import json

from pathlib import Path

from langformer import TranspilationOrchestrator
from langformer.runtime import RunSession


def test_run_session_records_units(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    session = RunSession(run_root)
    session.mark_unit_started("unit", {"language": "python"})
    session.log_event("unit_started", {"unit": "unit"})
    session.mark_unit_completed("unit", "success", {"result": "ok"})

    units_dir = session.units_dir
    unit_path = units_dir / "unit.json"
    assert unit_path.exists()
    payload = json.loads(unit_path.read_text())
    assert payload["status"] == "success"
    assert payload["result"]["result"] == "ok"

    events_path = session.events_path
    assert events_path.exists()
    events = [
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if line
    ]
    assert any(evt["kind"] == "unit_started" for evt in events)


def test_run_session_persist_files(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    session = RunSession(run_root)
    manifest = session.persist_files("unit", {Path("out.py"): "print('hi')"})
    session.mark_unit_completed(
        "unit", "success", {"manifest": manifest, "notes": {"attempt": 1}}
    )
    loaded = session.load_files("unit")
    assert Path("out.py") in loaded
    assert loaded[Path("out.py")] == "print('hi')"
    summary = session.write_summary()
    assert summary.exists()


def test_run_session_writes_metadata(tmp_path: Path):
    run_root = tmp_path / "runs"
    orchestrator = TranspilationOrchestrator(
        config={
            "transpilation": {
                "source_language": "python",
                "target_language": "python",
                "layout": {
                    "output": {"path": "transpiled_output.py", "kind": "file"}
                },
                "runtime": {"enabled": True, "run_root": str(run_root)},
                "verification": {"strategy": "exact_match"},
                "llm": {"provider": "echo"},
            }
        }
    )
    source = tmp_path / "input.py"
    source.write_text("def main():\n    return 1\n")
    target = tmp_path / "out.py"
    orchestrator.transpile_file(source, target)
    run_dirs = [
        path for path in run_root.iterdir() if (path / "meta.json").exists()
    ]
    assert run_dirs, "expected run directory"
    meta = run_dirs[0] / "meta.json"
    assert meta.exists()
