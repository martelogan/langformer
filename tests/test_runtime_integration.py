import json

from pathlib import Path

from langformer import TranspilationOrchestrator


def test_runtime_session_writes_metadata(tmp_path: Path):
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
    units_dir = run_dirs[0] / "units"
    unit_files = list(units_dir.glob("*.json"))
    assert unit_files
    payload = json.loads(unit_files[0].read_text())
    assert payload["status"] == "success"
