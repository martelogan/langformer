from __future__ import annotations

from pathlib import Path

from langformer.worker.transpile_worker import run_worker


def test_run_worker_uses_event_stream(tmp_path: Path, monkeypatch) -> None:
    source_code = "def main():\n    return 3\n"
    streams_dir = tmp_path / "streams"

    class _StubAdapter:
        def __init__(
            self,
            model,
            store_responses,
            timeout_s,
            jsonl_path,
            on_delta=None,
            **kwargs,
        ):
            self.path = Path(jsonl_path)
            self.on_delta = on_delta

        def stream(self, **kwargs):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("[]", encoding="utf-8")
            if self.on_delta:
                self.on_delta("delta-chunk")
            return {
                "output_text": source_code,
                "response_id": "stubbed-stream",
            }

    monkeypatch.setattr(
        "langformer.worker.transpile_worker.EventAdapter",
        _StubAdapter,
    )

    payload = {
        "source_language": "python",
        "target_language": "python",
        "unit": {
            "id": "unit",
            "language": "python",
            "kind": "module",
            "source_code": source_code,
        },
        "context": {
            "target_language": "python",
            "layout": {
                "output": {"path": str(tmp_path / "out.py"), "kind": "file"}
            },
            "build": {},
            "api_mappings": {},
            "feature_spec": {},
        },
        "llm": {"provider": "echo"},
        "prompt_paths": ["langformer/prompts/templates"],
        "agent": {"max_retries": 1, "temperature_range": (0.2, 0.6)},
        "verification": {"strategy": "exact_match"},
        "events": {
            "enabled": True,
            "model": "stub",
            "store_responses": False,
            "timeout_s": 5,
            "base_dir": str(streams_dir),
            "variant_label": "worker_variant",
        },
        "variant_label": "worker_variant",
        "shared_digests_dir": str(tmp_path / "digests"),
    }

    result = run_worker(0, payload)

    assert result["success"]
    files = result["files"]
    assert files, "expected worker to return files"
    assert source_code in next(iter(files.values()))
    assert result["notes"]["stream"]["response_id"] == "stubbed-stream"
    jsonl_files = list(streams_dir.rglob("*.jsonl"))
    assert jsonl_files, "expected stream log to be written"
    console_log = streams_dir / "unit" / "worker_variant" / "console.log"
    assert console_log.exists()
    assert "delta-chunk" in console_log.read_text()
    assert result["notes"]["dedup"]["status"] == "unique"
    digest_files = list((tmp_path / "digests").glob("*.json"))
    assert digest_files, "expected digest registration"
