from __future__ import annotations

from pathlib import Path
from types import MethodType

from langformer.artifacts import ArtifactManager
from langformer.configuration import ArtifactSettings
from langformer.constants import (
    ARTIFACT_STAGE_ANALYZER,
    ARTIFACT_STAGE_TRANSPILER,
    ARTIFACT_STAGE_VERIFIER,
)
from langformer.orchestrator import TranspilationOrchestrator
from langformer.types import CandidatePatchSet, TranspileUnit, VerifyResult


def test_artifact_manager_tracks_manifest(tmp_path: Path) -> None:
    settings = ArtifactSettings(root=tmp_path / "artifacts")
    manager = ArtifactManager(settings)

    stage_dir = manager.stage_dir(ARTIFACT_STAGE_ANALYZER, "unit-1")
    assert stage_dir.exists()

    artifact_path = stage_dir / "summary.json"
    artifact_path.write_text("{}", encoding="utf-8")
    manager.register(
        ARTIFACT_STAGE_ANALYZER,
        "unit-1",
        artifact_path,
        metadata={"kind": "summary"},
    )

    manifest = manager.manifest_for("unit-1")
    assert ARTIFACT_STAGE_ANALYZER in manifest
    entry = manifest[ARTIFACT_STAGE_ANALYZER][0]
    assert entry["path"] == str(artifact_path)
    assert entry["metadata"] == {"kind": "summary"}

    manager.reset("unit-1", preserve=[ARTIFACT_STAGE_ANALYZER])
    manifest = manager.manifest_for("unit-1")
    assert ARTIFACT_STAGE_ANALYZER in manifest

    manager.reset("unit-1")
    assert manager.manifest_for("unit-1") == {}


def test_artifact_manager_reset_all(tmp_path: Path) -> None:
    settings = ArtifactSettings(root=tmp_path / "artifacts")
    manager = ArtifactManager(settings)
    stage_dir_a = manager.stage_dir(ARTIFACT_STAGE_ANALYZER, "unit-a")
    sample_a = stage_dir_a / "sample.json"
    sample_a.write_text("{}", encoding="utf-8")
    manager.register(
        ARTIFACT_STAGE_ANALYZER,
        "unit-a",
        sample_a,
        metadata={"kind": "sample"},
    )

    stage_dir_b = manager.stage_dir(ARTIFACT_STAGE_ANALYZER, "unit-b")
    sample_b = stage_dir_b / "sample.json"
    sample_b.write_text("{}", encoding="utf-8")
    manager.register(
        ARTIFACT_STAGE_ANALYZER,
        "unit-b",
        sample_b,
        metadata={"kind": "sample"},
    )
    manager.reset_all()
    assert manager.manifest_for("unit-a") == {}
    assert manager.manifest_for("unit-b") == {}


def test_candidate_tests_and_feedback() -> None:
    candidate = CandidatePatchSet(
        files={Path("out.py"): "print('hi')"},
        notes={"variant": "orchestrator"},
    )
    rel = candidate.add_test("tests/test_sample.py", "assert 1 == 1\n")
    assert rel in candidate.tests

    result = VerifyResult(
        passed=True,
        details={"strategy": "custom"},
    )
    result.details.add_test("tests/test_verify.py", "assert True\n")
    assert list(result.details.tests.keys())[0] == Path("tests/test_verify.py")


def test_orchestrator_records_stage_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    sample_input = tmp_path / "inputs" / "sample.py"
    sample_input.parent.mkdir(parents=True, exist_ok=True)
    sample_input.write_text("print('source')\n", encoding="utf-8")
    sample_output = tmp_path / "outputs" / "sample.rb"
    config = {
        "transpilation": {
            "source_language": "python",
            "target_language": "ruby",
            "layout": {
                "input": {"path": str(sample_input)},
                "output": {"path": str(sample_output)},
            },
            "llm": {"provider": "echo"},
            "runtime": {
                "enabled": False,
                "run_root": str(tmp_path / "runs"),
            },
            "artifacts": {"root": str(tmp_path / "artifacts")},
        }
    }
    orchestrator = TranspilationOrchestrator(config=config)

    def fake_analyze(self, source_code, unit_id, kind="module"):
        manager = self._artifact_manager
        stage_dir = manager.stage_dir(ARTIFACT_STAGE_ANALYZER, unit_id)
        artifact_path = stage_dir / "summary.txt"
        artifact_path.write_text("summary", encoding="utf-8")
        manager.register(
            ARTIFACT_STAGE_ANALYZER,
            unit_id,
            artifact_path,
            metadata={"kind": "summary"},
        )
        return [
            TranspileUnit(
                id=unit_id,
                language=self._language_plugin.name,
                source_code=source_code,
            )
        ]

    def fake_transpile(self, unit, ctx, verifier=None, **kwargs):
        manager = self._artifact_manager
        stage_dir = manager.stage_dir(ARTIFACT_STAGE_TRANSPILER, unit.id)
        artifact_path = stage_dir / "candidate.rb"
        artifact_path.write_text("puts 'draft'\n", encoding="utf-8")
        manager.register(
            ARTIFACT_STAGE_TRANSPILER,
            unit.id,
            artifact_path,
            metadata={"kind": "candidate"},
        )
        files = {ctx.layout.target_path(unit.id): "puts 'ok'\n"}
        return CandidatePatchSet(files=files, notes={"variant": "stub"})

    def fake_verify(self, unit, candidate, ctx):
        result = VerifyResult(
            passed=True,
            details={"strategy": "stub"},
        )
        result.details.add_test(
            "tests/test_transpiled.rb", "assert true\n"
        )
        return result

    monkeypatch.setattr(
        orchestrator.source_analyzer,
        "analyze",
        MethodType(fake_analyze, orchestrator.source_analyzer),
    )
    monkeypatch.setattr(
        orchestrator.transpiler_agent,
        "transpile",
        MethodType(fake_transpile, orchestrator.transpiler_agent),
    )
    monkeypatch.setattr(
        orchestrator.verification_agent,
        "verify",
        MethodType(fake_verify, orchestrator.verification_agent),
    )

    candidate = orchestrator._run_pipeline(
        "print('hi')",
        "unit_artifacts",
        verify=True,
    )

    manifest = orchestrator._artifact_manager.manifest_for(
        "unit_artifacts"
    )
    assert ARTIFACT_STAGE_ANALYZER in manifest
    assert ARTIFACT_STAGE_TRANSPILER in manifest
    assert ARTIFACT_STAGE_VERIFIER in manifest
    verifier_entry = manifest[ARTIFACT_STAGE_VERIFIER][0]
    assert Path(verifier_entry["path"]).exists()

    artifacts_note = candidate.notes["artifacts"]
    assert ARTIFACT_STAGE_ANALYZER in artifacts_note
    assert ARTIFACT_STAGE_TRANSPILER in artifacts_note
    assert ARTIFACT_STAGE_VERIFIER in artifacts_note
    assert Path("tests/test_transpiled.rb") in candidate.tests
