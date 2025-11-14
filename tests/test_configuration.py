from __future__ import annotations

from pathlib import Path

from langformer.configuration import build_transpilation_settings


def test_build_transpilation_settings_resolves_paths(tmp_path: Path) -> None:
    config = {
        "transpilation": {
            "source_language": "python",
            "target_language": "ruby",
            "llm": {
                "provider": "openai",
                "streaming": {"enabled": True, "log_root": "streams"},
            },
            "agents": {
                "max_retries": 4,
                "parallel_workers": 2,
                "temperature_range": [0.1, 0.4],
                "prompt_dir": "prompts/overrides",
            },
            "runtime": {"enabled": True, "run_root": "runs"},
            "verification": {
                "strategy": "execution_match",
                "test_inputs": [{"a": 1}],
                "runner": {"kind": "sandbox", "run_root": "ver/runs"},
            },
        }
    }
    settings = build_transpilation_settings(config, config_root=tmp_path)
    assert settings.agents.max_retries == 4
    assert settings.agents.parallel_workers == 2
    assert settings.agents.temperature_range == (0.1, 0.4)
    assert len(settings.agents.prompt_overrides) == 1
    assert (
        settings.agents.prompt_overrides[0]
        == (tmp_path / "prompts/overrides").resolve()
    )
    assert settings.runtime.enabled is True
    assert settings.runtime.run_root == (tmp_path / "runs").resolve()
    assert settings.llm.streaming.enabled is True
    assert settings.llm.streaming.log_root == (tmp_path / "streams").resolve()
    assert settings.verification.strategy == "execution_match"
    assert settings.verification.test_inputs == [{"a": 1}]
    assert settings.verification.sandbox is not None
    assert (
        settings.verification.sandbox.run_root
        == (tmp_path / "ver/runs").resolve()
    )


def test_artifact_settings_defaults(tmp_path: Path) -> None:
    config = {
        "transpilation": {
            "source_language": "python",
            "target_language": "python",
            "runtime": {"run_root": str(tmp_path / "runs")},
        }
    }
    settings = build_transpilation_settings(config, config_root=tmp_path)
    assert (
        settings.artifacts.root
        == (tmp_path / "runs" / "artifacts").resolve()
    )
    assert settings.artifacts.analyzer_dir == "analyzer"
    assert settings.artifacts.transpiler_dir == "transpiler"
    assert settings.artifacts.verifier_dir == "verifier"


def test_artifact_settings_override(tmp_path: Path) -> None:
    config = {
        "transpilation": {
            "source_language": "python",
            "target_language": "python",
            "runtime": {"run_root": str(tmp_path / "runs")},
            "artifacts": {
                "root": "custom/artifacts",
                "analyzer_dir": "analysis",
                "transpiler_dir": "codegen",
                "verifier_dir": "checks",
            },
        }
    }
    settings = build_transpilation_settings(config, config_root=tmp_path)
    assert (
        settings.artifacts.root
        == (tmp_path / "custom/artifacts").resolve()
    )
    assert settings.artifacts.analyzer_dir == "analysis"
    assert settings.artifacts.transpiler_dir == "codegen"
    assert settings.artifacts.verifier_dir == "checks"
