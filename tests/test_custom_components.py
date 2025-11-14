from __future__ import annotations

from pathlib import Path

from langformer.configuration import IntegrationSettings
from langformer.orchestrator import TranspilationOrchestrator
from langformer.types import (
    CandidatePatchSet,
    IntegrationContext,
    LayoutPlan,
    TranspileUnit,
    VerifyResult,
)


class CustomSourceAnalyzer:
    instantiated = False
    analyzed_units: list[str] = []

    def __init__(
        self, language_plugin, llm_config, *, config=None
    ) -> None:  # pragma: no cover - simple flag
        CustomSourceAnalyzer.instantiated = True
        self.language_plugin = language_plugin
        self.config = config or {}
        self.llm_config = llm_config

    def analyze(self, source_code: str, unit_id: str, kind: str = "module"):
        CustomSourceAnalyzer.analyzed_units.append(unit_id)
        return [
            TranspileUnit(
                id=f"{unit_id}_custom",
                language=self.language_plugin.name,
                kind=kind,
                source_code=source_code,
            )
        ]


class CustomContextBuilder:
    build_calls: int = 0

    def build(
        self,
        integration: IntegrationSettings,
        layout: LayoutPlan,
        artifacts=None,
    ) -> IntegrationContext:
        CustomContextBuilder.build_calls += 1
        return IntegrationContext(
            target_language=integration.target_language,
            layout=layout,
        )


class CustomTranspilerAgent:
    transpile_calls: int = 0

    def __init__(
        self, source_plugin, target_plugin, llm_config, **kwargs
    ) -> None:  # pragma: no cover - simple store
        self.source_plugin = source_plugin
        self.target_plugin = target_plugin
        self.llm_config = llm_config

    def transpile(
        self,
        unit: TranspileUnit,
        ctx: IntegrationContext,
        verifier=None,
        **kwargs,
    ) -> CandidatePatchSet:
        CustomTranspilerAgent.transpile_calls += 1
        return CandidatePatchSet(
            files={Path("dummy.py"): "print('custom')"},
            notes={"unit": unit.id},
        )

    def verify(
        self,
        unit: TranspileUnit,
        candidate: CandidatePatchSet,
        ctx: IntegrationContext,
    ) -> VerifyResult:
        return VerifyResult(passed=True, details={"custom": True})


class CustomVerificationAgent:
    verify_calls: int = 0

    def __init__(
        self, strategy, source_plugin, target_plugin, llm_config
    ) -> None:  # pragma: no cover - store only
        self.strategy = strategy
        self.source_plugin = source_plugin
        self.target_plugin = target_plugin
        self.llm_config = llm_config

    def verify(
        self,
        unit: TranspileUnit,
        candidate: CandidatePatchSet,
        ctx: IntegrationContext,
    ) -> VerifyResult:
        CustomVerificationAgent.verify_calls += 1
        return VerifyResult(passed=True, details={"unit": unit.id})


class CustomTargetIntegrator:
    integrate_calls: int = 0

    def integrate(
        self, candidate: CandidatePatchSet, destination: Path | None = None
    ) -> Path:
        CustomTargetIntegrator.integrate_calls += 1
        dest = Path(destination or "custom_out.py")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(next(iter(candidate.files.values())))
        return dest

    def combine_candidates(
        self, candidates, module_path: Path
    ) -> CandidatePatchSet:
        return CandidatePatchSet(
            files={module_path: "\n".join(str(c.files) for c in candidates)}
        )


def test_custom_components_config(tmp_path: Path):
    CustomSourceAnalyzer.analyzed_units = []
    CustomTranspilerAgent.transpile_calls = 0
    CustomVerificationAgent.verify_calls = 0
    CustomTargetIntegrator.integrate_calls = 0
    CustomContextBuilder.build_calls = 0

    module_path = __name__
    config = {
        "transpilation": {
            "source_language": "python",
            "target_language": "python",
            "layout": {
                "input": {"path": ""},
                "output": {"path": str(tmp_path / "out.py")},
            },
            "agents": {
                "parallel_workers": 1,
            },
                "verification": {"strategy": "exact_match"},
                "components": {
                    "source_analyzer": (
                        f"{module_path}.CustomSourceAnalyzer"
                    ),
                    "context_builder": (
                        f"{module_path}.CustomContextBuilder"
                    ),
                    "transpiler_agent": (
                        f"{module_path}.CustomTranspilerAgent"
                    ),
                    "verification_agent": (
                        f"{module_path}.CustomVerificationAgent"
                    ),
                    "target_integrator": (
                        f"{module_path}.CustomTargetIntegrator"
                    ),
                },
            }
        }

    orchestrator = TranspilationOrchestrator(config=config)
    assert isinstance(orchestrator.context_builder, CustomContextBuilder)
    source_file = tmp_path / "src.py"
    source_file.write_text("print('hi')")
    target_file = tmp_path / "result.py"

    orchestrator.transpile_file(source_file, target_file, verify=True)

    assert target_file.read_text() == "print('custom')"
    assert isinstance(orchestrator.source_analyzer, CustomSourceAnalyzer)
    assert CustomSourceAnalyzer.analyzed_units == ["src"]
    assert isinstance(orchestrator.transpiler_agent, CustomTranspilerAgent)
    assert CustomTranspilerAgent.transpile_calls == 1
    assert isinstance(orchestrator.verification_agent, CustomVerificationAgent)
    assert isinstance(orchestrator.integration_agent, CustomTargetIntegrator)
    assert CustomVerificationAgent.verify_calls == 1
    assert CustomTargetIntegrator.integrate_calls == 1
    assert CustomContextBuilder.build_calls == 1
