from pathlib import Path

from langformer.orchestration.target_integrator import TargetIntegrator
from langformer.types import CandidatePatchSet


def test_integration_agent_combines_candidates(tmp_path: Path) -> None:
    agent = TargetIntegrator()
    c1 = CandidatePatchSet(
        files={tmp_path / "a.py": "def one():\n    return 1"}
    )
    c2 = CandidatePatchSet(
        files={tmp_path / "b.py": "def two():\n    return 2"}
    )

    combined = agent.combine_candidates([c1, c2], tmp_path / "out.py")

    contents = combined.files[tmp_path / "out.py"]
    assert "def one" in contents
    assert "def two" in contents
