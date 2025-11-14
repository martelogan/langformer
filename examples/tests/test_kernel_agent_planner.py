from pathlib import Path

from examples.kernel_agent_delegate import KernelAgentAutoPlanner


def test_kernel_agent_planner_routes_direct(tmp_path: Path):
    source = tmp_path / "simple.py"
    source.write_text("def main(x):\n    return x + 1\n")
    cache_path = tmp_path / "cache.json"
    planner = KernelAgentAutoPlanner({"enforce": True, "cache_path": str(cache_path)})
    plan = planner.plan(source, {})
    assert plan.action == "delegate"
    assert plan.context["route"] == "kernel_agent"
    assert plan.context["route_source"] == "analysis"
    assert plan.context["solver"] == "kernelagent"

    # Second run should hit cache and still return kernel_agent route
    planner_cached = KernelAgentAutoPlanner({"enforce": True, "cache_path": str(cache_path)})
    cached_plan = planner_cached.plan(source, {})
    assert cached_plan.context["route_source"] == "cache"


def test_kernel_agent_planner_delegate(tmp_path: Path):
    source = tmp_path / "conv.py"
    source.write_text(
        "import torch.nn.functional as F\n"
        "def main(x):\n"
        "    return F.conv_transpose2d(x, x)\n"
    )
    planner = KernelAgentAutoPlanner({"enforce": True, "cache_path": str(tmp_path / "cache.json")})
    plan = planner.plan(source, {})
    assert plan.action == "delegate"
    assert plan.context["delegate"] == "kernel_agent_delegate"
    assert plan.context["solver"] == "fuser"
    assert plan.context["route"] == "fuser"


def test_kernel_agent_planner_force_delegate(tmp_path: Path):
    source = tmp_path / "relu.py"
    source.write_text("import torch\n\ndef main(x):\n    return torch.relu(x)\n")
    planner = KernelAgentAutoPlanner(
        {"force_delegate": True, "delegate_name": "custom_delegate", "cache_path": str(tmp_path / "cache.json")}
    )
    plan = planner.plan(source, {})
    assert plan.action == "delegate"
    assert plan.context["delegate"] == "custom_delegate"
