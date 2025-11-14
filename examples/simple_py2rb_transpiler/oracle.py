"""Custom oracle that compares the Python reference implementation with Ruby output."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import textwrap

from pathlib import Path
from typing import Any, Dict, List

from langformer.types import Oracle, VerifyResult
from langformer.verification.oracles import OracleRegistry


def register_oracle() -> None:
    """Register the example oracle with the global registry."""

    registry = OracleRegistry.get_registry()
    registry.register("simple_py2rb", _build_oracle)


def _build_oracle(config: Dict[str, Any]) -> Oracle:
    cases: List[Dict[str, Any]] = config.get("cases") or []
    if not cases:
        cases = [
            {
                "scores": [1, 4, 10],
                "min_size": 3,
                "metrics": {"alice": 3, "bob": 2},
                "threshold": 3,
                "fmt": "plain",
            }
        ]

    def verify(
        source_code: str, target_code: str, metadata: Dict[str, Any]
    ) -> VerifyResult:
        python_namespace = _load_python_functions(source_code)
        py_outputs = [
            _run_python_case(python_namespace, case) for case in cases
        ]
        try:
            rb_outputs = [_run_ruby_case(target_code, case) for case in cases]
        except Exception as exc:  # pragma: no cover - defensive path
            return VerifyResult(
                passed=False,
                details={"strategy": "simple_py2rb", "error": str(exc)},
            )

        passed = py_outputs == rb_outputs
        return VerifyResult(
            passed=passed,
            details={
                "strategy": "simple_py2rb",
                "python_outputs": py_outputs,
                "ruby_outputs": rb_outputs,
                "cases": cases,
                "unit": metadata.get("unit"),
            },
        )

    return Oracle(verify=verify)


def _load_python_functions(source_code: str) -> Dict[str, Any]:
    namespace: Dict[str, Any] = {}
    exec(source_code, namespace, namespace)
    required = ["normalize_scores", "generate_report"]
    missing = [name for name in required if name not in namespace]
    if missing:
        raise ValueError(
            f"Source module missing required functions: {missing}"
        )
    return namespace


def _run_python_case(
    namespace: Dict[str, Any], case: Dict[str, Any]
) -> Dict[str, Any]:
    normalize = namespace["normalize_scores"]
    report = namespace["generate_report"]
    norm = normalize(case["scores"], min_size=case["min_size"])
    rep = report(case["metrics"], threshold=case["threshold"], fmt=case["fmt"])
    if case.get("fmt") == "json":
        rep = json.loads(rep)
    return {"normalize": norm, "report": rep}


def _run_ruby_case(candidate: str, case: Dict[str, Any]) -> Dict[str, Any]:
    temp_dir = Path(tempfile.mkdtemp(prefix="py2rb-oracle-"))
    try:
        candidate_path = temp_dir / "candidate.rb"
        candidate_path.write_text(candidate, encoding="utf-8")
        runner_path = temp_dir / "runner.rb"
        runner_path.write_text(_RUBY_HARNESS, encoding="utf-8")
        payload = json.dumps(case)
        proc = subprocess.run(
            ["ruby", str(runner_path), payload],
            capture_output=True,
            text=True,
            check=True,
        )
        stdout = proc.stdout.strip()
        if not stdout:
            raise RuntimeError(
                proc.stderr.strip() or "Ruby runner produced no output"
            )
        data = json.loads(stdout)
        if case.get("fmt") == "json":
            data["report"] = json.loads(data["report"])
        return data
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


_RUBY_HARNESS = textwrap.dedent(
    """
    require "json"
    require_relative "candidate"

    payload = JSON.parse(ARGV[0])
    scores = payload["scores"]
    min_size = payload["min_size"]
    metrics = payload["metrics"]
    threshold = payload["threshold"]
    fmt = payload["fmt"]

    def call_entry(fn_name, *args, **kwargs)
      if defined?(Report) && Report.respond_to?(fn_name)
        return Report.public_send(fn_name, *args, **kwargs)
      elsif respond_to?(fn_name)
        return method(fn_name).call(*args, **kwargs)
      else
        raise "Unable to locate method '#{fn_name}'"
      end
    end

    normalize = call_entry(:normalize_scores, scores, min_size: min_size)
    report = call_entry(:generate_report, metrics, threshold: threshold, fmt: fmt)

    puts JSON.generate({normalize: normalize, report: report})
    """
).strip()
