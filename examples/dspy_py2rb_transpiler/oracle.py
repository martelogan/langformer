"""Oracle that executes python + ruby implementations for deterministic cases."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import textwrap

from pathlib import Path
from typing import Any, Dict, Iterable, List

from langformer.types import Oracle, VerifyResult
from langformer.verification.oracles import OracleRegistry


def register_oracle() -> None:
    """Expose the oracle under the ``dspy_py2rb`` identifier."""

    registry = OracleRegistry.get_registry()
    registry.register("dspy_py2rb", _build_oracle)


def _build_oracle(config: Dict[str, Any]) -> Oracle:
    cases: List[Dict[str, Any]] = config.get("cases") or [
        {"values": [1, 2, 3]},
        {"values": [5, 10]},
    ]
    ruby_tests: List[Path] = [
        Path(test_path) for test_path in config.get("ruby_tests", [])
    ]

    def verify(
        source_code: str,
        target_code: str,
        metadata: Dict[str, Any],
    ) -> VerifyResult:
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="dspy_py2rb_oracle"))
            candidate_path = temp_dir / "candidate.rb"
            candidate_path.write_text(target_code, encoding="utf-8")
            failures: List[str] = []
            details: Dict[str, Any] = {
                "strategy": "dspy_py2rb",
                "unit": metadata.get("unit"),
                "cases": cases,
            }
            try:
                _check_ruby_syntax(candidate_path)
                details["syntax"] = {"passed": True}
            except Exception as exc:
                failures.append(f"Ruby syntax error: {exc}")
                details["syntax"] = {"passed": False, "error": str(exc)}
            try:
                python_outputs = _run_python_cases(source_code, cases)
                try:
                    ruby_outputs = _run_ruby_cases(
                        candidate_path, temp_dir, cases
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    failures.append(f"Ruby execution failed: {exc}")
                    details["ruby_error"] = str(exc)
                    ruby_outputs = []
                mismatch = _capture_mismatch(
                    cases, python_outputs, ruby_outputs
                )
                details["python"] = python_outputs
                details["ruby"] = ruby_outputs
                if mismatch:
                    failures.append(
                        "Ruby outputs did not match Python results."
                    )
                    details["mismatch"] = mismatch
            except Exception as exc:  # pragma: no cover - defensive
                failures.append(f"Python execution failed: {exc}")
                details["python_error"] = str(exc)
            if ruby_tests:
                test_results = _run_ruby_tests(candidate_path, ruby_tests)
                details["ruby_tests"] = test_results
                failed_tests = [
                    test
                    for test in test_results
                    if test.get("status") != "passed"
                ]
                if failed_tests:
                    failures.append(
                        "Ruby tests failed: "
                        + ", ".join(test["name"] for test in failed_tests)
                    )
            passed = not failures
            if failures:
                details["failures"] = failures
            return VerifyResult(passed=passed, details=details)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return Oracle(verify=verify)


def _run_python_cases(source_code: str, cases: Iterable[Dict[str, Any]]):
    ns: Dict[str, Any] = {}
    exec(source_code, ns, ns)
    scale_scores = ns.get("scale_scores")
    render_report = ns.get("render_report")
    if not callable(scale_scores) or not callable(render_report):
        raise RuntimeError("Python source is missing scale_scores/render_report")
    outputs = []
    for case in cases:
        values = case["values"]
        scaled = scale_scores(values)
        report = render_report(values)
        outputs.append({"scaled": scaled, "report": report})
    return outputs


def _check_ruby_syntax(candidate_path: Path) -> None:
    proc = subprocess.run(
        ["ruby", "-c", str(candidate_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr.strip()
            or proc.stdout.strip()
            or "Ruby syntax check failed."
        )


def _run_ruby_cases(
    candidate_path: Path, temp_dir: Path, cases: Iterable[Dict[str, Any]]
):
    harness_path = temp_dir / "runner.rb"
    harness_path.write_text(_RUBY_HARNESS, encoding="utf-8")
    outputs = []
    for case in cases:
        payload = json.dumps(case)
        proc = subprocess.run(
            ["ruby", str(harness_path), payload],
            capture_output=True,
            text=True,
            check=True,
        )
        if not proc.stdout.strip():
            raise RuntimeError(proc.stderr.strip() or "Ruby produced no output")
        outputs.append(json.loads(proc.stdout.strip()))
    return outputs


def _run_ruby_tests(candidate_path: Path, tests: Iterable[Path]):
    if not tests:
        return []
    env = os.environ.copy()
    env["CANDIDATE_PATH"] = str(candidate_path.resolve())
    results = []
    for test_path in tests:
        proc = subprocess.run(
            ["ruby", str(test_path)],
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            output = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
            results.append(
                {
                    "name": test_path.name,
                    "status": "failed",
                    "message": output,
                }
            )
        else:
            results.append(
                {
                    "name": test_path.name,
                    "status": "passed",
                    "message": proc.stdout.strip(),
                }
            )
    return results


def _capture_mismatch(
    cases: Iterable[Dict[str, Any]],
    python_outputs: Iterable[Dict[str, Any]],
    ruby_outputs: Iterable[Dict[str, Any]],
) -> Dict[str, Any] | None:
    case_list = list(cases)
    py = list(python_outputs)
    rb = list(ruby_outputs)
    for idx, case in enumerate(case_list):
        expected = py[idx] if idx < len(py) else None
        actual = rb[idx] if idx < len(rb) else None
        if expected != actual:
            return {"case": case, "python": expected, "ruby": actual}
    return None


_RUBY_HARNESS = textwrap.dedent(
    """
    require "json"
    require_relative "candidate"

    payload = JSON.parse(ARGV[0])
    values = payload["values"]

    unless defined?(Report)
      raise "Report module not defined"
    end

    scaled = Report.scale_scores(values)
    report = Report.render_report(values)

    puts JSON.generate({scaled: scaled, report: report})
    """
).strip()
