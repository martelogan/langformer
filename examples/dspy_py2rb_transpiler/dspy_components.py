"""DSPy-style modules and feedback optimizer for the Py→Rb example."""

from __future__ import annotations

import json

from typing import Any, Callable, Dict, List, Sequence

from langformer.artifacts import ArtifactManager
from langformer.constants import ARTIFACT_STAGE_TRANSPILER

try:  # pragma: no cover - optional dependency
    import dspy
except ImportError as exc:  # pragma: no cover - fail loudly
    raise RuntimeError(
        "DSPy is required for examples/dspy_py2rb_transpiler. "
        "Install it via `pip install dspy`."
    ) from exc


class ContextSummarySignature(dspy.Signature):  # type: ignore[misc]
    source_code = dspy.InputField(
        desc="Complete Python module that needs to be translated."
    )
    api_contract = dspy.InputField(
        desc="Docstring-level summary of the public API."
    )
    summary = dspy.OutputField(
        desc="Concise description of Python responsibilities and invariants."
    )


class Py2RbSignature(dspy.Signature):  # type: ignore[misc]
    python_summary = dspy.InputField(
        desc="Short narrative explaining what the Python module does."
    )
    api_contract = dspy.InputField(
        desc="Function/class signatures plus docstring highlights."
    )
    style_guide = dspy.InputField(
        desc="Ruby-specific style rules, namespace expectations, and API notes."
    )
    verifier_requirements = dspy.InputField(
        desc="Concrete behaviors/tests gathered from verification feedback."
    )
    verification_feedback = dspy.InputField(
        desc="Structured log from the latest oracle run (failures, mismatches, tests)."
    )
    source_code = dspy.InputField(
        desc="Original Python implementation for reference when translating."
    )
    output = dspy.OutputField(
        desc="Complete Ruby module/class that satisfies every requirement."
    )


class Py2RbProgram(dspy.Module):  # type: ignore[misc]
    """Two-stage DSPy-style program (summarize → transpile)."""

    def __init__(self) -> None:
        super().__init__()
        self.summarize = dspy.ChainOfThought(ContextSummarySignature)
        self.transpile = dspy.ChainOfThought(Py2RbSignature)

    def forward(  # type: ignore[override]
        self,
        source_code: str,
        *,
        unit_metadata: Dict[str, Any] | None = None,
        verification_feedback: List[Dict[str, Any]] | None = None,
    ):
        metadata = unit_metadata or {}
        summary_result = self.summarize(
            source_code=source_code,
            api_contract=metadata.get("api_contract", ""),
        )
        summary_text = getattr(summary_result, "summary", source_code)
        style_text = "\n".join(metadata.get("style_guide", []))
        requirements_text = _format_requirements(
            metadata.get("verifier_requirements")
        )
        feedback_text = _format_feedback(verification_feedback)
        return self.transpile(
            python_summary=summary_text,
            source_code=source_code,
            api_contract=metadata.get("api_contract", ""),
            style_guide=style_text,
            verifier_requirements=requirements_text,
            verification_feedback=feedback_text,
        )


def build_py2rb_program() -> Py2RbProgram:
    return Py2RbProgram()


class DSPyFeedbackOptimizer:
    """Collects verifier metrics and feeds them back into DSPy tasks."""

    def __init__(
        self,
        *,
        artifact_manager: ArtifactManager | None = None,
        compile_fn: Callable[[Callable[[], dspy.Module]], Callable[[], dspy.Module]]
        | None = None,
    ) -> None:
        self._artifact_manager = artifact_manager
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._compile_fn = compile_fn

    def compile_program(
        self, builder: Callable[[], dspy.Module]
    ) -> Callable[[], dspy.Module]:
        if self._compile_fn:
            return self._compile_fn(builder)
        return builder

    def record_feedback(
        self,
        unit_id: str,
        attempt: int,
        verification_note: Dict[str, Any],
    ) -> None:
        entry = {
            "attempt": attempt,
            "passed": bool(verification_note.get("passed")),
            "failures": list(verification_note.get("failures") or []),
            "details": verification_note,
        }
        self._history.setdefault(unit_id, []).append(entry)
        if not self._artifact_manager:
            return
        stage_dir = self._artifact_manager.stage_dir(
            ARTIFACT_STAGE_TRANSPILER, unit_id
        )
        payload = json.dumps(entry, indent=2)
        artifact_path = stage_dir / f"dspy_feedback_{attempt:02d}.json"
        artifact_path.write_text(payload, encoding="utf-8")
        self._artifact_manager.register(
            ARTIFACT_STAGE_TRANSPILER,
            unit_id,
            artifact_path,
            metadata={"attempt": attempt, "passed": entry["passed"]},
        )

    def build_overrides(self, unit_id: str) -> Dict[str, Any]:
        history = self._history.get(unit_id)
        if not history:
            return {}
        latest = history[-1]
        overrides: Dict[str, Any] = {}
        hints: List[str] = []
        for entry in reversed(history):
            for failure in entry.get("failures", []):
                hints.append(failure)
            if hints:
                break
        if hints:
            overrides["hints"] = hints
        details = latest["details"]
        overrides["verification_feedback"] = [details]
        style_hints = _style_hints_from_feedback(details)
        if style_hints:
            overrides["style_guide"] = style_hints
        requirement_summary = _requirements_from_feedback(details)
        if requirement_summary:
            overrides["verifier_requirements"] = requirement_summary
        return overrides


def _format_feedback(
    entries: Sequence[Dict[str, Any]] | None,
) -> str:
    if not entries:
        return ""
    blocks: List[str] = []
    for entry in entries:
        lines: List[str] = []
        if entry.get("failures"):
            lines.append("Failures: " + "; ".join(entry["failures"]))
        mismatch = entry.get("mismatch")
        if mismatch:
            case = mismatch.get("case")
            python = mismatch.get("python")
            ruby = mismatch.get("ruby")
            lines.append(
                "Mismatch case: "
                f"input={case}, python={python}, ruby={ruby}"
            )
        tests = entry.get("ruby_tests") or []
        failed = [
            test for test in tests if test.get("status") != "passed"
        ]
        if failed:
            for test in failed:
                lines.append(
                    f"Ruby test {test.get('name')} failed: "
                    f"{test.get('message')}"
                )
        syntax = entry.get("syntax")
        if syntax and not syntax.get("passed"):
            lines.append(f"Syntax error: {syntax.get('error')}")
        if not lines and entry:
            lines.append(json.dumps(entry, indent=2, sort_keys=True))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_requirements(raw: Dict[str, Any] | None) -> str:
    if not raw:
        return ""
    lines: List[str] = []
    cases = raw.get("cases") or []
    if cases:
        lines.append("Verification cases:")
        for case in cases:
            lines.append(f"- {case}")
    tests = raw.get("ruby_tests") or []
    if tests:
        lines.append("Ruby tests to satisfy:")
        for test in tests:
            lines.append(f"- {test}")
    notes = raw.get("notes") or []
    if notes:
        lines.append("Additional notes:")
        for note in notes:
            lines.append(f"- {note}")
    return "\n".join(lines)


def _style_hints_from_feedback(details: Dict[str, Any]) -> List[str]:
    hints: List[str] = []
    syntax = details.get("syntax")
    if syntax and not syntax.get("passed"):
        hints.append(
            "Candidate must compile cleanly (fix syntax error: "
            f"{syntax.get('error')})."
        )
    mismatch = details.get("mismatch")
    if mismatch:
        case = mismatch.get("case")
        hints.append(
            f"Ensure the Ruby implementation produces the same scaled values "
            f"and report as Python for {case}."
        )
    ruby_tests = details.get("ruby_tests") or []
    for test in ruby_tests:
        if test.get("status") != "passed":
            hints.append(
                f"Fix {test.get('name')}: {test.get('message')}"
            )
    return hints


def _requirements_from_feedback(details: Dict[str, Any]) -> Dict[str, Any]:
    requirements: Dict[str, List[str]] = {}
    cases = details.get("cases") or []
    python_outputs = details.get("python") or []
    ruby_outputs = details.get("ruby") or []
    for idx, case in enumerate(cases):
        expected = python_outputs[idx] if idx < len(python_outputs) else None
        actual = ruby_outputs[idx] if idx < len(ruby_outputs) else None
        if expected:
            requirements.setdefault("cases", []).append(
                f"Input {case} → scaled={expected.get('scaled')} "
                f"report={expected.get('report')}"
            )
        if actual and expected and actual != expected:
            requirements.setdefault("cases", []).append(
                f"Mismatch observed: Ruby returned {actual} for {case}"
            )
    ruby_tests = details.get("ruby_tests") or []
    for test in ruby_tests:
        name = test.get("name")
        status = test.get("status")
        message = test.get("message")
        requirements.setdefault("ruby_tests", []).append(
            f"{name}: {status} ({message})"
        )
    return requirements
