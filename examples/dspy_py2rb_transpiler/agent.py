"""Custom DSPy-powered transpiler agent for the Py→Rb example."""

from __future__ import annotations

import ast
import json
import logging
import re

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

try:
    import dspy
except ImportError as exc:  # pragma: no cover - defensive
    raise RuntimeError(
        "DSPy is required for Py2RbDSPyTranspilerAgent. "
        "Install it via `pip install dspy`."
    ) from exc

from examples.dspy_py2rb_transpiler.dspy_components import (
    DSPyFeedbackOptimizer,
    build_py2rb_program,
)
from langformer.agents.base import LLMConfig
from langformer.agents.transpiler import (
    TranspilerAgent,
    build_prompt_task_spec,
)
from langformer.languages.base import LanguagePlugin
from langformer.types import (
    CandidatePatchSet,
    IntegrationContext,
    TranspileUnit,
)

# TODO: ideally, we lean fully on DSPy signatures without any hardcoded prompts.
DEFAULT_STYLE_GUIDE: tuple[str, ...] = (
    "Mirror the Python module's public API exactly—including method names, "
    "arguments, and return semantics.",
    "Prefer a Ruby module/class namespace rather than defining helpers on "
    "the top level.",
    "Keep the output deterministic: no debug logging, puts statements, or "
    "markdown formatting.",
    "Translate Python collections and truthiness rules idiomatically "
    "(lists → Arrays, dicts → Hashes, booleans → true/false).",
)


class LangformerDSPyLM(dspy.BaseLM):  # type: ignore[misc]
    """Wrap Langformer providers so DSPy modules can consume them."""

    def __init__(self, provider, *, temperature: float = 0.2) -> None:
        super().__init__(model="langformer/dspy", temperature=temperature)
        self._provider = provider

    def forward(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
        if messages:
            user_prompt = "\n".join(
                msg.get("content", "") for msg in messages if isinstance(msg, dict)
            )
        else:
            user_prompt = prompt or ""
        temperature = kwargs.get("temperature", self.kwargs.get("temperature"))
        text = self._provider.generate(
            user_prompt,
            metadata={"source": "dspy"},
            temperature=temperature,
        )
        usage = {
            "prompt_tokens": len(user_prompt),
            "completion_tokens": len(text),
            "total_tokens": len(user_prompt) + len(text),
        }
        message = SimpleNamespace(role="assistant", content=text)
        choice = SimpleNamespace(
            index=0,
            message=message,
            finish_reason="stop",
        )
        return SimpleNamespace(
            id="langformer-dspy",
            model="langformer/dspy",
            usage=usage,
            choices=[choice],
        )


class Py2RbDSPyTranspilerAgent(TranspilerAgent):
    """Transpiler that routes prompt specs through a DSPy program + optimizer."""

    def __init__(
        self,
        source_plugin: LanguagePlugin,
        target_plugin: LanguagePlugin,
        *,
        llm_config: LLMConfig,
        module_factory: Optional[Callable[[], Any]] = None,
        optimizer: DSPyFeedbackOptimizer | None = None,
        max_retries: int = 2,
        parallel_workers: int = 1,
        temperature_range: tuple[float, float] = (0.2, 0.6),
        event_adapter_factory=None,
        prompt_renderer=None,
        prompt_template_map=None,
    ) -> None:
        self._source_plugin = source_plugin
        self._target_plugin = target_plugin
        self._llm_config = llm_config
        self._artifact_manager = llm_config.artifact_manager
        self._module_factory = module_factory or build_py2rb_program
        self._optimizer = optimizer or DSPyFeedbackOptimizer(
            artifact_manager=self._artifact_manager
        )
        self._program_factory = self._optimizer.compile_program(
            self._module_factory
        )
        self._max_retries = max(1, max_retries)
        self._event_factory = event_adapter_factory
        self._logger = logging.getLogger(__name__)
        self._configure_dspy()

    def _configure_dspy(self) -> None:
        try:
            dspy.configure(
                lm=LangformerDSPyLM(
                    self._llm_config.provider,
                    temperature=self._llm_config.provider.__dict__.get(
                        "temperature", 0.2
                    ),
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.warning("Failed to configure DSPy LM: %s", exc)

    def transpile(
        self,
        unit: TranspileUnit,
        ctx: IntegrationContext,
        verifier=None,
        *,
        event_factory=None,
        variant_label=None,
        dedup_handler=None,
        cancel_event=None,
    ) -> CandidatePatchSet:
        target_path = ctx.layout.target_path(unit.id)
        feedback: Optional[str] = None
        previous_code: Optional[str] = None
        for attempt in range(1, self._max_retries + 1):
            if (
                cancel_event is not None
                and getattr(cancel_event, "is_set", lambda: False)()
            ):
                raise RuntimeError("Transpiler canceled by orchestrator")
            spec = build_prompt_task_spec(
                self._source_plugin,
                self._target_plugin,
                unit,
                ctx,
                feedback,
                attempt,
                previous_code,
            )
            module_name = _derive_ruby_namespace(unit, ctx)
            spec.metadata["ruby_namespace"] = module_name
            spec.metadata.setdefault(
                "style_guide", _build_style_guide(ctx, module_name)
            )
            spec.metadata.setdefault(
                "verifier_requirements",
                _build_requirement_outline(ctx),
            )
            spec.metadata["api_contract"] = _describe_python_api(
                unit.source_code or ""
            )
            overrides = self._optimizer.build_overrides(unit.id)
            style_overrides = None
            requirement_overrides: Dict[str, Any] | None = None
            if overrides:
                style_overrides = overrides.pop("style_guide", None)
                requirement_overrides = overrides.pop(
                    "verifier_requirements", None
                )
                spec.metadata = _merge_metadata(spec.metadata, overrides)
            if style_overrides:
                spec.metadata.setdefault(
                    "style_guide", _build_style_guide(ctx, module_name)
                )
                spec.metadata["style_guide"].extend(style_overrides)
            if requirement_overrides:
                base_requirements = spec.metadata.setdefault(
                    "verifier_requirements", {}
                )
                spec.metadata["verifier_requirements"] = _merge_metadata(
                    base_requirements, requirement_overrides
                )

            program = self._program_factory()
            result = program(
                source_code=unit.source_code or "",
                unit_metadata={
                    "api_contract": spec.metadata.get("api_contract", ""),
                    "style_guide": spec.metadata.get("style_guide", []),
                    "hints": spec.metadata.get("hints", []),
                    "verifier_requirements": spec.metadata.get(
                        "verifier_requirements", {}
                    ),
                    "ruby_namespace": spec.metadata.get("ruby_namespace", ""),
                },
                verification_feedback=spec.metadata.get(
                    "verification_feedback"
                ),
            )
            code = getattr(result, "output", "")
            notes: Dict[str, Any] = {
                "attempt": attempt,
                "prompt_task": {"kind": spec.kind, "engine": "dspy_py2rb"},
            }
            candidate = CandidatePatchSet(
                files={target_path: code}, notes=notes
            )
            previous_code = code
            if verifier is None:
                return candidate
            verification = verifier.verify(unit, candidate, ctx)
            verification_note = verification.feedback.to_dict()
            candidate.notes["verification"] = verification_note
            self._optimizer.record_feedback(unit.id, attempt, verification_note)
            self._logger.info(
                "Verifier result for unit %s (attempt %d): passed=%s, feedback=%s",
                unit.id,
                attempt,
                verification.passed,
                verification_note,
            )
            if verification.passed:
                return candidate
            feedback = json.dumps(verification_note)
        raise RuntimeError(
            f"DSPy agent failed to transpile unit {unit.id} within retry budget"
        )


def _merge_metadata(
    base: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and isinstance(base.get(key), dict)
        ):
            base[key] = _merge_metadata(
                base.get(key, {}).copy(), value
            )
        else:
            base[key] = value
    return base


def _build_style_guide(
    ctx: IntegrationContext, module_name: str
) -> List[str]:
    feature_spec = ctx.feature_spec or {}
    style = list(DEFAULT_STYLE_GUIDE)
    if module_name:
        style.append(
            f"Wrap the translated helpers inside a Ruby module or class "
            f"named `{module_name}`."
        )
    extra = feature_spec.get("style_guide")
    if isinstance(extra, (list, tuple)):
        style.extend(str(entry) for entry in extra if entry)
    return style


def _build_requirement_outline(ctx: IntegrationContext) -> Dict[str, Any]:
    feature_spec = ctx.feature_spec or {}
    outline: Dict[str, List[str]] = {}
    for case in feature_spec.get("verification_cases") or []:
        try:
            payload = json.dumps(case, sort_keys=True)
        except TypeError:
            payload = str(case)
        outline.setdefault("cases", []).append(
            f"Ruby results must match the Python implementation for input {payload}."
        )
    for test_path in feature_spec.get("ruby_tests") or []:
        outline.setdefault("ruby_tests", []).append(
            f"ruby {test_path} must pass after requiring the generated file."
        )
    extra_notes = feature_spec.get("verifier_notes")
    if isinstance(extra_notes, (list, tuple)):
        outline.setdefault("notes", []).extend(
            str(entry) for entry in extra_notes if entry
        )
    return outline


def _derive_ruby_namespace(unit: TranspileUnit, ctx: IntegrationContext) -> str:
    feature_spec = ctx.feature_spec or {}
    explicit = feature_spec.get("ruby_namespace")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    slug = unit.id or "transpiled_module"
    parts = re.split(r"[^0-9a-zA-Z]+", slug)
    tokens = [token for token in parts if token]
    if not tokens:
        return "TranspiledModule"
    return "".join(token.capitalize() for token in tokens)


def _describe_python_api(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "Python module could not be parsed."
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            params = [arg.arg for arg in node.args.args]
            doc = ast.get_docstring(node) or ""
            doc_line = doc.splitlines()[0] if doc else ""
            signature = f"{node.name}({', '.join(params)})"
            if doc_line:
                lines.append(f"{signature}: {doc_line}")
            else:
                lines.append(signature)
        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            doc_line = doc.splitlines()[0] if doc else ""
            header = f"class {node.name}"
            lines.append(f"{header}: {doc_line or 'no docstring provided.'}")
    return "\n".join(lines) if lines else "Module defines no public functions."
