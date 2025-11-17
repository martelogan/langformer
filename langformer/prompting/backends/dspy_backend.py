"""PromptTaskEngine backed by a minimal DSPy module."""

from __future__ import annotations

from typing import Callable, Optional

from langformer.prompting.backends.base import PromptTaskEngine
from langformer.prompting.types import PromptTaskResult, PromptTaskSpec

try:  # pragma: no cover - optional dependency
    import dspy
except ImportError:  # pragma: no cover - optional dependency
    dspy = None


class BasicDSPyTranspiler(PromptTaskEngine):
    """Simple DSPy-based prompt engine for transpilation tasks."""

    def __init__(
        self,
        module_factory: Optional[Callable[[], "dspy.Module"]] = None,
    ) -> None:
        if module_factory is None and dspy is None:
            raise RuntimeError(
                "DSPy is not installed. Install the 'dspy' package to use "
                "BasicDSPyTranspiler or provide a custom module_factory."
            )
        self._module_factory = module_factory or self._default_factory

    def _default_factory(self) -> "dspy.Module":
        if dspy is None:  # pragma: no cover - guarded by __init__
            raise RuntimeError(
                "DSPy is not installed. Install the 'dspy' package to use "
                "BasicDSPyTranspiler."
            )

        class TranspileSignature(dspy.Signature):
            source_code = dspy.InputField()
            source_language = dspy.InputField()
            target_language = dspy.InputField()
            unit_metadata = dspy.InputField()
            hints = dspy.InputField()
            verification_feedback = dspy.InputField()
            output = dspy.OutputField(desc="Target-language code")

        class TranspileModule(dspy.Module):
            def __init__(self) -> None:
                super().__init__()
                self._step = dspy.ChainOfThought(TranspileSignature)

            def forward(self, **kwargs):
                return self._step(**kwargs)

        return TranspileModule()

    def run(self, spec: PromptTaskSpec) -> PromptTaskResult:
        metadata = spec.metadata
        module = self._module_factory()
        result = module(
            source_code=metadata.get("source_code", ""),
            source_language=metadata.get("source_language", ""),
            target_language=metadata.get("target_language", ""),
            unit_metadata=metadata.get("unit_metadata", {}),
            hints=metadata.get("hints", []),
            verification_feedback=metadata.get(
                "verification_feedback", []
            ),
        )
        text = getattr(result, "output", str(result))
        return PromptTaskResult(
            output_type="code",
            output=text or "",
            metadata={
                "source": "task_engine",
                "engine": type(module).__name__,
            },
        )
