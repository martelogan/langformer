"""Lightweight Python language plugin used for early scaffolding."""

from __future__ import annotations

import ast

from typing import Any, Dict, List, Optional

from langformer.languages.base import LanguagePlugin
from langformer.types import TranspileUnit


class LightweightPythonLanguagePlugin(LanguagePlugin):
    """Parses and executes Python code for analysis and verification."""

    language_name = "python"

    def parse(self, source_code: str) -> ast.AST:
        return ast.parse(source_code)

    def compile(self, code: str) -> bool:
        compile(code, "<langformer>", "exec")
        return True

    def execute(
        self, code: str, inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        namespace: Dict[str, Any] = {}
        exec(code, {}, namespace)
        inputs = inputs or {}
        if "main" in namespace and callable(namespace["main"]):
            return {"result": namespace["main"](**inputs)}
        return namespace

    def partition_units(
        self,
        source_code: str,
        unit_id: str,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[TranspileUnit]]:
        cfg = config or {}
        if not cfg.get("split_functions"):
            return None
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return None

        units: List[TranspileUnit] = []
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.FunctionDef):
                snippet = ast.get_source_segment(source_code, node) or ""
                units.append(
                    TranspileUnit(
                        id=f"{unit_id}:{node.name}",
                        language=self.name,
                        kind="function",
                        source_code=snippet,
                        source_ast=node,
                        metadata={
                            "partition": "python_function",
                            "function": node.name,
                        },
                    )
                )
        return units or None


__all__ = ["LightweightPythonLanguagePlugin"]
