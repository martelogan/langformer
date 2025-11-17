"""DSPy-powered Python→Ruby transpiler example."""

from examples.dspy_py2rb_transpiler.agent import Py2RbDSPyTranspilerAgent
from examples.dspy_py2rb_transpiler.oracle import register_oracle

__all__ = ["Py2RbDSPyTranspilerAgent", "register_oracle"]
