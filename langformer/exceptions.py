"""Custom exceptions for the transpilation framework."""


class TranspilationError(RuntimeError):
    """Base exception for orchestration failures."""


class TranspilationAttemptError(TranspilationError):
    """Raised when the transpiler exhausts all retries without success."""


class VerificationFailedError(TranspilationError):
    """Raised when verification reports a failure for a candidate."""
