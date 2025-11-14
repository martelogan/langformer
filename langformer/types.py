"""Core dataclasses used throughout the Langformer framework."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Protocol,
    cast,
)

if TYPE_CHECKING:  # pragma: no cover
    from langformer.artifacts import ArtifactManager


@dataclass
class AnalyzerMetadata(MutableMapping[str, Any]):
    """Structured metadata produced by the analyzer."""

    data: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterable[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def update(self, other: Dict[str, Any]) -> None:
        self.data.update(other)

    def setdefault(self, key: str, default: Any = None) -> Any:
        return self.data.setdefault(key, default)

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()


@dataclass
class TranspilerMetadata(MutableMapping[str, Any]):
    """Structured metadata produced by the transpiler."""

    data: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterable[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def update(self, other: Dict[str, Any]) -> None:
        self.data.update(other)


@dataclass
class VerificationFeedback(MutableMapping[str, Any]):
    """Structured verification feedback, including generated tests."""

    data: Dict[str, Any] = field(default_factory=dict)
    tests: Dict[Path, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tests = {Path(path): contents for path, contents in self.tests.items()}

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterable[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def update(self, other: Dict[str, Any]) -> None:
        self.data.update(other)

    def add_test(self, relative_path: Path | str, contents: str) -> Path:
        path = Path(relative_path)
        self.tests[path] = contents
        return path

    def setdefault(self, key: str, default: Any = None) -> Any:
        return self.data.setdefault(key, default)

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()


class VerifyResultProtocol(Protocol):
    """Protocol used to avoid circular imports within type annotations."""

    passed: bool


@dataclass
class TranspileUnit:
    """Description of a unit of work that needs to be transpiled."""

    id: str
    language: str
    kind: str = "module"
    source_code: Optional[str] = None
    source_ast: Optional[Any] = None
    behavior: Optional[Any] = None
    semspec: Optional[Any] = None
    entrypoints: Optional[List[str]] = None
    metadata: AnalyzerMetadata | Dict[str, Any] = field(
        default_factory=AnalyzerMetadata
    )

    def __post_init__(self) -> None:
        if isinstance(self.metadata, dict):
            self.metadata = AnalyzerMetadata(self.metadata)


@dataclass
class Oracle:
    """Wrapper that executes a custom oracle to compare behaviors."""

    verify: Callable[[str, str, Dict[str, Any]], VerifyResultProtocol]


@dataclass
class LayoutPlan:
    """Encapsulates layout metadata and resolved output helpers."""

    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.raw = deepcopy(self.raw)
        self._output = dict(self.raw.get("output") or {})
        self._module_path = self.raw.get("module_path")

    def as_dict(self) -> Dict[str, Any]:
        return deepcopy(self.raw)

    def target_path(self, unit_id: str) -> Path:
        output = self._output
        resolved = output.get("resolved_path")
        if resolved:
            return Path(resolved)
        kind = str(output.get("kind", "file")).lower()
        if kind == "directory":
            base_dir = Path(output.get("path") or ".")
            filename = output.get("filename")
            if not filename:
                extension = output.get("extension")
                if extension:
                    extension = str(extension)
                    if not extension.startswith("."):
                        extension = f".{extension}"
                    filename = f"{unit_id}{extension}"
                else:
                    filename = f"{unit_id}.out"
            return base_dir / filename
        raw_path = output.get("path") or self._module_path
        if raw_path:
            return Path(str(raw_path))
        extension = output.get("extension")
        if extension:
            extension = str(extension)
            if not extension.startswith("."):
                extension = f".{extension}"
            return Path(f"{unit_id}{extension}")
        return Path(f"{unit_id}.out")


@dataclass
class IntegrationContext:
    """Target language context plus runtime and layout metadata."""

    target_language: str
    runtime_adapter: Optional[str] = None
    contract: Optional[Any] = None
    layout: LayoutPlan = field(default_factory=LayoutPlan)
    build: Dict[str, Any] = field(default_factory=dict)
    oracle: Optional[Oracle] = None
    api_mappings: Dict[str, str] = field(default_factory=dict)
    feature_spec: Dict[str, bool] = field(default_factory=dict)
    artifacts: Optional["ArtifactManager"] = None


@dataclass
class CandidatePatchSet:
    """Container for generated target code artifacts."""

    files: Dict[Path, str] = field(default_factory=dict)
    patches: List[str] = field(default_factory=list)
    notes: TranspilerMetadata | Dict[str, Any] = field(
        default_factory=TranspilerMetadata
    )
    tests: Dict[Path, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._ensure_metadata()
        self.files = {
            Path(path): contents for path, contents in self.files.items()
        }
        self.tests = {
            Path(path): contents for path, contents in self.tests.items()
        }

    @property
    def metadata(self) -> TranspilerMetadata:
        return self._ensure_metadata()

    @metadata.setter
    def metadata(self, value: TranspilerMetadata | Dict[str, Any]) -> None:
        self.notes = (
            value
            if isinstance(value, TranspilerMetadata)
            else TranspilerMetadata(value)
        )

    def add_test(self, relative_path: Path | str, contents: str) -> Path:
        path = Path(relative_path)
        self.tests[path] = contents
        return path

    def _ensure_metadata(self) -> TranspilerMetadata:
        if isinstance(self.notes, dict):
            self.notes = TranspilerMetadata(self.notes)
        return self.notes


@dataclass
class VerifyResult:
    """Outcome from running a verification strategy."""

    passed: bool
    details: VerificationFeedback | Dict[str, Any] = field(
        default_factory=VerificationFeedback
    )
    cost: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.details, dict):
            self.details = VerificationFeedback(self.details)

    @property
    def feedback(self) -> VerificationFeedback:
        return cast(VerificationFeedback, self.details)
