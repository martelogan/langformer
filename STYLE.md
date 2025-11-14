# Langformer Style Guide

This project keeps things simple: follow the tooling that already ships with the repo and you’ll match the expected style automatically.

## Formatting & Imports
- Line length is 79 characters. When in doubt, wrap manually instead of relying on auto-formatters.
- Run `uv run ruff check --select I --fix` before committing. This applies the canonical import ordering (isort-equivalent).
- After the import pass, run `uv run ruff check .` to catch lint issues.

## Types & Protocols
- Prefer `dataclasses` or small typed structs instead of passing loose dictionaries through the orchestrator layers.
- When using `typing.Protocol`, provide an ellipsis (`...`) inside method bodies so Pyright doesn’t flag missing returns.
- Keep `typing` hints concrete. If a function can return `VerifyResult`, return a real `VerifyResult`, not a bare protocol implementation.

## Lint, Type Check, and Tests
- `make lint` runs the Ruff import pass plus `ruff check .`.
- `make typecheck` runs Pyright with the settings defined in `pyproject.toml`; it focuses on `langformer` and the `examples/simple_py2rb_transpiler` example to avoid vendored noise.
- `make pylint` executes `uv run pylint --errors-only langformer` for lightweight static checks.
- `make test` runs the full pytest suite (framework + examples).

## Recommended Local Workflow
```bash
uv pip install -e ".[dev]"
make lint
make typecheck
make pylint
make test
```

Following these commands keeps code style, imports, and types aligned with the rest of the project without memorizing any bespoke rules.

## Artifact manager
- The orchestrator provisions a per-run `ArtifactManager` with stage-specific directories. Configure it via `transpilation.artifacts` (`root`, `analyzer_dir`, `transpiler_dir`, `verifier_dir`).
- Every agent receives the manager through `llm_config.artifact_manager`; verification strategies also see it via `IntegrationContext.artifacts`.
- Use `stage_dir(stage, unit_id)` to obtain a directory, write your files there, then call `register(stage, unit_id, path, metadata=...)` so the orchestrator can surface them in `candidate.notes['artifacts']` and the run metadata.
- Analyzer metadata/access: use `unit.metadata[...]`. Transpiler metadata lives on `candidate.metadata` (alias `candidate.notes`), and verification feedback is exposed via `VerifyResult.feedback`.
- Generated tests should be attached to `VerifyResult.feedback.tests`; the orchestrator writes them under the verifier artifact directory and mirrors them in `candidate.tests`.
