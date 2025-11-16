# Repository Guidelines

## Project Structure & Module Organization
Core runtime code (agents, configuration objects, LLM adapters, artifact helpers) lives in `langformer/`. Tests are grouped under `tests/` with pytest modules named `test_*.py`. Example pipelines and custom plugins live in `examples/`, while runtime YAMLs stay under `configs/` and diagrams/assets under `assets/`. Keep new modules beside their subsystem and mirror the same layout in `tests/`.

## Build, Test, and Development Commands
- `uv pip install -e ".[dev]"`: provision the tooling stack used in CI.
- `make lint`: Ruff import pass plus repo-wide linting.
- `make typecheck` / `make pylint`: Pyright and `pylint --errors-only langformer`.
- `make test` (or `uv run pytest -q tests/`): execute the regression suite.
- `uv run python examples/simple_py2rb_transpiler/run.py --config configs/simple_py2rb.yaml --verify`: exercise the reference pipeline; switch to `uv run langform … --verify` for CLI-driven runs.

## Coding Style & Naming Conventions
Follow PEP 8, 4-space indentation, and docstrings for public APIs. `STYLE.md` caps lines at 79 characters, so wrap long prompts manually. Modules/files stay snake_case, classes PascalCase, cli flags kebab-case. Run `uv run ruff check --select I --fix` before `uv run ruff check .` to keep imports and linting consistent. Prefer dataclasses or typed structs for data shared between agents and avoid passing loose dicts across orchestrator layers.

## Testing Guidelines
Pytest drives the suite; add `tests/test_<module>.py` whenever you introduce a new agent, runner, or CLI surface. Use fixtures to seed deterministic artifacts and reuse helpers in `tests/simple_py2rb_stub.py` for example-specific scaffolding. Verification work should assert both `VerifyResult.status` and emitted feedback/tests. Keep coverage even with new behavior—hook additions need at least one success and one failure path test. Run `make test` and `uv run pyright` before pushing.

## Commit Guidelines
History favors short, imperative commits (`Add MVP prototype of langformer framework`, `Bump required python to >3.10`). Keep each commit focused and lint-tested. Document config or API tweaks in `langformer_spec.md` or the affected example README, and avoid bundling unrelated refactors.

## Configuration & Security Tips
Place runtime knobs in YAML under `configs/` and override prompt templates via `prompt_dir` so diffs stay reviewable. Long-lived artifacts land in `.transpile_runs/` (gitignored); scrub logs before sharing. Never commit secrets in git.
