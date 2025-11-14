# Langformer Technical Specification

## 1. Scope & Positioning

Langformer is a **language-agnostic transpilation framework** that converts
source programs into equivalent implementations in a different language or
runtime. It combines LLM-driven agents, deterministic orchestration, execution
grounded verification, and a plugin-first architecture so downstream projects
can register their own language plugins, prompts, planners, verification
oracles, and delegates without forking the core.

Ideas from [KernelAgent](https://github.com/meta-pytorch/KernelAgent) inspired
the original flow (LLM → verify → refine), but langformer generalizes its surface areas:

| Topic | KernelAgent (legacy) | Langformer (current) |
| --- | --- | --- |
| Focus | CUDA/Triton kernel generation | Arbitrary language/runtime transpilation |
| Structure | Monolith, hard-coded planners | Modular `langformer/` core w/ plugin registry |
| Config | YAML + scattered defaults | Typed dataclasses + CLI overrides |
| Verification | CUDA runtime specific | Strategy registry + sandbox runners + custom oracles |
| Prompts | Inline strings per target | Jinja2 template library + prompt-fill registry |
| Artifacts | Ad-hoc temp dirs | Stage-aware `ArtifactManager`, RunSession manifests |

---

## 2. Repository Layout

```
langformer_spec.md        ← this document
langformer/               ← framework code
  agents/                 ← analyzer / transpiler / verifier bases
  artifacts.py            ← ArtifactManager
  configuration.py        ← typed config dataclasses
  languages/              ← default language plugins
  llm/                    ← provider adapters & utils
  logging/                ← event adapters, stream dispatcher
  orchestration/          ← context builder, target integrator
  orchestrator.py         ← TranspilationOrchestrator entry point
  prompts/                ← templates + fill registry
  runtime/                ← RunSession, worker/dedup helpers
  verification/           ← strategies, oracles, sandbox runners
  worker/                 ← worker payloads and sub-process helper
examples/                 ← runnable samples (simple_py2rb, kernel delegate, WIP)
tests/                    ← unit/integration tests + reusable stubs
configs/                  ← example YAML configs
README_OLD.md             ← historical doc (pre-rebrand)
```

---

## 3. Architecture Overview

### 3.1 Control Plane

```
┌────────────────────────────────────────────────────┐
│            TranspilationOrchestrator               │
│  • Load config & CLI overrides                     │
│  • Import plugin modules                           │
│  • Instantiate language plugins + components       │
│  • Build IntegrationContext (layout + feature_spec)│
│  • Kick off agents, manage RunSession + artifacts  │
└───────────────┬────────────────────────────────────┘
                │ units
        ┌───────▼────────┐
        │ AnalyzerAgent   │ → `TranspileUnit` (+ metadata)
        └───────┬────────┘
                │ units
        ┌───────▼────────┐
        │ LLMTranspiler  │ → `CandidatePatchSet`
        │ Agent          │   (prompts, retries, stream events)
        └───────┬────────┘
                │ candidates
        ┌───────▼────────┐
        │ Verification   │ → `VerifyResult`
        │ Agent          │   (strategies/oracles)
        └───────┬────────┘
                │ verified candidate
        ┌───────▼────────┐
        │ TargetIntegrator│ → final files per layout
        └────────────────┘
```

### 3.2 Runtime / Artifact Flow

```
┌──────────────┐   analyzer artifacts   ┌────────────────────┐
│ ArtifactMgr  │<-----------------------│ Analyzer / plugins  │
│ (per stage)  │----------------------->│ Transpiler / oracle │
└──────┬───────┘                        └──────┬──────────────┘
       │ manifest                               │
┌──────▼────────────────────┐         ┌─────────▼────────────┐
│ RunSession (.transpile/…) │↔events↔│ CLI / clients / CI    │
│  • metadata.json          │         └───────────────────────┘
│  • per-unit status        │
│  • persisted candidates   │
└───────────────────────────┘
```

RunSession enables resumable jobs and audit trails; ArtifactManager ensures each
stage writes to a predictable directory (`analyzer/<unit>/…`,
`transpiler/<unit>/…`, `verifier/<unit>/…`) and records manifests that later
populate `candidate.notes["artifacts"]` and `RunSession` metadata.

---

## 4. Component Responsibilities

| Component | Key Classes | Responsibilities |
| --- | --- | --- |
| Language Plugins | `langformer.languages.*` | Parse/execute/compile source and target languages, optionally partition units. |
| Analyzer Agent | `DefaultAnalyzerAgent` | Partition source into `TranspileUnit`s, attach metadata/AST, write analyzer artifacts. |
| Transpiler Agent | `LLMTranspilerAgent` | Render prompts (`PromptManager` + `prompt_fills`), call LLM provider, manage retries/variants, optional streaming logs, inline or worker-driven verification hooks. |
| Verifier Agent | `DefaultVerificationAgent` + strategies | Execute configured strategy (exact/structural/execution/custom oracle); return `VerifyResult` with structured feedback + optional generated tests. |
| Target Integrator | `TargetIntegrator` | Materialize candidate files to layout outputs, optionally merge multi-unit snippets. |
| Context Builder | `ContextBuilder` | Turn typed `IntegrationSettings` + `LayoutPlan` into `IntegrationContext` (feature_spec, build info, oracle). |
| Artifact Manager | `ArtifactManager` | Stage-aware directories and manifests with reset/preserve semantics per unit. |
| RunSession | `langformer.runtime.RunSession` | Orchestrator audit log: metadata, events, candidate files, resume support, stream roots. |
| Worker Pipeline | `WorkerManager`, `worker/transpile_worker.py` | Spawn isolated processes to run the same analyzer/transpiler/verifier logic when `parallel_workers > 1` with verification disabled (multivariate generation). |
| CLI | `langformer/cli.py` (`langform`) | Thin wrapper over `TranspilationOrchestrator.transpile_file`, plus planner/delegate invocation for KernelAgent-style workflows. |

---

## 5. Configuration Model

`langformer.configuration.build_transpilation_settings` normalizes YAML into
dataclasses. Key sections:

- **Languages**: `source_language`, `target_language` (string keys into
  `LANGUAGE_PLUGINS`; applications can register new keys via
  `register_language_plugin`).
- **Layout**: declarative input/output (kind, path, filename, extension) and
  `relative_to` (`cwd`, `config`, `run_root`). `LayoutPlan.target_path(unit_id)`
  resolves per-unit outputs and respects `feature_spec` (e.g., module_path).
- **Integration**: `feature_spec`, build info, API mappings, optional runtime
  adapter string, custom contract data. Passed into
  `ContextBuilder.build -> IntegrationContext`.
- **LLM**: provider, model, base URL, API key env, streaming config (log root,
  mode, timeout, worker mode, `store_responses`), temperature bounds.
- **Agents**: `max_retries`, `parallel_workers`, `prompt_dir` overrides,
  worker manager toggles.
- **Runtime**: `.transpile_runs` root, enable/disable session logging, resume.
- **Artifacts**: root directory + per-stage subdir names.
- **Verification**: strategy name, optional exact/structural options, execution
  runner configuration (sandbox mode, timeouts, command template), oracle
  definition.
- **Components**: dotted paths overriding analyzer, context builder, transpiler,
  verification agent, or target integrator.
- **Plugins**: `paths` (directory globs) and `modules` (import strings) to load
  planners, delegates, prompt fills, oracles, etc.

CLI flags (`langform …`) override YAML keys (e.g., `--llm-provider`, `--model`,
`--config`, `--planner`).

---

## 6. Prompt System

1. **Templates** live in `langformer/prompts/templates/*.j2` (`transpile`,
   `refine`, `guidelines`, `tests`). Users can add override directories via
   `transpilation.agents.prompt_dir`; `PromptManager` enforces precedence.
2. **Prompt fills** come from `langformer.prompts.fills.prompt_fills`, a
   registry of callables receiving `PromptFillContext` (unit, integration
   context, attempt, feedback, previous code, source/target plugin handles).
3. **Extensibility**: applications register extra payload providers (e.g.,
   translation hints for Ruby) through `prompt_fills.register(...)` or helper
   APIs (`register_target_language_hints`, `register_translation_hints`).
4. **Discoverability**: every injected instruction flows through the registry
   so prompts remain auditable; templates never embed hard-coded language
   strings besides general scaffolding.

---

## 7. LLM Providers & Runtime

- `langformer.llm.providers.load_provider` inspects `llm.provider` / `model`,
  base URL, API key env var, reasoning settings, etc., and returns either a
  first-party adapter (OpenAI, Anthropic, Relay) or the deterministic
  `EchoProvider` (mirrors prompts; used for dry runs and tests).
- `LLMConfig` objects package the provider, `PromptManager`, search paths, and
  `ArtifactManager` references for agents.
- Streaming: `StreamDispatcher` writes jsonl + console logs under either the
  orchestrator or worker stream root, tied to RunSession directories when
  runtime logging is enabled.
- Worker fan-out: `WorkerManager` serializes `WorkerPayload`s (unit, context,
  LLM config, verification settings) and spawns workers via
  `worker/transpile_worker.py`, useful for multi-variant exploration with
  central dedup/aggregation.

---

## 8. Verification & Oracles

- Strategies (`langformer.verification.strategies`):
  - `ExactMatchStrategy`: literal diff of source vs. candidate outputs.
  - `StructuralMatchStrategy`: AST / IR comparison.
  - `ExecutionMatchStrategy`: run source + target with test inputs using
    sandbox runners (`langformer.runtime.runner.*`).
- Custom oracles register via `OracleRegistry.register(name, factory)` and can
  inspect source and target code. Example:
  `examples/simple_py2rb_transpiler/oracle.py` loads deterministic cases to
  compare Python vs. Ruby outputs.
- `VerifyResult.feedback` is a dict-like object; verifiers add metadata and
  generated tests via `feedback.add_test(path, contents)`.
- Orchestrator writes verifier tests into the artifact tree and mirrors them in
  `candidate.tests` for later consumption.

---

## 9. Artifact & RunSession Semantics

- **ArtifactManager**
  - `stage_dir(stage, unit_id)` gives deterministic paths.
  - `register(stage, unit_id, path, metadata)` records manifest entries.
  - `reset(unit_id, preserve=...)` clears prior entries while optionally keeping
    analyzer outputs (useful when retries occur).
- **RunSession**
  - Directories: `orchestrator/`, `workers/`, `digests/`, `streams/`.
  - Methods: `mark_unit_started/completed`, `log_event`, `persist_files`,
    `write_metadata`, `write_summary`, `list_units`, `load_files`.
  - Resume: if `runtime.resume` or `run_id` is supplied, orchestrator repopulates
    `resume_candidates` by reading successful units from RunSession.

---

## 10. Extensibility Map

| Extension Point | How to Register | Example |
| --- | --- | --- |
| Language Plugin | `register_language_plugin("swift", SwiftPlugin)` | Add Swift analyzer/executor. |
| Analyzer/Transpiler/Verifier | Set `transpilation.components.*` to dotted path | `mypkg.agents.CustomTranspiler`. |
| Prompt Templates | Add directory to `transpilation.agents.prompt_dir` | CI injecting company-wide prompts. |
| Prompt Payloads | `prompt_fills.register(func)` or `register_translation_hints` | Ruby-specific hints (advanced example). |
| Oracles / Strategies | `OracleRegistry.register("foo", factory)` or custom `VerificationStrategy` | Domain-specific behavior tests. |
| Planners / Delegates | Add modules under `transpilation.plugins.modules` or directories in `paths` | `examples.kernel_agent_delegate`. |
| Runner integrations | Provide custom `integration.runtime_adapter` + runner settings | Domain runtime executors. |
| CLI commands | `langformer/cli.py` already exposes `langform`; new scripts can import `TranspilationOrchestrator`. |

---

## 11. Example Scenarios

1. **Simple Py→Rb (`examples/simple_py2rb_transpiler`)**
   - Demonstrates layout-driven IO, OpenAI provider defaults, and a custom
     verification oracle. Deterministic tests use `tests/simple_py2rb_stub`.
2. **KernelAgent Delegate (`examples/kernel_agent_delegate`)**
   - Planner writes router strategy into plan context (`context["solver"]`);
     delegate dispatches to either the Langformer pipeline or the vendored
     KernelAgent solver. Shows how Langformer can orchestrate external systems.
3. **Advanced WIP Py→Rb (`examples/advanced_wip_py2rb_transpiler`)**
   - Sandbox for richer prompt customizations and translator hints without
     destabilizing the simple example.

---

Langformer is now the primary surface for new integrations. KernelAgent remains
available as a demonstrative dependency but is no longer a first-class path.
