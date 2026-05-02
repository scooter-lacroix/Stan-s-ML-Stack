# Rusty Stack Master Spec Bible

**Date:** 2026-04-24
**Status:** Drafted from approved design and tightened after internal review
**Scope:** Master architecture, policy, migration, and execution authority for the Rusty Stack rebrand and update platform work.

## Goal

Create the canonical design authority for:
- `rusty update` as the validated-by-default component and manifest update command
- `rusty upgrade` as the Rusty Stack application/runtime upgrade command
- migration of stack logic into Rust shared crates
- the first Windows application foundation
- opt-in anonymous telemetry and a 180-second minimum stability benchmark

## Approved Constraints

- Validated combinations are user-facing by default; experimental combinations exist only behind explicit opt-in.
- Version truth is hybrid: in-repo manifests are the safe baseline, optional signed remote manifests can override them.
- `rusty update` updates installed Rusty Stack components and manifests only.
- `rusty upgrade` updates the Rusty Stack application/runtime.
- Update UX is scan-first, user-selectable, and guardrailed: users may untick components before apply.
- Windows support is dual-path by design: native where possible, WSL2-backed where required, but users should not need terminal or Linux knowledge.
- Telemetry is opt-in, anonymous, hardware-focused, and sent directly with minimal user friction.
- Parallel implementation is allowed only when file-ownership and dependency cross-pollution risk are both below 10%.
- TDD is mandatory for implementation work.
- Tzar PASS is required before a milestone advances the handoff ledger.

## Non-Goals

- Claiming support for unvalidated hardware or version combinations.
- Replacing every legacy shell/Python execution path in a single rewrite wave.
- Shipping a Windows-native ROCm stack where upstream support does not exist.
- Collecting personal data, project data, prompts, or filesystem-identifying telemetry.

## Current System Map

| Area | Current Source | What Exists Today | Why It Matters |
| --- | --- | --- | --- |
| Primary Rust entrypoint | `rusty-stack/src/main.rs#L24-L127` | Rust TUI bootstraps the app, locates `scripts/`, and sets repo context. | This is the current frontend shell for future shared-core integration. |
| Update CLI wrapper | `rusty-stack/src/bin/update.rs#L1-L58` | Thin binary wrapper that only shells out to `scripts/update_stack.sh`. | Must become a real Rust command surface instead of a delegator. |
| Update UX and batching | `scripts/update_stack.sh#L34-L307` | Help, list, interactive menu, selection parsing, `--all`, and batch execution live in shell. | This is the behavioral source material for `rusty update` scan/plan/apply flow. |
| Installed-component detection | `scripts/lib/update_helper.sh#L306-L341` | Detects ROCm, Python packages, git-based apps, `rocm-smi`, and `.mlstack_env`. | This logic must migrate into the Rust scan engine and capability resolver. |
| Component update dispatch | `scripts/lib/update_helper.sh#L348-L401` | Maps component IDs to installer scripts, handles special skips, and forces installers. | This becomes the initial adapter layer before full Rust executors exist. |
| Install orchestration | `rusty-stack/src/installer.rs#L51-L279` | Rust orchestrates component execution, environment prep, verification, and progress events. | This is the strongest existing foundation for the shared Rust orchestrator. |
| Environment canonicalization | `rusty-stack/src/installer.rs#L325-L430` | Ensures `.mlstack_env` exists and stays normalized across ROCm, GPU, and Python settings. | This is a strong candidate for extraction into shared platform/runtime crates. |
| ROCm version fallback | `rusty-stack/src/installer.rs#L580-L585` | `detect_rocm_version()` falls back to `7.2.0` if `/opt/rocm/.info/version` is unavailable. | Version fallback policy must move into manifest-aware Rust logic. |
| Full-stack verification | `rusty-stack/src/installer.rs#L587-L760` | Runs verification commands, benchmark-target checks, and builds reports. | This becomes the basis for post-update verification and telemetry evidence. |
| Component verification | `rusty-stack/src/installer.rs#L762-L920` | Verifies component-specific install outcomes and assembles reports. | This is the natural basis for per-component update validation gates. |
| Product positioning | `README.md#L121-L220` | Documents current ROCm channels, Rust TUI, and current install story. | Public docs must be updated as the architecture becomes real. |
| Windows position today | `docs/guides/beginners_guide.md#L636-L637` | Explicitly says the stack is designed for Linux and Windows is not the recommended path. | The Windows foundation track starts from almost no dedicated product support. |

## Command Contracts

### `rusty update`

`rusty update` is the component and manifest update command.

It must:
- detect hardware, installed components, current versions, backend mode, and manifest freshness
- load the in-repo manifest baseline
- optionally fetch and verify a signed remote override manifest
- compute eligible updates and classify them as safe, guarded, blocked, candidate, or experimental-opt-in
- present a plan with preselected defaults and allow users to untick components
- require explicit confirmation before risky system-level work
- run baseline verification after apply and produce a user-readable summary
- never update the Rusty Stack application binary itself

### `rusty upgrade`

`rusty upgrade` is the Rusty Stack application/runtime upgrade command.

It must:
- update the Rusty Stack binary/runtime separately from component updates
- reuse manifest trust and validation policy where applicable
- stop `rusty update` from applying work when the current runtime is too old for the effective manifest
- remain operationally separate from `rusty update`

## Target Architecture

### Shared Rust Workspace

Recommended crate split:
- `rusty-stack-core`: canonical types for components, manifests, hardware profiles, validation tiers, update plans, verification evidence, telemetry payloads, and platform capability descriptors
- `rusty-stack-orchestrator`: scan, manifest resolution, compatibility filtering, planning, guarded execution, and verification orchestration
- `rusty-stack-platform`: Linux-native, Windows-native, and WSL-backed execution bridges behind a single contract
- `rusty-stack-telemetry`: stress-test runner, anonymization, consent, and submission client
- frontends: existing Rust TUI, `rusty update`, `rusty upgrade`, and the future Windows app

### Manifest Trust Contract

Manifest layers:
1. bundled in-repo validated baseline
2. cached last-valid remote overlay
3. freshly fetched remote overlay when trust checks pass

Remote overlays are not full replacements.
They are field-level overlays on top of the baseline manifest.

A remote overlay is accepted only when:
- signature is valid
- schema version is supported
- anti-rollback checks pass using sequence/timestamp rules
- the document is not expired

If the runtime is too old to understand the effective manifest, `rusty update` must stop and direct the user to `rusty upgrade`.

### Validation State Contract

Validation states:
- `validated`
- `candidate`
- `experimental`
- `blocked`

Selection policy:
- `validated`: may be shown and preselected
- `candidate`: may be shown, but never preselected and never included in `--all-safe`
- `experimental`: hidden unless the user explicitly opts in
- `blocked`: never actionable in the normal flow

Promotion governance:
- metadata may advance to `candidate` through controlled upstream discovery workflows
- user-facing validation-state changes only take effect through signed manifest publication
- ROCm and other system-level promotions require explicit manual PASS

### Windows Architecture

The first Windows release path is dual-path by design:
- native Windows control app for UX, settings, update control, logs, and service launch
- native Windows execution where upstream support exists
- WSL2-backed Linux execution where ROCm or stack components require Linux
- resource-conscious background management, path bridging, and service exposure handled by the app, not the user
- local-only service exposure by default in v1

### Telemetry Architecture

Telemetry is opt-in and direct-submit.

The first telemetry system must:
- run a 180-second minimum stability benchmark only when the user explicitly opts into telemetry/stability mode
- gather anonymous hardware and performance metrics, especially GPU behavior
- submit only non-personal, structured data over HTTPS
- return a clear user thank-you on successful submission
- support low-cost secure intake that can grow to hundreds of submissions per week

Baseline verification remains part of normal update/install behavior and is distinct from the opt-in stress benchmark.

## What / How / Why Matrix

| Goal | Current File(s) | What Changes | How It Changes | Why It Changes |
| --- | --- | --- | --- | --- |
| Replace shell-only updater | `rusty-stack/src/bin/update.rs#L1-L58`, `scripts/update_stack.sh#L34-L307` | Move updater planning and execution into Rust. | Build a real Rust planner and keep script adapters temporarily. | Users need a trustworthy `rusty update`, not a wrapper around shell state. |
| Separate app/runtime upgrade from component update | `rusty-stack/src/bin/update.rs#L1-L58`, `README.md#L121-L220` | Formalize `rusty upgrade` as a distinct command and compatibility gate. | Introduce runtime-version compatibility checks in manifest resolution. | Prevents update/app-version drift and keeps trust boundaries explicit. |
| Centralize component detection | `scripts/lib/update_helper.sh#L306-L341` | Migrate scan logic into shared Rust types and platform probes. | Replace shell detection with Rust capability resolvers. | Windows, Linux, and future tooling need one source of truth. |
| Preserve validated install behavior during migration | `rusty-stack/src/installer.rs#L51-L279`, `scripts/lib/update_helper.sh#L348-L401` | Keep existing scripts as adapters while Rust replaces them incrementally. | Use execution contracts that can target Rust executors or legacy scripts. | Minimizes regression risk while migration proceeds. |
| Normalize environment and runtime metadata | `rusty-stack/src/installer.rs#L325-L430` | Extract env/runtime state management into shared platform logic. | Convert `.mlstack_env` normalization into reusable Rust services. | Update, install, verify, and Windows orchestration all need consistent runtime state. |
| Make validation policy manifest-aware | `rusty-stack/src/installer.rs#L580-L760`, `README.md#L123-L129` | Tie verification and channel/version truth to manifests. | Replace hard-coded fallbacks and scattered policy with manifest-driven rules. | Validation discipline is the product’s trust boundary. |
| Enable selection-based guarded updates | `scripts/update_stack.sh#L104-L249` | Recreate menu/list/selection logic in Rust. | Build scan-plan-apply UX with untick support, dependency rules, and risk classification. | Users want automation without losing control. |
| Start real Windows support | `docs/guides/beginners_guide.md#L636-L637` | Replace “Linux only in practice” with a true Windows control surface. | Build a native app over shared Rust contracts with WSL2 abstraction. | Users should not need terminals or Linux knowledge. |
| Expand validation through anonymous field evidence | `rusty-stack/src/installer.rs#L587-L920` | Add a stability benchmark and telemetry submission flow. | Reuse verification structures and add anonymized payload submission. | Broader user testing is required to grow validated coverage responsibly. |

## Execution Governance

### Blocking Track Order

1. Master specs and governance docs
2. Shared core contracts
3. `rusty update` scan/plan/apply flow
4. Rust migration waves
5. Telemetry and stability benchmark
6. Windows foundation backed by the shared core

### Parallelism Rule

Parallel execution is allowed only when:
- file overlap risk is below 10%
- interface churn risk is below 10%

Serial by default for:
- manifest schema and trust rules
- validation policy
- shared core types
- update planning and execution contracts
- Windows platform contract definition

### TDD Rule

Implementation plans must enforce strict red/green/refactor.
No production code may land before a failing test proves the next slice of required behavior.

### Tzar Rule

- Every milestone ends with a Tzar review.
- PASS advances the handoff ledger.
- FAIL blocks downstream work and creates mandatory remediation.
- The ledger must not advance on partial acceptance.

## Artifact Set Governed By This Bible

Approved companion specs:
- `docs/superpowers/specs/2026-04-24-rusty-stack-update-intelligence-spec.md`
- `docs/superpowers/specs/2026-04-24-rusty-stack-upgrade-spec.md`
- `docs/superpowers/specs/2026-04-24-rusty-stack-rust-migration-spec.md`
- `docs/superpowers/specs/2026-04-24-rusty-stack-windows-foundation-spec.md`
- `docs/superpowers/specs/2026-04-24-rusty-stack-telemetry-stability-spec.md`

Approved handoff docs:
- `docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md`
- `docs/superpowers/handoffs/2026-04-24-rusty-stack-review-index.md`

Implementation plans are intentionally deferred until these specs are reviewed and approved.

## LeIndex Operating Appendix

This project requires LeIndex-first codebase operations whenever LeIndex provides a suitable tool.

### Required Session Order

1. `leindex_index(project_path, force_reindex=true)`
2. `leindex_diagnostics(project_path)`
3. `leindex_project_map(project_path, ...)` as the first structural inspection step
4. targeted use of read/search/summary/context tools
5. edit preview or scoped impact checks where changes are high-risk
6. `leindex_git_status(project_path)` before closeout

### Practical Tool Inventory And Schema Notes

| Tool | Primary Use | Core Inputs |
| --- | --- | --- |
| `leindex_index` | Fresh project indexing | `project_path`, `force_reindex?` |
| `leindex_diagnostics` | Freshness, health, cache stats | `project_path?` |
| `leindex_project_map` | Directory/file map with complexity and optional symbols | `project_path?`, `path?`, `depth?`, `focus?`, `include_symbols?`, `limit?`, `sort_by?`, `token_budget?` |
| `leindex_read_file` | Exact file reads with line numbers and optional symbol map | `file_path`, `start_line?`, `end_line?`, `max_lines?`, `include_symbol_map?`, `project_path?` |
| `leindex_read_symbol` | Exact symbol source with callers/callees | `symbol`, `file_path?`, `include_dependencies?`, `project_path?`, `token_budget?` |
| `leindex_text_search` | Exact text search with file:line context | `query`, `scope?`, `include_globs?`, `exclude_globs?`, `context_lines?`, `is_regex?`, `max_results?`, `project_path?` |
| `leindex_search` | Semantic code search | `query`, `scope?`, `top_k?`, `search_mode?`, `project_path?` |
| `leindex_grep_symbols` | Symbol-name search | `pattern`, `scope?`, `type_filter?`, `mode?`, `include_context_lines?`, `include_source?`, `project_path?` |
| `leindex_symbol_lookup` | Callers/callees/data dependencies | `symbol` or `symbols`, `depth?`, `include_callers?`, `include_callees?`, `include_source?`, `scope?`, `project_path?` |
| `leindex_context` | PDG-local context expansion | `node_id`, `project_path?`, `token_budget?` |
| `leindex_deep_analyze` | Broad semantic + PDG analysis | `query`, `project_path?`, `token_budget?` |
| `leindex_file_summary` | File role, symbol inventory, complexity | `file_path`, `focus_symbol?`, `include_source?`, `project_path?`, `token_budget?` |
| `leindex_phase_analysis` | Five-phase additive analysis | `project_path?`, `path?`, `phase?`, `mode?`, `include_docs?`, `docs_mode?`, `top_n?`, `max_focus_files?` |
| `phase_analysis` | Alias for `leindex_phase_analysis` | same inputs as above |
| `leindex_edit_preview` | Pre-edit diff/risk preview | `file_path`, `old_text?`, `new_text?`, `project_path?` |
| `leindex_edit_apply` | File edits | `file_path`, `old_text?`, `new_text?`, `dry_run?`, `project_path?` |
| `leindex_rename_symbol` | Graph-aware symbol rename | `old_name`, `new_name`, `preview_only?`, `scope?`, `project_path?` |
| `leindex_impact_analysis` | Transitive change impact | `symbol`, `change_type?`, `depth?`, `project_path?` |
| `leindex_git_status` | Working tree plus structural impact | `project_path?`, `include_diff?`, `diff_context_lines?` |

### Repo-Specific LeIndex Guidance

- Scope searches to first-party paths like `rusty-stack`, `scripts`, `tests`, and `docs` before widening.
- Avoid allowing vendored `venv` and `site-packages` content to dominate semantic search results.
- Prefer line-precise mapping of current behavior before drafting migration tasks.
- Use shell fallback only when a LeIndex operation times out or does not support the required operation cleanly.

## Acceptance Condition For This Document

This master bible is ready to govern planning once:
- the user reviews and approves the written spec set
- the companion child specs are reviewed for consistency with this document
- the implementation plans are generated from the approved specs
