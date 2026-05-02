# Rusty Stack Update Intelligence Spec

**Date:** 2026-04-24
**Parent:** `docs/superpowers/specs/2026-04-24-rusty-stack-master-spec-bible.md`

## Goal

Define the behavior, trust model, and execution flow for `rusty update` so users can update installed Rusty Stack components and manifests with minimal effort while retaining explicit control over risky changes.

## Scope

Included:
- scan, plan, select, apply, verify lifecycle
- in-repo baseline manifest and signed remote override manifest
- compatibility and validation filtering
- untick/holdback behavior
- safe vs guarded vs blocked update classification
- Linux-first behavior with future Windows frontend reuse

Excluded:
- Rusty Stack binary/runtime self-upgrade (`rusty upgrade`)
- full retirement of all legacy script executors
- server-side telemetry ingest implementation details except where update flow depends on verification evidence

## Current State

- `rusty-stack/src/bin/update.rs#L1-L58` is only a shell wrapper.
- `scripts/update_stack.sh#L34-L307` contains the real user-facing update behavior.
- `scripts/lib/update_helper.sh#L306-L401` contains installed-component detection and script dispatch.

## Required User Experience

1. User runs `rusty update`.
2. Command scans the machine, installed components, platform mode, and manifest freshness.
3. Command loads the in-repo manifest and attempts to fetch a signed remote override.
4. Command computes eligible updates and groups them into:
   - safe user-space updates
   - guarded system-level updates
   - blocked or incompatible updates
   - experimental updates, shown only when the user explicitly opts in
5. User sees a plan with recommended defaults already selected.
6. User may untick any offered component.
7. Guarded updates require explicit confirmation before apply.
8. Verification runs after apply.
9. User receives a clear summary of success, failure, holdbacks, and reboot/service guidance.

## Effective Manifest Contract

### Baseline Manifest

The in-repo manifest is the safety net.
It must always exist and represent the last known-good validated matrix that can be trusted offline.

### Remote Override Manifest

The remote manifest is an overlay, not a full replacement.
It may:
- add new candidates
- promote candidates to validated
- block bad versions
- revise compatibility metadata
- add advisory notes and migration guidance

### Trust And Freshness Rules

The client accepts a remote override only if all are true:
- signature is valid
- schema version is supported by the current runtime
- the manifest is newer than the last accepted remote manifest or bundled baseline according to sequence/timestamp rules
- the manifest is not expired

If the current runtime is too old for the effective manifest, `rusty update` must stop and instruct the user to run `rusty upgrade`.

Client fallback order:
1. newly fetched valid remote overlay
2. cached last-valid remote overlay
3. bundled in-repo baseline only

If any remote validation check fails, the client must reject that remote payload and continue with the newest valid fallback.

## Validation And Compatibility Policy

Every offered update must be filtered through:
- hardware compatibility
- platform compatibility
- validation tier
- executor availability
- local install state

Validation tiers:
- `validated`: may be shown and preselected
- `candidate`: may be shown, but never preselected and never included in `--all-safe`
- `experimental`: hidden unless the user explicitly opts in
- `blocked`: never offered for apply

## Planning Model

The planner output must contain:
- component ID
- current version
- proposed version
- validation tier
- backend mode
- executor kind
- risk tier
- selection default
- explicit rationale string
- verification plan reference
- dependency relationships
- whether the item is isolation-safe for partial continuation

Recommended shared vocabulary:
- `backend_mode`: `linux-native`, `windows-native`, `wsl-backed-linux`
- `executor_kind`: `rust`, `legacy-script`, `external-package-manager`, `unsupported`

## Risk Classification

### Safe

Examples:
- user-space Python package updates already validated for the current hardware/backend
- manifest-only refreshes
- non-privileged git-based tool updates where the executor is validated

### Guarded

Examples:
- ROCm updates
- system package changes
- updates likely to require reboot, service restart, or driver/toolchain re-linking

### Blocked

Examples:
- unsupported hardware/backend combination
- missing executor path
- manifest-declared bad version
- unvalidated combinations outside experimental opt-in

## Selection And Dependency Rules

- Safe validated updates may be preselected.
- Guarded updates may be suggested but must not auto-apply without confirmation.
- Candidate updates may be visible but are never preselected.
- Experimental updates are excluded unless the user explicitly asks to see or include them.
- The planner must let the user untick any offered component before apply.
- The user cannot force-apply blocked updates through the normal flow.
- Required dependencies may not be deselected while a selected dependent remains selected.
- `rusty update <component...>` still performs a full scan and compatibility pass; direct targeting does not bypass policy.

## Apply Model

The apply engine must:
- execute selected items in dependency-safe order
- batch only when risk tier and executor policy allow it
- stop or isolate failures so dangerous later work does not continue blindly
- run baseline verification after component or batch completion
- record evidence for the summary and Tzar review trail

During migration, the apply engine may invoke legacy scripts through adapters as long as the planner and policy layers remain Rust-owned.

## Verification Model

The update system should reuse and extend the existing verification seams in `rusty-stack/src/installer.rs#L587-L920`.
Verification outputs must include:
- checks executed
- checks skipped and why
- final pass/fail status per updated component
- benchmark evidence only when a benchmark mode was explicitly invoked

The 180-second stability benchmark is outside the default update path.

## Non-Interactive And Machine-Readable Behavior

The command surface must support:
- `rusty update`
- `rusty update --scan-only`
- `rusty update --all-safe`
- `rusty update --include-experimental`
- `rusty update <component...>`
- non-interactive output suitable for automation
- stable machine-readable plan/status output for future TUI and Windows frontend reuse

Guarded non-interactive apply requires an explicit approval flag.
Without it, guarded items remain planned but unapplied.

## Failure Handling And Exit Semantics

- Manifest fetch failure: fall back to newest valid cached or bundled manifest.
- Signature failure: reject override and log the reason.
- Missing executor: mark component blocked, do not attempt apply.
- Failed guarded update: stop dependent work and present remediation guidance.
- Verification failure: component status remains failed even if the installer exited zero.
- Dependent work must stop when its prerequisite fails.
- Unrelated safe work may continue only when the planner marks it isolation-safe.

## Acceptance Criteria

This spec is satisfied when `rusty update` can:
- produce a manifest-aware plan in Rust
- resolve baseline, cached, and fetched manifest layers safely
- let users untick selected components without violating dependency rules
- distinguish safe, guarded, blocked, candidate, and experimental updates correctly
- support stable non-interactive and machine-readable planning behavior
- execute through validated adapters or Rust executors
- verify outcomes and report them coherently
- leave `rusty upgrade` outside its scope except for explicit runtime-version gating
