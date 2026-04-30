# Rusty Stack Rust Migration Spec

**Date:** 2026-04-24
**Parent:** `docs/superpowers/specs/2026-04-24-rusty-stack-master-spec-bible.md`

## Goal

Migrate stack orchestration, update intelligence, and validation-critical logic into Rust shared crates without destabilizing already-validated installation and verification flows.

## Migration Strategy

Adopt a shared-core, adapter-first migration.

That means:
- new behavior lands in Rust first
- legacy shell/Python execution remains available behind adapters where needed
- script retirement happens in narrow, verified waves
- no large rewrite wave is allowed to bypass validated behavior

## Current Migration Seams

| Current File | Current Responsibility | Migration Target |
| --- | --- | --- |
| `rusty-stack/src/bin/update.rs#L1-L58` | Thin CLI wrapper | Real Rust `rusty update` frontend |
| `scripts/update_stack.sh#L34-L307` | Update UX, selection, apply batching | `rusty-stack-orchestrator` planner/apply layer |
| `scripts/lib/update_helper.sh#L14-L341` | user-home resolution, detection, version lookups | `rusty-stack-platform` scanners + `rusty-stack-core` types |
| `scripts/lib/update_helper.sh#L348-L401` | component -> installer dispatch | adapter execution registry |
| `rusty-stack/src/installer.rs#L51-L279` | installation orchestration, progress, verification calls | split across orchestrator, platform, and verification crates |
| `rusty-stack/src/installer.rs#L325-L430` | `.mlstack_env` normalization | shared runtime/environment module |
| `rusty-stack/src/installer.rs#L587-L920` | verification and reporting | shared verification module |

## Target Workspace Shape

### `rusty-stack-core`

Owns:
- component identity and metadata
- manifest schema
- version and validation types
- update plan schema
- verification result schema
- telemetry payload schema
- platform/backend capability descriptors

### `rusty-stack-orchestrator`

Owns:
- scan-plan-apply flow
- risk classification
- dependency ordering
- selection state
- guarded execution rules
- verification orchestration

### `rusty-stack-platform`

Owns:
- Linux-native process execution
- Windows-native process execution
- WSL-backed execution and path/service bridging
- hardware detection
- environment and runtime state management

### `rusty-stack-telemetry`

Owns:
- stability benchmark control
- anonymous payload construction
- submission flow and local persistence if needed

## Retirement Waves

### Wave 1

Retire or replace low-risk, read-heavy shell logic first:
- installed-component detection
- version lookup
- selection-plan construction
- manifest freshness checks

### Wave 2

Migrate adapter registry and executor policy:
- component dispatch table
- installer capability metadata
- safe vs guarded execution classification

### Wave 3

Migrate component executors one domain at a time:
- user-space Python package updates
- git-based UI/service tools
- environment/runtime setup tasks
- system-level ROCm work last

## Retirement Rule

A legacy script path may only be retired when all are true:
- Rust replacement exists
- Rust replacement has failing tests written first and passing after implementation
- verification coverage is equivalent or stronger
- Tzar review for the milestone passes

## Boundary Rules

- Shared core types are serial-only work.
- Manifest schema changes are serial-only work.
- Platform contract changes are serial-only work.
- Adapter implementations may run in limited parallel only when they do not change shared type or planner contracts.

## Testing Expectations

Each migration slice must prove:
- the Rust path produces the same or stronger user-visible behavior
- version/validation decisions are preserved or improved
- failure handling remains explicit
- verification evidence is not weakened

## Acceptance Criteria

This spec is satisfied when the implementation plans can migrate the system in waves while preserving a usable, validated `rusty update` surface throughout the transition.
