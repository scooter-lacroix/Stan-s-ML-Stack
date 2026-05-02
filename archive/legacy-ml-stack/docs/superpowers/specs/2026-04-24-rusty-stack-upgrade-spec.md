# Rusty Stack Upgrade Spec

**Date:** 2026-04-24
**Parent:** `docs/superpowers/specs/2026-04-24-rusty-stack-master-spec-bible.md`

## Goal

Define `rusty upgrade` as the command that updates the Rusty Stack application/runtime itself, separately from component and manifest updates handled by `rusty update`.

## Scope

Included:
- Rusty Stack application/runtime upgrade policy
- compatibility boundary between current app version and effective manifests
- upgrade refusal rules when app/runtime is too old for current manifest contracts
- packaging/channel expectations for the Rusty Stack app itself

Excluded:
- component/package updates
- telemetry submission behavior except where runtime version gates submission
- Windows service orchestration details beyond app/runtime delivery

## Command Boundary

- `rusty update`: installed components + manifests
- `rusty upgrade`: Rusty Stack binary/runtime only

The two commands must stay operationally separate.
`rusty update` may instruct the user to run `rusty upgrade`, but it must not silently self-upgrade the application.

## Effective Manifest Compatibility Rule

Every effective manifest may declare:
- minimum supported Rusty Stack application/runtime version
- optional maximum known-compatible runtime version
- schema version required for planner/apply behavior

If the current app/runtime does not satisfy the effective manifest requirements, `rusty update` must stop before apply and instruct the user to run `rusty upgrade`.

## Upgrade Channels

The Rusty Stack app/runtime may have its own release channels, but the first implementation should default to a validated stable channel.
Candidate or experimental runtime releases must not become default without explicit policy and review.

## Baseline Manifest Interaction

A `rusty upgrade` may ship a newer bundled baseline manifest.
However:
- bundled baseline must not override a newer valid cached remote manifest unless trust/anti-rollback rules require it
- upgrading the app/runtime must preserve the last valid accepted remote manifest cache when compatible

## Failure Handling

- failed upgrade must not corrupt the current runnable app/runtime
- partial runtime replacement is unacceptable
- if the new app/runtime cannot start, the upgrade flow must preserve or restore the prior working version

## Acceptance Criteria

This spec is satisfied when implementation plans can enforce a clean operational separation between component updates and app/runtime upgrades, including manifest-version compatibility gating.
