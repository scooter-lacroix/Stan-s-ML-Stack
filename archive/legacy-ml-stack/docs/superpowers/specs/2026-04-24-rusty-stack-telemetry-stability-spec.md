# Rusty Stack Telemetry And Stability Spec

**Date:** 2026-04-24
**Parent:** `docs/superpowers/specs/2026-04-24-rusty-stack-master-spec-bible.md`

## Goal

Define an opt-in telemetry system and built-in 180-second minimum stability benchmark that expands validation coverage using anonymous hardware and performance evidence without collecting personal user data.

## Product Principles

- opt-in only
- minimal user friction
- anonymous by default
- structured data, not inbox noise
- security and low operational cost over convenience hacks like email
- designed for hundreds of submissions per week with room to grow

## Verification Boundary

Baseline post-update verification is always-on for `rusty update`.
The 180-second stability benchmark is not part of the default update path.

It runs only when:
- the user explicitly opts into telemetry/stability testing
- the user explicitly invokes a stability or benchmark mode
- a future approved workflow specifically requests it

This boundary prevents the update UX from turning every component update into a forced long-running benchmark.

## Stability Benchmark

The benchmark is not a vanity performance race.
It is a stability-focused validation tool.

### Minimum Requirements

- runtime of at least 180 seconds
- continuous or interval sampling of GPU-relevant metrics
- final result payload tied to manifest, version, platform mode, and verification context

### Metrics To Collect

At minimum:
- GPU model and architecture class
- ROCm version or backend/runtime equivalent
- CPU model family
- RAM tier
- OS/platform mode
- component versions under test
- GPU utilization trend
- VRAM usage trend
- thermal behavior
- clock behavior if available
- throttling or fault indicators if available
- benchmark throughput/latency summary
- verification outcomes associated with the tested components

## Privacy Boundaries

Must not collect:
- username
- email address
- home directory
- project paths
- prompts or input data
- file contents
- any direct personal identifier

The system may include an anonymous install fingerprint for de-duplication or longitudinal comparison as long as it is not user-identifying.

## Submission Model

The first UX is direct submit:
1. user opts in
2. app runs or selects the telemetry bundle
3. app sends the anonymized payload over HTTPS
4. app shows a warm thank-you message on success

No mandatory export/import loop is required in the first version.

## Intake Architecture

Email is explicitly rejected as the primary design.

Reasons:
- poor structure
- weak deduplication
- hard to secure and validate at scale
- noisy maintainer workflow
- poor automation hooks for version-validation promotion

Preferred intake architecture:
- minimal Rust ingest API
- append-only structured storage
- secure transport
- cheap deployment footprint
- simple maintainer review flow for browsing submissions by hardware/version/platform

If a constantly-running self-hosted service is not acceptable in practice, the design may fall back to a cheap managed or semi-managed endpoint, but only after preserving the same payload contract and privacy guarantees.

## Validation Governance

Telemetry produces recommendations, not direct client-side promotion.
User-facing validation state changes only through signed manifest publication.

Governance rules:
- telemetry may justify candidate review, holdbacks, or block recommendations
- repeated field failures may justify a signed remote block
- `validated` promotion still requires maintainer review and local validation evidence
- ROCm and other system-level promotions always require explicit manual PASS

## Security Requirements

- TLS for submission
- signed or versioned payload schema
- strict server-side validation
- bounded payload size
- no code execution based on submission content
- operational logging that avoids storing personal data

## Acceptance Criteria

This spec is satisfied when implementation plans can deliver all of the following:
- an opt-in 180-second minimum stability benchmark
- payloads that exclude personal and filesystem-identifying data
- low-friction direct submission and a thank-you confirmation
- structured intake suitable for hundreds of submissions per week
- telemetry evidence that informs validation policy only through signed manifest publication
