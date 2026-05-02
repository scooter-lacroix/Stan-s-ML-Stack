# Rusty Stack Windows Foundation Spec

**Date:** 2026-04-24
**Parent:** `docs/superpowers/specs/2026-04-24-rusty-stack-master-spec-bible.md`

## Goal

Define the first real Windows `.exe` as a user-facing Rusty Stack control app that can operate across native Windows capabilities and WSL2-backed Linux services without requiring users to understand terminals, WSL, or Linux internals.

## Product Promise

After installation, the Windows user should not need to:
- manually open WSL
- manually map paths between Windows and Linux
- manually forward ports or discover service URLs
- manually manage background processes for supported stack services
- understand which components are native and which are WSL-backed

## First Shipping Model

The architecture is dual-path by design:
- native Windows path where upstream or project-owned support exists
- WSL2-backed Linux path where components or ROCm dependencies require Linux

The first shippable path may lean heavily on WSL2-backed execution, but the app itself must feel like a native Windows product.

## Windows V1 Scope

Included in Windows v1:
- native control shell
- shared scan/plan/apply integration
- WSL detection plus guided provisioning and health checks
- service launch surfaces and endpoint exposure for supported components
- local logs and error summaries
- telemetry consent and submission UI

Explicitly out of scope for Windows v1 unless later approved in a child milestone:
- full greenfield install automation for every supported component
- arbitrary LAN-facing service hosting
- pretending Linux-only components are truly native Windows components

## Core Responsibilities Of The Windows App

- backend capability detection
- WSL2 provisioning and health checks when required
- update planning and execution through shared Rust contracts
- service management and launch surfaces
- log viewing and error summarization
- telemetry consent and submission UI
- safe exposure of local web services to Windows users

## Backend Routing Model

Each component must declare one or more execution modes:
- `windows-native`
- `wsl-backed-linux`
- `unsupported`

Routing decisions must consider:
- manifest support status
- current hardware
- current platform/backend capabilities
- validation tier
- required privileges

## WSL2 Abstraction Requirements

When WSL2 is required, the app must manage:
- distro detection or installation guidance
- resource-conscious startup and shutdown
- local path translation
- service endpoint discovery and bridging
- process liveness monitoring
- restart and recovery behavior

WSL2 should behave like an implementation detail, not a manual prerequisite workflow.

## Service Exposure Model

For web-based tools and services, the app must:
- launch or connect to the backend service
- expose the service through a stable Windows-facing endpoint or launcher
- present the user with a product-style `open` action instead of raw terminal commands

Security default:
- all first-version service exposure is local-only by default (`localhost` or equivalent loopback bridge)
- no LAN exposure is permitted in v1 unless a future spec explicitly introduces it

Target interaction model should feel closer to tools like ComfyUI launchers, Pinokio, LM Studio, text-generation-webui managers, Ollama, or similar desktop control surfaces than to a shell script wrapper.

## Update Integration

The Windows app must reuse shared Rust update contracts.
It must not create a second independent update ruleset.

That means:
- same manifest policy
- same validation tiers
- same safe/guarded/blocked logic
- same component selection model
- same verification evidence model

## Non-Goals For The First Windows Track

- claiming native Windows ROCm support where it does not actually exist
- replacing Linux-backed components with fake native wrappers that weaken reliability
- exposing raw Linux complexity in the default UX

## Acceptance Criteria

This spec is satisfied when implementation plans can deliver all of the following:
- a native Windows shell that reuses shared Rust planning contracts
- WSL-backed components launch without the user manually opening WSL
- supported local services open through a Windows-facing launcher or endpoint
- service exposure stays local-only by default
- telemetry consent and update summaries are surfaced in the app
