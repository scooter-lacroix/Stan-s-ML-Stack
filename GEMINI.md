# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand the "Stan's ML Stack" project.

## Project Overview

"Stan's ML Stack" is a comprehensive machine learning environment optimized for AMD GPUs, with a focus on large language models (LLMs) and deep learning. It provides a complete set of tools and libraries for training and deploying machine learning models. The stack is designed to work with AMD's ROCm platform, providing CUDA compatibility through HIP, allowing most CUDA-based machine learning code to run on AMD GPUs with minimal modifications.

The project is a Python package named `stans-ml-stack` and includes a collection of shell and Python scripts for installation, configuration, and verification.

## MCP Agent Mail — coordination for multi-agent workflows

What it is
- A mail-like layer that lets coding agents coordinate asynchronously via MCP tools and resources.
- Provides identities, inbox/outbox, searchable threads, and advisory file reservations, with human-auditable artifacts in Git.

Why it's useful
- Prevents agents from stepping on each other with explicit file reservations (leases) for files/globs.
- Keeps communication out of your token budget by storing messages in a per-project archive.
- Offers quick reads (`resource://inbox/...`, `resource://thread/...`) and macros that bundle common flows.

How to use effectively
1) Same repository
   - Register an identity: call `ensure_project`, then `register_agent` using this repo's absolute path as `project_key`.
   - Reserve files before you edit: `reserve_file_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true)` to signal intent and avoid conflict.
   - Communicate with threads: use `send_message(..., thread_id="FEAT-123")`; check inbox with `fetch_inbox` and acknowledge with `acknowledge_message`.
   - Read fast: `resource://inbox/{Agent}?project=<abs-path>&limit=20` or `resource://thread/{id}?project=<abs-path>&include_bodies=true`.
   - Tip: set `AGENT_NAME` in your environment so the pre-commit guard can block commits that conflict with others' active exclusive file reservations.

2) Across different repos in one project (e.g., Next.js frontend + FastAPI backend)
   - Option A (single project bus): register both sides under the same `project_key` (shared key/path). Keep reservation patterns specific (e.g., `frontend/**` vs `backend/**`).
   - Option B (separate projects): each repo has its own `project_key`; use `macro_contact_handshake` or `request_contact`/`respond_contact` to link agents, then message directly. Keep a shared `thread_id` (e.g., ticket key) across repos for clean summaries/audits.

Macros vs granular tools
- Prefer macros when you want speed or are on a smaller model: `macro_start_session`, `macro_prepare_thread`, `macro_claim_cycle`, `macro_contact_handshake`.
- Use granular tools when you need control: `register_agent`, `reserve_file_paths`, `send_message`, `fetch_inbox`, `acknowledge_message`.

Common pitfalls
- "from_agent not registered": always `register_agent` in the correct `project_key` first.
- "CLAIM_CONFLICT": adjust patterns, wait for expiry, or use a non-exclusive reservation when appropriate.
- Auth errors: if JWT+JWKS is enabled, include a bearer token with a `kid` that matches server JWKS; static bearer is used only when JWT is disabled.

## Building and Running

The project offers several methods for installation and setup:

### Installation

*   **PyPI:** The core package can be installed via pip:
    ```bash
    pip install stans-ml-stack
    ```
*   **Interactive Installer:** A curses-based UI for guided installation:
    ```bash
    ./scripts/install_ml_stack_curses.py
    ```
*   **Manual Installation:** Individual scripts are provided for manual installation of each component. The main script is:
    ```bash
    ./scripts/install_ml_stack.sh
    ```
*   **Docker:** A pre-built Docker image is available, and a `Dockerfile` is provided for custom builds.
    ```bash
    docker pull bartholemewii/stans-ml-stack:latest
    ```

### Environment Setup

*   An environment setup script configures the necessary environment variables:
    ```bash
    ./scripts/enhanced_setup_environment.sh
    source ~/.mlstack_env
    ```

### Verification

*   A verification script checks the installation of all components:
    ```bash
    ./scripts/enhanced_verify_installation.sh
    ```

### Testing

*   The project includes a `tests` directory with various tests. The main test runner is:
    ```bash
    ./tests/run_all_tests.sh
    ```

## Development Conventions

*   **Coding Style:**
    *   Python code should follow PEP 8.
    *   Shell scripts should be checked with `shellcheck`.
*   **Testing:**
    *   New features should include tests.
    *   The `tests` directory contains integration, performance, and verification tests.

## Key Files and Directories

*   `README.md`: The main documentation file with a detailed overview of the project.
*   `pyproject.toml`: Defines the Python package metadata and dependencies.
*   `setup.py`: The setup script for the Python package, including scripts and package data.
*   `requirements.txt`: A list of Python dependencies.
*   `VERSION`: Contains the version number of the package.
*   `scripts/`: Contains all the installation, configuration, and verification scripts.
    *   `install_ml_stack.sh`: The main installation script.
    *   `install_ml_stack_curses.py`: The interactive installer.
    *   `enhanced_setup_environment.sh`: The environment setup script.
    *   `enhanced_verify_installation.sh`: The verification script.
*   `stans_ml_stack/`: The source code for the Python package.
    *   `cli/`: Command-line interface scripts.
*   `tests/`: Contains all the tests for the project.
*   `docs/`: Contains the documentation for the project.
*   `Dockerfile`: For building the Docker image.
*   `docker-compose.yml`: For running the project with Docker Compose.


## MCP Agent Mail: coordination for multi-agent workflows

### What it is
- A mail-like layer that lets coding agents coordinate asynchronously via MCP tools and resources.
- Provides identities, inbox/outbox, searchable threads, and advisory file reservations, with human-auditable artifacts in Git.

### Why it's useful
- Prevents agents from stepping on each other with explicit file reservations (leases) for files/globs.
- Keeps communication out of your token budget by storing messages in a per-project archive.
- Offers quick reads (`resource://inbox/...`, `resource://thread/...`) and macros that bundle common flows.

### How to use effectively

**1) Same repository**
- Register an identity: call `ensure_project`, then `register_agent` using this repo's absolute path as `project_key`.
- Reserve files before you edit: `file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true)` to signal intent and avoid conflict.
- Communicate with threads: use `send_message(..., thread_id="FEAT-123")`; check inbox with `fetch_inbox` and acknowledge with `acknowledge_message`.
- Read fast: `resource://inbox/{Agent}?project=<abs-path>&limit=20` or `resource://thread/{id}?project=<abs-path>&include_bodies=true`.
- Tip: set `AGENT_NAME` in your environment so the pre-commit guard can block commits that conflict with others' active exclusive file reservations.

**2) Across different repos in one project (e.g., Next.js frontend + FastAPI backend)**
- Option A (single project bus): register both sides under the same `project_key` (shared key/path). Keep reservation patterns specific (e.g., `frontend/**` vs `backend/**`).
- Option B (separate projects): each repo has its own `project_key`; use `macro_contact_handshake` or `request_contact`/`respond_contact` to link agents, then message directly. Keep a shared `thread_id` (e.g., ticket key) across repos for clean summaries/audits.

**Macros vs granular tools**
- Prefer macros when you want speed or are on a smaller model: `macro_start_session`, `macro_prepare_thread`, `macro_file_reservation_cycle`, `macro_contact_handshake`.
- Use granular tools when you need control: `register_agent`, `file_reservation_paths`, `send_message`, `fetch_inbox`, `acknowledge_message`.

**Common pitfalls**
- "from_agent not registered": always `register_agent` in the correct `project_key` first.
- "FILE_RESERVATION_CONFLICT": adjust patterns, wait for expiry, or use a non-exclusive reservation when appropriate.
- Auth errors: if JWT+JWKS is enabled, include a bearer token with a `kid` that matches server JWKS; static bearer is used only when JWT is disabled.


## Integrating with Beads (dependency-aware task planning)

Beads provides a lightweight, dependency-aware issue database and a CLI (`bd`) for selecting "ready work," setting priorities, and tracking status. It complements MCP Agent Mail's messaging, audit trail, and file-reservation signals. Project: [steveyegge/beads](https://github.com/steveyegge/beads)

### Recommended conventions
- **Single source of truth**: Use **Beads** for task status/priority/dependencies; use **Agent Mail** for conversation, decisions, and attachments (audit).
- **Shared identifiers**: Use the Beads issue id (e.g., `bd-123`) as the Mail `thread_id` and prefix message subjects with `[bd-123]`.
- **Reservations**: When starting a `bd-###` task, call `file_reservation_paths(...)` for the affected paths; include the issue id in the `reason` and release on completion.

### Typical flow (agents)
1) **Pick ready work** (Beads)
   - `bd ready --json` → choose one item (highest priority, no blockers)
2) **Reserve edit surface** (Mail)
   - `file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true, reason="bd-123")`
3) **Announce start** (Mail)
   - `send_message(..., thread_id="bd-123", subject="[bd-123] Start: <short title>", ack_required=true)`
4) **Work and update**
   - Reply in-thread with progress and attach artifacts/images; keep the discussion in one thread per issue id
5) **Complete and release**
   - `bd close bd-123 --reason "Completed"` (Beads is status authority)
   - `release_file_reservations(project_key, agent_name, paths=["src/**"])`
   - Final Mail reply: `[bd-123] Completed` with summary and links

### Mapping cheat-sheet
- **Mail `thread_id`** ↔ `bd-###`
- **Mail subject**: `[bd-###] …`
- **File reservation `reason`**: `bd-###`
- **Commit messages (optional)**: include `bd-###` for traceability

### Event mirroring (optional automation)
- On `bd update --status blocked`, send a high-importance Mail message in thread `bd-###` describing the blocker.
- On Mail "ACK overdue" for a critical decision, add a Beads label (e.g., `needs-ack`) or bump priority to surface it in `bd ready`.

### Pitfalls to avoid
- Do not create or manage tasks in Mail; treat Beads as the single task queue.
- Always include `bd-###` in message `thread_id` to avoid ID drift across tools.


## Using bv as an AI sidecar

bv is a fast terminal UI for Beads projects (.beads/beads.jsonl). It renders lists/details and precomputes dependency metrics (PageRank, critical path, cycles, etc.) so you instantly see blockers and execution order. For agents, it's a graph sidecar: instead of parsing JSONL or risking hallucinated traversal, call the robot flags to get deterministic, dependency-aware outputs.

***IMPORTANT: As an agent, you must ONLY use bv with the robot flags, otherwise you'll get stuck in the interactive TUI that's intended for human usage only!***

- bv --robot-help — shows all AI-facing commands.
- bv --robot-insights — JSON graph metrics (PageRank, betweenness, HITS, critical path, cycles) with top-N summaries for quick triage.
- bv --robot-plan — JSON execution plan: parallel tracks, items per track, and unblocks lists showing what each item frees up.
- bv --robot-priority — JSON priority recommendations with reasoning and confidence.
- bv --robot-recipes — list recipes (default, actionable, blocked, etc.); apply via bv --recipe <name> to pre-filter/sort before other flags.
- bv --robot-diff --diff-since <commit|date> — JSON diff of issue changes, new/closed items, and cycles introduced/resolved.

Use these commands instead of hand-rolling graph logic; bv already computes the hard parts so agents can act safely and quickly.
