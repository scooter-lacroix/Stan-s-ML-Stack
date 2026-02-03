# Repository Guidelines

## Project Overview & Rebranding

**Rusty Stack** (formerly "Stan's ML Stack") is a comprehensive machine learning environment optimized for AMD GPUs. The project is undergoing a gradual rebranding to reflect its modern Rust-based TUI installer.

- **New Name**: Rusty Stack
- **Old Name**: Stan's ML Stack
- **Python Package**: `stans-ml-stack` (maintained for backward compatibility)
- **Primary Installer**: Rusty-Stack TUI (`rusty-stack/`)
- **Repository**: `scooter-lacroix/Stan-s-ML-Stack` (unchanged)

## Project Structure & Module Organization
Rusty Stack centers on `stans_ml_stack/` (Python package for backward compatibility) and `rusty-stack/` (primary TUI installer). The Python package houses CLI wrappers, core GPU installers, and utility modules. Automation helpers live under `scripts/`, echoed by verification harnesses in `tests/verification`. Reference material sits in `docs/` and `examples/`, while performance JSON and reports land in `benchmarks/`, `results/`, and `assets/`. Keep data-heavy artifacts out of version control and route runtime output to `results/`.

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

## Build, Test, and Development Commands
Create a development environment with `pip install -e .` (use `uv pip install -e .` when uv is available); this wires console entry points like `ml-stack-install`. For the primary installer, use `./scripts/run_rusty_stack.sh` to build and launch the Rusty-Stack TUI. Build distributable wheels via `python -m build`. Run focused Python tests with `pytest tests/`, or execute `./tests/run_all_tests.sh` for the full GPU, integration, and performance sweep. For faster feedback in CI, prefer `./tests/run_integration_tests.sh`. Use `ml-stack-verify` after system changes to confirm ROCm dependencies.

## Coding Style & Naming Conventions
Python code follows PEP 8 with four-space indents, descriptive module names (`verify_installation.py`), and module-level docstrings. Leverage type hints and dataclasses where they clarify installer state. Before committing, run `black .` and `isort .`, then lint with `pylint stans_ml_stack` and `mypy stans_ml_stack`. Rust code in `rusty-stack/` follows standard Rust conventions (`cargo fmt`, `cargo clippy`). Shell scripts should remain POSIX-compatible Bash, start with `set -euo pipefail`, and use uppercase environment variables plus kebab-case filenames (`install_flash_attention_ck.sh`).

## Testing Guidelines
Unit and smoke checks sit under `tests/` (`test_*.py` for pytest, `test_*.sh` for shell harnesses). Integration and verification suites assume AMD GPUs with ROCm; gate heavy runs behind feature flags or skip markers if hardware is unavailable. Performance benchmarks write JSON and Markdown into `benchmark_results.json` and `benchmark_report.md`; clean or ignore new artifacts. Keep coverage parity by adding tests alongside new installers and update the runner scripts when introducing flows.

## Commit & Pull Request Guidelines
Use clear, sentence-style subjects aligned with recent history ("Improve install scripts: …"). Group related code paths into a single commit and include context in the body when touching hardware-sensitive logic. Pull requests should summarize the scenario, link tracking issues, list verification commands, and attach logs or screenshots (for example, installer UI captures). Request review from infrastructure maintainers when modifying `scripts/`, `rusty-stack/`, or GPU verification pathways.
