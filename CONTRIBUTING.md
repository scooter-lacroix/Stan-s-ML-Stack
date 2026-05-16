# Contributing to Rusty Stack

Thank you for your interest in contributing to Rusty Stack! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Contributor Levels](#contributor-levels)
- [AI Usage Policy](#ai-usage-policy)
- [Guidelines for Contributors Using AI](#guidelines-for-contributors-using-ai)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Coding Guidelines](#coding-guidelines)
- [Naming Guidelines](#naming-guidelines)
- [Resources](#resources)

## Contributor Levels

### Contributors

Anyone who submits a pull request. Contributors:

- Submit pull requests that follow the guidelines in this document
- Respond to code review feedback in a timely manner
- Ensure their changes pass all CI checks before requesting review

### Collaborators (Triage)

Trusted contributors with additional repository access. Collaborators:

- Triage issues and pull requests
- Label and organize issues
- Review pull requests and provide feedback
- Merge approved pull requests

### Maintainers

Core project maintainers responsible for the overall direction and quality of Rusty Stack. Maintainers:

- Set project direction and priorities
- Make final decisions on architectural changes
- Review and approve significant changes
- Manage releases and versioning
- Enforce project guidelines and code of conduct

## AI Usage Policy (Anti-AI-slop Policy)

> [!IMPORTANT]
> This project does **not** accept pull requests that are fully or predominantly AI-generated. AI tools may be utilized solely in an assistive capacity.
>
> Read more: [CONTRIBUTING.md](CONTRIBUTING.md)

AI assistance is permissible only when the majority of the code is authored by a human contributor, with AI employed exclusively for corrections or to expand on verbose modifications that the contributor has already conceptualized.

### Permitted AI Usage

The following uses of AI tools are acceptable:

- **Learning and understanding**: Using AI to learn about ROCm, HIP, Rust patterns, or the codebase architecture
- **Review assistance**: Asking AI to help review your own code before submitting (the code must be yours)
- **Mechanical tasks**: Using AI for repetitive, mechanical transformations (renaming, formatting, generating boilerplate from your design)
- **Documentation drafts**: Using AI to help draft documentation for changes you have already made
- **Writing code when you designed the solution**: AI may be used to write code that expands on a solution you have fully conceptualized and designed — you must be able to explain every line of the resulting code and why it solves the problem

### Prohibited AI Usage

The following uses of AI tools are **not** accepted:

- **AI-written PR descriptions or commit messages**: All PR descriptions and commit messages must be written by the contributor
- **AI responses to code review feedback**: When a reviewer asks a question or provides feedback, the response must come from the contributor's own understanding — not an AI-generated explanation
- **Implementing features without understanding**: You must be able to explain every line of code in your PR. If you cannot explain why a piece of code exists or how it works, do not submit it
- **Automated AI commits**: No fully automated AI-driven commit pipelines. Every commit must represent human-reviewed and human-understood changes

### Guidelines for AI Coding Agents

If you are using an AI coding agent (e.g., Cursor, Copilot, Claude Code, or similar):

1. **Consider maintainer workload**: Maintainers review code to ensure quality and correctness. AI-generated code that requires extensive review to verify understanding wastes maintainer time. Write code that is easy to review because *you* understand it.

2. **Verify comprehension**: Before submitting a PR, verify that you understand every line of code. If an AI agent wrote something you cannot explain, remove it and write it yourself.

3. **Provide guidance, not solutions**: When using AI, provide detailed guidance about what you want and why — do not ask the AI to "implement feature X" and then submit the result verbatim. The thinking and design must be yours.

4. **Disclose AI usage**: Check the "AI meaningfully contributed to this PR" box in the PR template and describe how AI was used. Transparency builds trust.

## Pull Request Guidelines

### Before Submitting

1. **Search existing PRs**: Check if your change or a similar one has already been submitted. If it has, comment on the existing PR rather than creating a duplicate.

2. **Test your changes locally**:
   ```bash
   # Run the full test suite
   cd rusty-stack && cargo test -- --test-threads=8

   # Ensure formatting is clean
   cargo fmt --check

   # Ensure clippy is happy
   cargo clippy -- -D warnings
   ```

3. **One PR per feature or fix**: Keep pull requests focused. Do not combine unrelated changes in a single PR. Each PR should address one concern.

### PR Submission

1. **Title**: Use a clear, descriptive title that summarizes the change
2. **Description**: Write the PR description yourself (no AI-generated descriptions). Explain:
   - What the change does
   - Why the change is needed
   - How the change was tested
3. **Checklist**: Complete all items in the PR template checklist
4. **AI Disclosure**: Honestly disclose whether AI tools meaningfully contributed to the PR

### Review Process

- All PRs require at least one maintainer review
- Respond to review feedback yourself — do not paste reviewer comments into an AI and submit the response
- Be prepared to explain any part of your code during review
- Address all review comments before requesting re-review

## Coding Guidelines

### Rust Conventions

Rusty Stack is primarily a Rust project. Follow standard Rust conventions:

- **Formatting**: Run `cargo fmt` before every commit. Do not submit PRs with formatting changes mixed into feature code.
- **Linting**: Run `cargo clippy -- -D warnings` and address all warnings before submitting.
- **Naming**: Use `snake_case` for functions, variables, and modules. Use `PascalCase` for types and traits. See [Naming Guidelines](#naming-guidelines) below.
- **Error handling**: Use `Result<T, E>` with descriptive error types. Avoid `unwrap()` in production code; use `expect()` with a clear message or proper error propagation.
- **Documentation**: Add `///` doc comments to all public functions, structs, and traits.

### Follow Existing Patterns

- **Installer modules**: New component installers go in `src/installers/components/` and follow the pattern of existing installers (see `rocm.rs` or `pytorch.rs` for reference).
- **Shared infrastructure**: Reusable code goes in `src/installers/common/`. Check existing utilities before adding new ones.
- **Test placement**: Unit tests go in `#[cfg(test)]` modules within the source file. Integration tests go in `tests/`.
- **No new `extern crate`**: Do not add `extern crate` declarations. Add dependencies to `Cargo.toml` only.

### Python Code

For the Python package (`stans_ml_stack/`):

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Add docstrings to all public functions and classes
- Maintain the mirrored structure between `core/`, `extensions/`, and their `stans_ml_stack/` counterparts

### Shell Scripts

Remaining shell scripts (verification and benchmarks only):

- All scripts must be executable (`chmod +x`)
- Use `set -euo pipefail` at the top of every script
- Handle errors gracefully with meaningful error messages
- Source shared libraries from `scripts/lib/`

## Naming Guidelines

### Rust Naming

Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/naming.html):

| Element | Convention | Example |
|---------|-----------|---------|
| Crates | `snake_case` | `rusty_stack` |
| Modules | `snake_case` | `installers`, `platform` |
| Types | `PascalCase` | `LlamaCppInstaller`, `RocmChannel` |
| Traits | `PascalCase` | `Installer`, `HardwareDetector` |
| Functions | `snake_case` | `detect_gpu`, `install_component` |
| Constants | `SCREAMING_SNAKE_CASE` | `MAX_RETRIES`, `DEFAULT_INSTALL_PATH` |
| Statics | `SCREAMING_SNAKE_CASE` | `GLOBAL_CONFIG` |
| Type parameters | concise `PascalCase` | `T`, `E`, `RocmChan` |
| Lifetimes | short lowercase | `'a`, `'ctx` |

### File Naming

- Rust source files: `snake_case.rs` (e.g., `llama_cpp.rs`, `sealed_token.rs`)
- Module directories: `snake_case/` (e.g., `benchmark_runners/`)
- Test files: `snake_case.rs` in `tests/` directory
- Shell scripts: `snake_case.sh` (e.g., `verify_installation.sh`)

## Resources

### Rusty Stack Documentation

- [README.md](README.md) — Project overview and quick start
- [MIGRATION.md](MIGRATION.md) — Migration guide from legacy shell/Python to Rust
- [CLAUDE.md](CLAUDE.md) — Development guidelines and architecture overview
- [docs/](docs/) — Comprehensive project documentation
  - [Multi-Channel Guide](docs/MULTI_CHANNEL_GUIDE.md) — ROCm channel selection
  - [Beginners Guide](docs/guides/beginners_guide.md) — Getting started guide

### Rusty Llama

Rusty Stack includes **Rusty Llama**, our optimized llama.cpp runtime with TurboQuant compression, RDNA3 WMMA flash attention, and pre-built binary distribution for AMD GPUs. Install it through Rusty Stack:

```bash
rusty install llama-cpp
```

Docs: [https://github.com/scooter-lacroix/rusty-llama-docs](https://github.com/scooter-lacroix/rusty-llama-docs)

### External Resources

- [The Rust Programming Language](https://doc.rust-lang.org/book/) — Official Rust book
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) — Rust naming and design conventions
- [ROCm Documentation](https://rocm.docs.amd.com/) — AMD ROCm platform documentation
- [ratatui](https://ratatui.rs/) — Terminal UI framework used by Rusty Stack

---

By contributing to Rusty Stack, you agree that your contributions will be licensed under the [MIT License](LICENSE).
