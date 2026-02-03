# Claude Code Assistant Context

This document provides comprehensive context for Claude Code Assistant to understand and effectively work with the "Rusty Stack" project (formerly "Stan's ML Stack").

## Project Overview

**Rusty Stack** is a comprehensive machine learning environment optimized for AMD GPUs, with a focus on large language models (LLMs) and deep learning. It provides a complete set of tools and libraries for training and deploying machine learning models. The stack is designed to work with AMD's ROCm platform, providing CUDA compatibility through HIP, allowing most CUDA-based machine learning code to run on AMD GPUs with minimal modifications.

Formerly known as "Stan's ML Stack", the project is undergoing a gradual rebranding to reflect its modern Rust-based TUI installer.

### Key Characteristics
- **Primary Focus**: AMD GPU optimization (7900 XTX, 7800 XT, 7700 XT)
- **Core Technology**: ROCm platform with HIP for CUDA compatibility
- **Package Name**: `stans-ml-stack` (version 0.1.5) - maintained for backward compatibility
- **Primary Installer**: Rusty-Stack TUI (Rust + Ratatui)
- **Installation Methods**: One-line install, PyPI, Docker, Rusty-Stack TUI, manual scripts
- **Architecture**: Modular Python package + Rust TUI installer + shell script backend

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

## Repository Structure

```
Stan-s-ML-Stack/
├── rusty-stack/              # Primary TUI installer (Rust + Ratatui)
├── stans_ml_stack/           # Python package (backward compatibility)
│   ├── cli/                  # Command-line interface modules
│   │   ├── install.py        # Installation UI (curses-based, deprecated)
│   │   ├── verify.py         # Installation verification
│   │   └── repair.py         # Repair functionality
│   ├── installers/           # Component installation modules
│   │   ├── rocm.py          # ROCm installation
│   │   ├── pytorch.py       # PyTorch setup
│   │   ├── flash_attention.py # Flash Attention build
│   │   ├── megatron.py      # Megatron-LM setup
│   │   └── [other installers]
│   ├── core/                 # Core functionality
│   │   ├── pytorch/         # PyTorch utilities and distributed training
│   │   └── extensions/      # Extended component support (vLLM, Triton, etc.)
│   └── utils/               # Utility functions and benchmarks
├── scripts/                  # Shell installation scripts (backend for all installers)
├── docs/                     # Comprehensive documentation
├── tests/                    # Test suites
├── AGENTS.md                 # Repository guidelines (this file)
├── QWEN.md                   # Qwen agent context
├── GEMINI.md                 # Gemini agent context
├── CLAUDE.md                 # This file
├── pyproject.toml           # Python package configuration
└── VERSION                  # Package version (0.1.5)
```

## Development Conventions

### Build and Test Commands
- **Development install**: `pip install -e .` (or `uv pip install -e .`)
- **Build wheels**: `python -m build`
- **Run tests**: `pytest tests/` or `./tests/run_all_tests.sh`
- **Integration tests**: `./tests/run_integration_tests.sh`
- **Installation verification**: `ml-stack-verify`

### Coding Standards
- **Python**: PEP 8 compliance, four-space indents, descriptive module names
- **Type hints**: Required throughout the codebase
- **Shell scripts**: POSIX-compatible Bash, start with `set -euo pipefail`
- **Formatting**: Run `black .` and `isort .` before commits
- **Linting**: `pylint stans_ml_stack` and `mypy stans_ml_stack`

### Project Guidelines (from AGENTS.md)
- Module organization centers on `stans_ml_stack/` for CLI wrappers and core utilities
- Automation helpers in `scripts/`, verification in `tests/`
- Documentation in `docs/`, benchmark results in `benchmarks/` and `results/`
- Use clear, sentence-style commit subjects aligned with recent history
- Group related code paths into single commits with hardware-sensitive context

## CLI Commands and Entry Points

The Python package provides three main CLI commands (defined in pyproject.toml):

1. **ml-stack-install**: Launches the curses-based installation UI (deprecated)
2. **ml-stack-verify**: Verifies installation and environment setup
3. **ml-stack-repair**: Detects and fixes common installation issues

**Primary Installer**: Use `./scripts/run_rusty_stack.sh` to launch the Rusty-Stack TUI.

## Key Components and Technologies

### Core ML Stack Components
- **ROCm** (6.4.43482): AMD's GPU computing platform
- **PyTorch** (2.6.0+rocm6.4.43482): Deep learning framework with ROCm support
- **ONNX Runtime** (1.22.0): Cross-platform inference accelerator
- **MIGraphX** (2.12.0): AMD's graph optimization library
- **Flash Attention** (2.5.6): Efficient attention computation
- **RCCL**: ROCm Collective Communication Library
- **MPI** (Open MPI 5.0.7): Distributed computing
- **Megatron-LM**: Large language model training framework

### Extension Components
- **Triton** (3.2.0): Parallel programming compiler
- **BITSANDBYTES** (0.45.5): Model quantization
- **vLLM** (0.8.5): High-throughput LLM inference
- **vLLM Studio**: Web UI for vLLM model management (https://github.com/0xSero/vllm-studio)
- **ROCm SMI**: GPU monitoring and management
- **Weights & Biases** (0.19.9): Experiment tracking

## Installation Methods

### 1. One-Line Install (Recommended)
```bash
curl -fsSL https://raw.githubusercontent.com/scooter-lacroix/Stan-s-ML-Stack/main/scripts/install.sh | bash
```

### 2. Rusty-Stack TUI (Primary)
```bash
./scripts/run_rusty_stack.sh
```

### 3. PyPI Installation
```bash
pip install stans-ml-stack
```

### 4. Legacy Curses Installer (Deprecated)
```bash
./scripts/install_ml_stack_curses.py
```
- Features responsive UI with real-time feedback
- Automatically detects hardware and configures environment
- Provides component selection and progress tracking
- **Deprecated**: Use Rusty-Stack TUI instead

### 3. Docker Installation
```bash
# Pre-built image
docker pull bartholemewii/stans-ml-stack:latest
docker run --device=/dev/kfd --device=/dev/dri --group-add video -it bartholemewii/stans-ml-stack:latest

# Build from source
docker build -t stans-ml-stack .
```

### 4. Manual Installation
Individual scripts available for each component:
- `./scripts/install_rocm.sh`
- `./scripts/install_pytorch.sh`
- `./scripts/build_flash_attn_amd.sh`
- etc.

## Environment Configuration

### Automatic Setup
```bash
./scripts/enhanced_setup_environment.sh
source ~/.mlstack_env
```

### Key Environment Variables
- `ROCM_PATH=/opt/rocm`
- `HIP_VISIBLE_DEVICES=0,1`
- `CUDA_VISIBLE_DEVICES=0,1`
- `HSA_OVERRIDE_GFX_VERSION=11.0.0`
- `PYTORCH_ROCM_ARCH=GFX1100`

### Persistent Environment
```bash
sudo ./scripts/create_persistent_env.sh
```
Creates system-wide environment persistence and systemd services.

## Testing and Verification

### Verification Scripts
- **Standard**: `./scripts/enhanced_verify_installation.sh`
- **Custom**: `./scripts/custom_verify_installation.sh`
- **CLI**: `ml-stack-verify`

### Test Categories
- Unit tests in `tests/`
- Integration tests for ROCm-dependent components
- Performance benchmarks in `benchmarks/`
- GPU-specific tests with hardware detection

## Common Troubleshooting Areas

### Environment Issues
- GPU detection problems
- ROCm path configuration
- CUDA compatibility layer
- Memory allocation settings

### Build Issues
- Flash Attention compilation
- ONNX Runtime ROCm support
- Ninja build symlinks
- Python 3.13 compatibility workarounds

### Performance Optimization
- Memory tuning parameters
- Multi-GPU configuration
- Distributed training setup
- Benchmark collection and analysis

## Development Workflow

### When Making Changes
1. Use `TodoWrite` tool to plan multi-step tasks
2. Follow PEP 8 and type hints requirements
3. Add tests for new functionality
4. Update documentation as needed
5. Run verification scripts after environment changes
6. Use appropriate shell script conventions for new installers

### Code Review Guidelines
- Check hardware-sensitive logic carefully
- Verify environment variable handling
- Test with multiple GPU configurations
- Ensure backward compatibility where possible

## Documentation Structure

The project maintains extensive documentation across multiple guides:
- Beginner's guide for new users
- Component-specific installation guides
- Performance optimization guides
- Troubleshooting documentation
- Extension component guides

All documentation follows the patterns established in existing docs and maintains consistency with the AGENTS.md guidelines.

## Contact and Support

- **Author**: Stanley Chisango (Scooter Lacroix)
- **Email**: scooterlacroix@gmail.com
- **GitHub**: https://github.com/scooter-lacroix
- **Documentation**: Comprehensive guides in `docs/` directory

This context should enable Claude Code Assistant to effectively understand, navigate, and contribute to the Rusty Stack project while maintaining consistency with established conventions and requirements.