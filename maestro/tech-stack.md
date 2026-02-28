# Rusty Stack - Technology Stack

## Languages

| Language | Version | Purpose |
|----------|---------|---------|
| **Rust** | 2021 edition | TUI installer and utilities |
| **Python** | 3.10-3.13 | ML components and utilities |
| **Bash** | - | Installation and automation scripts |

## Rust Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| ratatui | 0.26 | Terminal UI framework |
| crossterm | 0.27 | Cross-platform terminal manipulation |
| anyhow | 1.0 | Error handling |
| serde | 1.0 | Serialization |
| sysinfo | 0.30 | System information |

## Core ML Frameworks

| Framework | Version | Purpose |
|-----------|---------|---------|
| **PyTorch** | 2.6.0+rocm6.4 | Deep learning with AMD GPU support |
| **ONNX Runtime** | 1.22.0 | Cross-platform inference |
| **MIGraphX** | 2.12.0 | AMD graph optimization |

## ML Extensions

| Extension | Purpose |
|-----------|---------|
| **Flash Attention** | High-performance attention (Triton + CK variants) |
| **Triton** | GPU kernel compiler for AMD |
| **bitsandbytes** | 8-bit quantization for AMD GPUs |
| **vLLM** | High-throughput LLM inference |
| **DeepSpeed** | Memory-efficient distributed training |
| **Megatron-LM** | Large language model training |
| **ComfyUI** | Node-based AI image generation with ROCm |

## Communication & Distributed Computing

| Component | Version | Purpose |
|-----------|---------|---------|
| **RCCL** | Latest | ROCm Collective Communications Library |
| **OpenMPI** | 5.0.7 | Message Passing Interface |

## Package Managers

| Manager | Purpose |
|---------|---------|
| **uv** | Primary Python package manager |
| **pip** | Fallback Python package manager |
| **cargo** | Rust package manager |

## Containerization

| Platform | Purpose |
|----------|---------|
| **Podman** | Primary container engine |
| **Docker** | Through Podman compatibility layer |

## Testing

| Type | Framework |
|------|-----------|
| Shell tests | Custom test framework |
| Python tests | pytest/unittest |

## Build Tools

| Tool | Purpose |
|------|---------|
| **CMake** | Native compilation |
| **cargo** | Rust builds |
| **setuptools** | Python packaging |

## ROCm Channels

| Channel | Version | Stability |
|---------|---------|-----------|
| **Legacy** | 6.4.3 | Production-proven |
| **Stable** | 7.1 | Production-ready (RDNA 3) |
| **Latest** | 7.2 | Cutting-edge (RDNA 4) |
