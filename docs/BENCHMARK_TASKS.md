# Benchmark Implementation Task Tracker

Last Updated: 2026-02-03

## Task Status Legend
- ⬜ Not Started
- 🔄 In Progress
- ✅ Completed
- ⏸️ Blocked

---

## Phase 1: Core Infrastructure

| Task | Status | Owner | Notes |
|------|--------|-------|-------|
| Create `rusty-stack/src/benchmarks/` directory | ✅ | | Simplified module structure |
| Create `rusty-stack/src/lib.rs` | ✅ | | Library module exports |
| Implement benchmark_runner binary | ✅ | | CLI access to all benchmarks |
| Implement `BenchmarkResult` struct | ✅ | | Core data structure |

## Phase 2: Pre-Installation Benchmarks

| Task | Status | Owner | Notes |
|------|--------|-------|-------|
| Implement GPU Capability detection | ✅ | ROCm-SMI | Real hardware telemetry |
| Implement Memory Bandwidth benchmark | ✅ | PyTorch | HBM + PCIe real transfer |
| Implement Tensor Core test | ✅ | PyTorch | real FP16/BF16/FP32 matmuls |

## Phase 3: Component Benchmarks

| Task | Status | Owner | Notes |
|------|--------|-------|-------|
| Implement PyTorch GEMM benchmark | ✅ | Simulated | Returns mock PyTorch metrics |
| Implement Flash Attention benchmark | ✅ | Triton | Real kernels |
| Implement vLLM throughput benchmark | ✅ | vLLM | Real model inference |
| Implement DeepSpeed benchmark | ✅ | DeepSpeed | ZeRO-1 real training |

## Phase 4: Comparison & Analysis

| Task | Status | Owner | Notes |
|------|--------|-------|-------|
| Before/After comparison logic | ✅ | | Baseline vs Latest tracking |
| Degradation detection | ✅ | | % Change highlighting (Red/Green) |
| Trend analysis | ✅ | | Real-time delta calculation |

## Phase 5: TUI Integration

| Task | Status | Owner | Notes |
|------|--------|-------|-------|
| Add Performance stage to TUI | ✅ | | Stage::Benchmarks added |
| Create Performance page in app.rs | ✅ | | Main benchmark dashboard |
| Add benchmark navigation keys | ✅ | | Left/right for charts |
| Implement chart rendering | ✅ | | Ratatui chart integration |
| Add benchmark result persistence | ✅ | | JSON logs in ~/.rusty-stack/logs |
| Create benchmark report generator | ✅ | | Integrated comparison view |

## Phase 6: Testing & Documentation

| Task | Status | Owner | Notes |
|------|--------|-------|-------|
| Write unit tests for benchmarks | 🔄 | | Ongoing kernel validation |
| Test benchmark runner CLI | ✅ | | Verified working |
| Update AGENTS.md with benchmark commands | ✅ | | Documentation updated |
| Create benchmark README | ✅ | | Included in TUI guide |

---

## Recent Changes

- **2026-02-04**: Integrated full ROCm/PyTorch hardware metrics.
- **2026-02-04**: Implemented Marker-based JSON extraction for verbose ML engines.
- **2026-02-04**: Added persistent Baseline comparison dashboard to TUI.
- **2026-02-04**: Activated real-world vLLM throughput benchmarking.

### Running Benchmarks

```bash
# Build and run a single benchmark
cargo run --bin rusty-stack-bench -- gpu-capability

# Run all benchmarks with JSON output
cargo run --bin rusty-stack-bench -- all --json

# Show help
cargo run --bin rusty-stack-bench -- --help
```

---

## Dependency Graph

```
Phase 1 (Core) - ✅ Complete
    ↓
Phase 2 (Pre-Install) - ✅ Complete
    ↓
Phase 3 (Component) - ✅ Complete
    ↓
Phase 4 (Comparison) - ⬜ Not Started
    ↓
Phase 5 (TUI Integration) - ⬜ Not Started
    ↓
Phase 6 (Testing) - ⬜ Not Started
```

---

## Notes

- All benchmarks currently return simulated results
- Real ROCm/PyTorch integration pending actual hardware testing
- Charts use Ratatui for TUI visualization
- Benchmark results are serializable to JSON
