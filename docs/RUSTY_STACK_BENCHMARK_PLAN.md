# Rusty-Stack Performance Benchmark Infrastructure Plan

## Executive Summary

This plan outlines a complete performance benchmark system for Rusty-Stack using **proper Ratatui Chart widgets** for visualization, not simple text with borders. The plan includes three benchmark tiers with rich chart-based output.

---

### 📊 Chart Architecture

The benchmark system will leverage Ratatui's `Chart` widget with `Dataset` components:

```rust
// Core chart types we'll use
use ratatui::widgets::{Chart, Dataset, Axis, Block, LegendPosition};
use ratatui::symbols::{Marker, GraphType};
use ratatui::style::{Color, Style, Modifier};
```

**Chart Configuration Pattern**:
- **Line Charts**: For time-series data (benchmark progress, multi-run averages)
- **Scatter Charts**: For discrete measurements (memory bandwidth at various sizes)
- **Bar Charts**: For before/after comparisons, component-specific results

---

### 📁 Proposed File Structure

```
rusty-stack/src/
├── benchmarks/
│   ├── mod.rs                    # Module exports
│   ├── pre_installation/         # GPU baseline benchmarks
│   │   ├── gpu_capability.rs
│   │   ├── memory_bandwidth.rs
│   │   └── tensor_core_test.rs
│   │
│   ├── component_specific/       # Per-component benchmarks
│   │   ├── pytorch_bench.rs
│   │   ├── triton_bench.rs
│   │   ├── flash_attention_bench.rs
│   │   └── vllm_bench.rs
│   │
│   └── gpu_performance/          # Low-level GPU tests
│       ├── gemm_bench.rs
│       └── tensor_ops_bench.rs
│
└── widgets/
    ├── mod.rs
    ├── benchmark_charts.rs       # Chart widget builders
    ├── comparison_view.rs        # Before/after comparison charts
    └── chart_animator.rs         # Animated chart updates
```

---

### 📈 Chart Types by Benchmark Category

#### 1. Pre-Installation Benchmarks (GPU Capability & Baseline)

| Benchmark | Chart Type | X-Axis | Y-Axis | Datasets |
|-----------|------------|--------|--------|----------|
| **Memory Bandwidth** | Line + Scatter | Size (MB) | Bandwidth (GB/s) | Copy, Read, Write |
| **Tensor Core Test** | Bar | Precision (FP16/BF16/TF32) | TFLOPS | Peak, Measured |
| **Compute Capability** | Gauge | N/A | Utilization % | Compute, Memory |

#### 2. Component-Specific Benchmarks

| Component | Chart Type | Purpose |
|-----------|------------|---------|
| **PyTorch** | Line chart | MatMul performance vs matrix size |
| **Flash Attention** | Bar chart | Attention speed comparison (FA vs Standard) |
| **vLLM** | Line + Error bars | Throughput (tok/s) vs batch size |
| **DeepSpeed** | Stacked bar | ZeRO memory savings |

#### 3. Before/After Comparison Charts

Two datasets: Before (Red) and After (Green) with visual indicators:
- **Green upward arrow**: improved (>5%)
- **Red downward arrow**: degraded (<-5%)
- **Yellow dash**: within 5% tolerance

---

### 🎨 Color Scheme for Benchmark Charts

| Data Type | Color | Usage |
|-----------|-------|-------|
| **Baseline/Before** | `Color::Red` | Pre-installation or comparison baseline |
| **Current/After** | `Color::Green` | Post-installation or improved state |
| **FP32** | `Color::White` | Standard precision |
| **FP16** | `Color::Cyan` | Half precision (Tensor Core) |
| **BF16** | `Color::Yellow` | Brain float16 |
| **TF32** | `Color::Magenta` | TensorFloat-32 |
| **Warning** | `Color::LightYellow` | Degradation detected |
| **Error** | `Color::LightRed` | Benchmark failure |

---

### 📊 TUI Performance Pages

#### Page 1: Pre-Installation Benchmarks
- Memory Bandwidth: Line chart showing HBM vs System RAM bandwidth across transfer sizes
- Tensor Core Performance: Bar chart comparing FP16/BF16/TF32 TFLOPS

#### Page 2: Component Performance (PyTorch)
- GEMM Performance: Line chart showing GFLOPS vs matrix size for FP32/FP16/BF16

#### Page 3: Before/After Comparison
- Performance Comparison: Overlaid line charts showing before/after metrics
- Visual alerts for degradations (>5% slower)
- Summary of improvements

---

### 📈 Data Structure for Chart Generation

```rust
// Core chart data structures
pub struct BenchmarkChartData {
    pub title: String,
    pub x_axis: AxisData,
    pub y_axis: AxisData,
    pub datasets: Vec<DatasetData>,
    pub legend_position: Option<LegendPosition>,
}

pub struct DatasetData {
    pub name: String,
    pub data_points: Vec<(f64, f64)>,  // (x, y) coordinates
    pub marker: Marker,
    pub graph_type: GraphType,
    pub color: Color,
}

pub struct AxisData {
    pub title: String,
    pub bounds: [f64; 2],  // [min, max]
    pub labels: Vec<String>,
}
```

---

### 📊 Benchmark Categories and Chart Mappings

| Category | Chart Type | Metrics Displayed | Update Pattern |
|----------|------------|-------------------|----------------|
| **GPU Capability** | Gauge + Bar | Compute units, VRAM, ROCm version | Single-shot |
| **Memory Bandwidth** | Line + Scatter | Copy/Read/Write vs size | Single-shot |
| **GEMM** | Line | GFLOPS vs matrix size | Single-shot |
| **Tensor Ops** | Bar | FP32/FP16/BF16/TF32 TFLOPS | Single-shot |
| **Flash Attention** | Bar | FA vs Standard (speed, memory) | Single-shot |
| **vLLM** | Line + Error | TPS vs batch size (with CI) | Repeated runs |
| **DeepSpeed** | Stacked Bar | Memory saved by ZeRO stage | Single-shot |
| **Comparison** | Line (2 datasets) | Before/After metrics | Two-phase |
| **Trend** | Multi-line | Historical performance | Accumulating |

---

### 🎯 Implementation Tasks

| Task | Status | Priority | Dependencies |
|------|--------|----------|--------------|
| Create `rusty-stack/src/benchmarks/mod.rs` | ⬜ | High | None |
| Create `rusty-stack/src/widgets/mod.rs` | ⬜ | High | None |
| Implement `BenchmarkChartData` structs | ⬜ | High | None |
| Implement Memory Bandwidth chart | ⬜ | High | GPU capability detection |
| Implement GEMM Performance chart | ⬜ | High | PyTorch installation |
| Implement Tensor Core chart | ⬜ | Medium | Memory bandwidth |
| Implement Flash Attention chart | ⬜ | Medium | Flash Attention install |
| Implement vLLM Throughput chart | ⬜ | Medium | vLLM installation |
| Implement Before/After Comparison | ⬜ | High | Pre & post benchmarks |
| Integrate with Performance TUI page | ⬜ | High | All benchmarks |
| Add chart animation for live updates | ⬜ | Low | Basic charts |
| Write benchmark result persistence | ⬜ | Medium | All benchmarks |

---

### 🔧 Key Implementation Details

#### Example: Memory Bandwidth Chart Builder

```rust
pub fn create_memory_benchmark_chart(
    results: &MemoryBenchmarkResults,
) -> Chart<'static> {
    let datasets = vec![
        Dataset::default()
            .name("HBM Copy")
            .data(&results.hbm_copy_points())
            .marker(Marker::Dot)
            .graph_type(GraphType::Line)
            .cyan(),
        Dataset::default()
            .name("HBM Read")
            .data(&results.hbm_read_points())
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .green(),
        Dataset::default()
            .name("System RAM")
            .data(&results.ram_points())
            .marker(Marker::HalfBlock)
            .graph_type(GraphType::Line)
            .red(),
    ];
    
    Chart::new(datasets)
        .block(Block::new().title("Memory Bandwidth Test"))
        .x_axis(Axis::default()
            .title("Transfer Size (MB)")
            .bounds([0.0, 1100.0])
            .labels(["64", "256", "512", "1024"]))
        .y_axis(Axis::default()
            .title("Bandwidth (GB/s)")
            .bounds([0.0, 1500.0])
            .labels(["0", "500", "1000", "1500"]))
        .legend_position(Some(LegendPosition::TopLeft))
}
```

#### Example: Comparison Chart Builder

```rust
pub fn create_comparison_chart(
    before: &BenchmarkResults,
    after: &BenchmarkResults,
) -> Chart<'static> {
    let datasets = vec![
        Dataset::default()
            .name("Before Installation")
            .data(&before.to_chart_points())
            .marker(Marker::HalfBlock)
            .graph_type(GraphType::Line)
            .red(),
        Dataset::default()
            .name("After Installation")
            .data(&after.to_chart_points())
            .marker(Marker::Dot)
            .graph_type(GraphType::Line)
            .green(),
    ];
    
    Chart::new(datasets)
        .block(Block::new().title("Installation Impact Analysis"))
        .x_axis(Axis::default()
            .title("Matrix Size")
            .bounds([0.0, 3.0])
            .labels(["512", "1024", "2048", "4096"]))
        .y_axis(Axis::default()
            .title("Performance (GFLOPS)")
            .bounds([0.0, 5000.0])
            .labels(["0", "1000", "2000", "3000", "4000", "5000"]))
        .legend_position(Some(LegendPosition::TopRight))
}
```

---

### 📝 Next Steps

1. Save this plan to `docs/RUSTY_STACK_BENCHMARK_PLAN.md`
2. Create directory structure for benchmarks
3. Implement core chart data structures
4. Build benchmark runners for each category
5. Integrate with Performance TUI page
6. Add persistence and comparison logic

---

**Plan Version**: 1.0  
**Last Updated**: 2026-02-03  
**Status**: Ready for Implementation
