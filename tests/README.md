# ML Stack Test Suite

This directory contains comprehensive tests for the Stan's ML Stack, organized into categories for easy maintenance and execution.

## Test Organization

```
tests/
├── integration/          # Cross-integration tests
├── verification/         # Installation verification tests
├── performance/          # Performance benchmarking scripts
├── run_integration_tests.sh    # Run all integration tests
├── run_performance_tests.sh    # Run performance benchmarks
├── run_all_tests.sh            # Run complete test suite
└── README.md             # This file
```

## Running Tests

### Quick Start

To run the complete test suite:

```bash
./tests/run_all_tests.sh
```

### Individual Test Categories

#### Integration Tests

Run cross-integration tests to verify component interactions:

```bash
./tests/run_integration_tests.sh
```

Or run individual integration tests:

```bash
cd tests/integration
./test_env_consistency.sh
./test_package_manager_integration.sh
# ... etc
```

#### Verification Tests

Run installation verification tests:

```bash
cd tests/verification
./custom_verify_installation.sh
./enhanced_verify_installation.sh
```

#### Performance Benchmarks

Run performance benchmarking:

```bash
./tests/run_performance_tests.sh
```

Or run individual benchmarks:

```bash
cd tests/performance
python benchmark_runner.py
python system_benchmark.py
python ml_stack_benchmark.py
```

### Prerequisites

- Bash shell
- Python 3.7+
- Required ML Stack components installed
- For performance tests: psutil, torch

### Test Results

- Integration and verification tests output to console and log files
- Performance benchmarks generate JSON results and comparison reports
- Results are saved in `benchmark_results.json` and `benchmark_report.md`

### Troubleshooting

- Ensure all ML Stack components are properly installed
- Check that Python dependencies are available
- For GPU tests, ensure CUDA/ROCm is properly configured
- Review log files for detailed error information

### Adding New Tests

- Place shell scripts in appropriate subdirectories
- Update runner scripts to include new tests
- Document new tests in this README
- Follow naming convention: `test_*.sh` for integration tests

## Continuous Integration

These tests are designed to be run in CI/CD pipelines. All tests should exit with code 0 on success, non-zero on failure.
