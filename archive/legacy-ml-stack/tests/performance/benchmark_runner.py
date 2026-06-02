#!/usr/bin/env python3
"""
Main benchmarking orchestrator for ML Stack performance tests.
Runs system and ML stack benchmarks and generates comparison reports.
"""

import json
import sys
import os
from typing import Dict, Any

# Add the performance directory to path so we can import the modules
sys.path.insert(0, os.path.dirname(__file__))

from system_benchmark import run as run_system_benchmark
from ml_stack_benchmark import run as run_ml_benchmark
from comparison_report import run as generate_comparison_report

def run_all_benchmarks() -> Dict[str, Any]:
    """Run all benchmark suites."""
    print("Running system benchmarks...")
    system_results = run_system_benchmark()

    print("Running ML stack benchmarks...")
    ml_results = run_ml_benchmark()

    # Combine results
    combined_results = {
        "timestamp": system_results.get("timestamp", ml_results.get("timestamp")),
        "system_benchmarks": system_results.get("benchmarks", {}),
        "ml_benchmarks": ml_results.get("benchmarks", {})
    }

    return combined_results

def main():
    """Main entry point."""
    print("Starting ML Stack Performance Benchmarks")
    print("=" * 50)

    results = run_all_benchmarks()

    # Save raw results
    results_file = "benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved to {results_file}")

    # Generate comparison report
    print("\nGenerating comparison report...")
    report = generate_comparison_report(results)
    print(report)

    # Save report
    report_file = "benchmark_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_file}")

    print("\nBenchmarking complete!")

if __name__ == "__main__":
    main()