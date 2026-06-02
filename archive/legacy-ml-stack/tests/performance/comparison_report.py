#!/usr/bin/env python3
"""
Generate before/after comparison reports for performance benchmarks.
"""

import json
import os
from typing import Dict, Any, Optional

def load_previous_results(filepath: str) -> Optional[Dict[str, Any]]:
    """Load previous benchmark results from file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def save_results(results: Dict[str, Any], filepath: str):
    """Save benchmark results to file."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

def compare_results(current: Dict[str, Any], previous: Dict[str, Any]) -> Dict[str, Any]:
    """Compare current and previous benchmark results."""
    comparison = {}

    def compare_dict(curr_dict: Dict[str, Any], prev_dict: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        comp = {}
        for key in curr_dict:
            if key in prev_dict:
                if isinstance(curr_dict[key], dict) and isinstance(prev_dict[key], dict):
                    comp[key] = compare_dict(curr_dict[key], prev_dict[key], f"{path}.{key}")
                elif isinstance(curr_dict[key], (int, float)) and isinstance(prev_dict[key], (int, float)):
                    curr_val = curr_dict[key]
                    prev_val = prev_dict[key]
                    if prev_val != 0:
                        change_percent = ((curr_val - prev_val) / prev_val) * 100
                        comp[key] = {
                            "current": curr_val,
                            "previous": prev_val,
                            "change_percent": change_percent,
                            "improvement": "better" if change_percent < 0 else "worse"
                        }
                    else:
                        comp[key] = {
                            "current": curr_val,
                            "previous": prev_val,
                            "change_percent": None
                        }
                else:
                    comp[key] = {
                        "current": curr_dict[key],
                        "previous": prev_dict[key]
                    }
            else:
                comp[key] = {"current": curr_dict[key], "previous": None}
        return comp

    if "benchmarks" in current and "benchmarks" in previous:
        comparison["benchmarks"] = compare_dict(current["benchmarks"], previous["benchmarks"])

    return comparison

def generate_report(comparison: Dict[str, Any]) -> str:
    """Generate a human-readable report from comparison."""
    report_lines = ["# Performance Benchmark Comparison Report\n"]

    def format_comparison(comp_dict: Dict[str, Any], indent: int = 0) -> list:
        lines = []
        for key, value in comp_dict.items():
            if isinstance(value, dict):
                if "change_percent" in value:
                    change = value.get("change_percent")
                    if change is not None:
                        status = "📈" if change > 0 else "📉"
                        lines.append("  " * indent + f"- {key}: {status} {change:.2f}% ({value['improvement']})")
                    else:
                        lines.append("  " * indent + f"- {key}: {value['current']} (prev: {value['previous']})")
                else:
                    lines.append("  " * indent + f"- {key}:")
                    lines.extend(format_comparison(value, indent + 1))
            else:
                lines.append("  " * indent + f"- {key}: {value}")
        return lines

    if "benchmarks" in comparison:
        report_lines.extend(format_comparison(comparison["benchmarks"]))

    return "\n".join(report_lines)

def run(current_results: Dict[str, Any], previous_file: str = "previous_benchmark.json") -> str:
    """Run comparison and generate report."""
    previous_results = load_previous_results(previous_file)

    if previous_results:
        comparison = compare_results(current_results, previous_results)
        report = generate_report(comparison)
    else:
        report = "# No previous results found. This is the baseline run.\n"
        report += json.dumps(current_results, indent=2)

    # Save current as previous for next run
    save_results(current_results, previous_file)

    return report

if __name__ == "__main__":
    # For testing, load from stdin or something, but for now, just print usage
    print("Usage: python comparison_report.py <current_results.json> [previous_file.json]")