#!/usr/bin/env python3
"""Shared helpers for Python entry points that shell out to Rusty CLI binaries."""

from __future__ import annotations

import subprocess
from pathlib import Path


def ensure_binary(binary_name: str) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    crate_dir = repo_root / "rusty-stack"
    binary = crate_dir / "target" / "release" / binary_name

    if not binary.exists():
        subprocess.run(["cargo", "build", "--release"], cwd=crate_dir, check=True)

    return str(binary)
