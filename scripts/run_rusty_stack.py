#!/usr/bin/env python3
"""Launch the Rusty-Stack TUI installer."""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    crate_dir = repo_root / "rusty-stack"
    binary = crate_dir / "target" / "release" / "Rusty-Stack"

    if not binary.exists():
        subprocess.run(["cargo", "build", "--release"], cwd=crate_dir, check=True)

    subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
