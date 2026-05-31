#!/usr/bin/env python3
"""Apply safe update repairs via the Rusty CLI."""

from __future__ import annotations

import subprocess

from archive.scripts.rusty_cli_common import ensure_binary


def main() -> None:
    subprocess.run([ensure_binary("rusty"), "update", "--all-safe", "--yes"], check=True)


if __name__ == "__main__":
    main()
