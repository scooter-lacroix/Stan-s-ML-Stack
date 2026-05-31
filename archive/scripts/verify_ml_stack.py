#!/usr/bin/env python3
"""Verify ML stack installation via the Rusty CLI."""

from __future__ import annotations

import subprocess

from archive.scripts.rusty_cli_common import ensure_binary


def main() -> None:
    subprocess.run([ensure_binary("rusty"), "verify"], check=True)


if __name__ == "__main__":
    main()
