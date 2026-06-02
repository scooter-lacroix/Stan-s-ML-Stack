"""PyPI entrypoints that install and invoke Rusty Stack Rust binaries."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib import metadata

CRATE_NAME = "rusty"
RUSTY_BIN = "rusty"
INSTALLER_BIN = "rusty-stack"


def _wrapper_version() -> str | None:
    for dist_name in ("Rusty-Stack", "rusty-stack"):
        try:
            return metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _resolve_bin(name: str) -> str | None:
    path = shutil.which(name)
    if path:
        return path

    cargo_home = os.environ.get("CARGO_HOME")
    if cargo_home:
        candidate = os.path.join(cargo_home, "bin", name)
    else:
        candidate = os.path.expanduser(os.path.join("~", ".cargo", "bin", name))
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def _resolve_cargo_bin(name: str) -> str | None:
    cargo_home = os.environ.get("CARGO_HOME")
    if cargo_home:
        candidate = os.path.join(cargo_home, "bin", name)
    else:
        candidate = os.path.expanduser(os.path.join("~", ".cargo", "bin", name))
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def _install_from_crates_io() -> None:
    cargo = shutil.which("cargo")
    if not cargo:
        raise RuntimeError(
            "cargo is required to install rusty from crates.io. "
            "Install Rust toolchain from https://rustup.rs/ and retry."
        )

    version = _wrapper_version()
    if version:
        cmd = [cargo, "install", "--locked", "--version", version, CRATE_NAME]
        result = subprocess.run(cmd)
        if result.returncode == 0:
            return
        raise RuntimeError(
            f"failed to install {CRATE_NAME}=={version} from crates.io. "
            "Exact-version install is required to keep wrapper and Rust binary versions in sync."
        )

    fallback_cmd = [cargo, "install", "--locked", CRATE_NAME]
    fallback = subprocess.run(fallback_cmd)
    if fallback.returncode != 0:
        raise RuntimeError("failed to install rusty from crates.io via cargo install")


def _bin_version_matches(path: str) -> bool:
    expected = _wrapper_version()
    if not expected:
        return True
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False

    if result.returncode != 0:
        return False
    output = f"{result.stdout} {result.stderr}"
    return expected in output


def _ensure_rusty() -> str:
    rusty = _resolve_bin(RUSTY_BIN)
    if rusty and _bin_version_matches(rusty):
        return rusty

    _install_from_crates_io()

    rusty = _resolve_cargo_bin(RUSTY_BIN) or _resolve_bin(RUSTY_BIN)
    if not rusty or not _bin_version_matches(rusty):
        raise RuntimeError("rusty binary not found after installation")
    return rusty


def _ensure_installer() -> str:
    installer = _resolve_bin(INSTALLER_BIN)
    if installer and _bin_version_matches(installer):
        return installer

    _install_from_crates_io()

    installer = _resolve_cargo_bin(INSTALLER_BIN) or _resolve_bin(INSTALLER_BIN)
    if not installer or not _bin_version_matches(installer):
        raise RuntimeError("rusty-stack binary not found after installation")
    return installer


def _exec(cmd: list[str]) -> int:
    return subprocess.run(cmd).returncode


def install_main() -> None:
    sys.exit(_exec([_ensure_installer()]))


def verify_main() -> None:
    sys.exit(_exec([_ensure_rusty(), "verify"]))


def repair_main() -> None:
    sys.exit(_exec([_ensure_rusty(), "update", "--all-safe", "--yes"]))
