"""
Repair CLI module for Stan's ML Stack.

This module provides repair functionality for ML stack components.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd: str, **kwargs) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)


def repair_rocm() -> bool:
    """Repair ROCm installation."""
    print("Repairing ROCm installation...")
    # For now, just run the repair script
    script_path = Path(__file__).parent.parent.parent / 'scripts' / 'repair_ml_stack.sh'
    if script_path.exists():
        try:
            result = run_command(str(script_path))
            return result.returncode == 0
        except Exception:
            pass
    return False


def repair_all() -> bool:
    """Repair all components."""
    print("Repairing ML stack...")
    return repair_rocm()


def main() -> int:
    """Main entry point for repair."""
    print("Stan\'s ML Stack Repair")
    print("=" * 30)

    success = repair_all()

    if success:
        print("✓ Repair completed successfully")
        return 0
    else:
        print("✗ Repair failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
