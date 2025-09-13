"""
Verification CLI module for Stan's ML Stack.

This module provides verification functionality for ML stack components.
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


def run_command(cmd: str, **kwargs) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)


def check_component(name: str, check_cmd: Optional[str], import_check: Optional[str] = None) -> Dict[str, Any]:
    """Check if a component is installed."""
    result = {
        'name': name,
        'installed': False,
        'version': None,
        'status': 'unknown'
    }

    # Check if command exists
    if check_cmd:
        try:
            proc = run_command(check_cmd)
            if proc.returncode == 0:
                result['installed'] = True
                result['status'] = 'installed'
                # Try to extract version from output
                if 'version' in proc.stdout.lower():
                    lines = proc.stdout.split('\n')
                    for line in lines:
                        if 'version' in line.lower():
                            result['version'] = line.split(':')[-1].strip()
                            break
        except Exception:
            pass

    # Check Python import
    if import_check and not result['installed']:
        try:
            __import__(import_check)
            result['installed'] = True
            result['status'] = 'installed'
        except ImportError:
            pass

    return result


def verify_rocm() -> Dict[str, Any]:
    """Verify ROCm installation."""
    return check_component('ROCm', 'rocminfo --version')


def verify_pytorch() -> Dict[str, Any]:
    """Verify PyTorch installation."""
    result = check_component('PyTorch', None, 'torch')
    if result['installed']:
        try:
            import torch
            result['version'] = torch.__version__
            result['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                result['gpu_count'] = torch.cuda.device_count()
                result['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(result['gpu_count'])]
        except Exception:
            pass
    return result


def verify_all() -> Dict[str, Dict[str, Any]]:
    """Verify all components."""
    components = {
        'rocm': verify_rocm(),
        'pytorch': verify_pytorch(),
    }

    return components


def print_status(component: Dict[str, Any]) -> None:
    """Print component status."""
    status = "✓" if component['installed'] else "✗"
    version = f" ({component['version']})" if component.get('version') else ""
    print(f"{status} {component['name']}{version}")


def main() -> int:
    """Main entry point for verification."""
    print("Stan\'s ML Stack Verification")
    print("=" * 40)

    components = verify_all()

    for comp in components.values():
        print_status(comp)

    # Summary
    installed = sum(1 for c in components.values() if c['installed'])
    total = len(components)
    print(f"\nInstalled: {installed}/{total} components")

    return 0 if installed == total else 1


if __name__ == "__main__":
    sys.exit(main())
