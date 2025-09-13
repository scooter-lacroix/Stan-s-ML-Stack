"""
Triton Installation Module for Stan's ML Stack.

This module provides Python functions for installing Triton with AMD optimizations.
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict, Any


class TritonInstaller:
    """Triton installation and management class."""

    def __init__(self):
        self.install_script = Path(__file__).parent.parent.parent / 'scripts' / 'install_triton.sh'

    def is_installed(self) -> bool:
        """Check if Triton is installed."""
        try:
            import triton
            return True
        except ImportError:
            return False

    def install(self, force: bool = False) -> bool:
        """
        Install Triton.

        Args:
            force: Force reinstallation

        Returns:
            True if successful, False otherwise
        """
        if self.is_installed() and not force:
            print("Triton is already installed. Use force=True to reinstall.")
            return True

        if not self.install_script.exists():
            print("Installation script not found")
            return False

        try:
            result = subprocess.run([str(self.install_script)], check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False

    def verify_installation(self) -> Dict[str, Any]:
        """Verify Triton installation."""
        result = {
            'installed': self.is_installed(),
            'version': None
        }

        if result['installed']:
            try:
                import triton
                result['version'] = getattr(triton, '__version__', None)
            except (ImportError, AttributeError):
                pass

        return result


def install_triton(force: bool = False) -> bool:
    """Convenience function to install Triton."""
    installer = TritonInstaller()
    return installer.install(force=force)


def verify_triton() -> Dict[str, Any]:
    """Verify Triton installation."""
    installer = TritonInstaller()
    return installer.verify_installation()