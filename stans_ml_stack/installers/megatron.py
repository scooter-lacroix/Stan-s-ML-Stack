"""
Megatron Installation Module for Stan's ML Stack.

This module provides Python functions for installing Megatron-LM with AMD optimizations.
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict, Any


class MegatronInstaller:
    """Megatron installation and management class."""

    def __init__(self):
        self.install_script = Path(__file__).parent.parent.parent / 'scripts' / 'install_megatron.sh'

    def is_installed(self) -> bool:
        """Check if Megatron is installed."""
        try:
            import megatron
            return True
        except ImportError:
            # Check for common megatron installations
            import sys
            for path in sys.path:
                if 'megatron' in str(path).lower():
                    return True
            return False

    def install(self, force: bool = False) -> bool:
        """
        Install Megatron.

        Args:
            force: Force reinstallation

        Returns:
            True if successful, False otherwise
        """
        if self.is_installed() and not force:
            print("Megatron is already installed. Use force=True to reinstall.")
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
        """Verify Megatron installation."""
        result = {
            'installed': self.is_installed(),
            'version': None
        }

        return result


def install_megatron(force: bool = False) -> bool:
    """Convenience function to install Megatron."""
    installer = MegatronInstaller()
    return installer.install(force=force)


def verify_megatron() -> Dict[str, Any]:
    """Verify Megatron installation."""
    installer = MegatronInstaller()
    return installer.verify_installation()