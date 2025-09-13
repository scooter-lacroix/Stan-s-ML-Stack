"""
VLLM Installation Module for Stan's ML Stack.

This module provides Python functions for installing VLLM with AMD optimizations.
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict, Any


class VLLMInstaller:
    """VLLM installation and management class."""

    def __init__(self):
        self.install_script = Path(__file__).parent.parent.parent / 'scripts' / 'install_vllm.sh'

    def is_installed(self) -> bool:
        """Check if VLLM is installed."""
        try:
            import vllm
            return True
        except ImportError:
            return False

    def install(self, force: bool = False) -> bool:
        """
        Install VLLM.

        Args:
            force: Force reinstallation

        Returns:
            True if successful, False otherwise
        """
        if self.is_installed() and not force:
            print("VLLM is already installed. Use force=True to reinstall.")
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
        """Verify VLLM installation."""
        result = {
            'installed': self.is_installed(),
            'version': None
        }

        if result['installed']:
            try:
                import vllm
                result['version'] = getattr(vllm, '__version__', None)
            except (ImportError, AttributeError):
                pass

        return result


def install_vllm(force: bool = False) -> bool:
    """Convenience function to install VLLM."""
    installer = VLLMInstaller()
    return installer.install(force=force)


def verify_vllm() -> Dict[str, Any]:
    """Verify VLLM installation."""
    installer = VLLMInstaller()
    return installer.verify_installation()