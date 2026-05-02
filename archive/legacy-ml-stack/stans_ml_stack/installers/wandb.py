"""
Weights & Biases Installation Module for Stan's ML Stack.

This module provides Python functions for installing Weights & Biases (wandb).
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict, Any


class WandbInstaller:
    """Weights & Biases installation and management class."""

    def __init__(self):
        self.install_script = Path(__file__).parent.parent.parent / 'scripts' / 'install_wandb.sh'

    def is_installed(self) -> bool:
        """Check if wandb is installed."""
        try:
            import wandb
            return True
        except ImportError:
            return False

    def install(self, force: bool = False) -> bool:
        """
        Install Weights & Biases.

        Args:
            force: Force reinstallation

        Returns:
            True if successful, False otherwise
        """
        if self.is_installed() and not force:
            print("Weights & Biases is already installed. Use force=True to reinstall.")
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
        """Verify Weights & Biases installation."""
        result = {
            'installed': self.is_installed(),
            'version': None
        }

        if result['installed']:
            try:
                import wandb
                result['version'] = getattr(wandb, '__version__', None)
            except (ImportError, AttributeError):
                pass

        return result


def install_wandb(force: bool = False) -> bool:
    """Convenience function to install Weights & Biases."""
    installer = WandbInstaller()
    return installer.install(force=force)


def verify_wandb() -> Dict[str, Any]:
    """Verify Weights & Biases installation."""
    installer = WandbInstaller()
    return installer.verify_installation()