"""
Flash Attention Installation Module for Stan's ML Stack.

This module provides Python functions for installing Flash Attention optimized for AMD GPUs.
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict, Any


class FlashAttentionInstaller:
    """Flash Attention installation and management class."""

    def __init__(self):
        self.build_script = Path(__file__).parent.parent.parent / 'scripts' / 'build_flash_attn_amd.sh'

    def is_installed(self) -> bool:
        """Check if Flash Attention is installed."""
        try:
            import flash_attn
            return True
        except ImportError:
            return False

    def install(self, force: bool = False) -> bool:
        """
        Install Flash Attention.

        Args:
            force: Force reinstallation

        Returns:
            True if successful, False otherwise
        """
        if self.is_installed() and not force:
            print("Flash Attention is already installed. Use force=True to reinstall.")
            return True

        if not self.build_script.exists():
            print("Build script not found")
            return False

        try:
            result = subprocess.run([str(self.build_script)], check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False

    def verify_installation(self) -> Dict[str, Any]:
        """Verify Flash Attention installation."""
        result = {
            'installed': self.is_installed(),
            'version': None
        }

        if result['installed']:
            try:
                import flash_attn
                result['version'] = getattr(flash_attn, '__version__', None)
            except (ImportError, AttributeError):
                pass

        return result


def install_flash_attention(force: bool = False) -> bool:
    """Convenience function to install Flash Attention."""
    installer = FlashAttentionInstaller()
    return installer.install(force=force)


def verify_flash_attention() -> Dict[str, Any]:
    """Verify Flash Attention installation."""
    installer = FlashAttentionInstaller()
    return installer.verify_installation()