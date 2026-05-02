"""
FastVideo Installation Module for Stan's ML Stack.

This module provides Python functions for installing FastVideo optimized for AMD GPUs.
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict, Any


class FastVideoInstaller:
    """FastVideo installation and management class."""

    def __init__(self):
        self.build_script = Path(__file__).parent.parent.parent / 'scripts' / 'build_fastvideo_rocm.sh'

    def is_installed(self) -> bool:
        """Check if FastVideo is installed."""
        try:
            import fastvideo
            return True
        except ImportError:
            return False

    def install(self, force: bool = False) -> bool:
        """
        Install FastVideo.

        Args:
            force: Force reinstallation

        Returns:
            True if successful, False otherwise
        """
        if self.is_installed() and not force:
            print("FastVideo is already installed. Use force=True to reinstall.")
            return True

        if not self.build_script.exists():
            print(f"Build script not found at {self.build_script}")
            return False

        try:
            print(f"Running FastVideo build script: {self.build_script}")
            result = subprocess.run([str(self.build_script)], check=True, capture_output=True, text=True)
            print("FastVideo build script stdout:")
            print(result.stdout)
            print("FastVideo build script stderr:")
            print(result.stderr)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"Error installing FastVideo: {e}")
            print("FastVideo build script stdout:")
            print(e.stdout)
            print("FastVideo build script stderr:")
            print(e.stderr)
            return False

    def verify_installation(self) -> Dict[str, Any]:
        """Verify FastVideo installation."""
        result = {
            'installed': self.is_installed(),
            'version': None
        }

        if result['installed']:
            try:
                import fastvideo
                result['version'] = getattr(fastvideo, '__version__', None)
            except (ImportError, AttributeError):
                pass

        return result


def install_fastvideo(force: bool = False) -> bool:
    """Convenience function to install FastVideo."""
    installer = FastVideoInstaller()
    return installer.install(force=force)


def verify_fastvideo() -> Dict[str, Any]:
    """Verify FastVideo installation."""
    installer = FastVideoInstaller()
    return installer.verify_installation()
