"""
PyTorch with ROCm Installation Module for Stan's ML Stack.

This module provides Python functions for installing PyTorch with ROCm support.
"""

import os
import sys
import subprocess
import shutil
import re
from pathlib import Path
from typing import Optional, Dict, Any


class PyTorchROCmInstaller:
    """PyTorch with ROCm installation and management class."""

    def __init__(self):
        self.rocm_path = "/opt/rocm"

    def detect_rocm_version(self) -> Optional[str]:
        """Detect ROCm version."""
        try:
            result = subprocess.run(['rocminfo'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'rocm version' in line.lower():
                        parts = line.split(':')
                        if len(parts) > 1:
                            return parts[1].strip()
        except FileNotFoundError:
            pass

        # Check directory listing
        rocm_dirs = list(Path('/opt').glob('rocm-*'))
        if rocm_dirs:
            version_match = re.search(r'(\d+\.\d+\.\d+)', str(rocm_dirs[0]))
            if version_match:
                return version_match.group(1)

        return None

    def is_pytorch_installed(self) -> bool:
        """Check if PyTorch is installed."""
        try:
            import torch
            return True
        except ImportError:
            return False

    def has_rocm_support(self) -> bool:
        """Check if PyTorch has ROCm support."""
        try:
            import torch
            return hasattr(torch.version, 'hip') and torch.version.hip is not None
        except (ImportError, AttributeError):
            return False

    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except (ImportError, AttributeError):
            return False

    def get_pytorch_version(self) -> Optional[str]:
        """Get PyTorch version."""
        try:
            import torch
            return torch.__version__
        except ImportError:
            return None

    def get_rocm_pytorch_version(self) -> Optional[str]:
        """Get PyTorch ROCm version."""
        try:
            import torch
            if hasattr(torch.version, 'hip'):
                return torch.version.hip
            return None
        except (ImportError, AttributeError):
            return None

    def install(self, method: str = 'auto', force: bool = False) -> bool:
        """
        Install PyTorch with ROCm support.

        Args:
            method: Installation method ('global', 'venv', 'auto')
            force: Force reinstallation

        Returns:
            True if successful, False otherwise
        """
        if self.is_pytorch_installed() and not force:
            print("PyTorch is already installed. Use force=True to reinstall.")
            return True

        rocm_version = self.detect_rocm_version()
        if not rocm_version:
            print("ROCm not detected. Please install ROCm first.")
            return False

        # For now, delegate to the shell script
        script_path = Path(__file__).parent.parent.parent / 'scripts' / 'install_pytorch_rocm.sh'
        if script_path.exists():
            cmd = [str(script_path)]
            if force:
                cmd.append('--force')
            if method != 'auto':
                cmd.extend(['--method', method])

            try:
                result = subprocess.run(cmd, check=True)
                return result.returncode == 0
            except subprocess.CalledProcessError:
                return False

        print("Installation script not found")
        return False

    def get_environment_variables(self) -> Dict[str, str]:
        """Get PyTorch ROCm environment variables."""
        env_vars = {
            'HSA_TOOLS_LIB': '0',
            'HSA_OVERRIDE_GFX_VERSION': '11.0.0',
            'PYTORCH_ROCM_ARCH': 'gfx1100',
            'ROCM_PATH': self.rocm_path,
            'PATH': f"{self.rocm_path}/bin:$PATH",
            'LD_LIBRARY_PATH': f"{self.rocm_path}/lib:$LD_LIBRARY_PATH"
        }

        # Check for rocprofiler
        if Path(f"{self.rocm_path}/lib/librocprofiler-sdk-tool.so").exists():
            env_vars['HSA_TOOLS_LIB'] = f"{self.rocm_path}/lib/librocprofiler-sdk-tool.so"

        return env_vars

    def verify_installation(self) -> Dict[str, Any]:
        """Verify PyTorch with ROCm installation."""
        result = {
            'installed': self.is_pytorch_installed(),
            'version': self.get_pytorch_version(),
            'rocm_support': self.has_rocm_support(),
            'rocm_version': self.get_rocm_pytorch_version(),
            'gpu_available': self.is_gpu_available(),
            'gpu_count': 0,
            'gpus': []
        }

        if result['installed'] and result['gpu_available']:
            try:
                import torch
                result['gpu_count'] = torch.cuda.device_count()
                result['gpus'] = [torch.cuda.get_device_name(i) for i in range(result['gpu_count'])]
            except Exception:
                pass

        return result

    def test_gpu_operations(self) -> bool:
        """Test basic GPU operations."""
        try:
            import torch
            if not torch.cuda.is_available():
                return False

            # Test tensor creation and operations
            x = torch.ones(10, device='cuda')
            y = x + 1
            return bool(torch.all(y == 2).item())
        except Exception:
            return False


def install_pytorch_rocm(method: str = 'auto', force: bool = False) -> bool:
    """Convenience function to install PyTorch with ROCm."""
    installer = PyTorchROCmInstaller()
    return installer.install(method=method, force=force)


def get_pytorch_rocm_env() -> Dict[str, str]:
    """Get PyTorch ROCm environment variables."""
    installer = PyTorchROCmInstaller()
    return installer.get_environment_variables()


def verify_pytorch_rocm() -> Dict[str, Any]:
    """Verify PyTorch with ROCm installation."""
    installer = PyTorchROCmInstaller()
    return installer.verify_installation()


def test_pytorch_gpu() -> bool:
    """Test PyTorch GPU operations."""
    installer = PyTorchROCmInstaller()
    return installer.test_gpu_operations()