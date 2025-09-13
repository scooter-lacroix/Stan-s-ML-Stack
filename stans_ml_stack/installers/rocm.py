"""
ROCm Installation Module for Stan's ML Stack.

This module provides Python functions for installing and managing AMD ROCm platform.
"""

import os
import sys
import subprocess
import shutil
import re
from pathlib import Path
from typing import Optional, Dict, Any


class ROCmInstaller:
    """ROCm installation and management class."""

    def __init__(self):
        self.rocm_path = "/opt/rocm"
        self.rocm_version = None

    def detect_package_manager(self) -> str:
        """Detect the system's package manager."""
        package_managers = ['dnf', 'apt-get', 'yum', 'pacman', 'zypper']

        for pm in package_managers:
            if shutil.which(pm):
                if pm == 'apt-get':
                    return 'apt'
                return pm

        return 'unknown'

    def detect_gpu_architecture(self) -> str:
        """Detect GPU architecture using rocminfo."""
        try:
            result = subprocess.run(['rocminfo'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'gfx' in line.lower():
                        # Extract gfx version
                        import re
                        match = re.search(r'gfx(\d+)', line)
                        if match:
                            return f"gfx{match.group(1)}"
        except FileNotFoundError:
            pass

        # Default fallback
        return "gfx1100"

    def get_rocm_version(self) -> Optional[str]:
        """Get the installed ROCm version."""
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

    def is_installed(self) -> bool:
        """Check if ROCm is installed."""
        return shutil.which('rocminfo') is not None or Path(self.rocm_path).exists()

    def install(self, method: str = 'standard', force: bool = False) -> bool:
        """
        Install ROCm.

        Args:
            method: Installation method ('minimal', 'standard', 'full', 'custom')
            force: Force reinstallation

        Returns:
            True if successful, False otherwise
        """
        if self.is_installed() and not force:
            print("ROCm is already installed. Use force=True to reinstall.")
            return True

        package_manager = self.detect_package_manager()
        if package_manager == 'unknown':
            print("Unsupported package manager")
            return False

        # For now, delegate to the shell script
        script_path = Path(__file__).parent.parent.parent / 'scripts' / 'install_rocm.sh'
        if script_path.exists():
            cmd = [str(script_path)]
            if force:
                cmd.append('--force')

            try:
                result = subprocess.run(cmd, check=True)
                return result.returncode == 0
            except subprocess.CalledProcessError:
                return False

        print("Installation script not found")
        return False

    def get_environment_variables(self) -> Dict[str, str]:
        """Get ROCm environment variables."""
        gpu_arch = self.detect_gpu_architecture()

        env_vars = {
            'HSA_TOOLS_LIB': '0',
            'HSA_OVERRIDE_GFX_VERSION': gpu_arch.replace('gfx', ''),
            'PYTORCH_ROCM_ARCH': gpu_arch,
            'ROCM_PATH': self.rocm_path,
            'PATH': f"{self.rocm_path}/bin:$PATH",
            'LD_LIBRARY_PATH': f"{self.rocm_path}/lib:$LD_LIBRARY_PATH"
        }

        # Check for rocprofiler
        if Path(f"{self.rocm_path}/lib/librocprofiler-sdk-tool.so").exists():
            env_vars['HSA_TOOLS_LIB'] = f"{self.rocm_path}/lib/librocprofiler-sdk-tool.so"

        return env_vars

    def verify_installation(self) -> Dict[str, Any]:
        """Verify ROCm installation."""
        result = {
            'installed': self.is_installed(),
            'version': self.get_rocm_version(),
            'gpu_count': 0,
            'gpus': []
        }

        if result['installed']:
            try:
                rocminfo_result = subprocess.run(['rocminfo'], capture_output=True, text=True)
                if rocminfo_result.returncode == 0:
                    gpu_count = rocminfo_result.stdout.count('Device Type:.*GPU')
                    result['gpu_count'] = gpu_count

                    # Extract GPU names
                    lines = rocminfo_result.stdout.split('\n')
                    for i, line in enumerate(lines):
                        if 'Marketing Name' in line and i > 0:
                            gpu_name = line.split(':')[-1].strip()
                            result['gpus'].append(gpu_name)
            except FileNotFoundError:
                pass

        return result


def install_rocm(method: str = 'standard', force: bool = False) -> bool:
    """Convenience function to install ROCm."""
    installer = ROCmInstaller()
    return installer.install(method=method, force=force)


def get_rocm_env() -> Dict[str, str]:
    """Get ROCm environment variables."""
    installer = ROCmInstaller()
    return installer.get_environment_variables()


def verify_rocm() -> Dict[str, Any]:
    """Verify ROCm installation."""
    installer = ROCmInstaller()
    return installer.verify_installation()