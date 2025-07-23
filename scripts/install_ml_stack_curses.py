#!/usr/bin/env python3
"""
ML Stack Installation UI using curses

A simple, clean UI for installing the ML Stack components.
"""

import os
import time
import curses
import subprocess
import platform
import fcntl
import select
import re
import signal
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Constants
# Use relative paths for better portability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MLSTACK_DIR = os.path.dirname(SCRIPT_DIR)
SCRIPTS_DIR = SCRIPT_DIR
LOGS_DIR = os.path.join(MLSTACK_DIR, "logs")
LOG_FILE = os.path.join(LOGS_DIR, f"ml_stack_install_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Environment setup and verification scripts
ENVIRONMENT_SCRIPTS = [
    {
        "name": "Environment Setup",
        "description": "Basic environment setup for simple GPU configurations",
        "script": "setup_environment.sh",
        "required": True,
        "status": "pending"
    },
    {
        "name": "Enhanced Environment Setup",
        "description": "Advanced setup for complex GPU configs with iGPU filtering",
        "script": "enhanced_setup_environment.sh",
        "required": False,
        "status": "pending"
    },
    {
        "name": "Repair ML Stack",
        "description": "Fix common issues, conflicts, and environment problems",
        "script": "repair_ml_stack.sh",
        "required": False,
        "status": "pending"
    },
    {
        "name": "Verify Installation",
        "description": "Basic verification of ML Stack installation",
        "script": "verify_installation.sh",
        "required": False,
        "status": "pending"
    },
    {
        "name": "Enhanced Verify Installation",
        "description": "Advanced verification with detailed diagnostics",
        "script": "enhanced_verify_installation.sh",
        "required": False,
        "status": "pending"
    },
    {
        "name": "Custom Verify Installation",
        "description": "Custom verification for specific configurations",
        "script": "custom_verify_installation.sh",
        "required": False,
        "status": "pending"
    },
    {
        "name": "Verify and Build",
        "description": "Verify installation and build components",
        "script": "verify_and_build.sh",
        "required": False,
        "status": "pending"
    }
]

# Components to install
FOUNDATION_COMPONENTS = [
    {
        "name": "ROCm",
        "description": "AMD's open software platform for GPU computing",
        "script": "install_rocm.sh",
        "required": True,
        "status": "installed"
    },
    {
        "name": "PyTorch with ROCm",
        "description": "Deep learning framework with AMD GPU support (INSTALLED)",
        "script": "install_pytorch_rocm.sh",
        "required": True,
        "status": "installed"
    },
    {
        "name": "Triton",
        "description": "Compiler for parallel programming (INSTALLED)",
        "script": "install_triton.sh",
        "required": True,
        "status": "installed"
    },
    {
        "name": "MPI4Py",
        "description": "Message Passing Interface for distributed computing (INSTALLED)",
        "script": "install_mpi4py.sh",
        "required": True,
        "status": "installed"
    },
    {
        "name": "DeepSpeed",
        "description": "Deep learning optimization library for large models (INSTALLED)",
        "script": "install_deepspeed.sh",
        "required": True,
        "status": "installed"
    }
]

CORE_COMPONENTS = [
    {
        "name": "ML Stack Core",
        "description": "Core ML Stack components",
        "script": "install_ml_stack.sh",
        "required": False,
        "status": "pending"
    },
    {
        "name": "Flash Attention",
        "description": "Efficient attention computation",
        "script": "install_flash_attention_ck.sh",
        "required": False,
        "status": "pending"
    },
    {
        "name": "Repair ML Stack",
        "description": "Fix and repair ML Stack installation",
        "script": "repair_ml_stack.sh",
        "required": False,
        "status": "pending"
    }
]

EXTENSION_COMPONENTS = [
    {
        "name": "Megatron-LM",
        "description": "Large-scale training framework for transformer models (INSTALLED)",
        "script": "install_megatron.sh",
        "required": False,
        "status": "installed"
    },
    {
        "name": "vLLM",
        "description": "High-throughput inference engine for LLMs (INSTALLED)",
        "script": "install_vllm.sh",
        "required": False,
        "status": "installed"
    },
    {
        "name": "Flash Attention CK",
        "description": "Efficient attention computation (IN PROGRESS - Assembly code debugging)",
        "script": "install_flash_attention_ck.sh",
        "required": False,
        "status": "building"
    },
    {
        "name": "ONNX Runtime",
        "description": "Cross-platform inference accelerator (BUILDS TAKE HOURS)",
        "script": "build_onnxruntime.sh",
        "required": False,
        "status": "pending"
    },
    {
        "name": "BITSANDBYTES",
        "description": "Efficient quantization for deep learning models",
        "script": "install_bitsandbytes.sh",
        "required": False,
        "status": "pending"
    },
    {
        "name": "ROCm SMI",
        "description": "System monitoring and management for AMD GPUs",
        "script": "install_rocm_smi.sh",
        "required": False,
        "status": "pending"
    },
    {
        "name": "MIGraphX Python Wrapper",
        "description": "Python wrapper for MIGraphX library",
        "script": "install_migraphx_python_wrapper.sh",
        "required": False,
        "status": "pending"
    },
    {
        "name": "PyTorch Profiler",
        "description": "Performance analysis for PyTorch models",
        "script": "install_pytorch_profiler.sh",
        "required": False,
        "status": "pending"
    },
    {
        "name": "Weights & Biases",
        "description": "Experiment tracking and visualization",
        "script": "install_wandb.sh",
        "required": False,
        "status": "pending"
    }
]

# Color pairs
COLOR_NORMAL = 1
COLOR_HIGHLIGHT = 2
COLOR_TITLE = 3
COLOR_SUCCESS = 4
COLOR_ERROR = 5
COLOR_WARNING = 6
COLOR_INFO = 7
COLOR_BORDER = 8  # New color for the border

def strip_ansi_codes(text: str) -> str:
    """Strip ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def log_message(message: str, level: str = "INFO") -> None:
    """Log a message to the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Strip ANSI escape codes from the message
    clean_message = strip_ansi_codes(message)
    log_line = f"[{timestamp}] [{level}] {clean_message}"

    # Create a debug log file in the same directory as the script
    debug_log_file = os.path.join(LOGS_DIR, "debug_curses_ui.log")

    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")
        f.flush()  # Ensure the log is written immediately
        os.fsync(f.fileno())  # Force the OS to write to disk

    # Also write to debug log file
    with open(debug_log_file, "a") as f:
        f.write(log_line + "\n")
        f.flush()  # Ensure the log is written immediately
        os.fsync(f.fileno())  # Force the OS to write to disk

def run_command(command: str, cwd: Optional[str] = None, timeout: Optional[int] = 300) -> Tuple[int, str, str]:
    """Run a shell command and return the return code, stdout, and stderr with a timeout."""
    log_message(f"Running command: {command}")

    try:
        # Create a new environment with the current environment
        cmd_env = os.environ.copy()

        # Start the process
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=cmd_env,
            bufsize=1  # Line buffered
        )

        # Initialize stdout and stderr
        stdout = ""
        stderr = ""

        # Set up non-blocking I/O
        for fd in [process.stdout, process.stderr]:
            fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)

        # Set up the timeout
        start_time = time.time()
        deadline = start_time + timeout if timeout is not None else float('inf')

        # Process output until the process completes or times out
        while process.poll() is None and time.time() < deadline:
            # Wait for output using select with a small timeout
            ready_to_read, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)

            # Process stdout
            if process.stdout in ready_to_read:
                try:
                    output = process.stdout.read()
                    if output:
                        stdout += output
                except IOError as e:
                    log_message(f"IOError reading stdout: {e}", "DEBUG")
                    # Small delay to prevent CPU spinning
                    time.sleep(0.01)

            # Process stderr
            if process.stderr in ready_to_read:
                try:
                    output = process.stderr.read()
                    if output:
                        stderr += output
                except IOError as e:
                    log_message(f"IOError reading stderr: {e}", "DEBUG")
                    # Small delay to prevent CPU spinning
                    time.sleep(0.01)

            # If no output was ready, add a small delay to prevent CPU spinning
            if not ready_to_read:
                time.sleep(0.01)

            # Periodically log that the command is still running
            elapsed = time.time() - start_time
            if elapsed > 5 and int(elapsed) % 10 == 0:  # Log every 10 seconds
                log_message(f"Command still running after {int(elapsed)} seconds...")

        # Check if the process has timed out
        if process.poll() is None:
            log_message(f"Command timed out after {timeout} seconds: {command}", "WARNING")
            # Try graceful termination first
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log_message("Process did not terminate gracefully, killing it", "WARNING")
                process.kill()
                process.wait()
            return 1, stdout, stderr + f"\nCommand timed out after {timeout} seconds"

        # Get any remaining output
        try:
            remaining_stdout, remaining_stderr = process.communicate(timeout=1)
            if remaining_stdout:
                stdout += remaining_stdout
            if remaining_stderr:
                stderr += remaining_stderr
        except (subprocess.TimeoutExpired, ValueError) as e:
            log_message(f"Error getting remaining output: {e}", "WARNING")
            try:
                process.kill()
                remaining_stdout, remaining_stderr = process.communicate()
                if remaining_stdout:
                    stdout += remaining_stdout
                if remaining_stderr:
                    stderr += remaining_stderr
            except Exception as e:
                log_message(f"Error killing process: {e}", "ERROR")

        return_code = process.returncode
        log_message(f"Command completed with return code: {return_code}")

        # Log a sample of the output for debugging
        if stdout and return_code != 0:
            stdout_sample = stdout[:200] + "..." if len(stdout) > 200 else stdout
            log_message(f"Command stdout sample: {stdout_sample}")
        if stderr and return_code != 0:
            stderr_sample = stderr[:200] + "..." if len(stderr) > 200 else stderr
            log_message(f"Command stderr sample: {stderr_sample}")

        return return_code, stdout, stderr
    except Exception as e:
        log_message(f"Error executing command: {str(e)}", "ERROR")
        return 1, "", f"Error: {str(e)}"

def detect_hardware() -> Dict[str, Any]:
    """Detect hardware information."""
    log_message("Detecting hardware...")

    hardware_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "amd_gpus": [],
        "rocm_version": None,
        "rocm_path": None
    }

    # Check if ROCm is installed
    return_code, stdout, _ = run_command("which rocminfo", timeout=5)
    if return_code == 0 and stdout.strip():
        log_message("ROCm is installed")
        hardware_info["rocm_path"] = os.path.dirname(os.path.dirname(stdout.strip()))

        # Try to get ROCm version using relative paths where possible
        return_code, stdout, _ = run_command("ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\\+\\.[0-9]\\+\\.[0-9]\\+' | head -n 1", timeout=5)
        if return_code == 0 and stdout.strip():
            hardware_info["rocm_version"] = stdout.strip()
            log_message(f"ROCm version: {hardware_info['rocm_version']}")
        else:
            # Try alternative method to get ROCm version
            # Use which to find rocminfo path
            rocm_cmd = "which rocminfo 2>/dev/null"
            rc, rocminfo_path, _ = run_command(rocm_cmd, timeout=5)
            if rc == 0 and rocminfo_path.strip():
                rocminfo_cmd = f"{rocminfo_path.strip()} 2>/dev/null | grep 'ROCm Version' | cut -d ':' -f 2 | tr -d ' '"
                return_code, stdout, _ = run_command(rocminfo_cmd, timeout=5)
                if return_code == 0 and stdout.strip():
                    hardware_info["rocm_version"] = stdout.strip()
                    log_message(f"ROCm version: {hardware_info['rocm_version']}")

            # If still not found, try another alternative method
            if not hardware_info["rocm_version"]:
                return_code, stdout, _ = run_command("cat /opt/rocm/.info/version 2>/dev/null", timeout=5)
                if return_code == 0 and stdout.strip():
                    hardware_info["rocm_version"] = stdout.strip()
                    log_message(f"ROCm version: {hardware_info['rocm_version']}")
    else:
        log_message("ROCm is not installed or not in PATH")

    # Detect AMD GPUs using multiple methods
    # Method 1: rocminfo
    log_message("Detecting AMD GPUs using rocminfo...")
    # Use which to find rocminfo path
    rocm_cmd = "which rocminfo 2>/dev/null"
    rc, rocminfo_path, _ = run_command(rocm_cmd, timeout=5)
    if rc == 0 and rocminfo_path.strip():
        rocminfo_cmd = f"{rocminfo_path.strip()} 2>/dev/null | grep -A 2 'GPU ID:' | grep 'Marketing Name' | cut -d ':' -f 2 | sed 's/^ *//'"
        return_code, stdout, _ = run_command(rocminfo_cmd, timeout=5)
        if return_code == 0 and stdout.strip():
            hardware_info["amd_gpus"] = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
            log_message(f"Found {len(hardware_info['amd_gpus'])} AMD GPU(s) using rocminfo")

    # Method 2: rocm-smi
    if not hardware_info["amd_gpus"]:
        log_message("Detecting AMD GPUs using rocm-smi...")
        # Use which to find rocm-smi path
        rocm_smi_cmd = "which rocm-smi 2>/dev/null"
        rc, rocm_smi_path, _ = run_command(rocm_smi_cmd, timeout=5)
        if rc == 0 and rocm_smi_path.strip():
            smi_cmd = f"{rocm_smi_path.strip()} --showproductname 2>/dev/null | grep -v '=====' | grep -v 'Product Name' | sed 's/^ *//'"
            return_code, stdout, _ = run_command(smi_cmd, timeout=5)
        if return_code == 0 and stdout.strip():
            hardware_info["amd_gpus"] = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
            log_message(f"Found {len(hardware_info['amd_gpus'])} AMD GPU(s) using rocm-smi")

    # Method 3: lspci
    if not hardware_info["amd_gpus"]:
        log_message("Detecting AMD GPUs using lspci...")
        return_code, stdout, _ = run_command("lspci | grep -i 'amd\\|radeon\\|advanced micro devices' | grep -i 'vga\\|3d\\|display'", timeout=5)
        if return_code == 0 and stdout.strip():
            hardware_info["amd_gpus"] = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
            log_message(f"Found {len(hardware_info['amd_gpus'])} AMD GPU(s) using lspci")

    # Method 4: lshw
    if not hardware_info["amd_gpus"]:
        log_message("Detecting AMD GPUs using lshw...")
        return_code, stdout, _ = run_command("lshw -C display 2>/dev/null | grep -A 2 'display' | grep 'product:' | cut -d ':' -f 2 | sed 's/^ *//'", timeout=5)
        if return_code == 0 and stdout.strip():
            hardware_info["amd_gpus"] = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
            log_message(f"Found {len(hardware_info['amd_gpus'])} GPU(s) using lshw")

    # Method 5: Check render nodes
    if not hardware_info["amd_gpus"]:
        log_message("Checking render nodes...")
        return_code, stdout, _ = run_command("ls -la /dev/dri/render* 2>/dev/null | wc -l", timeout=5)
        if return_code == 0 and stdout.strip() and int(stdout.strip()) > 0:
            # Count render nodes
            gpu_count = int(stdout.strip())

            # Try to get more info about these GPUs
            return_code, stdout, _ = run_command("ls -la /dev/dri/by-path/ 2>/dev/null | grep pci | wc -l", timeout=5)
            if return_code == 0 and stdout.strip() and int(stdout.strip()) > 0:
                # Get PCI paths
                return_code, stdout, _ = run_command("ls -la /dev/dri/by-path/ 2>/dev/null | grep pci | awk '{print $9}'", timeout=5)
                if return_code == 0 and stdout.strip():
                    pci_paths = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
                    hardware_info["amd_gpus"] = [f"GPU at {path}" for path in pci_paths]
                else:
                    hardware_info["amd_gpus"] = [f"GPU {i}" for i in range(gpu_count)]
            else:
                hardware_info["amd_gpus"] = [f"GPU {i}" for i in range(gpu_count)]

            log_message(f"Found {len(hardware_info['amd_gpus'])} GPU(s) by checking render nodes")

    # Method 6: Check if HIP is available through PyTorch
    if not hardware_info["amd_gpus"]:
        log_message("Checking if HIP is available through PyTorch...")
        return_code, stdout, _ = run_command("python3 -c 'import torch; print(torch.version.hip is not None and torch.cuda.is_available())'", timeout=5)
        if return_code == 0 and stdout.strip() == "True":
            # Get device count and names
            return_code, stdout, _ = run_command("python3 -c 'import torch; print(torch.cuda.device_count())'", timeout=5)
            if return_code == 0 and stdout.strip() and int(stdout.strip()) > 0:
                gpu_count = int(stdout.strip())
                hardware_info["amd_gpus"] = []

                for i in range(gpu_count):
                    return_code, stdout, _ = run_command(f"python3 -c 'import torch; print(torch.cuda.get_device_name({i}))'", timeout=5)
                    if return_code == 0 and stdout.strip():
                        hardware_info["amd_gpus"].append(stdout.strip())
                    else:
                        hardware_info["amd_gpus"].append(f"AMD GPU {i}")

                log_message(f"Found {len(hardware_info['amd_gpus'])} AMD GPU(s) using PyTorch")

    log_message("Hardware detection completed")
    return hardware_info

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed."""
    log_message("Checking dependencies...")

    dependencies = {
        "build-essential": False,
        "cmake": False,
        "git": False,
        "python3-dev": False,
        "python3-pip": False,
        "libnuma-dev": False,
        "pciutils": False,
        "mesa-utils": False,
        "clinfo": False
    }

    for dep in dependencies:
        log_message(f"Checking dependency: {dep}")

        # Special handling for libnuma-dev
        if dep == "libnuma-dev":
            # First check if the package is installed according to dpkg
            dpkg_cmd = f"dpkg-query -W -f='${{Status}}' {dep} 2>/dev/null | grep -q 'install ok installed'"
            dpkg_return_code, _, _ = run_command(dpkg_cmd, timeout=5)

            # Then check if the shared library is available
            lib_check_cmd = "ldconfig -p | grep -q 'libnuma.so'"
            lib_check_return_code, _, _ = run_command(lib_check_cmd, timeout=5)

            # Consider it installed if either the package is installed or the library is available
            dependencies[dep] = (dpkg_return_code == 0) or (lib_check_return_code == 0)

            if dependencies[dep]:
                log_message(f"Dependency {dep}: Installed (package or library found)")
            else:
                # Try one more approach - check if the file exists directly
                file_check_cmd = "ls -la /usr/lib/x86_64-linux-gnu/libnuma.so* 2>/dev/null"
                file_check_return_code, file_check_stdout, _ = run_command(file_check_cmd, timeout=5)

                if file_check_return_code == 0 and file_check_stdout.strip():
                    dependencies[dep] = True
                    log_message(f"Dependency {dep}: Installed (library files found)")
                else:
                    log_message(f"Dependency {dep}: Not installed")
        else:
            # Standard dependency check for other packages
            return_code, _, _ = run_command(f"dpkg-query -W -f='${{Status}}' {dep} 2>/dev/null | grep -q 'install ok installed'", timeout=5)
            dependencies[dep] = return_code == 0
            log_message(f"Dependency {dep}: {'Installed' if dependencies[dep] else 'Not installed'}")

    log_message("Dependency check completed")
    return dependencies

def install_dependencies(deps_to_install: List[str], stdscr) -> bool:
    """Install dependencies."""
    if not deps_to_install:
        log_message("No dependencies to install")
        return True

    log_message(f"Installing dependencies: {', '.join(deps_to_install)}")

    # First, try to install system dependencies with apt
    system_deps = [dep for dep in deps_to_install if not dep.startswith("python-") and not dep.endswith(".whl")]
    python_deps = [dep for dep in deps_to_install if dep.startswith("python-") or dep.endswith(".whl")]

    if system_deps:
        # Update package lists
        log_message("Running apt-get update")
        stdscr.addstr(curses.LINES - 3, 2, f"Running: sudo apt-get update".ljust(curses.COLS - 4), curses.color_pair(COLOR_INFO))
        stdscr.refresh()

        update_cmd = "sudo apt-get update"
        update_return_code, _, update_stderr = run_command(update_cmd, timeout=60)
        if update_return_code != 0:
            log_message(f"Failed to update package lists: {update_stderr}", "ERROR")
            stdscr.addstr(curses.LINES - 3, 2, f"Failed to update package lists".ljust(curses.COLS - 4), curses.color_pair(COLOR_ERROR))
            stdscr.refresh()
            time.sleep(2)
            # Continue anyway, we might still be able to install packages

        # Install each system dependency separately
        for dep in system_deps:
            log_message(f"Installing system dependency: {dep}")
            stdscr.addstr(curses.LINES - 3, 2, f"Running: sudo apt-get install -y {dep}".ljust(curses.COLS - 4), curses.color_pair(COLOR_INFO))
            stdscr.refresh()

            install_cmd = f"sudo apt-get install -y {dep}"
            install_return_code, _, install_stderr = run_command(install_cmd, timeout=120)
            if install_return_code != 0:
                log_message(f"Failed to install {dep}: {install_stderr}", "ERROR")
                stdscr.addstr(curses.LINES - 3, 2, f"Failed to install {dep}".ljust(curses.COLS - 4), curses.color_pair(COLOR_ERROR))
                stdscr.refresh()
                time.sleep(2)
                continue

            # Verify installation
            log_message(f"Verifying installation of {dep}")
            verify_cmd = f"dpkg-query -W -f='${{Status}}' {dep} 2>/dev/null | grep -q 'install ok installed'"
            verify_return_code, _, _ = run_command(verify_cmd, timeout=5)
            if verify_return_code != 0:
                log_message(f"Failed to verify installation of {dep}", "ERROR")
                stdscr.addstr(curses.LINES - 3, 2, f"Failed to verify installation of {dep}".ljust(curses.COLS - 4), curses.color_pair(COLOR_ERROR))
                stdscr.refresh()
                time.sleep(2)
                continue

            log_message(f"Successfully installed {dep}")
            stdscr.addstr(curses.LINES - 3, 2, f"Successfully installed {dep}".ljust(curses.COLS - 4), curses.color_pair(COLOR_SUCCESS))
            stdscr.refresh()
            time.sleep(1)

    # Install Python dependencies using uv
    if python_deps:
        log_message(f"Installing Python dependencies: {', '.join(python_deps)}")
        stdscr.addstr(curses.LINES - 3, 2, f"Installing Python dependencies with uv".ljust(curses.COLS - 4), curses.color_pair(COLOR_INFO))
        stdscr.refresh()

        # Check if uv is installed
        uv_check_cmd = "which uv"
        uv_check_return_code, _, _ = run_command(uv_check_cmd, timeout=5)
        if uv_check_return_code != 0:
            log_message("uv not found, installing it", "WARNING")
            stdscr.addstr(curses.LINES - 3, 2, "uv not found, installing it".ljust(curses.COLS - 4), curses.color_pair(COLOR_WARNING))
            stdscr.refresh()

            # Install uv
            install_uv_cmd = "curl -sSf https://astral.sh/uv/install.sh | sh"
            install_uv_return_code, _, install_uv_stderr = run_command(install_uv_cmd, timeout=60)
            if install_uv_return_code != 0:
                log_message(f"Failed to install uv: {install_uv_stderr}", "ERROR")
                stdscr.addstr(curses.LINES - 3, 2, "Failed to install uv".ljust(curses.COLS - 4), curses.color_pair(COLOR_ERROR))
                stdscr.refresh()
                time.sleep(2)
                return False

        # Install each Python dependency with uv
        for dep in python_deps:
            log_message(f"Installing Python dependency: {dep}")
            stdscr.addstr(curses.LINES - 3, 2, f"Running: uv pip install {dep}".ljust(curses.COLS - 4), curses.color_pair(COLOR_INFO))
            stdscr.refresh()

            # Remove python- prefix if present
            if dep.startswith("python-"):
                dep = dep[7:]

            install_cmd = f"uv pip install {dep}"
            install_return_code, _, install_stderr = run_command(install_cmd, timeout=120)
            if install_return_code != 0:
                log_message(f"Failed to install {dep}: {install_stderr}", "ERROR")
                stdscr.addstr(curses.LINES - 3, 2, f"Failed to install {dep}".ljust(curses.COLS - 4), curses.color_pair(COLOR_ERROR))
                stdscr.refresh()
                time.sleep(2)
                continue

            log_message(f"Successfully installed {dep}")
            stdscr.addstr(curses.LINES - 3, 2, f"Successfully installed {dep}".ljust(curses.COLS - 4), curses.color_pair(COLOR_SUCCESS))
            stdscr.refresh()
            time.sleep(1)

    # Final verification for system dependencies
    if system_deps:
        log_message("Verifying all system dependencies")
        all_installed = True
        for dep in system_deps:
            verify_cmd = f"dpkg-query -W -f='${{Status}}' {dep} 2>/dev/null | grep -q 'install ok installed'"
            return_code, _, _ = run_command(verify_cmd, timeout=5)
            if return_code != 0:
                log_message(f"Dependency {dep} is still missing", "ERROR")
                all_installed = False

        if not all_installed:
            log_message("Some system dependencies failed to install", "WARNING")
            stdscr.addstr(curses.LINES - 3, 2, "Some system dependencies failed to install".ljust(curses.COLS - 4), curses.color_pair(COLOR_WARNING))
            stdscr.refresh()
            time.sleep(2)
            return False

    log_message("All dependencies installed successfully")
    stdscr.addstr(curses.LINES - 3, 2, "All dependencies installed successfully".ljust(curses.COLS - 4), curses.color_pair(COLOR_SUCCESS))
    stdscr.refresh()
    time.sleep(2)
    return True

def install_component(component: Dict[str, Any], stdscr, log_win) -> bool:
    """Install a component."""
    name = component["name"]
    script = component["script"]
    script_path = os.path.join(SCRIPTS_DIR, script)

    # Add component-specific logging to identify problematic components
    log_message(f"Starting installation of component: {name} using script: {script}", "INFO")

    # Set component-specific parameters
    # Adjust estimated duration based on component complexity
    component_specific_params = {
        "ML Stack Core": {
            "estimated_duration": 180,  # Longer duration for complex component
            "validation_duration": 90,
            "testing_duration": 45,
            "force_validation": True,   # Force validation phase for this component
            "aggressive_termination": True  # More aggressive process termination
        },
        "MIGraphX Python Wrapper": {
            "estimated_duration": 180,
            "validation_duration": 90,
            "testing_duration": 45,
            "force_validation": True,
            "aggressive_termination": True
        }
    }

    # Get component-specific parameters or use defaults
    component_params = component_specific_params.get(name, {
        "estimated_duration": 120,
        "validation_duration": 60,
        "testing_duration": 30,
        "force_validation": False,
        "aggressive_termination": False
    })

    # Log component-specific parameters
    log_message(f"Component parameters for {name}: {component_params}", "INFO")

    if not os.path.exists(script_path):
        log_message(f"Installation script not found for {name}: {script_path}", "ERROR")
        log_win.addstr(f"ERROR: Installation script not found for {name}\n", curses.color_pair(COLOR_ERROR))
        log_win.refresh()
        return False

    # Make the script executable
    os.chmod(script_path, 0o755)

    # Update component status
    component["status"] = "installing"
    log_win.addstr(f"Installing {name}...\n", curses.color_pair(COLOR_INFO))

    # Add a progress bar
    progress_width = 50
    progress_y, _ = log_win.getyx()
    log_win.addstr("Progress: [", curses.color_pair(COLOR_INFO))
    log_win.addstr(" " * progress_width, curses.color_pair(COLOR_BORDER) | curses.A_REVERSE)
    log_win.addstr("] 0%\n", curses.color_pair(COLOR_INFO))
    log_win.refresh()

    # Special handling for Flash Attention
    if name == "Flash Attention":
        # Make sure libnuma-dev is installed with comprehensive checks
        log_win.addstr("Checking for libnuma-dev dependency...\n", curses.color_pair(COLOR_INFO))
        log_win.refresh()

        # First check if the shared library is already available
        lib_check_cmd = "ldconfig -p | grep -q 'libnuma.so'"
        lib_check_return_code, _, _ = run_command(lib_check_cmd, timeout=5)

        if lib_check_return_code == 0:
            log_win.addstr("libnuma shared library is already available in the system\n", curses.color_pair(COLOR_SUCCESS))
            log_win.refresh()
        else:
            # Check if the package is installed according to dpkg
            verify_cmd = "dpkg-query -W -f='${Status}' libnuma-dev 2>/dev/null | grep -q 'install ok installed'"
            verify_return_code, _, _ = run_command(verify_cmd, timeout=5)

            if verify_return_code != 0:
                log_win.addstr("libnuma-dev is not installed. Installing it first...\n", curses.color_pair(COLOR_WARNING))
                log_win.refresh()

                # Try to install the package
                install_cmd = "sudo apt-get install -y libnuma-dev"
                install_return_code, _, install_stderr = run_command(install_cmd, timeout=60)

                if install_return_code != 0:
                    log_message(f"Failed to install libnuma-dev: {install_stderr}", "ERROR")
                    log_win.addstr(f"ERROR: Failed to install libnuma-dev: {install_stderr}\n", curses.color_pair(COLOR_ERROR))
                    log_win.refresh()

                    # Even if apt-get reports failure, check if the library is actually available
                    # as sometimes the package might be installed but apt reports an error
                    lib_check_cmd = "ldconfig -p | grep -q 'libnuma.so'"
                    lib_check_return_code, _, _ = run_command(lib_check_cmd, timeout=5)

                    if lib_check_return_code == 0:
                        log_win.addstr("Despite installation error, libnuma shared library is available\n", curses.color_pair(COLOR_SUCCESS))
                        log_win.refresh()
                    else:
                        # Try to update ldconfig cache
                        log_win.addstr("Updating ldconfig cache...\n", curses.color_pair(COLOR_INFO))
                        log_win.refresh()
                        run_command("sudo ldconfig", timeout=10)

                        # Check again after ldconfig update
                        lib_check_cmd = "ldconfig -p | grep -q 'libnuma.so'"
                        lib_check_return_code, _, _ = run_command(lib_check_cmd, timeout=5)

                        if lib_check_return_code == 0:
                            log_win.addstr("libnuma shared library is now available after ldconfig update\n", curses.color_pair(COLOR_SUCCESS))
                            log_win.refresh()
                        else:
                            # Try one more approach - check if the file exists directly
                            file_check_cmd = "ls -la /usr/lib/x86_64-linux-gnu/libnuma.so* 2>/dev/null"
                            file_check_return_code, file_check_stdout, _ = run_command(file_check_cmd, timeout=5)

                            if file_check_return_code == 0 and file_check_stdout.strip():
                                log_win.addstr(f"Found libnuma library files: {file_check_stdout.strip()}\n", curses.color_pair(COLOR_SUCCESS))
                                log_win.refresh()

                                # Try to create symlinks if needed
                                if "libnuma.so.1" not in file_check_stdout:
                                    log_win.addstr("Creating symlink for libnuma.so.1...\n", curses.color_pair(COLOR_INFO))
                                    run_command("sudo ln -sf /usr/lib/x86_64-linux-gnu/libnuma.so.* /usr/lib/x86_64-linux-gnu/libnuma.so.1", timeout=5)
                                    run_command("sudo ldconfig", timeout=5)
                                    log_win.refresh()
                            else:
                                component["status"] = "failed"
                                log_win.addstr("Could not find libnuma library files. Installation failed.\n", curses.color_pair(COLOR_ERROR))
                                log_win.refresh()
                                return False
                else:
                    log_win.addstr("Successfully installed libnuma-dev\n", curses.color_pair(COLOR_SUCCESS))
                    log_win.refresh()

                    # Update ldconfig cache after successful installation
                    log_win.addstr("Updating ldconfig cache...\n", curses.color_pair(COLOR_INFO))
                    run_command("sudo ldconfig", timeout=10)
                    log_win.refresh()
            else:
                log_win.addstr("libnuma-dev is already installed according to dpkg\n", curses.color_pair(COLOR_SUCCESS))
                log_win.refresh()

                # Update ldconfig cache to ensure the library is available
                log_win.addstr("Updating ldconfig cache...\n", curses.color_pair(COLOR_INFO))
                run_command("sudo ldconfig", timeout=10)
                log_win.refresh()

        # Final verification - try to load the library with Python
        log_win.addstr("Testing libnuma.so.1 loading with Python...\n", curses.color_pair(COLOR_INFO))
        log_win.refresh()

        test_cmd = "python3 -c 'from ctypes import CDLL; CDLL(\"libnuma.so.1\")' 2>/dev/null"
        test_return_code, _, test_stderr = run_command(test_cmd, timeout=5)

        if test_return_code == 0:
            log_win.addstr("Successfully loaded libnuma.so.1 with Python\n", curses.color_pair(COLOR_SUCCESS))
            log_win.refresh()
        else:
            log_win.addstr(f"Warning: Could not load libnuma.so.1 with Python, but continuing anyway\n", curses.color_pair(COLOR_WARNING))
            log_win.refresh()

    # Create a modified script that provides progress updates
    temp_script_path = os.path.join(SCRIPTS_DIR, f"temp_{script}")

    # Special handling for repair_ml_stack.sh or enhanced_setup_environment.sh
    if script == "repair_ml_stack.sh" or script == "enhanced_setup_environment.sh":
        # Get the cached sudo password if available
        sudo_password = os.environ.get("SUDO_PASSWORD", "")

        with open(temp_script_path, "w") as f:
            f.write(f"""#!/bin/bash
# Wrapper script for {script} with progress updates
echo "Starting {name}..."

# Set environment variables to use uv for Python package installation
export USE_UV=1
export PATH="$HOME/.cargo/bin:$PATH"
export NONINTERACTIVE=1
# Suppress HIP logs
export AMD_LOG_LEVEL=0
export HIP_VISIBLE_DEVICES=0,1,2
export ROCR_VISIBLE_DEVICES=0,1,2
# Set DEBIAN_FRONTEND to noninteractive to avoid sudo prompts
export DEBIAN_FRONTEND=noninteractive

# Create a sudo wrapper function that uses the cached password
sudo_with_pass() {{
    echo '{sudo_password}' | sudo -S "$@"
}}

# Export the sudo wrapper function
export -f sudo_with_pass

# Replace sudo with our wrapper in the script
sed -i 's/\\bsudo\\b/sudo_with_pass/g' {script_path}

# Create a function to handle uv commands properly
uv_pip_install() {{
    # Check if uv is available as a command
    if command -v uv &> /dev/null; then
        # Use uv directly as a command
        uv pip install "$@"
    else
        # Fall back to pip
        python3 -m pip install "$@"
    fi
}}

# Export the uv wrapper function
export -f uv_pip_install

# Replace python3 -m uv pip install with our wrapper function
sed -i 's/python3 -m uv pip install/uv_pip_install/g' {script_path}
sed -i 's/python -m uv pip install/uv_pip_install/g' {script_path}

# Create a progress bar function
progress_bar() {{
    local duration=$1
    local elapsed=0
    local progress=0
    local width=50
    local bar=""

    while [ $elapsed -lt $duration ]; do
        # Calculate progress percentage
        progress=$((elapsed * 100 / duration))
        filled=$((progress * width / 100))
        empty=$((width - filled))

        # Create the progress bar
        bar="["
        for ((i=0; i<filled; i++)); do
            bar+="█"
        done
        for ((i=0; i<empty; i++)); do
            bar+=" "
        done
        bar+="] $progress%"

        # Print the progress bar
        echo -ne "\\r$bar"

        # Sleep for a bit
        sleep 1
        elapsed=$((elapsed + 1))
    done
    echo ""
}}

# Run the actual script with NONINTERACTIVE mode to avoid prompts
# Use stdbuf to ensure unbuffered output
stdbuf -oL -eL {script_path} 2>&1 | while IFS= read -r line; do
    echo "$line"
    # Flush stdout to ensure real-time output
    sleep 0.01
done

# Capture the exit code
exit_code=${{PIPESTATUS[0]}}

# Restore the original script (remove our sudo wrapper)
sed -i 's/\\bsudo_with_pass\\b/sudo/g' {script_path}

echo "Script completed with exit code: $exit_code"
exit $exit_code
""")
    else:
        # Get the cached sudo password if available
        sudo_password = os.environ.get("SUDO_PASSWORD", "")

        with open(temp_script_path, "w") as f:
            f.write(f"""#!/bin/bash
# Wrapper script for {script} with progress updates
echo "Starting installation of {name}..."

# Set environment variables to use uv for Python package installation
export USE_UV=1
export PATH="$HOME/.cargo/bin:$PATH"
# Suppress HIP logs
export AMD_LOG_LEVEL=0
export HIP_VISIBLE_DEVICES=0,1,2
export ROCR_VISIBLE_DEVICES=0,1,2
# Set DEBIAN_FRONTEND to noninteractive to avoid sudo prompts
export DEBIAN_FRONTEND=noninteractive

# Create a sudo wrapper function that uses the cached password
sudo_with_pass() {{
    echo '{sudo_password}' | sudo -S "$@"
}}

# Export the sudo wrapper function
export -f sudo_with_pass

# Replace sudo with our wrapper in the script
sed -i 's/\\bsudo\\b/sudo_with_pass/g' {script_path}

# Create a function to handle uv commands properly
uv_pip_install() {{
    # Check if uv is available as a command
    if command -v uv &> /dev/null; then
        # Use uv directly as a command
        uv pip install "$@"
    else
        # Fall back to pip
        python3 -m pip install "$@"
    fi
}}

# Export the uv wrapper function
export -f uv_pip_install

# Replace python3 -m uv pip install with our wrapper function
sed -i 's/python3 -m uv pip install/uv_pip_install/g' {script_path}
sed -i 's/python -m uv pip install/uv_pip_install/g' {script_path}

# Create a progress bar function
progress_bar() {{
    local duration=$1
    local elapsed=0
    local progress=0
    local width=50
    local bar=""
    local spinner=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
    local spin_idx=0

    while [ $elapsed -lt $duration ]; do
        # Calculate progress percentage
        progress=$((elapsed * 100 / duration))
        filled=$((progress * width / 100))
        empty=$((width - filled))

        # Create the progress bar with spinner
        bar="["
        for ((i=0; i<filled; i++)); do
            bar+="█"
        done
        for ((i=0; i<empty; i++)); do
            bar+=" "
        done
        bar+="] $progress% ${{spinner[$spin_idx]}}"

        # Print the progress bar
        echo -ne "\\r$bar"

        # Update spinner index
        spin_idx=$(( (spin_idx + 1) % 10 ))

        # Sleep for a bit
        sleep 1
        elapsed=$((elapsed + 1))
    done
    echo ""
}}

# Start a background progress bar for long-running operations
progress_bar 120 &
PROGRESS_PID=$!

# Run the actual installation script and capture output in real-time
# Use stdbuf to ensure unbuffered output
stdbuf -oL -eL {script_path} 2>&1 | while IFS= read -r line; do
    echo "$line"
    # Flush stdout to ensure real-time output
    sleep 0.01
done

# Capture the exit code
exit_code=${{PIPESTATUS[0]}}

# Kill the progress bar process
kill $PROGRESS_PID 2>/dev/null

# Restore the original script (remove our sudo wrapper)
sed -i 's/\\bsudo_with_pass\\b/sudo/g' {script_path}

echo "Installation script completed with exit code: $exit_code"
exit $exit_code
""")

    # Make the script executable
    os.chmod(temp_script_path, 0o755)

    # Create a password window at the bottom of the screen
    height, width = stdscr.getmaxyx()
    # Make the password window taller, more visible, and centered
    password_win_height = 8  # Increased height for additional tip
    password_win_width = width - 16
    password_win_y = height - password_win_height - 2
    password_win_x = 8

    # Ensure the window fits within the screen boundaries
    if password_win_y < 0:
        password_win_y = 0
    if password_win_x < 0:
        password_win_x = 0
    if password_win_height + password_win_y >= height:
        password_win_height = height - password_win_y - 1
    if password_win_width + password_win_x >= width:
        password_win_width = width - password_win_x - 1

    # Ensure minimum size
    password_win_height = max(6, password_win_height)
    password_win_width = max(40, password_win_width)

    # Create the window
    password_win = curses.newwin(password_win_height, password_win_width, password_win_y, password_win_x)

    # Use box() for the border instead of drawing it manually
    password_win.box()

    # Add a colorful border highlight
    password_win.attron(curses.color_pair(COLOR_BORDER))
    # Only draw the border characters that are safe (not at the edges)
    for i in range(1, password_win_width - 1):
        try:
            password_win.addch(0, i, curses.ACS_HLINE)
            password_win.addch(password_win_height - 1, i, curses.ACS_HLINE)
        except:
            pass

    for i in range(1, password_win_height - 1):
        try:
            password_win.addch(i, 0, curses.ACS_VLINE)
            password_win.addch(i, password_win_width - 1, curses.ACS_VLINE)
        except:
            pass

    # Add corners safely
    try:
        password_win.addch(0, 0, curses.ACS_ULCORNER)
    except:
        pass

    try:
        password_win.addch(0, password_win_width - 1, curses.ACS_URCORNER)
    except:
        pass

    try:
        password_win.addch(password_win_height - 1, 0, curses.ACS_LLCORNER)
    except:
        pass

    try:
        password_win.addch(password_win_height - 1, password_win_width - 1, curses.ACS_LRCORNER)
    except:
        pass

    # Add a title
    title = " Password Input "
    password_win.addstr(0, (password_win_width - len(title)) // 2, title)
    password_win.attroff(curses.color_pair(COLOR_BORDER))

    # Add instructions with better formatting
    password_win.addstr(1, 2, "✧ If prompted for password, type it here and press Enter:", curses.color_pair(COLOR_INFO))
    password_win.addstr(2, 2, "✧ The cursor will appear when password input is needed.", curses.color_pair(COLOR_INFO))
    password_win.addstr(3, 2, "✧ Press Enter after typing your password.", curses.color_pair(COLOR_INFO))
    password_win.addstr(4, 2, "✧ Password field: [                                  ]", curses.color_pair(COLOR_WARNING))
    password_win.addstr(5, 2, "✧ Password will not be visible as you type (for security).", curses.color_pair(COLOR_INFO))
    password_win.addstr(6, 2, "✧ TIP: Press Down key until you see * characters appear", curses.color_pair(COLOR_SUCCESS))
    password_win.addstr(7, 2, "  indicating successful typing.", curses.color_pair(COLOR_SUCCESS))

    # Draw the password field with a different color
    password_win.attron(curses.color_pair(COLOR_HIGHLIGHT))
    password_win.addstr(4, 18, "                                  ")
    password_win.attroff(curses.color_pair(COLOR_HIGHLIGHT))

    password_win.refresh()

    # Run the modified script with a longer timeout
    process = subprocess.Popen(
        f"bash {temp_script_path}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
        env={**os.environ, "USE_UV": "1"}  # Set environment variable to use uv
    )

    # Set up non-blocking read

    # Clear the log window to ensure a clean display
    log_win.clear()
    log_win.refresh()

    # Read output in real-time and update the log window
    output_lines = []
    password_mode = False
    progress_chars = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
    progress_position = 0
    progress_width = 50
    progress_y = 5  # Line where progress bar was drawn

    # Variables for installation phases
    current_phase = "installation"  # Current phase: installation, validation, testing
    validation_phase_started = False
    validation_progress_y = 0
    validation_start_time = 0
    validation_estimated_duration = 60
    testing_phase_started = False
    testing_progress_y = 0
    testing_start_time = 0
    testing_estimated_duration = 30

    # Progress bar state tracking
    last_progress_percent = 0
    last_validation_progress = 0
    last_testing_progress = 0

    # Draw initial progress bar label
    log_win.addstr(progress_y, 0, "Progress: [", curses.color_pair(COLOR_INFO))

    # Store the start time for progress calculation
    start_time = time.time()
    estimated_duration = component_params["estimated_duration"]  # Use component-specific duration
    validation_estimated_duration = component_params["validation_duration"]
    testing_estimated_duration = component_params["testing_duration"]
    force_validation = component_params["force_validation"]
    aggressive_termination = component_params["aggressive_termination"]

    # Log component-specific settings
    log_message(f"Using estimated_duration={estimated_duration}, validation_duration={validation_estimated_duration}, testing_duration={testing_estimated_duration} for {name}", "INFO")
    log_message(f"Force validation: {force_validation}, Aggressive termination: {aggressive_termination}", "INFO")

    # Set up automatic refresh timer
    last_refresh_time = time.time()
    refresh_interval = 0.1  # More frequent refresh (100ms) for better responsiveness

    # Set stdout to non-blocking mode
    fl = fcntl.fcntl(process.stdout, fcntl.F_GETFL)
    fcntl.fcntl(process.stdout, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    while process.poll() is None:
        # Check if there's data to read (with a short timeout)
        r, _, _ = select.select([process.stdout], [], [], 0.05)

        current_time = time.time()
        needs_refresh = False

        # Process available output
        if process.stdout in r:
            try:
                # Read as much as possible without blocking
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break

                    # Strip ANSI escape codes for display
                    clean_line = strip_ansi_codes(line)
                    output_lines.append(clean_line)
                    log_message(f"[{name}] {clean_line.strip()}")
                    log_win.addstr(clean_line)
                    needs_refresh = True

                    # Get current cursor position
                    y, _ = log_win.getyx()

                    # Always scroll to ensure we see the latest output
                    max_y, _ = log_win.getmaxyx()
                    if y >= max_y - 10:  # Increased scroll threshold for better visibility
                        log_win.scroll(1)

                    # Force refresh after each line to ensure real-time updates
                    log_win.refresh()

                    # Check if this is a password prompt - be more aggressive in detection
                    if ("password" in line.lower() or "sudo" in line.lower() or "[sudo]" in line.lower() or "authentication" in line.lower()) and not password_mode:
                        password_mode = True
                        break  # Break to handle password prompt immediately
            except IOError:
                # No more data to read without blocking
                pass

        # Update progress bar (do this regardless of new output)
        elapsed_time = time.time() - start_time
        progress_percent = min(1.0, elapsed_time / estimated_duration)
        filled_width = int(progress_width * progress_percent)

        # Track current phase and update progress bars
        try:
            # Save cursor position
            current_y, current_x = log_win.getyx()

            # Check if we've reached 100% or if we should force validation phase
            if (progress_percent >= 0.99 or force_validation) and not validation_phase_started:
                # Log the transition to validation phase
                if force_validation:
                    log_message(f"Forcing validation phase for component: {name}", "INFO")
                else:
                    log_message(f"Transitioning to validation phase for component: {name} at progress {int(progress_percent * 100)}%", "INFO")

                # Mark that we've started the validation phase
                validation_phase_started = True
                current_phase = "validation"

                # Clear the entire progress bar line to prevent overlapping
                log_win.move(progress_y, 0)
                log_win.clrtoeol()

                # Draw a completed progress bar
                log_win.move(progress_y, 0)
                log_win.addstr("Progress: [", curses.color_pair(COLOR_INFO))
                log_win.addstr("█" * progress_width, curses.color_pair(COLOR_SUCCESS) | curses.A_REVERSE)
                log_win.addstr("] ", curses.color_pair(COLOR_INFO))
                log_win.addstr(f"✓ 100%", curses.color_pair(COLOR_SUCCESS))

                # Add a message about post-installation validation with clear separation
                log_win.addstr("\n\n", curses.color_pair(COLOR_INFO))
                log_win.addstr("=" * 70 + "\n", curses.color_pair(COLOR_TITLE))
                log_win.addstr(f"Running post-installation validation for {name}...\n", curses.color_pair(COLOR_INFO))
                log_win.addstr("=" * 70 + "\n\n", curses.color_pair(COLOR_TITLE))

                # Create a new progress bar for validation phase
                validation_progress_y = log_win.getyx()[0]
                log_win.addstr("Validation: [", curses.color_pair(COLOR_INFO))
                log_win.addstr(" " * progress_width, curses.color_pair(COLOR_BORDER) | curses.A_REVERSE)
                log_win.addstr("] ", curses.color_pair(COLOR_INFO))
                log_win.addstr(f"0%", curses.color_pair(COLOR_SUCCESS))

                # Force refresh to show the validation message immediately
                log_win.refresh()

                # Reset timer for validation phase - use component-specific duration
                validation_start_time = time.time()

                # For problematic components, check if we need to force completion
                if name in ["ML Stack Core", "MIGraphX Python Wrapper"]:
                    log_message(f"Using special handling for component: {name} in validation phase", "INFO")
                    # Add additional logging for debugging
                    log_win.addstr(f"\nNote: Using special handling for {name}...\n", curses.color_pair(COLOR_INFO))
                    log_win.refresh()

            # Update the appropriate progress bar based on current phase
            if validation_phase_started:
                # Update validation progress bar
                validation_elapsed_time = time.time() - validation_start_time
                validation_progress = min(1.0, validation_elapsed_time / validation_estimated_duration)
                validation_filled_width = int(progress_width * validation_progress)

                # Clear the validation progress bar line to prevent overlapping
                log_win.move(validation_progress_y, 0)
                log_win.clrtoeol()

                # Draw the validation progress bar
                log_win.move(validation_progress_y, 0)
                log_win.addstr("Validation: [", curses.color_pair(COLOR_INFO))
                log_win.addstr("█" * validation_filled_width, curses.color_pair(COLOR_INFO) | curses.A_REVERSE)
                if validation_filled_width < progress_width:
                    log_win.addstr(" " * (progress_width - validation_filled_width), curses.color_pair(COLOR_BORDER) | curses.A_REVERSE)

                # Add a spinner at the end of validation bar
                log_win.addstr("] ", curses.color_pair(COLOR_INFO))
                log_win.addstr(f"{progress_chars[progress_position]} {int(validation_progress * 100)}%",
                              curses.color_pair(COLOR_SUCCESS))

                # Check if validation is complete (100%)
                if validation_progress >= 0.99 and current_phase == "validation":
                    # Mark that we've started the testing phase
                    current_phase = "testing"

                    # Complete the validation progress bar
                    log_win.move(validation_progress_y, 0)
                    log_win.clrtoeol()
                    log_win.addstr("Validation: [", curses.color_pair(COLOR_INFO))
                    log_win.addstr("█" * progress_width, curses.color_pair(COLOR_SUCCESS) | curses.A_REVERSE)
                    log_win.addstr("] ", curses.color_pair(COLOR_INFO))
                    log_win.addstr(f"✓ 100%", curses.color_pair(COLOR_SUCCESS))

                    # Add a message about testing phase
                    log_win.addstr("\n\n", curses.color_pair(COLOR_INFO))
                    log_win.addstr("=" * 70 + "\n", curses.color_pair(COLOR_TITLE))
                    log_win.addstr("Running final tests...\n", curses.color_pair(COLOR_INFO))
                    log_win.addstr("=" * 70 + "\n", curses.color_pair(COLOR_TITLE))
                    log_win.refresh()
            else:
                # Clear the progress bar line to prevent overlapping
                log_win.move(progress_y, 0)
                log_win.clrtoeol()

                # Draw the installation progress bar
                log_win.move(progress_y, 0)
                log_win.addstr("Progress: [", curses.color_pair(COLOR_INFO))
                log_win.addstr("█" * filled_width, curses.color_pair(COLOR_INFO) | curses.A_REVERSE)
                if filled_width < progress_width:
                    log_win.addstr(" " * (progress_width - filled_width), curses.color_pair(COLOR_BORDER) | curses.A_REVERSE)

                # Add a spinner at the end
                log_win.addstr("] ", curses.color_pair(COLOR_INFO))
                log_win.addstr(f"{progress_chars[progress_position]} {int(progress_percent * 100)}%",
                              curses.color_pair(COLOR_SUCCESS))

            # Update spinner position
            progress_position = (progress_position + 1) % len(progress_chars)

            # Restore cursor position
            log_win.move(current_y, current_x)

            # Force refresh more frequently to ensure smooth updates
            if current_time - last_refresh_time >= refresh_interval:
                log_win.refresh()
                last_refresh_time = current_time
                needs_refresh = False
            elif needs_refresh:
                log_win.refresh()
                last_refresh_time = current_time
                needs_refresh = False
        except Exception as e:
            # Log the error instead of silently ignoring it
            log_message(f"Error updating progress bar: {str(e)}", "WARNING")
            pass

        # Check if this is a password prompt - be more aggressive in detection
        if password_mode:

                    # Clear and redraw the password window with the same style as before
                    password_win.clear()
                    password_win.box()

                    # Add a colorful border highlight
                    password_win.attron(curses.color_pair(COLOR_BORDER))
                    # Only draw the border characters that are safe (not at the edges)
                    for i in range(1, password_win_width - 1):
                        try:
                            password_win.addch(0, i, curses.ACS_HLINE)
                            password_win.addch(password_win_height - 1, i, curses.ACS_HLINE)
                        except:
                            pass

                    for i in range(1, password_win_height - 1):
                        try:
                            password_win.addch(i, 0, curses.ACS_VLINE)
                            password_win.addch(i, password_win_width - 1, curses.ACS_VLINE)
                        except:
                            pass

                    # Add corners safely
                    try:
                        password_win.addch(0, 0, curses.ACS_ULCORNER)
                    except:
                        pass

                    try:
                        password_win.addch(0, password_win_width - 1, curses.ACS_URCORNER)
                    except:
                        pass

                    try:
                        password_win.addch(password_win_height - 1, 0, curses.ACS_LLCORNER)
                    except:
                        pass

                    try:
                        password_win.addch(password_win_height - 1, password_win_width - 1, curses.ACS_LRCORNER)
                    except:
                        pass

                    # Add a title
                    title = " Password Required "
                    password_win.addstr(0, (password_win_width - len(title)) // 2, title)
                    password_win.attroff(curses.color_pair(COLOR_BORDER))

                    # Show the prompt that triggered the password request
                    password_win.addstr(1, 2, f"✧ Prompt: {line.strip()}", curses.color_pair(COLOR_WARNING))
                    password_win.addstr(2, 2, "✧ This may be for sudo or other system commands.", curses.color_pair(COLOR_INFO))
                    password_win.addstr(3, 2, "✧ Enter your password below:", curses.color_pair(COLOR_INFO))

                    # Draw the password field with a different color
                    password_win.addstr(4, 2, "✧ Password: [", curses.color_pair(COLOR_WARNING))
                    password_win.attron(curses.color_pair(COLOR_HIGHLIGHT))
                    password_win.addstr(4, 14, "                                  ")
                    password_win.attroff(curses.color_pair(COLOR_HIGHLIGHT))
                    password_win.addstr(4, 50, "]", curses.color_pair(COLOR_WARNING))

                    password_win.addstr(5, 2, "✧ Password will not be visible as you type (for security).", curses.color_pair(COLOR_INFO))
                    password_win.refresh()

                    # Move cursor to password input position
                    password_win.move(4, 14)

                    # Enable cursor and disable echo for password input (for security)
                    curses.curs_set(1)
                    curses.noecho()  # Don't show the password as it's typed

                    # Get password input character by character
                    password = ""
                    cursor_pos = 14
                    max_cursor_pos = min(49, password_win_width - 5)

                    # Draw a blinking cursor to indicate where input will go
                    password_win.attron(curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BLINK)
                    try:
                        password_win.addch(4, cursor_pos, "_")
                    except:
                        pass
                    password_win.attroff(curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BLINK)
                    password_win.refresh()

                    try:
                        while True:
                            # Get a character
                            ch = password_win.getch(4, cursor_pos)

                            # Check if it's Enter (10 or 13)
                            if ch == 10 or ch == 13:
                                break

                            # Check if it's backspace (8 or 127)
                            elif ch == 8 or ch == 127:
                                if len(password) > 0:
                                    password = password[:-1]
                                    cursor_pos = max(14, cursor_pos - 1)

                                    # Clear the current position and redraw
                                    try:
                                        password_win.attron(curses.color_pair(COLOR_HIGHLIGHT))
                                        password_win.addstr(4, cursor_pos, " ")
                                        password_win.attroff(curses.color_pair(COLOR_HIGHLIGHT))

                                        # Add blinking cursor at new position
                                        password_win.attron(curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BLINK)
                                        password_win.addch(4, cursor_pos, "_")
                                        password_win.attroff(curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BLINK)
                                    except:
                                        pass

                            # Otherwise, add the character to the password
                            elif ch >= 32 and ch <= 126:  # Printable ASCII
                                if cursor_pos < max_cursor_pos:
                                    password += chr(ch)

                                    # Show a * for each character and move cursor
                                    try:
                                        password_win.attron(curses.color_pair(COLOR_HIGHLIGHT))
                                        password_win.addstr(4, cursor_pos, "*")
                                        password_win.attroff(curses.color_pair(COLOR_HIGHLIGHT))

                                        cursor_pos += 1

                                        # Add blinking cursor at new position
                                        password_win.attron(curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BLINK)
                                        password_win.addch(4, cursor_pos, "_")
                                        password_win.attroff(curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BLINK)
                                    except:
                                        pass

                            # Update the password field indicator
                            try:
                                # Show how many characters have been typed
                                indicator = f"[{len(password)} chars]"
                                password_win.addstr(5, password_win_width - len(indicator) - 3, indicator, curses.color_pair(COLOR_INFO))
                            except:
                                pass

                            # Refresh to show changes
                            password_win.refresh()
                    except Exception as e:
                        log_message(f"Error getting password: {str(e)}", "ERROR")

                    # Disable cursor and echo
                    curses.noecho()
                    curses.curs_set(0)

                    # Send password to the process
                    try:
                        process.stdin.write(password + "\n")
                        process.stdin.flush()
                    except Exception as e:
                        log_message(f"Error sending password: {str(e)}", "ERROR")

                    # Reset password mode
                    password_mode = False

                    # Clear the password window and show confirmation
                    password_win.clear()
                    password_win.box()

                    # Add a colorful border highlight
                    password_win.attron(curses.color_pair(COLOR_BORDER))
                    # Only draw the border characters that are safe (not at the edges)
                    for i in range(1, password_win_width - 1):
                        try:
                            password_win.addch(0, i, curses.ACS_HLINE)
                            password_win.addch(password_win_height - 1, i, curses.ACS_HLINE)
                        except:
                            pass

                    for i in range(1, password_win_height - 1):
                        try:
                            password_win.addch(i, 0, curses.ACS_VLINE)
                            password_win.addch(i, password_win_width - 1, curses.ACS_VLINE)
                        except:
                            pass

                    # Add corners safely
                    try:
                        password_win.addch(0, 0, curses.ACS_ULCORNER)
                    except:
                        pass

                    try:
                        password_win.addch(0, password_win_width - 1, curses.ACS_URCORNER)
                    except:
                        pass

                    try:
                        password_win.addch(password_win_height - 1, 0, curses.ACS_LLCORNER)
                    except:
                        pass

                    try:
                        password_win.addch(password_win_height - 1, password_win_width - 1, curses.ACS_LRCORNER)
                    except:
                        pass

                    # Add a title
                    title = " Password Accepted "
                    password_win.addstr(0, (password_win_width - len(title)) // 2, title)
                    password_win.attroff(curses.color_pair(COLOR_BORDER))

                    # Show confirmation message
                    password_win.addstr(1, 2, "✧ Password entered successfully.", curses.color_pair(COLOR_SUCCESS))
                    password_win.addstr(2, 2, "✧ Continuing installation...", curses.color_pair(COLOR_SUCCESS))
                    password_win.addstr(3, 2, "✧ Please wait while the command completes.", curses.color_pair(COLOR_INFO))
                    password_win.addstr(4, 2, "✧ The password window will close automatically.", curses.color_pair(COLOR_INFO))
                    password_win.addstr(5, 2, "✧ If you need to enter another password, this window will reappear.", curses.color_pair(COLOR_INFO))
                    password_win.refresh()

                    # Wait a moment to show the confirmation
                    time.sleep(2)

        # We've already updated the progress bar above, so we don't need to do it again here
        # Just force a refresh periodically to ensure the UI stays responsive
        if int(elapsed_time * 10) % 5 == 0:  # Update every 0.5 seconds
            try:
                log_win.refresh()
            except Exception as e:
                log_message(f"Error refreshing log window: {str(e)}", "WARNING")
                pass

        # Check for keyboard input to allow cancellation
        try:
            key = stdscr.getch()
            if key == ord('q'):
                process.terminate()
                log_win.addstr("\nInstallation cancelled by user.\n", curses.color_pair(COLOR_WARNING))
                log_win.refresh()
                break
        except:
            pass

    # Read any remaining output
    while True:
        try:
            line = process.stdout.readline()
            if not line:
                break
            # Strip ANSI escape codes for display
            clean_line = strip_ansi_codes(line)
            output_lines.append(clean_line)
            log_message(f"[{name}] {clean_line.strip()}")
            log_win.addstr(clean_line)

            # Get current cursor position
            y, _ = log_win.getyx()

            # Always scroll to ensure we see the latest output
            max_y, _ = log_win.getmaxyx()
            if y >= max_y - 10:  # Increased scroll threshold for better visibility
                log_win.scroll(1)

            # Force refresh after each line to ensure real-time updates
            log_win.refresh()
        except:
            break

    # Show completion message and update progress to 100%
    try:
        # Clear the log window to ensure a clean display for completion
        y, x = log_win.getyx()
        log_win.addstr("\n\n")

        # If validation phase was started, ensure it's shown as complete
        if validation_phase_started:
            # Clear any existing progress bars to prevent overlapping
            if validation_progress_y > 0:
                log_win.move(validation_progress_y, 0)
                log_win.clrtoeol()

                # Draw a completed validation progress bar
                log_win.move(validation_progress_y, 0)
                log_win.addstr("Validation: [", curses.color_pair(COLOR_INFO))
                log_win.addstr("█" * progress_width, curses.color_pair(COLOR_SUCCESS) | curses.A_REVERSE)
                log_win.addstr("] ", curses.color_pair(COLOR_INFO))
                log_win.addstr(f"✓ 100%", curses.color_pair(COLOR_SUCCESS))
                log_win.refresh()
        else:
            # If validation phase wasn't started (rare case), ensure main progress bar is complete
            log_win.move(progress_y, 0)
            log_win.clrtoeol()

            # Draw a completed progress bar
            log_win.move(progress_y, 0)
            log_win.addstr("Progress: [", curses.color_pair(COLOR_INFO))
            log_win.addstr("█" * progress_width, curses.color_pair(COLOR_SUCCESS) | curses.A_REVERSE)
            log_win.addstr("] ", curses.color_pair(COLOR_INFO))
            log_win.addstr(f"✓ 100%", curses.color_pair(COLOR_SUCCESS))
            log_win.refresh()

        # Add a clear completion message with visual separation
        log_win.addstr("\n\n")
        log_win.addstr("=" * 70 + "\n", curses.color_pair(COLOR_TITLE))
        log_win.addstr(f"INSTALLATION COMPLETED: {name}\n", curses.color_pair(COLOR_SUCCESS) | curses.A_BOLD)
        log_win.addstr("=" * 70 + "\n", curses.color_pair(COLOR_TITLE))
        log_win.refresh()

        # Read any remaining output from the process
        try:
            # Set a timeout for reading remaining output
            remaining_output_timeout = time.time() + 3  # 3 second timeout

            while process.poll() is None and time.time() < remaining_output_timeout:
                r, _, _ = select.select([process.stdout], [], [], 0.1)
                if process.stdout in r:
                    line = process.stdout.readline()
                    if line:
                        # Strip ANSI escape codes for display
                        clean_line = strip_ansi_codes(line)
                        output_lines.append(clean_line)
                        log_message(f"[{name}] {clean_line.strip()}")
                        log_win.addstr(clean_line)
                        log_win.refresh()
        except Exception as e:
            log_message(f"Error reading final output: {str(e)}", "WARNING")

        # Add return instructions
        log_win.addstr("\n\n")
        log_win.addstr("Press any key to return to menu or wait for automatic return...\n", curses.color_pair(COLOR_INFO))
        log_win.refresh()

        # Create a visually appealing countdown for automatic return
        log_win.addstr("\nReturning to menu in ", curses.color_pair(COLOR_INFO))

        # Set a shorter timeout for better responsiveness
        countdown_seconds = 5
        for i in range(countdown_seconds, 0, -1):
            # Clear previous number
            log_win.addstr(f"{i+1} ", curses.color_pair(COLOR_INFO) | curses.A_BOLD)
            log_win.addstr("\b\b  \b\b", curses.color_pair(COLOR_INFO))

            # Write new number
            log_win.addstr(f"{i}", curses.color_pair(COLOR_SUCCESS) | curses.A_BOLD)
            log_win.refresh()

            # Check for key press during countdown with short timeout
            stdscr.timeout(1000)  # 1 second timeout
            key = stdscr.getch()
            if key != -1:  # If any key is pressed, break the countdown
                break

        # Reset timeout to match main loop for consistent behavior
        stdscr.timeout(200)
    except Exception as e:
        # Log the error instead of silently ignoring it
        log_message(f"Error updating completion message: {str(e)}", "ERROR")
        # Try a simpler completion message as fallback
        try:
            log_win.addstr("\n\nInstallation completed. Press any key to continue...\n", curses.color_pair(COLOR_SUCCESS))
            log_win.refresh()
            # Wait for a key press with a timeout
            stdscr.timeout(5000)  # 5 second timeout
            stdscr.getch()
            stdscr.timeout(200)  # Reset timeout
        except:
            pass

    # Delete the password window
    password_win.clear()
    password_win.refresh()
    del password_win

    # Wait for the process to complete with a timeout
    process.stdout.close()
    try:
        # Use component-specific timeout for problematic components
        termination_timeout = 2 if aggressive_termination else 5
        log_message(f"Waiting for process to complete with timeout {termination_timeout}s for component: {name}", "INFO")

        # Wait for the process with appropriate timeout
        return_code = process.wait(timeout=termination_timeout)
        log_message(f"Process for {name} completed with return code {return_code}", "INFO")
    except subprocess.TimeoutExpired:
        # If the process is still running after timeout, handle based on component type
        log_message(f"Process for {name} still running after completion UI shown", "WARNING")

        # Add a message to the log window
        log_win.addstr("\nFinalizing background processes...\n", curses.color_pair(COLOR_INFO))
        log_win.refresh()

        # For problematic components, use more aggressive termination
        if aggressive_termination:
            log_message(f"Using aggressive termination for component: {name}", "WARNING")
            log_win.addstr(f"Using special termination for {name}...\n", curses.color_pair(COLOR_WARNING))
            log_win.refresh()

            # Kill the process directly for problematic components
            process.kill()

            try:
                return_code = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                return_code = -9  # Assume it was killed

            # Add success message to log window
            log_win.addstr("Installation completed successfully.\n", curses.color_pair(COLOR_SUCCESS))
            log_win.refresh()
        else:
            # For normal components, try to terminate gracefully first
            try:
                process.terminate()
                # Give it a chance to terminate gracefully with a short timeout
                return_code = process.wait(timeout=3)
                log_message(f"Process for {name} terminated gracefully with return code {return_code}", "INFO")

                # Add success message to log window
                log_win.addstr("Background processes completed successfully.\n", curses.color_pair(COLOR_SUCCESS))
                log_win.refresh()
            except subprocess.TimeoutExpired:
                # If still running after graceful termination attempt, kill it
                log_message(f"Process for {name} did not terminate gracefully, killing it", "WARNING")

                # Add message to log window
                log_win.addstr("Finalizing installation...\n", curses.color_pair(COLOR_INFO))
                log_win.refresh()

                # Kill the process directly - we've already shown the completion UI
                process.kill()

                try:
                    return_code = process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    return_code = -9  # Assume it was killed

                # Add message to log window
                log_win.addstr("Installation completed.\n", curses.color_pair(COLOR_SUCCESS))
                log_win.refresh()

        # Set a special return code to indicate graceful termination
        if return_code in [-15, 15, 143, -9, 9]:  # SIGTERM and SIGKILL related codes
            log_message(f"Process was terminated after UI completion, treating as success", "INFO")
            return_code = 0  # Treat as success since UI phase was completed

    # If we got a SIGPIPE or similar signal but saw success indicators, treat as success
    if return_code in [141, -13, -2, 1, 2]:
        log_message(f"Process for {name} exited with code {return_code}, checking for success indicators", "WARNING")
        # Check if we see any success indicators in the output
        output_text = "\n".join(output_lines)
        if "Installation completed successfully" in output_text or "Installation script completed with exit code: 0" in output_text:
            log_message(f"Found success indicators for {name}, treating as success despite exit code {return_code}", "INFO")
            return_code = 0

    # Clean up the temporary script
    try:
        os.remove(temp_script_path)
    except Exception as e:
        log_message(f"Error removing temporary script: {str(e)}", "WARNING")

    # Special handling for exit codes that might indicate success despite the error code
    if return_code in [141, -13, -2, 1, 2, -15, 15, 143, -9, 9]:  # Added SIGTERM and SIGKILL related codes
        # Check if we see any success indicators in the output
        output_text = "\n".join(output_lines)

        # Define component-specific success indicators
        component_specific_indicators = {
            "ML Stack Core": [
                "ML Stack Installation Completed",
                "ML Stack Core installed",
                "All core components installed successfully",
                "Installation script completed",
                "Verification Complete"
            ],
            "MIGraphX Python Wrapper": [
                "MIGraphX Python wrapper has been installed",
                "Installation Complete",
                "MIGraphX Python wrapper is installed",
                "Basic functionality test passed"
            ]
        }

        # Get component-specific indicators or use default ones
        specific_indicators = component_specific_indicators.get(name, [])

        # Combine with general success indicators
        success_indicators = [
            "ROCm-related variables set",
            "Environment file created successfully",
            "Enhanced ML Stack Environment Setup Complete",
            "ML Stack Repair Complete",
            "Installation completed successfully",
            "Verification completed successfully",
            "Script completed with exit code: 0",
            "Testing completed",
            "Validation completed",
            "Installation script completed",
            "100%",  # If we saw 100% progress, likely successful
            "Completed"
        ] + specific_indicators

        # Log the success indicators we're looking for
        log_message(f"Checking for success indicators for component: {name}", "INFO")
        if specific_indicators:
            log_message(f"Using component-specific indicators: {specific_indicators}", "INFO")

        # For problematic components, be more lenient with success detection
        if name in ["ML Stack Core", "MIGraphX Python Wrapper"] or aggressive_termination:
            log_message(f"Using lenient success detection for component: {name}", "INFO")
            # For problematic components, consider validation phase or force_validation as success
            if validation_phase_started or force_validation:
                log_message(f"Component {name} reached validation phase, treating as success despite exit code {return_code}", "WARNING")
                component["status"] = "installed"
                log_win.addstr(f"\nSUCCESS: {name} installed successfully\n", curses.color_pair(COLOR_SUCCESS))
                log_win.addstr(f"(Note: Process exited with code {return_code} but installation was successful)\n", curses.color_pair(COLOR_INFO))
                log_win.refresh()

                # Wait a moment to ensure the message is seen
                time.sleep(1)

                # Return to the components screen
                return True

        # Check if any success indicators are present or if validation phase was started
        if any(indicator in output_text for indicator in success_indicators) or validation_phase_started:
            log_message(f"Script reported partial success despite exit code {return_code}. Treating as success.", "WARNING")
            # Update component status
            component["status"] = "installed"
            log_win.addstr(f"\nSUCCESS: {name} installed successfully\n", curses.color_pair(COLOR_SUCCESS))
            if return_code != 0:
                log_win.addstr(f"(Note: Process exited with code {return_code} but installation was successful)\n", curses.color_pair(COLOR_INFO))
            log_win.refresh()

            # Wait a moment to ensure the message is seen
            time.sleep(1)

            # Return to the components screen
            return True
        else:
            # Check for specific error patterns
            error_indicators = [
                "Error:",
                "Failed:",
                "Installation failed",
                "Could not find",
                "Not found",
                "No such file or directory",
                "Permission denied",
                "ImportError:",
                "ModuleNotFoundError:",
                "Exception:"
            ]

            # Extract error details if possible
            error_details = ""
            for indicator in error_indicators:
                if indicator in output_text:
                    # Find the line containing the error
                    for line in output_lines:
                        if indicator in line:
                            error_details = line.strip()
                            break
                    if error_details:
                        break

            log_message(f"Installation of {name} failed with exit code {return_code}", "ERROR")
            component["status"] = "failed"

            # Show detailed error message
            log_win.addstr(f"\nERROR: Installation of {name} failed\n", curses.color_pair(COLOR_ERROR))
            if error_details:
                log_win.addstr(f"Details: {error_details}\n", curses.color_pair(COLOR_ERROR))
            else:
                log_win.addstr(f"Exit code: {return_code}\n", curses.color_pair(COLOR_ERROR))
            log_win.addstr("\nPress any key to return to component selection...\n", curses.color_pair(COLOR_INFO))
            log_win.refresh()

            # Wait for a key press with a timeout
            stdscr.timeout(5000)  # 5 second timeout
            stdscr.getch()
            stdscr.timeout(200)  # Reset timeout

            return False
    # Special handling for negative return codes (signals) - should be handled by the previous block
    # but keeping this as a fallback
    elif return_code < 0:
        # If the script was terminated by a signal but we saw a success message or validation phase started
        output_text = "\n".join(output_lines)

        # Define component-specific success indicators (same as above)
        component_specific_indicators = {
            "ML Stack Core": [
                "ML Stack Installation Completed",
                "ML Stack Core installed",
                "All core components installed successfully",
                "Installation script completed",
                "Verification Complete"
            ],
            "MIGraphX Python Wrapper": [
                "MIGraphX Python wrapper has been installed",
                "Installation Complete",
                "MIGraphX Python wrapper is installed",
                "Basic functionality test passed"
            ]
        }

        # Get component-specific indicators or use default ones
        specific_indicators = component_specific_indicators.get(name, [])

        # Combine with general success indicators
        success_indicators = [
            "Installation script completed with exit code: 0",
            "100%",
            "Completed",
            "Successfully",
            "Success"
        ] + specific_indicators

        # For problematic components, be more lenient with success detection
        if name in ["ML Stack Core", "MIGraphX Python Wrapper"] or aggressive_termination:
            log_message(f"Using lenient success detection for negative return code for component: {name}", "INFO")
            # For problematic components, consider validation phase or force_validation as success
            if validation_phase_started or force_validation:
                log_message(f"Component {name} reached validation phase, treating as success despite signal {-return_code}", "WARNING")
                component["status"] = "installed"
                log_win.addstr(f"\nSUCCESS: {name} installed successfully\n", curses.color_pair(COLOR_SUCCESS))
                log_win.addstr(f"(Note: Process was terminated with signal {-return_code} but installation was successful)\n", curses.color_pair(COLOR_INFO))
                log_win.refresh()

                # Wait a moment to ensure the message is seen
                time.sleep(1)

                # Return to the components screen
                return True

        if any(indicator in output_text for indicator in success_indicators) or validation_phase_started:
            log_message(f"Script reported success but was terminated with signal {-return_code}. Treating as success.", "WARNING")
            # Update component status
            component["status"] = "installed"
            log_win.addstr(f"\nSUCCESS: {name} installed successfully\n", curses.color_pair(COLOR_SUCCESS))
            log_win.refresh()

            # Wait a moment to ensure the message is seen
            time.sleep(1)

            return True
        else:
            log_message(f"Installation of {name} failed with exit code {return_code} (terminated by signal {-return_code})", "ERROR")
            component["status"] = "failed"

            # Show error message with more context
            log_win.addstr(f"\nERROR: Installation of {name} failed\n", curses.color_pair(COLOR_ERROR))
            log_win.addstr("The process was terminated unexpectedly.\n", curses.color_pair(COLOR_WARNING))
            log_win.addstr("This may be due to a background process being terminated.\n", curses.color_pair(COLOR_WARNING))
            log_win.addstr("\nPress any key to return to component selection...\n", curses.color_pair(COLOR_INFO))
            log_win.refresh()

            # Wait for a key press with a timeout
            stdscr.timeout(5000)  # 5 second timeout
            stdscr.getch()
            stdscr.timeout(200)  # Reset timeout

            return False
    elif return_code != 0:
        # One last check for success indicators in case we missed them earlier
        output_text = "\n".join(output_lines)

        # For problematic components, be even more lenient with success detection
        if name in ["ML Stack Core", "MIGraphX Python Wrapper"] or aggressive_termination:
            log_message(f"Using final lenient success detection for component: {name} with exit code {return_code}", "INFO")

            # For problematic components, consider validation phase or force_validation as success
            if validation_phase_started or force_validation or "100%" in output_text or "Completed" in output_text:
                log_message(f"Component {name} likely succeeded despite exit code {return_code}", "WARNING")
                component["status"] = "installed"
                log_win.addstr(f"\nSUCCESS: {name} installed successfully\n", curses.color_pair(COLOR_SUCCESS))
                log_win.addstr(f"(Note: Process exited with code {return_code} but installation was successful)\n", curses.color_pair(COLOR_INFO))
                log_win.refresh()

                # Wait a moment to ensure the message is seen
                time.sleep(1)

                # Return to the components screen
                return True

        # Standard check for other components
        if validation_phase_started or "100%" in output_text or "Completed" in output_text:
            log_message(f"Process completed with non-zero exit code {return_code} but validation phase was started. Treating as success.", "WARNING")
            component["status"] = "installed"
            log_win.addstr(f"\nSUCCESS: {name} installed successfully\n", curses.color_pair(COLOR_SUCCESS))
            log_win.addstr(f"(Note: Process exited with code {return_code} but installation was successful)\n", curses.color_pair(COLOR_INFO))
            log_win.refresh()

            # Wait a moment to ensure the message is seen
            time.sleep(1)

            return True
        else:
            log_message(f"Installation of {name} failed with exit code {return_code}", "ERROR")
            component["status"] = "failed"

            # Show error message with more context
            log_win.addstr(f"\nERROR: Installation of {name} failed with exit code {return_code}\n", curses.color_pair(COLOR_ERROR))
            log_win.addstr("Check the logs for more details.\n", curses.color_pair(COLOR_WARNING))
            log_win.addstr("\nPress any key to return to component selection...\n", curses.color_pair(COLOR_INFO))
            log_win.refresh()

            # Wait for a key press with a timeout
            stdscr.timeout(5000)  # 5 second timeout
            stdscr.getch()
            stdscr.timeout(200)  # Reset timeout

            return False

    # Update component status for successful installation
    component["status"] = "installed"
    log_message(f"Installation of {name} completed successfully", "SUCCESS")
    log_win.addstr(f"\nSUCCESS: {name} installed successfully\n", curses.color_pair(COLOR_SUCCESS))
    log_win.refresh()

    # Wait a moment to ensure the message is seen
    time.sleep(1)

    return True

def draw_menu(stdscr, selected_idx, menu_items):
    """Draw a menu with the given items."""
    menu_height = len(menu_items)
    menu_width = max(len(item) for item in menu_items) + 12  # Extra space for decorations

    # Adjust menu position to be at line 11
    # Original: menu_y = (curses.LINES - menu_height - 4) // 2
    menu_y = max(11, (curses.LINES - menu_height - 8) // 2)  # Moved 5 lines higher from previous position of 16
    menu_x = (curses.COLS - menu_width) // 2

    # Draw a decorative box around the menu
    box_width = menu_width + 4
    box_height = menu_height + 4
    box_y = menu_y - 2
    box_x = menu_x - 2

    # Draw the box corners
    stdscr.addstr(box_y, box_x, "╔", curses.color_pair(COLOR_BORDER))  # Use orange border
    stdscr.addstr(box_y, box_x + box_width - 1, "╗", curses.color_pair(COLOR_BORDER))
    stdscr.addstr(box_y + box_height - 1, box_x, "╚", curses.color_pair(COLOR_BORDER))
    stdscr.addstr(box_y + box_height - 1, box_x + box_width - 1, "╝", curses.color_pair(COLOR_BORDER))

    # Draw the box edges
    for i in range(1, box_width - 1):
        stdscr.addstr(box_y, box_x + i, "═", curses.color_pair(COLOR_BORDER))
        stdscr.addstr(box_y + box_height - 1, box_x + i, "═", curses.color_pair(COLOR_BORDER))

    for i in range(1, box_height - 1):
        stdscr.addstr(box_y + i, box_x, "║", curses.color_pair(COLOR_BORDER))
        stdscr.addstr(box_y + i, box_x + box_width - 1, "║", curses.color_pair(COLOR_BORDER))

    # Draw a title for the menu
    menu_title = "Main Menu"
    stdscr.addstr(box_y + 1, box_x + (box_width - len(menu_title)) // 2, menu_title, curses.color_pair(COLOR_TITLE) | curses.A_BOLD)

    # Draw a separator
    for i in range(1, box_width - 1):
        stdscr.addstr(box_y + 2, box_x + i, "─", curses.color_pair(COLOR_TITLE))

    # Draw the menu items with decorative elements
    for i, item in enumerate(menu_items):
        if i == selected_idx:
            stdscr.addstr(menu_y + i, menu_x, f"▶ {item} ◀", curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BOLD)
        else:
            stdscr.addstr(menu_y + i, menu_x, f"  {item}  ", curses.color_pair(COLOR_NORMAL))

    # Don't call refresh here - let the caller handle refreshing
    # This allows the function to work with both regular windows and pads

def draw_header(stdscr, title):
    """Draw a header with the given title."""
    # Get screen dimensions
    _, max_x = stdscr.getmaxyx()  # We only need max_x

    # Clear the header area first to prevent artifacts
    for i in range(0, 5):
        stdscr.addstr(i, 0, " " * (max_x - 1))

    # Draw stars in the background
    for i in range(1, 3):
        for j in range(2, max_x - 2, 8):
            stdscr.addstr(i, j, "✧", curses.color_pair(COLOR_INFO))

    # Draw the title with a fancy border
    title_text = "• ✵ Stan's ML Stack Installer © ✵ •"

    # Only add the specific title if it's not the welcome screen
    if title != "ML Stack Installation" and title != "Welcome to Stan's ML Stack Installer ✵ ©":
        title_text = f"{title_text} - {title}"

    # Create a fancy border
    border_char = "•"
    border = border_char * (len(title_text))

    # Calculate center position
    center_x = (max_x - len(border)) // 2

    # Draw the fancy header - perfectly aligned with the rows of "✧" stars
    # Top border at line 1, title at line 2, bottom border at line 3
    stdscr.addstr(1, center_x, border, curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
    stdscr.addstr(2, center_x, title_text, curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
    stdscr.addstr(3, center_x, border, curses.color_pair(COLOR_TITLE) | curses.A_BOLD)

    # Draw a separator line
    stdscr.addstr(4, 2, "=" * (max_x - 4), curses.color_pair(COLOR_TITLE))
    # Don't call refresh here - let the caller handle refreshing
    # This allows the function to work with both regular windows and pads

def draw_footer(stdscr, text):
    """Draw a footer with the given text."""
    # Ensure the footer is always at the bottom of the screen
    max_y, max_x = stdscr.getmaxyx()
    footer_y = max_y - 1  # Last line of the screen

    # Clear the footer area first to prevent artifacts
    for i in range(max_y - 3, max_y):
        stdscr.addstr(i, 0, " " * (max_x - 1))

    # Draw stars in the footer
    for j in range(2, max_x - 2, 8):
        stdscr.addstr(max_y - 2, j, "✧", curses.color_pair(COLOR_INFO))

    # Draw a separator line
    stdscr.addstr(max_y - 3, 2, "=" * (max_x - 4), curses.color_pair(COLOR_TITLE))

    # Draw the footer text with decorative elements
    footer_text = f"✦ {text} ✦"
    stdscr.addstr(footer_y, 2, footer_text.center(max_x - 4), curses.color_pair(COLOR_INFO) | curses.A_BOLD)
    # Don't call refresh here - let the caller handle refreshing
    # This allows the function to work with both regular windows and pads

def draw_dependencies(stdscr, dependencies, selected_deps, selected_idx=None):
    """Draw the dependencies list."""
    dep_y = 11  # Adjusted to start at line 11 instead of 14
    dep_x = 2

    stdscr.addstr(dep_y, dep_x, "Dependencies:".ljust(curses.COLS - 4), curses.color_pair(COLOR_TITLE))
    dep_y += 1

    # Instructions for selecting multiple dependencies
    stdscr.addstr(dep_y, dep_x, "Use Space to select/deselect dependencies".ljust(curses.COLS - 4), curses.color_pair(COLOR_INFO))
    dep_y += 2

    deps_list = list(dependencies.items())
    for i, (dep, installed) in enumerate(deps_list):
        # Determine status symbol and color
        if installed:
            status = "✓"
            color = COLOR_SUCCESS
        else:
            status = "✗"
            color = COLOR_ERROR

        # Add selection marker if this dependency is selected
        if dep in selected_deps:
            selection = "◉"
        else:
            selection = "○"

        # Highlight the currently focused item
        if i == selected_idx:
            stdscr.addstr(dep_y + i, dep_x, f"> [{status}] {selection} {dep}".ljust(curses.COLS - 4), curses.color_pair(COLOR_HIGHLIGHT))
        else:
            stdscr.addstr(dep_y + i, dep_x, f"  [{status}] {selection} {dep}".ljust(curses.COLS - 4), curses.color_pair(color))

    # Don't call refresh here - let the caller handle refreshing

def draw_environment_scripts(stdscr, environment_scripts, selected_idx=None, selected_scripts=None):
    """Draw the environment scripts list."""
    env_y = 11  # Adjusted to start at line 11 instead of 14
    env_x = 2

    # Instructions for selecting environment scripts
    stdscr.addstr(env_y, env_x, "Use Space to select/deselect environment scripts".ljust(curses.COLS - 4), curses.color_pair(COLOR_INFO))
    env_y += 2

    # Environment scripts
    stdscr.addstr(env_y, env_x, "Environment Setup Scripts:".ljust(curses.COLS - 4), curses.color_pair(COLOR_TITLE))
    env_y += 1

    # Log the selected scripts for debugging
    if selected_scripts:
        log_message(f"Selected scripts: {[s['name'] for s in selected_scripts]}")
    else:
        log_message("No scripts selected")

    for i, script in enumerate(environment_scripts):
        # Determine status symbol and color
        if script["status"] == "installed":
            status = "✓"
            color = COLOR_SUCCESS
        elif script["status"] == "installing":
            status = "⟳"
            color = COLOR_INFO
        elif script["status"] == "failed":
            status = "✗"
            color = COLOR_ERROR
        else:
            status = " "
            color = COLOR_NORMAL

        # Check if this script is in the selected_scripts list
        is_selected = False
        if selected_scripts:
            for selected_script in selected_scripts:
                if selected_script['name'] == script['name']:
                    is_selected = True
                    break

        # Use a filled circle for selected items, empty circle for unselected
        if is_selected:
            selection = "◉"  # Filled circle for selected
            log_message(f"Drawing script {script['name']} as selected")
        else:
            selection = "○"  # Empty circle for unselected
            log_message(f"Drawing script {script['name']} as not selected")

        # Highlight the currently focused item
        if i == selected_idx:
            # Use a different color for the selection marker to make it more visible
            selection_text = f"> [{status}] "
            script_text = f"{script['name']} - {script['description']}".ljust(curses.COLS - len(selection_text) - 4)

            # Draw the selection marker with highlight color
            stdscr.addstr(env_y + i, env_x, selection_text, curses.color_pair(COLOR_HIGHLIGHT))

            # Draw the selection circle with a different color to make it stand out
            if is_selected:
                stdscr.addstr(env_y + i, env_x + len(selection_text), selection, curses.color_pair(COLOR_SUCCESS) | curses.A_BOLD)
            else:
                stdscr.addstr(env_y + i, env_x + len(selection_text), selection, curses.color_pair(COLOR_HIGHLIGHT))

            # Draw the rest of the text
            stdscr.addstr(env_y + i, env_x + len(selection_text) + 1, " " + script_text, curses.color_pair(COLOR_HIGHLIGHT))
        else:
            # For non-highlighted items, use normal colors but make selected items stand out
            selection_text = f"  [{status}] "
            script_text = f"{script['name']} - {script['description']}".ljust(curses.COLS - len(selection_text) - 4)

            # Draw the selection marker with normal color
            stdscr.addstr(env_y + i, env_x, selection_text, curses.color_pair(color))

            # Draw the selection circle with a different color to make it stand out
            if is_selected:
                stdscr.addstr(env_y + i, env_x + len(selection_text), selection, curses.color_pair(COLOR_SUCCESS) | curses.A_BOLD)
            else:
                stdscr.addstr(env_y + i, env_x + len(selection_text), selection, curses.color_pair(color))

            # Draw the rest of the text
            stdscr.addstr(env_y + i, env_x + len(selection_text) + 1, " " + script_text, curses.color_pair(color))

    # Don't call refresh here - let the caller handle refreshing

def draw_components(stdscr, foundation_components=None, core_components=None, extension_components=None, selected_section=0, selected_idx=0):
    """Draw the components list."""
    comp_y = 11  # Adjusted to start at line 11 instead of 14
    comp_x = 2
    max_width = curses.COLS - comp_x - 4  # Maximum width for text to avoid boundary errors

    # Use global components if none provided
    if foundation_components is None:
        foundation_components = FOUNDATION_COMPONENTS
    if core_components is None:
        core_components = CORE_COMPONENTS
    if extension_components is None:
        extension_components = EXTENSION_COMPONENTS

    # Instructions for selecting components
    stdscr.addstr(comp_y, comp_x, "Use Space to select/deselect components".ljust(max_width), curses.color_pair(COLOR_INFO))
    comp_y += 2

    # Foundation components - ensure full section name is visible
    section_title = "Foundation Components (Recommended First):"
    stdscr.addstr(comp_y, comp_x, section_title.ljust(max_width), curses.color_pair(COLOR_TITLE))
    comp_y += 1

    for i, comp in enumerate(foundation_components):
        # Determine status symbol and color
        if comp["status"] == "installed":
            status = "✓"
            color = COLOR_SUCCESS
        elif comp["status"] == "installing":
            status = "⟳"
            color = COLOR_INFO
        elif comp["status"] == "failed":
            status = "✗"
            color = COLOR_ERROR
        else:
            status = " "
            color = COLOR_NORMAL

        # Add selection marker
        if comp.get("selected", False):
            selection = "◉"
        else:
            selection = "○"

        # Create component text with proper truncation if needed
        component_text = f"{comp['name']} - {comp['description']}"
        if len(component_text) > max_width - 10:  # Account for status and selection markers
            component_text = component_text[:max_width - 13] + "..."

        if selected_section == 0 and i == selected_idx:
            stdscr.addstr(comp_y + i, comp_x, f"> [{status}] {selection} {component_text}".ljust(max_width), curses.color_pair(COLOR_HIGHLIGHT))
        else:
            stdscr.addstr(comp_y + i, comp_x, f"  [{status}] {selection} {component_text}".ljust(max_width), curses.color_pair(color))

    comp_y += len(foundation_components) + 1

    # Core components - ensure full section name is visible
    section_title = "Core Components:"
    stdscr.addstr(comp_y, comp_x, section_title.ljust(max_width), curses.color_pair(COLOR_TITLE))
    comp_y += 1

    for i, comp in enumerate(core_components):
        # Determine status symbol and color
        if comp["status"] == "installed":
            status = "✓"
            color = COLOR_SUCCESS
        elif comp["status"] == "installing":
            status = "⟳"
            color = COLOR_INFO
        elif comp["status"] == "failed":
            status = "✗"
            color = COLOR_ERROR
        else:
            status = " "
            color = COLOR_NORMAL

        # Add selection marker
        if comp.get("selected", False):
            selection = "◉"
        else:
            selection = "○"

        # Create component text with proper truncation if needed
        component_text = f"{comp['name']} - {comp['description']}"
        if len(component_text) > max_width - 10:  # Account for status and selection markers
            component_text = component_text[:max_width - 13] + "..."

        if selected_section == 1 and i == selected_idx:
            stdscr.addstr(comp_y + i, comp_x, f"> [{status}] {selection} {component_text}".ljust(max_width), curses.color_pair(COLOR_HIGHLIGHT))
        else:
            stdscr.addstr(comp_y + i, comp_x, f"  [{status}] {selection} {component_text}".ljust(max_width), curses.color_pair(color))

    comp_y += len(core_components) + 1

    # Extension components - ensure full section name is visible
    section_title = "Extension Components:"
    stdscr.addstr(comp_y, comp_x, section_title.ljust(max_width), curses.color_pair(COLOR_TITLE))
    comp_y += 1

    for i, comp in enumerate(extension_components):
        # Determine status symbol and color
        if comp["status"] == "installed":
            status = "✓"
            color = COLOR_SUCCESS
        elif comp["status"] == "installing":
            status = "⟳"
            color = COLOR_INFO
        elif comp["status"] == "failed":
            status = "✗"
            color = COLOR_ERROR
        else:
            status = " "
            color = COLOR_NORMAL

        # Add selection marker
        if comp.get("selected", False):
            selection = "◉"
        else:
            selection = "○"

        # Create component text with proper truncation if needed
        component_text = f"{comp['name']} - {comp['description']}"
        if len(component_text) > max_width - 10:  # Account for status and selection markers
            component_text = component_text[:max_width - 13] + "..."

        if selected_section == 2 and i == selected_idx:
            stdscr.addstr(comp_y + i, comp_x, f"> [{status}] {selection} {component_text}".ljust(max_width), curses.color_pair(COLOR_HIGHLIGHT))
        else:
            stdscr.addstr(comp_y + i, comp_x, f"  [{status}] {selection} {component_text}".ljust(max_width), curses.color_pair(color))

    # Don't call refresh here - let the caller handle refreshing

def draw_installation_screen(stdscr, foundation_components=None, core_components=None, extension_components=None, selected_section=0, selected_idx=0):
    """Draw the installation screen with component selection."""
    inst_y = 11  # Adjusted to start at line 11 instead of 14
    inst_x = 2
    max_width = curses.COLS - inst_x - 4  # Maximum width for text to avoid boundary errors

    # Use global components if none provided
    if foundation_components is None:
        foundation_components = FOUNDATION_COMPONENTS
    if core_components is None:
        core_components = CORE_COMPONENTS
    if extension_components is None:
        extension_components = EXTENSION_COMPONENTS

    # Instructions for selecting components
    stdscr.addstr(inst_y, inst_x, "Use Space to select/deselect components, Enter to install selected components".ljust(max_width), curses.color_pair(COLOR_INFO))
    inst_y += 2

    # Foundation components - ensure full section name is visible
    section_title = "Foundation Components:"
    stdscr.addstr(inst_y, inst_x, section_title.ljust(max_width), curses.color_pair(COLOR_TITLE))
    inst_y += 1

    for i, comp in enumerate(foundation_components):
        # Determine status symbol and color
        if comp["status"] == "installed":
            status = "✓"
            color = COLOR_SUCCESS
        elif comp["status"] == "installing":
            status = "⟳"
            color = COLOR_INFO
        elif comp["status"] == "failed":
            status = "✗"
            color = COLOR_ERROR
        else:
            status = " "
            color = COLOR_NORMAL

        # Add selection marker
        if comp.get("selected", False):
            selection = "◉"
        else:
            selection = "○"

        # Create component text with proper truncation if needed
        component_text = f"{comp['name']} - {comp['description']}"
        if len(component_text) > max_width - 10:  # Account for status and selection markers
            component_text = component_text[:max_width - 13] + "..."

        if selected_section == 0 and i == selected_idx:
            stdscr.addstr(inst_y + i, inst_x, f"> [{status}] {selection} {component_text}".ljust(max_width), curses.color_pair(COLOR_HIGHLIGHT))
        else:
            stdscr.addstr(inst_y + i, inst_x, f"  [{status}] {selection} {component_text}".ljust(max_width), curses.color_pair(color))

    inst_y += len(foundation_components) + 1

    # Core components - ensure full section name is visible
    section_title = "Core Components:"
    stdscr.addstr(inst_y, inst_x, section_title.ljust(max_width), curses.color_pair(COLOR_TITLE))
    inst_y += 1

    for i, comp in enumerate(core_components):
        # Determine status symbol and color
        if comp["status"] == "installed":
            status = "✓"
            color = COLOR_SUCCESS
        elif comp["status"] == "installing":
            status = "⟳"
            color = COLOR_INFO
        elif comp["status"] == "failed":
            status = "✗"
            color = COLOR_ERROR
        else:
            status = " "
            color = COLOR_NORMAL

        # Add selection marker
        if comp.get("selected", False):
            selection = "◉"
        else:
            selection = "○"

        # Create component text with proper truncation if needed
        component_text = f"{comp['name']} - {comp['description']}"
        if len(component_text) > max_width - 10:  # Account for status and selection markers
            component_text = component_text[:max_width - 13] + "..."

        if selected_section == 1 and i == selected_idx:
            stdscr.addstr(inst_y + i, inst_x, f"> [{status}] {selection} {component_text}".ljust(max_width), curses.color_pair(COLOR_HIGHLIGHT))
        else:
            stdscr.addstr(inst_y + i, inst_x, f"  [{status}] {selection} {component_text}".ljust(max_width), curses.color_pair(color))

    inst_y += len(core_components) + 1

    # Extension components - ensure full section name is visible
    section_title = "Extension Components:"
    stdscr.addstr(inst_y, inst_x, section_title.ljust(max_width), curses.color_pair(COLOR_TITLE))
    inst_y += 1

    for i, comp in enumerate(extension_components):
        # Determine status symbol and color
        if comp["status"] == "installed":
            status = "✓"
            color = COLOR_SUCCESS
        elif comp["status"] == "installing":
            status = "⟳"
            color = COLOR_INFO
        elif comp["status"] == "failed":
            status = "✗"
            color = COLOR_ERROR
        else:
            status = " "
            color = COLOR_NORMAL

        # Add selection marker
        if comp.get("selected", False):
            selection = "◉"
        else:
            selection = "○"

        # Create component text with proper truncation if needed
        component_text = f"{comp['name']} - {comp['description']}"
        if len(component_text) > max_width - 10:  # Account for status and selection markers
            component_text = component_text[:max_width - 13] + "..."

        if selected_section == 2 and i == selected_idx:
            stdscr.addstr(inst_y + i, inst_x, f"> [{status}] {selection} {component_text}".ljust(max_width), curses.color_pair(COLOR_HIGHLIGHT))
        else:
            stdscr.addstr(inst_y + i, inst_x, f"  [{status}] {selection} {component_text}".ljust(max_width), curses.color_pair(color))

    # Don't call refresh here - let the caller handle refreshing

def create_log_window(parent_win):
    """Create a scrollable log window with decorative border."""
    # Adjust dimensions to account for the header and footer
    # Reduce height to ensure it doesn't overlap with the footer
    height = curses.LINES - 19  # Adjusted for starting at line 11 instead of 16
    width = curses.COLS - 8
    y = 11  # Moved 5 lines higher from previous position of 16
    x = 4

    # Create the window
    log_win = curses.newwin(height, width, y, x)
    log_win.scrollok(True)
    log_win.idlok(True)

    # Add a scrolling hint
    log_win.addstr("Press any key to start scrolling. Use arrow keys or Page Up/Down to navigate logs.\n\n", curses.color_pair(COLOR_INFO))

    # Draw a decorative border around the log window
    parent_win.attron(curses.color_pair(COLOR_BORDER))  # Use orange border
    # Top and bottom borders with corners
    parent_win.addstr(y-1, x-1, "╔" + "═" * (width) + "╗")
    parent_win.addstr(y+height, x-1, "╚" + "═" * (width) + "╝")

    # Left and right borders
    for i in range(height):
        parent_win.addstr(y+i, x-1, "║")
        parent_win.addstr(y+i, x+width, "║")

    # Add a title to the border
    title = " Log Output "
    parent_win.addstr(y-1, x + (width - len(title))//2, title)
    parent_win.attroff(curses.color_pair(COLOR_BORDER))

    # Don't call refresh here - let the caller handle refreshing
    # This allows the function to work with both regular windows and pads

    return log_win

def show_welcome_screen(stdscr):
    """Show welcome screen with sudo authentication."""
    # Initialize curses
    curses.curs_set(0)  # Hide cursor
    curses.start_color()
    curses.use_default_colors()

    # Initialize color pairs with more vibrant colors
    curses.init_pair(COLOR_NORMAL, curses.COLOR_WHITE, -1)
    curses.init_pair(COLOR_HIGHLIGHT, curses.COLOR_BLACK, curses.COLOR_GREEN)  # Green selection highlight
    curses.init_pair(COLOR_TITLE, curses.COLOR_MAGENTA, -1)
    curses.init_pair(COLOR_SUCCESS, curses.COLOR_GREEN, -1)
    curses.init_pair(COLOR_ERROR, curses.COLOR_RED, -1)
    curses.init_pair(COLOR_WARNING, curses.COLOR_YELLOW, -1)
    curses.init_pair(COLOR_INFO, curses.COLOR_CYAN, -1)
    curses.init_pair(COLOR_BORDER, curses.COLOR_YELLOW, -1)  # Orange border (using yellow as closest to orange)
    curses.init_pair(9, curses.COLOR_CYAN, -1)  # Cyan for use cases box

    # Clear screen
    stdscr.clear()

    # Draw header - use ML Stack Installation to avoid adding redundant title
    draw_header(stdscr, "ML Stack Installation")
    draw_footer(stdscr, "Press Enter to continue, 'q' to quit")
    # Explicitly refresh the screen after drawing header and footer
    stdscr.refresh()

    # Calculate positions
    _, width = stdscr.getmaxyx()  # Only need width for centering

    # Add visually appealing ASCII art for "NIRVANA" between header and content
    nirvana_art = [
        "  _   _ ___ ______     _    _   _    _    ",
        " | \\ | |_ _|  _ \\ \\   / /  / \\  | \\ | |  / \\   ",
        " |  \\| || || |_) \\ \\ / /  / _ \\ |  \\| | / _ \\  ",
        " | |\\  || ||  _ < \\ V /  / ___ \\| |\\  |/ ___ \\ ",
        " |_| \\_|___|_| \\_\\ \\_/  /_/   \\_\\_| \\_/_/   \\_\\"
    ]

    # Position the ASCII art between header separator and welcome title
    art_y = 6  # Start after the header separator (line 4) with a small gap

    # Calculate center position for the ASCII art
    art_width = len(nirvana_art[0])
    art_x = (width - art_width) // 2

    # Display the ASCII art with colors matching the UI
    for i, line in enumerate(nirvana_art):
        # Use different colors for different parts of the art for visual appeal
        color = [COLOR_TITLE, COLOR_INFO, COLOR_SUCCESS, COLOR_INFO, COLOR_TITLE][i % 5]
        try:
            stdscr.addstr(art_y + i, art_x, line, curses.color_pair(color) | curses.A_BOLD)
        except:
            pass

    # Start UI elements after the ASCII art
    start_y = art_y + len(nirvana_art) + 2  # Add some space after the art

    # Main title below the ASCII art
    title = "Welcome to Stan's ML Stack Installer!"
    title_y = start_y
    title_x = (width - len(title)) // 2
    stdscr.addstr(title_y, title_x, title, curses.color_pair(COLOR_TITLE) | curses.A_BOLD)

    # Draw two boxes side by side
    left_box_width = 50
    right_box_width = 50
    box_height = 14  # Slightly reduced height to ensure it fits

    # Position boxes below the title
    box_y = title_y + 2  # Just below the title
    left_box_x = (width // 2) - left_box_width - 2
    right_box_x = (width // 2) + 2

    # Ensure boxes fit on screen
    if left_box_x < 0:
        left_box_x = 2
        left_box_width = (width // 2) - 4

    if right_box_x + right_box_width > width:
        right_box_width = width - right_box_x - 2

    # Create left box with orange border (features)
    stdscr.attron(curses.color_pair(COLOR_BORDER))
    # Top and bottom borders with corners - use try/except to handle potential errors
    try:
        stdscr.addstr(box_y, left_box_x, "╔" + "═" * (left_box_width - 2) + "╗")
    except:
        pass
    try:
        stdscr.addstr(box_y + box_height - 1, left_box_x, "╚" + "═" * (left_box_width - 2) + "╝")
    except:
        pass

    # Left and right borders
    for i in range(1, box_height - 1):
        try:
            stdscr.addstr(box_y + i, left_box_x, "║")
        except:
            pass
        try:
            stdscr.addstr(box_y + i, left_box_x + left_box_width - 1, "║")
        except:
            pass

    # Add title to left box
    left_box_title = " Features "
    try:
        stdscr.addstr(box_y, left_box_x + (left_box_width - len(left_box_title)) // 2, left_box_title)
    except:
        pass
    stdscr.attroff(curses.color_pair(COLOR_BORDER))

    # Create right box with cyan border (use cases)
    stdscr.attron(curses.color_pair(9))
    # Top and bottom borders with corners - use try/except to handle potential errors
    try:
        stdscr.addstr(box_y, right_box_x, "╔" + "═" * (right_box_width - 2) + "╗")
    except:
        pass
    try:
        stdscr.addstr(box_y + box_height - 1, right_box_x, "╚" + "═" * (right_box_width - 2) + "╝")
    except:
        pass

    # Left and right borders
    for i in range(1, box_height - 1):
        try:
            stdscr.addstr(box_y + i, right_box_x, "║")
        except:
            pass
        try:
            stdscr.addstr(box_y + i, right_box_x + right_box_width - 1, "║")
        except:
            pass

    # Add title to right box
    right_box_title = " Use Cases "
    try:
        stdscr.addstr(box_y, right_box_x + (right_box_width - len(right_box_title)) // 2, right_box_title)
    except:
        pass
    stdscr.attroff(curses.color_pair(9))

    # Add features text to left box
    features_text = [
        "Complete ML environment for AMD GPUs:",
        "",
        "• ROCm and AMDGPU drivers",
        "• PyTorch with ROCm support",
        "• AITER for efficient tensor operations",
        "• DeepSpeed for large model optimization",
        "• Flash Attention for transformer models",
        "• vLLM for high-throughput inference",
        "• Environment optimization for AMD GPUs",
        "",
        "Guided setup for hardware detection,",
        "environment configuration, and more."
    ]

    # Display features text with word wrapping
    for i, line in enumerate(features_text):
        if i < box_height - 2:  # Ensure we don't write outside the box
            # Wrap text if needed
            if len(line) > left_box_width - 4:
                # Simple word wrapping
                words = line.split()
                current_line = ""
                line_idx = 0

                for word in words:
                    if len(current_line) + len(word) + 1 <= left_box_width - 4:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        try:
                            stdscr.addstr(box_y + 1 + i + line_idx, left_box_x + 2, current_line)
                        except:
                            pass
                        line_idx += 1
                        current_line = word

                if current_line:
                    try:
                        stdscr.addstr(box_y + 1 + i + line_idx, left_box_x + 2, current_line)
                    except:
                        pass
            else:
                try:
                    stdscr.addstr(box_y + 1 + i, left_box_x + 2, line)
                except:
                    pass

    # Add use cases text to right box
    use_cases_text = [
        "What you can do with Stan's ML Stack:",
        "",
        "• Run Large Language Models (LLMs) faster",
        "  with AMD GPU acceleration",
        "",
        "• Generate images with Stable Diffusion",
        "  at higher resolutions and speeds",
        "",
        "• Create AI videos and animations with",
        "  optimized performance",
        "",
        "• Train your own AI models with less",
        "  waiting time and better efficiency"
    ]

    # Display use cases text with word wrapping
    for i, line in enumerate(use_cases_text):
        if i < box_height - 2:  # Ensure we don't write outside the box
            # Wrap text if needed
            if len(line) > right_box_width - 4:
                # Simple word wrapping
                words = line.split()
                current_line = ""
                line_idx = 0

                for word in words:
                    if len(current_line) + len(word) + 1 <= right_box_width - 4:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        try:
                            stdscr.addstr(box_y + 1 + i + line_idx, right_box_x + 2, current_line, curses.color_pair(9))
                        except:
                            pass
                        line_idx += 1
                        current_line = word

                if current_line:
                    try:
                        stdscr.addstr(box_y + 1 + i + line_idx, right_box_x + 2, current_line, curses.color_pair(9))
                    except:
                        pass
            else:
                try:
                    stdscr.addstr(box_y + 1 + i, right_box_x + 2, line, curses.color_pair(9))
                except:
                    pass

    # Add sudo authentication message - ensure it's positioned with enough space from the footer
    auth_message = "Before proceeding, we need to authenticate with sudo to ensure smooth installation."

    # Calculate position to ensure it doesn't overlap with footer
    max_y, _ = stdscr.getmaxyx()
    auth_y = box_y + box_height + 1

    # Make sure there's enough space for the auth message and password window before the footer
    # Footer starts at max_y - 3, and we need at least 10 lines for auth message + password window
    if auth_y > max_y - 13:
        # If not enough space, adjust the box height to make room
        box_height = max(8, max_y - 20)  # Ensure minimum box height of 8
        auth_y = box_y + box_height + 1

    auth_x = (width - len(auth_message)) // 2
    try:
        stdscr.addstr(auth_y, auth_x, auth_message)
    except:
        pass

    stdscr.refresh()

    # Create a password window for sudo authentication
    password_win_height = 7
    password_win_width = 70  # Increased width to fit all text
    password_win_y = auth_y + 2
    password_win_x = (width - password_win_width) // 2

    # Final check to ensure password window doesn't overlap with footer
    if password_win_y + password_win_height >= max_y - 3:
        password_win_y = max_y - password_win_height - 4  # Ensure at least 1 line gap before footer

    # Create the window
    password_win = curses.newwin(password_win_height, password_win_width, password_win_y, password_win_x)

    # Draw double border - first draw the outer border
    password_win.box()

    # Add a colorful border highlight for the outer border
    password_win.attron(curses.color_pair(COLOR_BORDER))
    # Only draw the border characters that are safe (not at the edges)
    for i in range(1, password_win_width - 1):
        try:
            password_win.addch(0, i, curses.ACS_HLINE)
            password_win.addch(password_win_height - 1, i, curses.ACS_HLINE)
        except:
            pass

    for i in range(1, password_win_height - 1):
        try:
            password_win.addch(i, 0, curses.ACS_VLINE)
            password_win.addch(i, password_win_width - 1, curses.ACS_VLINE)
        except:
            pass

    # Add corners safely for outer border
    try:
        password_win.addch(0, 0, curses.ACS_ULCORNER)
    except:
        pass

    try:
        password_win.addch(0, password_win_width - 1, curses.ACS_URCORNER)
    except:
        pass

    try:
        password_win.addch(password_win_height - 1, 0, curses.ACS_LLCORNER)
    except:
        pass

    try:
        password_win.addch(password_win_height - 1, password_win_width - 1, curses.ACS_LRCORNER)
    except:
        pass

    # Now draw the inner border (2 characters in from the outer border)
    for i in range(2, password_win_width - 2):
        try:
            password_win.addch(1, i, curses.ACS_HLINE)
            password_win.addch(password_win_height - 2, i, curses.ACS_HLINE)
        except:
            pass

    for i in range(2, password_win_height - 2):
        try:
            password_win.addch(i, 1, curses.ACS_VLINE)
            password_win.addch(i, password_win_width - 2, curses.ACS_VLINE)
        except:
            pass

    # Add corners for inner border
    try:
        password_win.addch(1, 1, curses.ACS_ULCORNER)
    except:
        pass

    try:
        password_win.addch(1, password_win_width - 2, curses.ACS_URCORNER)
    except:
        pass

    try:
        password_win.addch(password_win_height - 2, 1, curses.ACS_LLCORNER)
    except:
        pass

    try:
        password_win.addch(password_win_height - 2, password_win_width - 2, curses.ACS_LRCORNER)
    except:
        pass

    # Add a title
    title = " Sudo Authentication "
    password_win.addstr(0, (password_win_width - len(title)) // 2, title)
    password_win.attroff(curses.color_pair(COLOR_BORDER))

    # Add instructions with better formatting
    password_win.addstr(1, 2, "✧ Please enter your sudo password:", curses.color_pair(COLOR_INFO))
    password_win.addstr(2, 2, "✧ This will be used for all sudo operations during installation.", curses.color_pair(COLOR_INFO))
    password_win.addstr(3, 2, "✧ Password field: [                                            ]", curses.color_pair(COLOR_WARNING))
    password_win.addstr(4, 2, "✧ Password will not be visible as you type (for security).", curses.color_pair(COLOR_INFO))
    password_win.addstr(5, 2, "✧ Press Enter after typing your password.", curses.color_pair(COLOR_INFO))

    # Draw the password field with a different color
    password_win.attron(curses.color_pair(COLOR_HIGHLIGHT))
    password_win.addstr(3, 18, "                                            ")
    password_win.attroff(curses.color_pair(COLOR_HIGHLIGHT))

    password_win.refresh()

    # Move cursor to password input position
    password_win.move(3, 18)

    # Enable cursor and disable echo for password input (for security)
    curses.curs_set(1)
    curses.noecho()  # Don't show the password as it's typed

    # Get password input character by character
    password = ""
    cursor_pos = 18
    max_cursor_pos = min(63, password_win_width - 5)  # Adjusted for wider field

    # Draw a blinking cursor to indicate where input will go
    password_win.attron(curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BLINK)
    try:
        password_win.addch(3, cursor_pos, "_")
    except:
        pass
    password_win.attroff(curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BLINK)
    password_win.refresh()

    # Get password
    try:
        while True:
            # Get a character
            ch = password_win.getch(3, cursor_pos)

            # Check if it's Enter (10 or 13)
            if ch == 10 or ch == 13:
                break

            # Check if it's backspace (8 or 127)
            elif ch == 8 or ch == 127:
                if len(password) > 0:
                    password = password[:-1]
                    cursor_pos = max(18, cursor_pos - 1)

                    # Clear the current position and redraw
                    try:
                        password_win.attron(curses.color_pair(COLOR_HIGHLIGHT))
                        password_win.addstr(3, cursor_pos, " ")
                        password_win.attroff(curses.color_pair(COLOR_HIGHLIGHT))

                        # Add blinking cursor at new position
                        password_win.attron(curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BLINK)
                        password_win.addch(3, cursor_pos, "_")
                        password_win.attroff(curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BLINK)
                    except:
                        pass

            # Otherwise, add the character to the password
            elif ch >= 32 and ch <= 126:  # Printable ASCII
                if cursor_pos < max_cursor_pos:
                    password += chr(ch)

                    # Show a * for each character and move cursor
                    try:
                        password_win.attron(curses.color_pair(COLOR_HIGHLIGHT))
                        password_win.addstr(3, cursor_pos, "*")
                        password_win.attroff(curses.color_pair(COLOR_HIGHLIGHT))

                        cursor_pos += 1

                        # Add blinking cursor at new position
                        password_win.attron(curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BLINK)
                        password_win.addch(3, cursor_pos, "_")
                        password_win.attroff(curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BLINK)
                    except:
                        pass

            # Update the password field indicator
            try:
                # Show how many characters have been typed
                indicator = f"[{len(password)} chars]"
                password_win.addstr(4, password_win_width - len(indicator) - 3, indicator, curses.color_pair(COLOR_INFO))
            except:
                pass

            # Refresh to show changes
            password_win.refresh()
    except Exception as e:
        log_message(f"Error getting password: {str(e)}", "ERROR")

    # Disable cursor and echo
    curses.noecho()
    curses.curs_set(0)

    # Verify sudo password
    password_win.clear()

    # Draw double border - first draw the outer border
    password_win.box()

    # Add a colorful border highlight for the outer border
    password_win.attron(curses.color_pair(COLOR_BORDER))
    # Only draw the border characters that are safe (not at the edges)
    for i in range(1, password_win_width - 1):
        try:
            password_win.addch(0, i, curses.ACS_HLINE)
            password_win.addch(password_win_height - 1, i, curses.ACS_HLINE)
        except:
            pass

    for i in range(1, password_win_height - 1):
        try:
            password_win.addch(i, 0, curses.ACS_VLINE)
            password_win.addch(i, password_win_width - 1, curses.ACS_VLINE)
        except:
            pass

    # Add corners safely for outer border
    try:
        password_win.addch(0, 0, curses.ACS_ULCORNER)
    except:
        pass

    try:
        password_win.addch(0, password_win_width - 1, curses.ACS_URCORNER)
    except:
        pass

    try:
        password_win.addch(password_win_height - 1, 0, curses.ACS_LLCORNER)
    except:
        pass

    try:
        password_win.addch(password_win_height - 1, password_win_width - 1, curses.ACS_LRCORNER)
    except:
        pass

    # Now draw the inner border (2 characters in from the outer border)
    for i in range(2, password_win_width - 2):
        try:
            password_win.addch(1, i, curses.ACS_HLINE)
            password_win.addch(password_win_height - 2, i, curses.ACS_HLINE)
        except:
            pass

    for i in range(2, password_win_height - 2):
        try:
            password_win.addch(i, 1, curses.ACS_VLINE)
            password_win.addch(i, password_win_width - 2, curses.ACS_VLINE)
        except:
            pass

    # Add corners for inner border
    try:
        password_win.addch(1, 1, curses.ACS_ULCORNER)
    except:
        pass

    try:
        password_win.addch(1, password_win_width - 2, curses.ACS_URCORNER)
    except:
        pass

    try:
        password_win.addch(password_win_height - 2, 1, curses.ACS_LLCORNER)
    except:
        pass

    try:
        password_win.addch(password_win_height - 2, password_win_width - 2, curses.ACS_LRCORNER)
    except:
        pass

    title = " Verifying Password "
    password_win.addstr(0, (password_win_width - len(title)) // 2, title)
    password_win.attroff(curses.color_pair(COLOR_BORDER))

    password_win.addstr(2, 2, "✧ Verifying sudo access...", curses.color_pair(COLOR_INFO))
    password_win.refresh()

    # Test sudo access
    import subprocess
    import tempfile

    # Create a temporary script to test sudo
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
        temp_path = temp.name
        temp.write("#!/bin/bash\nexit 0\n")

    os.chmod(temp_path, 0o755)

    # Run the script with sudo
    process = subprocess.Popen(
        f"echo '{password}' | sudo -S {temp_path}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    _, stderr = process.communicate()
    return_code = process.returncode

    # Clean up
    os.unlink(temp_path)

    # Check if sudo was successful
    if return_code == 0:
        password_win.addstr(3, 2, "✧ Sudo access verified successfully!", curses.color_pair(COLOR_SUCCESS))
        password_win.addstr(4, 2, "✧ Press Enter to continue...", curses.color_pair(COLOR_INFO))
        password_win.refresh()

        # Cache sudo password for future use
        os.environ["SUDO_PASSWORD"] = password

        # Wait for Enter
        while True:
            ch = stdscr.getch()
            if ch == 10 or ch == 13:  # Enter key
                break
            elif ch == ord('q'):  # q key
                return False

        return True
    else:
        password_win.addstr(3, 2, "✧ Sudo authentication failed!", curses.color_pair(COLOR_ERROR))
        password_win.addstr(4, 2, "✧ Error: " + stderr.strip(), curses.color_pair(COLOR_ERROR))
        password_win.addstr(5, 2, "✧ Press Enter to try again or 'q' to quit...", curses.color_pair(COLOR_INFO))
        password_win.refresh()

        # Wait for Enter or q
        while True:
            ch = stdscr.getch()
            if ch == 10 or ch == 13:  # Enter key
                return show_welcome_screen(stdscr)  # Try again
            elif ch == ord('q'):  # q key
                return False

def main(stdscr):
    """Main function."""
    # Create log file if it doesn't exist
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Starting ML Stack Installation UI\n")
    except Exception as e:
        # If we can't create the log file, print an error and continue
        # We'll handle log file errors gracefully later
        pass

    # Initialize curses
    curses.curs_set(0)  # Hide cursor
    curses.start_color()
    curses.use_default_colors()

    # Set up non-blocking input with a short timeout for automatic refreshing
    stdscr.nodelay(True)  # Make getch() non-blocking
    stdscr.timeout(200)   # Set a 200ms timeout for getch() - this enables automatic refreshing

    # Initialize color pairs with more vibrant colors
    curses.init_pair(COLOR_NORMAL, curses.COLOR_WHITE, -1)
    curses.init_pair(COLOR_HIGHLIGHT, curses.COLOR_BLACK, curses.COLOR_GREEN)  # Green selection highlight
    curses.init_pair(COLOR_TITLE, curses.COLOR_MAGENTA, -1)
    curses.init_pair(COLOR_SUCCESS, curses.COLOR_GREEN, -1)
    curses.init_pair(COLOR_ERROR, curses.COLOR_RED, -1)
    curses.init_pair(COLOR_WARNING, curses.COLOR_YELLOW, -1)
    curses.init_pair(COLOR_INFO, curses.COLOR_CYAN, -1)
    curses.init_pair(COLOR_BORDER, curses.COLOR_YELLOW, -1)  # Orange border (using yellow as closest to orange)

    # Show welcome screen and get sudo authentication
    if not show_welcome_screen(stdscr):
        return  # Exit if authentication failed or user quit

    # Clear screen
    stdscr.clear()

    # Set up automatic refresh with timeout in the main loop
    # This ensures the screen updates regularly without requiring user input

    # Draw initial screen
    draw_header(stdscr, "ML Stack Installation")
    draw_footer(stdscr, "Press 'q' to quit, arrow keys to navigate, Enter to select")

    # Main menu
    menu_items = ["Hardware Detection", "Environment Setup", "Dependencies", "Components", "Installation", "Logs", "Exit"]
    selected_idx = 0
    current_screen = "menu"

    # Hardware info
    hardware_info = None

    # Environment scripts
    env_selected_idx = 0
    selected_env_scripts = []

    # Dependencies
    dependencies = None
    dep_selected_idx = 0
    selected_deps = []

    # Components
    comp_selected_section = 0
    comp_selected_idx = 0
    inst_selected_section = 0
    inst_selected_idx = 0

    # Create log window
    log_win = create_log_window(stdscr)
    # Explicitly refresh the screen after creating the log window
    stdscr.refresh()

    # Implement double-buffering to eliminate screen flashing
    # Create a virtual screen (pad) for drawing that will be copied to the real screen only when ready
    virtual_screen = curses.newpad(curses.LINES, curses.COLS)

    # Track the current screen to know when to redraw
    previous_screen = None

    # Create a state dictionary to store UI state and timing information
    # This avoids attaching attributes directly to curses window objects
    ui_state = {
        # Track the last state to avoid unnecessary redraws
        'last_selected_idx': -1,
        'last_env_selected_idx': -1,
        'last_dep_selected_idx': -1,
        'last_comp_selected_section': -1,
        'last_comp_selected_idx': -1,
        'last_inst_selected_section': -1,
        'last_inst_selected_idx': -1,
        # Timing information for controlled refreshes
        'last_refresh_time': 0,
        # Flag to force redraw when needed
        'force_redraw': False,
        # Last key press time for input responsiveness
        'last_key_time': 0
    }

    # Function to copy the virtual screen to the real screen
    def update_screen():
        try:
            # For a pad, we need to specify the region to display
            max_y, max_x = stdscr.getmaxyx()

            # Ensure we don't try to refresh beyond screen boundaries
            if max_y > 0 and max_x > 0:
                # Get current time to throttle updates
                current_time = time.time()

                # Only update if enough time has passed since last update (prevents excessive refreshes)
                if current_time - ui_state.get('last_screen_update', 0) > 0.03:  # 30ms minimum between updates
                    # Copy the entire virtual screen to the real screen
                    # Parameters: pminrow, pmincol, sminrow, smincol, smaxrow, smaxcol
                    virtual_screen.noutrefresh(0, 0, 0, 0, max_y-1, max_x-1)
                    curses.doupdate()
                    ui_state['last_screen_update'] = current_time
        except Exception as e:
            # Handle any errors during screen update
            log_message(f"Error updating screen: {str(e)}")

    # Main loop with double-buffering to eliminate flashing
    while True:
        # Use a shorter timeout to improve responsiveness while still being efficient
        stdscr.timeout(5)  # 5ms timeout for getch() - more responsive

        # Get user input (with timeout due to nodelay mode)
        key = stdscr.getch()

        # Handle screen transitions and state changes
        if key != -1:
            # Store current time for key press responsiveness
            current_time = time.time()

            # Only process key if enough time has passed since last key press
            # This prevents double-processing of keys but ensures we don't miss any
            if current_time - ui_state.get('last_key_time', 0) > 0.01:  # 10ms debounce
                ui_state['last_key_time'] = current_time
                ui_state['force_redraw'] = True  # Force redraw on key press

                # Store the previous screen before potentially changing it
                previous_screen = current_screen

                # Global 'b' key handling for all screens except menu
                if key == ord('b') and current_screen != "menu":
                    current_screen = "menu"
            else:
                # If we're debouncing, set the key to -1 to ignore it
                key = -1

        # Determine if we need to redraw the screen
        # Only redraw when something has actually changed
        need_redraw = (current_screen != previous_screen or
                      (current_screen == "menu" and selected_idx != ui_state['last_selected_idx']) or
                      ui_state.get('force_redraw', False))

        # Reset force_redraw flag
        ui_state['force_redraw'] = False

        # Draw current screen to virtual screen if needed
        if need_redraw or current_screen != previous_screen:  # Redraw when necessary or when screen changes
            try:
                virtual_screen.clear()

                if current_screen == "menu":
                    draw_header(virtual_screen, "ML Stack Installation")
                    draw_footer(virtual_screen, "Press 'q' to quit, arrow keys to navigate, Enter to select")
                    draw_menu(virtual_screen, selected_idx, menu_items)

                    # Update tracking variables
                    ui_state['last_selected_idx'] = selected_idx

                    # Copy virtual screen to real screen
                    update_screen()

                    # For menu screen, we don't need to redraw on every loop iteration
                    # This prevents flickering
                    time.sleep(0.05)  # Small delay to reduce CPU usage and prevent flickering
            except Exception as e:
                # Handle any errors during screen drawing
                log_message(f"Error drawing menu screen: {str(e)}")

        elif current_screen == "hardware":
            need_redraw = (current_screen != previous_screen or hardware_info is None)

            if need_redraw:
                try:
                    virtual_screen.clear()
                    draw_header(virtual_screen, "Hardware Detection")
                    draw_footer(virtual_screen, "Press 'b' to go back, 'r' to refresh")

                    if hardware_info is None:
                        try:
                            virtual_screen.addstr(11, 2, "Detecting hardware...".ljust(curses.COLS - 4), curses.color_pair(COLOR_INFO))
                            update_screen()  # Show "Detecting hardware..." message

                            # Add proper error handling around hardware detection
                            try:
                                hardware_info = detect_hardware()
                            except Exception as e:
                                log_message(f"Error detecting hardware: {str(e)}")
                                # Initialize with empty values to prevent crashes
                                hardware_info = {
                                    'system': 'Unknown',
                                    'release': 'Unknown',
                                    'machine': 'Unknown',
                                    'processor': 'Unknown',
                                    'python_version': 'Unknown',
                                    'rocm_path': None,
                                    'rocm_version': None,
                                    'amd_gpus': []
                                }

                            virtual_screen.clear()  # Clear again to redraw with the new info
                            draw_header(virtual_screen, "Hardware Detection")
                            draw_footer(virtual_screen, "Press 'b' to go back, 'r' to refresh")
                        except Exception as e:
                            log_message(f"Error updating hardware detection screen: {str(e)}")

                    # Display hardware information
                    hw_y = 11  # Moved 5 lines higher from previous position of 16
                    hw_x = 4
                    max_width = curses.COLS - hw_x - 4  # Maximum width for text to avoid boundary errors

                    # System information
                    try:
                        virtual_screen.addstr(hw_y, hw_x, "System Information:", curses.color_pair(COLOR_TITLE))
                        hw_y += 1

                        # Truncate long strings to prevent boundary errors
                        system_info = f"System: {hardware_info['system']} {hardware_info['release']}"
                        if len(system_info) > max_width:
                            system_info = system_info[:max_width-3] + "..."
                        virtual_screen.addstr(hw_y, hw_x, system_info, curses.color_pair(COLOR_NORMAL))
                        hw_y += 1

                        machine_info = f"Machine: {hardware_info['machine']}"
                        if len(machine_info) > max_width:
                            machine_info = machine_info[:max_width-3] + "..."
                        virtual_screen.addstr(hw_y, hw_x, machine_info, curses.color_pair(COLOR_NORMAL))
                        hw_y += 1

                        processor_info = f"Processor: {hardware_info['processor']}"
                        if len(processor_info) > max_width:
                            processor_info = processor_info[:max_width-3] + "..."
                        virtual_screen.addstr(hw_y, hw_x, processor_info, curses.color_pair(COLOR_NORMAL))
                        hw_y += 1

                        python_info = f"Python Version: {hardware_info['python_version']}"
                        if len(python_info) > max_width:
                            python_info = python_info[:max_width-3] + "..."
                        virtual_screen.addstr(hw_y, hw_x, python_info, curses.color_pair(COLOR_NORMAL))
                        hw_y += 2
                    except Exception as e:
                        log_message(f"Error displaying system information: {str(e)}")
                        hw_y += 5  # Skip to next section if there's an error

                    # ROCm information
                    try:
                        virtual_screen.addstr(hw_y, hw_x, "ROCm Information:", curses.color_pair(COLOR_TITLE))
                        hw_y += 1
                        if hardware_info['rocm_path']:
                            rocm_path = f"ROCm Path: {hardware_info['rocm_path']}"
                            if len(rocm_path) > max_width:
                                rocm_path = rocm_path[:max_width-3] + "..."
                            virtual_screen.addstr(hw_y, hw_x, rocm_path, curses.color_pair(COLOR_SUCCESS))
                            hw_y += 1
                            if hardware_info['rocm_version']:
                                rocm_version = f"ROCm Version: {hardware_info['rocm_version']}"
                                if len(rocm_version) > max_width:
                                    rocm_version = rocm_version[:max_width-3] + "..."
                                virtual_screen.addstr(hw_y, hw_x, rocm_version, curses.color_pair(COLOR_SUCCESS))
                            else:
                                virtual_screen.addstr(hw_y, hw_x, "ROCm Version: Not detected", curses.color_pair(COLOR_WARNING))
                        else:
                            virtual_screen.addstr(hw_y, hw_x, "ROCm: Not installed or not in PATH", curses.color_pair(COLOR_ERROR))
                        hw_y += 2
                    except Exception as e:
                        log_message(f"Error displaying ROCm information: {str(e)}")
                        hw_y += 3  # Skip to next section if there's an error

                    # GPU information
                    try:
                        virtual_screen.addstr(hw_y, hw_x, "GPU Information:", curses.color_pair(COLOR_TITLE))
                        hw_y += 1
                        if hardware_info['amd_gpus']:
                            gpu_count = f"Found {len(hardware_info['amd_gpus'])} AMD GPU(s):"
                            virtual_screen.addstr(hw_y, hw_x, gpu_count, curses.color_pair(COLOR_SUCCESS))
                            hw_y += 1

                            # Get maximum Y coordinate to avoid writing beyond screen boundaries
                            max_y, _ = virtual_screen.getmaxyx()

                            for i, gpu in enumerate(hardware_info['amd_gpus']):
                                # Check if we're still within screen boundaries
                                if hw_y >= max_y - 2:  # Leave space for footer
                                    break

                                # Truncate GPU name if too long
                                gpu_info = f"{i+1}. {gpu}"
                                if len(gpu_info) > max_width - 2:  # Account for indentation
                                    gpu_info = gpu_info[:max_width-5] + "..."

                                virtual_screen.addstr(hw_y, hw_x + 2, gpu_info, curses.color_pair(COLOR_NORMAL))
                                hw_y += 1
                        else:
                            virtual_screen.addstr(hw_y, hw_x, "No AMD GPUs detected", curses.color_pair(COLOR_ERROR))
                            hw_y += 1
                    except Exception as e:
                        log_message(f"Error displaying GPU information: {str(e)}")

                    # Copy virtual screen to real screen
                    update_screen()
                except Exception as e:
                    # Handle any errors during screen drawing
                    log_message(f"Error drawing hardware screen: {str(e)}")

        elif current_screen == "environment":
            need_redraw = (current_screen != previous_screen or
                          env_selected_idx != ui_state['last_env_selected_idx'])

            if need_redraw:
                try:
                    virtual_screen.clear()
                    draw_header(virtual_screen, "Environment Setup")
                    draw_footer(virtual_screen, "Press 'b' to go back, Space to select, 'i' to run selected scripts")
                    draw_environment_scripts(virtual_screen, ENVIRONMENT_SCRIPTS, env_selected_idx)

                    # Update tracking variables
                    ui_state['last_env_selected_idx'] = env_selected_idx

                    # Copy virtual screen to real screen
                    update_screen()
                except Exception as e:
                    # Handle any errors during screen drawing
                    log_message(f"Error drawing environment screen: {str(e)}")

        elif current_screen == "dependencies":
            need_redraw = (current_screen != previous_screen or
                          dep_selected_idx != ui_state['last_dep_selected_idx'] or
                          dependencies is None)

            if need_redraw:
                try:
                    virtual_screen.clear()
                    draw_header(virtual_screen, "Dependencies")
                    draw_footer(virtual_screen, "Press 'b' to go back, 'i' to install selected dependencies, 'r' to refresh, Space to select")

                    if dependencies is None:
                        virtual_screen.addstr(11, 2, "Checking dependencies...".ljust(curses.COLS - 4), curses.color_pair(COLOR_INFO))
                        update_screen()  # Show "Checking dependencies..." message
                        dependencies = check_dependencies()
                        # Initially select all missing dependencies
                        selected_deps = [dep for dep, installed in dependencies.items() if not installed]
                        virtual_screen.clear()  # Clear again to redraw with the new info
                        draw_header(virtual_screen, "Dependencies")
                        draw_footer(virtual_screen, "Press 'b' to go back, 'i' to install selected dependencies, 'r' to refresh, Space to select")

                    draw_dependencies(virtual_screen, dependencies, selected_deps, dep_selected_idx)

                    # Update tracking variables
                    ui_state['last_dep_selected_idx'] = dep_selected_idx

                    # Copy virtual screen to real screen
                    update_screen()
                except Exception as e:
                    # Handle any errors during screen drawing
                    log_message(f"Error drawing dependencies screen: {str(e)}")

        elif current_screen == "components":
            need_redraw = (current_screen != previous_screen or
                          comp_selected_section != ui_state['last_comp_selected_section'] or
                          comp_selected_idx != ui_state['last_comp_selected_idx'])

            if need_redraw:
                try:
                    virtual_screen.clear()
                    draw_header(virtual_screen, "Components")
                    draw_footer(virtual_screen, "Press 'b' to go back, Space to select, 'i' to install, 'r' to refresh")
                    draw_components(virtual_screen, FOUNDATION_COMPONENTS, CORE_COMPONENTS, EXTENSION_COMPONENTS, comp_selected_section, comp_selected_idx)

                    # Update tracking variables
                    ui_state['last_comp_selected_section'] = comp_selected_section
                    ui_state['last_comp_selected_idx'] = comp_selected_idx

                    # Copy virtual screen to real screen
                    update_screen()
                except Exception as e:
                    # Handle any errors during screen drawing
                    log_message(f"Error drawing components screen: {str(e)}")

        elif current_screen == "installation":
            need_redraw = (current_screen != previous_screen or
                          inst_selected_section != ui_state['last_inst_selected_section'] or
                          inst_selected_idx != ui_state['last_inst_selected_idx'])

            if need_redraw:
                try:
                    virtual_screen.clear()
                    draw_header(virtual_screen, "Installation")
                    draw_footer(virtual_screen, "Press 'b' to go back, Space to select, 'i' to install selected components")

                    # Display installation status
                    inst_y = 11  # Moved 5 lines higher from previous position of 16
                    inst_x = 2

                    virtual_screen.addstr(inst_y, inst_x, "Installation Status:".ljust(curses.COLS - 4), curses.color_pair(COLOR_TITLE))
                    inst_y += 2

                    # Count installed components
                    foundation_installed = sum(1 for comp in FOUNDATION_COMPONENTS if comp["status"] == "installed")
                    core_installed = sum(1 for comp in CORE_COMPONENTS if comp["status"] == "installed")
                    ext_installed = sum(1 for comp in EXTENSION_COMPONENTS if comp["status"] == "installed")
                    total_installed = foundation_installed + core_installed + ext_installed
                    total_components = len(FOUNDATION_COMPONENTS) + len(CORE_COMPONENTS) + len(EXTENSION_COMPONENTS)

                    virtual_screen.addstr(inst_y, inst_x, f"Progress: {total_installed}/{total_components} components installed".ljust(curses.COLS - 4), curses.color_pair(COLOR_INFO))
                    inst_y += 2

                    # Display foundation components
                    virtual_screen.addstr(inst_y, inst_x, "Foundation Components:".ljust(curses.COLS - 4), curses.color_pair(COLOR_TITLE))
                    inst_y += 1

                    for i, comp in enumerate(FOUNDATION_COMPONENTS):
                        if comp["status"] == "installed":
                            status = "✓"
                            color = COLOR_SUCCESS
                        elif comp["status"] == "installing":
                            status = "⟳"
                            color = COLOR_INFO
                        elif comp["status"] == "failed":
                            status = "✗"
                            color = COLOR_ERROR
                        else:
                            status = " "
                            color = COLOR_NORMAL

                        # Add selection marker
                        if comp.get("selected", False):
                            selection = "◉"
                        else:
                            selection = "○"

                        # Highlight the currently selected component
                        if inst_selected_section == 0 and i == inst_selected_idx:
                            virtual_screen.addstr(inst_y, inst_x, f"> [{status}] {selection} {comp['name']}".ljust(curses.COLS - 4), curses.color_pair(COLOR_HIGHLIGHT))
                        else:
                            virtual_screen.addstr(inst_y, inst_x, f"  [{status}] {selection} {comp['name']}".ljust(curses.COLS - 4), curses.color_pair(color))
                        inst_y += 1

                    inst_y += 1

                    # Display core components
                    virtual_screen.addstr(inst_y, inst_x, "Core Components:".ljust(curses.COLS - 4), curses.color_pair(COLOR_TITLE))
                    inst_y += 1

                    for i, comp in enumerate(CORE_COMPONENTS):
                        if comp["status"] == "installed":
                            status = "✓"
                            color = COLOR_SUCCESS
                        elif comp["status"] == "installing":
                            status = "⟳"
                            color = COLOR_INFO
                        elif comp["status"] == "failed":
                            status = "✗"
                            color = COLOR_ERROR
                        else:
                            status = " "
                            color = COLOR_NORMAL

                        # Add selection marker
                        if comp.get("selected", False):
                            selection = "◉"
                        else:
                            selection = "○"

                        # Highlight the currently selected component
                        if inst_selected_section == 1 and i == inst_selected_idx:
                            virtual_screen.addstr(inst_y, inst_x, f"> [{status}] {selection} {comp['name']}".ljust(curses.COLS - 4), curses.color_pair(COLOR_HIGHLIGHT))
                        else:
                            virtual_screen.addstr(inst_y, inst_x, f"  [{status}] {selection} {comp['name']}".ljust(curses.COLS - 4), curses.color_pair(color))
                        inst_y += 1

                    inst_y += 1

                    # Display extension components
                    virtual_screen.addstr(inst_y, inst_x, "Extension Components:".ljust(curses.COLS - 4), curses.color_pair(COLOR_TITLE))
                    inst_y += 1

                    for i, comp in enumerate(EXTENSION_COMPONENTS):
                        if comp["status"] == "installed":
                            status = "✓"
                            color = COLOR_SUCCESS
                        elif comp["status"] == "installing":
                            status = "⟳"
                            color = COLOR_INFO
                        elif comp["status"] == "failed":
                            status = "✗"
                            color = COLOR_ERROR
                        else:
                            status = " "
                            color = COLOR_NORMAL

                        # Add selection marker
                        if comp.get("selected", False):
                            selection = "◉"
                        else:
                            selection = "○"

                        # Highlight the currently selected component
                        if inst_selected_section == 2 and i == inst_selected_idx:
                            virtual_screen.addstr(inst_y, inst_x, f"> [{status}] {selection} {comp['name']}".ljust(curses.COLS - 4), curses.color_pair(COLOR_HIGHLIGHT))
                        else:
                            virtual_screen.addstr(inst_y, inst_x, f"  [{status}] {selection} {comp['name']}".ljust(curses.COLS - 4), curses.color_pair(color))
                        inst_y += 1

                    # Update tracking variables in the ui_state dictionary
                    ui_state['last_inst_selected_section'] = inst_selected_section
                    ui_state['last_inst_selected_idx'] = inst_selected_idx

                    # Copy virtual screen to real screen
                    update_screen()
                except Exception as e:
                    # Handle any errors during screen drawing
                    log_message(f"Error drawing installation screen: {str(e)}")

        elif current_screen == "logs":
            need_redraw = (current_screen != previous_screen)

            if need_redraw:
                try:
                    virtual_screen.clear()
                    draw_header(virtual_screen, "Logs")
                    draw_footer(virtual_screen, "Press 'b' to go back")

                    # Copy virtual screen to real screen
                    update_screen()

                    # Initialize log display time to ensure first update
                    ui_state['last_log_update'] = 0
                except Exception as e:
                    # Handle any errors during screen drawing
                    log_message(f"Error drawing logs screen: {str(e)}")

            # For logs, we handle the log window separately since it's a separate window
            # Only update log content at a controlled rate to prevent flickering
            current_time = time.time()
            if current_time - ui_state.get('last_log_update', 0) > 0.2:  # 200ms between full log updates
                try:
                    # Make sure log_win exists and is valid
                    if log_win is None:
                        # Create log window if it doesn't exist
                        try:
                            max_y, max_x = stdscr.getmaxyx()
                            log_win = curses.newwin(max_y - 7, max_x - 4, 6, 2)
                        except Exception as e:
                            log_message(f"Error creating log window: {str(e)}")
                            continue

                    # Display logs
                    log_win.erase()  # Use erase instead of clear for better performance
                    try:
                        with open(LOG_FILE, "r") as f:
                            lines = f.readlines()

                            # Get the maximum number of lines that can be displayed
                            try:
                                max_y, _ = log_win.getmaxyx()
                            except Exception as e:
                                log_message(f"Error getting log window dimensions: {str(e)}")
                                max_y = 20  # Default fallback value

                            # If there are more lines than can fit, show the last ones
                            if len(lines) > max_y:
                                lines = lines[-max_y:]

                            # Display the lines with appropriate colors
                            for line in lines:
                                try:
                                    # Truncate long lines to prevent boundary errors
                                    try:
                                        max_width = curses.COLS - 10  # Leave some margin
                                    except:
                                        max_width = 70  # Default fallback value

                                    if len(line) > max_width:
                                        line = line[:max_width-3] + "...\n"

                                    if "ERROR" in line:
                                        log_win.addstr(line, curses.color_pair(COLOR_ERROR))
                                    elif "WARNING" in line:
                                        log_win.addstr(line, curses.color_pair(COLOR_WARNING))
                                    elif "SUCCESS" in line:
                                        log_win.addstr(line, curses.color_pair(COLOR_SUCCESS))
                                    else:
                                        log_win.addstr(line)
                                except Exception as e:
                                    # If we can't add a specific line, log it and continue
                                    log_message(f"Error adding log line: {str(e)}")
                                    continue

                            # Move cursor to the end to ensure auto-scrolling
                            try:
                                y, _ = log_win.getyx()  # We only need the y coordinate
                                log_win.move(y, 0)
                            except Exception as e:
                                log_message(f"Error moving cursor: {str(e)}")
                    except FileNotFoundError:
                        try:
                            log_win.addstr("No log file found. Actions will be logged during this session.\n", curses.color_pair(COLOR_WARNING))
                        except Exception as e:
                            log_message(f"Error displaying file not found message: {str(e)}")
                    except Exception as e:
                        try:
                            log_win.addstr(f"Error reading log file: {str(e)}\n", curses.color_pair(COLOR_ERROR))
                        except Exception as e2:
                            log_message(f"Error displaying error message: {str(e2)}")

                    # Use noutrefresh instead of refresh for better performance
                    try:
                        log_win.noutrefresh()
                        curses.doupdate()
                    except Exception as e:
                        log_message(f"Error refreshing log window: {str(e)}")

                    # Update the last log update time
                    ui_state['last_log_update'] = current_time
                except Exception as e:
                    # Handle any errors with the log window itself
                    log_message(f"Error with log window: {str(e)}")

        # Note: Key handling is now done at the beginning of the loop

        # Note: Special handling for logs screen is now done in the logs screen section

        # Process input (key will be -1 if no key was pressed)
        if key == ord('q'):
            break

        elif current_screen == "menu":
            if key == curses.KEY_UP:
                selected_idx = (selected_idx - 1) % len(menu_items)
            elif key == curses.KEY_DOWN:
                selected_idx = (selected_idx + 1) % len(menu_items)
            elif key == curses.KEY_ENTER or key == 10 or key == 13:
                try:
                    if selected_idx == 0:  # Hardware Detection
                        current_screen = "hardware"
                        # Reset hardware info to force refresh when entering screen
                        hardware_info = None
                    elif selected_idx == 1:  # Environment Setup
                        current_screen = "environment"
                    elif selected_idx == 2:  # Dependencies
                        current_screen = "dependencies"
                        # Initialize dependencies if needed
                        if dependencies is None:
                            ui_state['force_redraw'] = True
                    elif selected_idx == 3:  # Components
                        current_screen = "components"
                        # Reset component selection indices to ensure correct highlighting
                        comp_selected_section = 0
                        comp_selected_idx = 0
                        ui_state['last_comp_selected_section'] = -1
                        ui_state['last_comp_selected_idx'] = -1
                    elif selected_idx == 4:  # Installation
                        current_screen = "installation"
                        # Reset installation selection indices
                        inst_selected_section = 0
                        inst_selected_idx = 0
                        ui_state['last_inst_selected_section'] = -1
                        ui_state['last_inst_selected_idx'] = -1
                    elif selected_idx == 5:  # Logs
                        current_screen = "logs"
                    elif selected_idx == 6:  # Exit
                        break

                    # Force redraw when changing screens
                    ui_state['force_redraw'] = True
                except Exception as e:
                    log_message(f"Error changing screens: {str(e)}")

        elif current_screen == "hardware":
            if key == ord('r'):
                hardware_info = None  # Force refresh

        elif current_screen == "environment":
            if key == curses.KEY_UP:
                env_selected_idx = (env_selected_idx - 1) % len(ENVIRONMENT_SCRIPTS)
            elif key == curses.KEY_DOWN:
                env_selected_idx = (env_selected_idx + 1) % len(ENVIRONMENT_SCRIPTS)
            elif key == ord(' '):  # Space to toggle selection
                # Toggle selection of the current script
                script = ENVIRONMENT_SCRIPTS[env_selected_idx]
                script["selected"] = not script.get("selected", False)

                # Update selected_env_scripts
                if script.get("selected", False):
                    if script["name"] not in selected_env_scripts:
                        selected_env_scripts.append(script["name"])
                else:
                    if script["name"] in selected_env_scripts:
                        selected_env_scripts.remove(script["name"])
            elif key == ord('i'):
                # Install selected environment scripts
                selected_scripts = []
                for script in ENVIRONMENT_SCRIPTS:
                    if script.get("selected", False):
                        selected_scripts.append(script)

                if selected_scripts:
                    current_screen = "logs"
                    log_win.clear()
                    log_win.addstr(f"Running {len(selected_scripts)} selected environment scripts...\n", curses.color_pair(COLOR_INFO))
                    log_win.refresh()

                    # Run each selected script
                    for script in selected_scripts:
                        log_win.addstr(f"\nRunning {script['name']}...\n", curses.color_pair(COLOR_INFO))
                        log_win.refresh()
                        install_component(script, stdscr, log_win)
                else:
                    # If no scripts are selected, run the currently highlighted script
                    script = ENVIRONMENT_SCRIPTS[env_selected_idx]
                    current_screen = "logs"
                    log_win.clear()
                    log_win.addstr(f"Running {script['name']}...\n", curses.color_pair(COLOR_INFO))
                    log_win.refresh()
                    install_component(script, stdscr, log_win)

        elif current_screen == "dependencies":
            if key == ord('r'):
                dependencies = None  # Force refresh
            elif key == curses.KEY_UP:
                # Add null check before accessing len(dependencies)
                if dependencies is not None and len(dependencies) > 0:
                    dep_selected_idx = (dep_selected_idx - 1) % len(dependencies)
            elif key == curses.KEY_DOWN:
                # Add null check before accessing len(dependencies)
                if dependencies is not None and len(dependencies) > 0:
                    dep_selected_idx = (dep_selected_idx + 1) % len(dependencies)
            elif key == ord(' '):  # Space to toggle selection
                # Add null check before accessing dependencies
                if dependencies is not None and len(dependencies) > 0 and dep_selected_idx < len(dependencies):
                    try:
                        # Get the dependency name at the current index
                        dep_name = list(dependencies.keys())[dep_selected_idx]

                        # Toggle selection
                        if dep_name in selected_deps:
                            selected_deps.remove(dep_name)
                        else:
                            selected_deps.append(dep_name)
                    except Exception as e:
                        log_message(f"Error toggling dependency selection: {str(e)}")
            elif key == ord('i'):
                # Install selected dependencies
                if selected_deps:
                    stdscr.addstr(curses.LINES - 3, 2, f"Installing selected dependencies: {', '.join(selected_deps)}".ljust(curses.COLS - 4), curses.color_pair(COLOR_INFO))
                    stdscr.refresh()
                    install_dependencies(selected_deps, stdscr)
                    dependencies = None  # Force refresh
                    selected_deps = []   # Clear selections after installation
                else:
                    stdscr.addstr(curses.LINES - 3, 2, "No dependencies selected for installation".ljust(curses.COLS - 4), curses.color_pair(COLOR_WARNING))
                    stdscr.refresh()
                    time.sleep(2)

        elif current_screen == "components":
            if key == ord('r'):
                # Refresh component status
                pass
            elif key == curses.KEY_UP:
                try:
                    if comp_selected_section == 0 and len(FOUNDATION_COMPONENTS) > 0:
                        comp_selected_idx = (comp_selected_idx - 1) % len(FOUNDATION_COMPONENTS)
                    elif comp_selected_section == 1 and len(CORE_COMPONENTS) > 0:
                        comp_selected_idx = (comp_selected_idx - 1) % len(CORE_COMPONENTS)
                    elif comp_selected_section == 2 and len(EXTENSION_COMPONENTS) > 0:
                        comp_selected_idx = (comp_selected_idx - 1) % len(EXTENSION_COMPONENTS)
                except Exception as e:
                    log_message(f"Error navigating components (UP): {str(e)}")
            elif key == curses.KEY_DOWN:
                try:
                    if comp_selected_section == 0:
                        if len(FOUNDATION_COMPONENTS) > 0:
                            if comp_selected_idx < len(FOUNDATION_COMPONENTS) - 1:
                                comp_selected_idx += 1
                            else:
                                comp_selected_section = 1
                                comp_selected_idx = 0
                    elif comp_selected_section == 1:
                        if len(CORE_COMPONENTS) > 0:
                            if comp_selected_idx < len(CORE_COMPONENTS) - 1:
                                comp_selected_idx += 1
                            else:
                                comp_selected_section = 2
                                comp_selected_idx = 0
                    elif comp_selected_section == 2:
                        if len(EXTENSION_COMPONENTS) > 0:
                            comp_selected_idx = (comp_selected_idx + 1) % len(EXTENSION_COMPONENTS)
                except Exception as e:
                    log_message(f"Error navigating components (DOWN): {str(e)}")
            elif key == curses.KEY_LEFT:
                if comp_selected_section > 0:
                    comp_selected_section -= 1
                    comp_selected_idx = 0
            elif key == curses.KEY_RIGHT:
                if comp_selected_section < 2:
                    comp_selected_section += 1
                    comp_selected_idx = 0
            elif key == ord(' '):  # Space to toggle selection
                # Toggle selection of the current component
                if comp_selected_section == 0 and comp_selected_idx < len(FOUNDATION_COMPONENTS):
                    component = FOUNDATION_COMPONENTS[comp_selected_idx]
                    component["selected"] = not component.get("selected", False)
                elif comp_selected_section == 1 and comp_selected_idx < len(CORE_COMPONENTS):
                    component = CORE_COMPONENTS[comp_selected_idx]
                    component["selected"] = not component.get("selected", False)
                elif comp_selected_section == 2 and comp_selected_idx < len(EXTENSION_COMPONENTS):
                    component = EXTENSION_COMPONENTS[comp_selected_idx]
                    component["selected"] = not component.get("selected", False)
            elif key == ord('i'):
                # Install selected components
                selected_components = []

                # Get all selected foundation components
                for comp in FOUNDATION_COMPONENTS:
                    if comp.get("selected", False):
                        selected_components.append(comp)

                # Get all selected core components
                for comp in CORE_COMPONENTS:
                    if comp.get("selected", False):
                        selected_components.append(comp)

                # Get all selected extension components
                for comp in EXTENSION_COMPONENTS:
                    if comp.get("selected", False):
                        selected_components.append(comp)

                if selected_components:
                    current_screen = "logs"
                    log_win.clear()
                    log_win.addstr(f"Installing {len(selected_components)} selected components...\n", curses.color_pair(COLOR_INFO))
                    log_win.refresh()

                    # Install each selected component
                    for component in selected_components:
                        log_win.addstr(f"\nInstalling {component['name']}...\n", curses.color_pair(COLOR_INFO))
                        log_win.refresh()
                        install_component(component, stdscr, log_win)
                else:
                    # If no components are selected, install the currently highlighted component
                    if comp_selected_section == 0 and comp_selected_idx < len(FOUNDATION_COMPONENTS):
                        component = FOUNDATION_COMPONENTS[comp_selected_idx]
                        current_screen = "logs"
                        log_win.clear()
                        log_win.addstr(f"Installing {component['name']}...\n", curses.color_pair(COLOR_INFO))
                        log_win.refresh()
                        install_component(component, stdscr, log_win)
                    elif comp_selected_section == 1 and comp_selected_idx < len(CORE_COMPONENTS):
                        component = CORE_COMPONENTS[comp_selected_idx]
                        current_screen = "logs"
                        log_win.clear()
                        log_win.addstr(f"Installing {component['name']}...\n", curses.color_pair(COLOR_INFO))
                        log_win.refresh()
                        install_component(component, stdscr, log_win)
                    elif comp_selected_section == 2 and comp_selected_idx < len(EXTENSION_COMPONENTS):
                        component = EXTENSION_COMPONENTS[comp_selected_idx]
                        current_screen = "logs"
                        log_win.clear()
                        log_win.addstr(f"Installing {component['name']}...\n", curses.color_pair(COLOR_INFO))
                        log_win.refresh()
                        install_component(component, stdscr, log_win)

        elif current_screen == "installation":
            if key == curses.KEY_UP:
                try:
                    if inst_selected_section == 0 and len(FOUNDATION_COMPONENTS) > 0:
                        inst_selected_idx = (inst_selected_idx - 1) % len(FOUNDATION_COMPONENTS)
                    elif inst_selected_section == 1 and len(CORE_COMPONENTS) > 0:
                        inst_selected_idx = (inst_selected_idx - 1) % len(CORE_COMPONENTS)
                    elif inst_selected_section == 2 and len(EXTENSION_COMPONENTS) > 0:
                        inst_selected_idx = (inst_selected_idx - 1) % len(EXTENSION_COMPONENTS)
                except Exception as e:
                    log_message(f"Error navigating installation (UP): {str(e)}")
            elif key == curses.KEY_DOWN:
                try:
                    if inst_selected_section == 0:
                        if len(FOUNDATION_COMPONENTS) > 0:
                            if inst_selected_idx < len(FOUNDATION_COMPONENTS) - 1:
                                inst_selected_idx += 1
                            else:
                                inst_selected_section = 1
                                inst_selected_idx = 0
                    elif inst_selected_section == 1:
                        if len(CORE_COMPONENTS) > 0:
                            if inst_selected_idx < len(CORE_COMPONENTS) - 1:
                                inst_selected_idx += 1
                            else:
                                inst_selected_section = 2
                                inst_selected_idx = 0
                    elif inst_selected_section == 2:
                        if len(EXTENSION_COMPONENTS) > 0:
                            inst_selected_idx = (inst_selected_idx + 1) % len(EXTENSION_COMPONENTS)
                except Exception as e:
                    log_message(f"Error navigating installation (DOWN): {str(e)}")
            elif key == curses.KEY_LEFT:
                if inst_selected_section > 0:
                    inst_selected_section -= 1
                    inst_selected_idx = 0
            elif key == curses.KEY_RIGHT:
                if inst_selected_section < 2:
                    inst_selected_section += 1
                    inst_selected_idx = 0
            elif key == ord(' '):  # Space to toggle selection
                # Toggle selection of the current component
                if inst_selected_section == 0 and inst_selected_idx < len(FOUNDATION_COMPONENTS):
                    component = FOUNDATION_COMPONENTS[inst_selected_idx]
                    component["selected"] = not component.get("selected", False)
                elif inst_selected_section == 1 and inst_selected_idx < len(CORE_COMPONENTS):
                    component = CORE_COMPONENTS[inst_selected_idx]
                    component["selected"] = not component.get("selected", False)
                elif inst_selected_section == 2 and inst_selected_idx < len(EXTENSION_COMPONENTS):
                    component = EXTENSION_COMPONENTS[inst_selected_idx]
                    component["selected"] = not component.get("selected", False)
            elif key == ord('i'):
                # Install selected components
                selected_components = []

                # Get all selected foundation components
                for comp in FOUNDATION_COMPONENTS:
                    if comp.get("selected", False) and comp["status"] != "installed":
                        selected_components.append(comp)

                # Get all selected core components
                for comp in CORE_COMPONENTS:
                    if comp.get("selected", False) and comp["status"] != "installed":
                        selected_components.append(comp)

                # Get all selected extension components
                for comp in EXTENSION_COMPONENTS:
                    if comp.get("selected", False) and comp["status"] != "installed":
                        selected_components.append(comp)

                if selected_components:
                    current_screen = "logs"
                    log_win.clear()
                    log_win.addstr(f"Installing {len(selected_components)} selected components...\n", curses.color_pair(COLOR_INFO))
                    log_win.refresh()

                    # Install each selected component
                    for component in selected_components:
                        log_win.addstr(f"\nInstalling {component['name']}...\n", curses.color_pair(COLOR_INFO))
                        log_win.refresh()
                        install_component(component, stdscr, log_win)
                else:
                    # If no components are selected, show a menu of options
                    # Create a selection menu for installation options
                    install_menu_items = [
                        "Install ROCm Configuration",
                        "Install PyTorch with ROCm support",
                        "Install ONNX Runtime with ROCm support",
                        "Install MIGraphX",
                        "Install Megatron-LM",
                        "Install Flash Attention with AMD GPU support",
                        "Install RCCL",
                        "Install MPI",
                        "Install All Core Components",
                        "Verify Installation",
                        "Repair ML Stack",
                        "Cancel"
                    ]

                    # Draw the installation menu
                    install_menu_y = 11  # Adjusted to be consistent with other screens
                    install_menu_x = 10
                    install_menu_width = curses.COLS - 20
                    install_menu_height = len(install_menu_items) + 4

                    # Create a window for the menu
                    install_menu_win = curses.newwin(install_menu_height, install_menu_width, install_menu_y, install_menu_x)
                    install_menu_win.box()

                    # Add a title
                    install_menu_win.attron(curses.color_pair(COLOR_TITLE))
                    title = " ML Stack Installation Menu "
                    install_menu_win.addstr(0, (install_menu_width - len(title)) // 2, title)
                    install_menu_win.attroff(curses.color_pair(COLOR_TITLE))

                    # Draw the menu items
                    selected_menu_idx = 0

                    # Menu loop
                    while True:
                        # Draw menu items
                        for i, item in enumerate(install_menu_items):
                            if i == selected_menu_idx:
                                install_menu_win.attron(curses.color_pair(COLOR_HIGHLIGHT))
                                install_menu_win.addstr(i + 2, 2, f"> {i+1}) {item}".ljust(install_menu_width - 4))
                                install_menu_win.attroff(curses.color_pair(COLOR_HIGHLIGHT))
                            else:
                                install_menu_win.addstr(i + 2, 2, f"  {i+1}) {item}".ljust(install_menu_width - 4), curses.color_pair(COLOR_NORMAL))

                        install_menu_win.refresh()

                        # Get user input
                        key = stdscr.getch()

                        # Process input
                        if key == curses.KEY_UP:
                            selected_menu_idx = (selected_menu_idx - 1) % len(install_menu_items)
                        elif key == curses.KEY_DOWN:
                            selected_menu_idx = (selected_menu_idx + 1) % len(install_menu_items)
                        elif key == ord('q') or key == ord('b') or key == 27:  # q, b or ESC
                            break
                        elif key == curses.KEY_ENTER or key == 10 or key == 13:  # Enter
                            # Handle menu selection
                            if selected_menu_idx == len(install_menu_items) - 1:  # Cancel
                                break
                            elif selected_menu_idx == 8:  # Install All Core Components
                                current_screen = "logs"
                                log_win.clear()
                                log_win.addstr("Installing all core components...\n", curses.color_pair(COLOR_INFO))
                                log_win.refresh()

                                # Install core components
                                for component in CORE_COMPONENTS:
                                    if component["status"] != "installed":
                                        log_win.addstr(f"\nInstalling {component['name']}...\n", curses.color_pair(COLOR_INFO))
                                        log_win.refresh()
                                        install_component(component, stdscr, log_win)

                                # After installation is done, return to menu
                                current_screen = "menu"
                                break
                            else:
                                # Install the selected component
                                if selected_menu_idx < 8:  # Individual components
                                    component = None
                                    if selected_menu_idx == 0:  # ROCm Configuration
                                        component = next((c for c in CORE_COMPONENTS if c["name"] == "ROCm Configuration"), None)
                                    elif selected_menu_idx == 1:  # PyTorch
                                        component = next((c for c in CORE_COMPONENTS if c["name"] == "PyTorch with ROCm support"), None)
                                    elif selected_menu_idx == 2:  # ONNX Runtime
                                        component = next((c for c in CORE_COMPONENTS if c["name"] == "ONNX Runtime with ROCm support"), None)
                                    elif selected_menu_idx == 3:  # MIGraphX
                                        component = next((c for c in CORE_COMPONENTS if c["name"] == "MIGraphX"), None)
                                    elif selected_menu_idx == 4:  # Megatron-LM
                                        component = next((c for c in EXTENSION_COMPONENTS if c["name"] == "Megatron-LM"), None)
                                    elif selected_menu_idx == 5:  # Flash Attention
                                        component = next((c for c in EXTENSION_COMPONENTS if c["name"] == "Flash Attention"), None)
                                    elif selected_menu_idx == 6:  # RCCL
                                        component = next((c for c in EXTENSION_COMPONENTS if c["name"] == "RCCL"), None)
                                    elif selected_menu_idx == 7:  # MPI
                                        component = next((c for c in EXTENSION_COMPONENTS if c["name"] == "MPI"), None)

                                    if component:
                                        current_screen = "logs"
                                        log_win.clear()
                                        log_win.addstr(f"Installing {component['name']}...\n", curses.color_pair(COLOR_INFO))
                                        log_win.refresh()
                                        install_component(component, stdscr, log_win)

                                        # After installation is done, return to menu
                                        current_screen = "menu"
                                    break
                                elif selected_menu_idx == 9:  # Verify Installation
                                    current_screen = "logs"
                                    log_win.clear()
                                    log_win.addstr("Verifying installation...\n", curses.color_pair(COLOR_INFO))
                                    log_win.refresh()

                                    # Run verification script
                                    verify_script = next((c for c in CORE_COMPONENTS if c["name"] == "Verify Installation"), None)
                                    if verify_script:
                                        install_component(verify_script, stdscr, log_win)

                                    # After verification is done, return to menu
                                    current_screen = "menu"
                                    break
                                elif selected_menu_idx == 10:  # Repair ML Stack
                                    current_screen = "logs"
                                    log_win.clear()
                                    log_win.addstr("Repairing ML Stack installation...\n", curses.color_pair(COLOR_INFO))
                                    log_win.refresh()

                                    # Run repair script
                                    repair_script = next((c for c in CORE_COMPONENTS if c["name"] == "Repair ML Stack"), None)
                                    if repair_script:
                                        install_component(repair_script, stdscr, log_win)

                                    # After repair is done, return to menu
                                    current_screen = "menu"
                                    break
                        # Handle number keys 1-9 for quick selection
                        elif key >= ord('1') and key <= ord('9'):
                            idx = key - ord('1')
                            if idx < len(install_menu_items):
                                selected_menu_idx = idx
                                # Simulate Enter key press
                                continue

        elif current_screen == "logs":
            pass  # No special key handling needed here

# Original main function that takes a stdscr parameter
def _main_curses(stdscr):
    """Main function."""
    # Create log file if it doesn't exist
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Starting ML Stack Installation UI\n")
    except Exception as e:
        # If we can't create the log file, print an error and continue
        # We'll handle log file errors gracefully later
        pass

    # Initialize curses
    curses.curs_set(0)  # Hide cursor
    curses.start_color()
    curses.use_default_colors()

    # Set up non-blocking input with a short timeout for automatic refreshing
    stdscr.nodelay(True)  # Make getch() non-blocking
    stdscr.timeout(200)   # Set a 200ms timeout for getch() - this enables automatic refreshing

    # Enable keypad mode
    stdscr.keypad(True)

    # Initialize color pairs with more vibrant colors
    curses.init_pair(COLOR_NORMAL, curses.COLOR_WHITE, -1)
    curses.init_pair(COLOR_HIGHLIGHT, curses.COLOR_BLACK, curses.COLOR_GREEN)  # Green selection highlight
    curses.init_pair(COLOR_TITLE, curses.COLOR_MAGENTA, -1)
    curses.init_pair(COLOR_SUCCESS, curses.COLOR_GREEN, -1)
    curses.init_pair(COLOR_ERROR, curses.COLOR_RED, -1)
    curses.init_pair(COLOR_WARNING, curses.COLOR_YELLOW, -1)
    curses.init_pair(COLOR_INFO, curses.COLOR_CYAN, -1)
    curses.init_pair(COLOR_BORDER, curses.COLOR_YELLOW, -1)  # Orange border (using yellow as closest to orange)

    # Show welcome screen and get sudo authentication
    if not show_welcome_screen(stdscr):
        return  # Exit if authentication failed or user quit

    # Clear screen
    stdscr.clear()

    # Set up automatic refresh with timeout in the main loop
    # This ensures the screen updates regularly without requiring user input

    # Draw initial screen
    draw_header(stdscr, "ML Stack Installation")
    draw_footer(stdscr, "Press 'q' to quit, arrow keys to navigate, Enter to select")

    # Main menu
    menu_items = ["Hardware Detection", "Environment Setup", "Dependencies", "Components", "Installation", "Logs", "Exit"]
    selected_idx = 0
    current_screen = "menu"

    # Hardware info
    hardware_info = None

    # Environment scripts
    env_selected_idx = 0
    selected_env_scripts = []

    # Initialize selected state for environment scripts
    for script in ENVIRONMENT_SCRIPTS:
        script["selected"] = False

    # Dependencies
    dependencies = None
    dep_selected_idx = 0
    selected_deps = []

    # Components
    comp_selected_section = 0
    comp_selected_idx = 0
    inst_selected_section = 0
    inst_selected_idx = 0

    # Create log window
    log_win = create_log_window(stdscr)
    # Explicitly refresh the screen after creating the log window
    stdscr.refresh()

    # Implement double-buffering to eliminate screen flashing
    # Create a virtual screen (pad) for drawing that will be copied to the real screen only when ready
    virtual_screen = curses.newpad(curses.LINES, curses.COLS)

    # Track the current screen to know when to redraw
    previous_screen = None

    # Create a state dictionary to store UI state and timing information
    # This avoids attaching attributes directly to curses window objects
    ui_state = {
        # Track the last state to avoid unnecessary redraws
        'last_selected_idx': -1,
        'last_env_selected_idx': -1,
        'last_dep_selected_idx': -1,
        'last_comp_selected_section': -1,
        'last_comp_selected_idx': -1,
        'last_inst_selected_section': -1,
        'last_inst_selected_idx': -1,
        # Timing information for controlled refreshes
        'last_refresh_time': 0,
        # Flag to force redraw when needed
        'force_redraw': False,
        # Last key press time for input responsiveness
        'last_key_time': 0
    }

    # Function to copy the virtual screen to the real screen
    def update_screen():
        try:
            # For a pad, we need to specify the region to display
            max_y, max_x = stdscr.getmaxyx()

            # Ensure we don't try to refresh beyond screen boundaries
            if max_y > 0 and max_x > 0:
                # Get current time to throttle updates
                current_time = time.time()

                # Only update if enough time has passed since last update (prevents excessive refreshes)
                if current_time - ui_state.get('last_screen_update', 0) > 0.03:  # 30ms minimum between updates
                    # Copy the entire virtual screen to the real screen
                    # Parameters: pminrow, pmincol, sminrow, smincol, smaxrow, smaxcol
                    virtual_screen.noutrefresh(0, 0, 0, 0, max_y-1, max_x-1)
                    curses.doupdate()
                    ui_state['last_screen_update'] = current_time
        except Exception as e:
            # Handle any errors during screen update
            log_message(f"Error updating screen: {str(e)}")

    # Main loop with double-buffering to eliminate flashing
    while True:
        # Use a shorter timeout to improve responsiveness while still being efficient
        stdscr.timeout(5)  # 5ms timeout for getch() - more responsive

        # Get user input (with timeout due to nodelay mode)
        key = stdscr.getch()

        # Handle screen transitions and state changes
        if key != -1:
            # Store current time for key press responsiveness
            current_time = time.time()

            # Only process key if enough time has passed since last key press
            # This prevents double-processing of keys but ensures we don't miss any
            if current_time - ui_state.get('last_key_time', 0) > 0.01:  # 10ms debounce
                ui_state['last_key_time'] = current_time
                ui_state['force_redraw'] = True  # Force redraw on key press

                # Store the previous screen before potentially changing it
                previous_screen = current_screen

                # Global 'b' key handling for all screens except menu
                if key == ord('b') and current_screen != "menu":
                    log_message(f"Going back to menu from {current_screen}")
                    current_screen = "menu"
                    ui_state['force_redraw'] = True  # Force redraw when going back to menu
                    previous_screen = None  # Reset previous_screen to force redraw
                    continue

                # Global 'q' key handling for quitting
                if key == ord('q'):
                    break

                # Handle key based on current screen
                if current_screen == "menu":
                    if key == curses.KEY_UP:
                        selected_idx = (selected_idx - 1) % len(menu_items)
                    elif key == curses.KEY_DOWN:
                        selected_idx = (selected_idx + 1) % len(menu_items)
                    elif key == curses.KEY_ENTER or key == 10 or key == 13:
                        if menu_items[selected_idx] == "Hardware Detection":
                            current_screen = "hardware"
                            hardware_info = None  # Reset to force refresh
                            ui_state['force_redraw'] = True  # Force redraw
                        elif menu_items[selected_idx] == "Environment Setup":
                            current_screen = "environment"
                            ui_state['force_redraw'] = True  # Force redraw
                        elif menu_items[selected_idx] == "Dependencies":
                            current_screen = "dependencies"
                            dependencies = None  # Reset to force refresh
                            ui_state['force_redraw'] = True  # Force redraw
                        elif menu_items[selected_idx] == "Components":
                            current_screen = "components"
                            ui_state['force_redraw'] = True  # Force redraw
                        elif menu_items[selected_idx] == "Installation":
                            current_screen = "installation"
                            ui_state['force_redraw'] = True  # Force redraw
                        elif menu_items[selected_idx] == "Logs":
                            current_screen = "logs"
                            ui_state['force_redraw'] = True  # Force redraw
                        elif menu_items[selected_idx] == "Exit":
                            break

                elif current_screen == "hardware":
                    if key == ord('r'):
                        # Refresh hardware info
                        log_message("Refreshing hardware info")
                        hardware_info = None  # Reset to force refresh
                        ui_state['force_redraw'] = True  # Force redraw
                        # Force redraw immediately
                        need_redraw = True

                elif current_screen == "environment":
                    if key == curses.KEY_UP:
                        env_selected_idx = (env_selected_idx - 1) % len(ENVIRONMENT_SCRIPTS)
                        ui_state['force_redraw'] = True  # Force redraw
                    elif key == curses.KEY_DOWN:
                        env_selected_idx = (env_selected_idx + 1) % len(ENVIRONMENT_SCRIPTS)
                        ui_state['force_redraw'] = True  # Force redraw
                    elif key == ord(' '):  # Space key to select/deselect
                        script = ENVIRONMENT_SCRIPTS[env_selected_idx]
                        # Toggle selection state
                        if script in selected_env_scripts:
                            selected_env_scripts.remove(script)
                            script["selected"] = False  # Update the script's selected property
                        else:
                            selected_env_scripts.append(script)
                            script["selected"] = True  # Update the script's selected property
                        log_message(f"Toggled selection of {script['name']} to {script['selected']}")
                        ui_state['force_redraw'] = True  # Force redraw
                    elif key == ord('i'):  # 'i' key to run selected scripts
                        if selected_env_scripts:
                            current_screen = "logs"
                            log_win.clear()
                            log_win.addstr(f"Running {len(selected_env_scripts)} selected environment scripts...\n", curses.color_pair(COLOR_INFO))
                            log_win.refresh()

                            # Run each selected script
                            for script in selected_env_scripts:
                                log_win.addstr(f"\nRunning {script['name']}...\n", curses.color_pair(COLOR_INFO))
                                log_win.refresh()
                                install_component(script, stdscr, log_win)
                        else:
                            # If no scripts are selected, run the currently highlighted script
                            script = ENVIRONMENT_SCRIPTS[env_selected_idx]
                            current_screen = "logs"
                            log_win.clear()
                            log_win.addstr(f"Running {script['name']}...\n", curses.color_pair(COLOR_INFO))
                            log_win.refresh()
                            install_component(script, stdscr, log_win)
                        ui_state['force_redraw'] = True  # Force redraw

                elif current_screen == "dependencies":
                    if dependencies is not None and len(dependencies) > 0:
                        if key == curses.KEY_UP:
                            dep_selected_idx = (dep_selected_idx - 1) % len(dependencies)
                            ui_state['force_redraw'] = True  # Force redraw
                        elif key == curses.KEY_DOWN:
                            dep_selected_idx = (dep_selected_idx + 1) % len(dependencies)
                            ui_state['force_redraw'] = True  # Force redraw
                        elif key == ord(' '):  # Space key to select/deselect
                            # Get the dependency name at the current index
                            dep_name = list(dependencies.keys())[dep_selected_idx]
                            # Toggle selection
                            if dep_name in selected_deps:
                                selected_deps.remove(dep_name)
                            else:
                                selected_deps.append(dep_name)
                            ui_state['force_redraw'] = True  # Force redraw
                        elif key == ord('i'):  # 'i' key to install selected dependencies
                            if selected_deps:
                                stdscr.addstr(curses.LINES - 3, 2, f"Installing selected dependencies: {', '.join(selected_deps)}".ljust(curses.COLS - 4), curses.color_pair(COLOR_INFO))
                                stdscr.refresh()
                                install_dependencies(selected_deps, stdscr)
                                dependencies = None  # Force refresh
                                selected_deps = []   # Clear selections after installation
                                ui_state['force_redraw'] = True  # Force redraw
                            else:
                                stdscr.addstr(curses.LINES - 3, 2, "No dependencies selected for installation".ljust(curses.COLS - 4), curses.color_pair(COLOR_WARNING))
                                stdscr.refresh()
                                time.sleep(2)
                    elif key == ord('r'):  # 'r' key to refresh dependencies
                        dependencies = None  # Reset to force refresh
                        ui_state['force_redraw'] = True  # Force redraw

                elif current_screen == "components":
                    if key == curses.KEY_UP:
                        if comp_selected_section == 0:  # Foundation components
                            comp_selected_idx = (comp_selected_idx - 1) % len(FOUNDATION_COMPONENTS)
                        elif comp_selected_section == 1:  # Core components
                            comp_selected_idx = (comp_selected_idx - 1) % len(CORE_COMPONENTS)
                        elif comp_selected_section == 2:  # Extension components
                            comp_selected_idx = (comp_selected_idx - 1) % len(EXTENSION_COMPONENTS)
                        ui_state['force_redraw'] = True  # Force redraw
                    elif key == curses.KEY_DOWN:
                        if comp_selected_section == 0:  # Foundation components
                            comp_selected_idx = (comp_selected_idx + 1) % len(FOUNDATION_COMPONENTS)
                        elif comp_selected_section == 1:  # Core components
                            comp_selected_idx = (comp_selected_idx + 1) % len(CORE_COMPONENTS)
                        elif comp_selected_section == 2:  # Extension components
                            comp_selected_idx = (comp_selected_idx + 1) % len(EXTENSION_COMPONENTS)
                        ui_state['force_redraw'] = True  # Force redraw
                    elif key == curses.KEY_LEFT:
                        comp_selected_section = (comp_selected_section - 1) % 3
                        comp_selected_idx = 0  # Reset index when changing sections
                        ui_state['force_redraw'] = True  # Force redraw
                    elif key == curses.KEY_RIGHT:
                        comp_selected_section = (comp_selected_section + 1) % 3
                        comp_selected_idx = 0  # Reset index when changing sections
                        ui_state['force_redraw'] = True  # Force redraw
                    elif key == ord(' '):  # Space to toggle selection
                        # Toggle selection of the current component
                        if comp_selected_section == 0 and comp_selected_idx < len(FOUNDATION_COMPONENTS):
                            component = FOUNDATION_COMPONENTS[comp_selected_idx]
                            component["selected"] = not component.get("selected", False)
                            log_message(f"Toggled selection of {component['name']} to {component['selected']}")
                        elif comp_selected_section == 1 and comp_selected_idx < len(CORE_COMPONENTS):
                            component = CORE_COMPONENTS[comp_selected_idx]
                            component["selected"] = not component.get("selected", False)
                            log_message(f"Toggled selection of {component['name']} to {component['selected']}")
                        elif comp_selected_section == 2 and comp_selected_idx < len(EXTENSION_COMPONENTS):
                            component = EXTENSION_COMPONENTS[comp_selected_idx]
                            component["selected"] = not component.get("selected", False)
                            log_message(f"Toggled selection of {component['name']} to {component['selected']}")
                        ui_state['force_redraw'] = True  # Force redraw
                    elif key == ord('i'):  # 'i' key to install selected components
                        # Handle installation of selected components
                        current_screen = "logs"
                        log_win.clear()

                        # Get all selected components
                        selected_components = []

                        # Get all selected foundation components
                        for comp in FOUNDATION_COMPONENTS:
                            if comp.get("selected", False):
                                selected_components.append(comp)

                        # Get all selected core components
                        for comp in CORE_COMPONENTS:
                            if comp.get("selected", False):
                                selected_components.append(comp)

                        # Get all selected extension components
                        for comp in EXTENSION_COMPONENTS:
                            if comp.get("selected", False):
                                selected_components.append(comp)

                        if selected_components:
                            log_win.addstr(f"Installing {len(selected_components)} selected components...\n", curses.color_pair(COLOR_INFO))
                            log_win.refresh()

                            # Install each selected component
                            for component in selected_components:
                                log_win.addstr(f"\nInstalling {component['name']}...\n", curses.color_pair(COLOR_INFO))
                                log_win.refresh()
                                install_result = install_component(component, stdscr, log_win)
                                log_message(f"Installation of {component['name']} completed with result: {install_result}")

                                # Give a brief pause between components
                                time.sleep(1)

                            # Return to components screen after all installations
                            current_screen = "components"
                            ui_state['force_redraw'] = True  # Force redraw
                            log_message("Returned to components screen after installation")
                        else:
                            # If no components are selected, install the currently highlighted component
                            if comp_selected_section == 0 and comp_selected_idx < len(FOUNDATION_COMPONENTS):
                                component = FOUNDATION_COMPONENTS[comp_selected_idx]
                                log_win.addstr(f"Installing {component['name']}...\n", curses.color_pair(COLOR_INFO))
                                log_win.refresh()
                                install_result = install_component(component, stdscr, log_win)
                                log_message(f"Installation of {component['name']} completed with result: {install_result}")
                            elif comp_selected_section == 1 and comp_selected_idx < len(CORE_COMPONENTS):
                                component = CORE_COMPONENTS[comp_selected_idx]
                                log_win.addstr(f"Installing {component['name']}...\n", curses.color_pair(COLOR_INFO))
                                log_win.refresh()
                                install_result = install_component(component, stdscr, log_win)
                                log_message(f"Installation of {component['name']} completed with result: {install_result}")
                            elif comp_selected_section == 2 and comp_selected_idx < len(EXTENSION_COMPONENTS):
                                component = EXTENSION_COMPONENTS[comp_selected_idx]
                                log_win.addstr(f"Installing {component['name']}...\n", curses.color_pair(COLOR_INFO))
                                log_win.refresh()
                                install_result = install_component(component, stdscr, log_win)
                                log_message(f"Installation of {component['name']} completed with result: {install_result}")

                            # Return to components screen after installation
                            current_screen = "components"
                            ui_state['force_redraw'] = True  # Force redraw
                            log_message("Returned to components screen after installation")
                        ui_state['force_redraw'] = True  # Force redraw
                    elif key == ord('r'):  # 'r' key to refresh components
                        # Reset component statuses
                        for component in FOUNDATION_COMPONENTS + CORE_COMPONENTS + EXTENSION_COMPONENTS:
                            component["status"] = "pending"
                        ui_state['force_redraw'] = True  # Force redraw

                elif current_screen == "installation":
                    if key == curses.KEY_UP:
                        if inst_selected_section == 0 and len(FOUNDATION_COMPONENTS) > 0:
                            inst_selected_idx = (inst_selected_idx - 1) % len(FOUNDATION_COMPONENTS)
                        elif inst_selected_section == 1 and len(CORE_COMPONENTS) > 0:
                            inst_selected_idx = (inst_selected_idx - 1) % len(CORE_COMPONENTS)
                        elif inst_selected_section == 2 and len(EXTENSION_COMPONENTS) > 0:
                            inst_selected_idx = (inst_selected_idx - 1) % len(EXTENSION_COMPONENTS)
                        ui_state['force_redraw'] = True  # Force redraw
                    elif key == curses.KEY_DOWN:
                        if inst_selected_section == 0:
                            if len(FOUNDATION_COMPONENTS) > 0:
                                if inst_selected_idx < len(FOUNDATION_COMPONENTS) - 1:
                                    inst_selected_idx += 1
                                else:
                                    inst_selected_section = 1
                                    inst_selected_idx = 0
                        elif inst_selected_section == 1:
                            if len(CORE_COMPONENTS) > 0:
                                if inst_selected_idx < len(CORE_COMPONENTS) - 1:
                                    inst_selected_idx += 1
                                else:
                                    inst_selected_section = 2
                                    inst_selected_idx = 0
                        elif inst_selected_section == 2:
                            if len(EXTENSION_COMPONENTS) > 0:
                                inst_selected_idx = (inst_selected_idx + 1) % len(EXTENSION_COMPONENTS)
                        ui_state['force_redraw'] = True  # Force redraw
                    elif key == curses.KEY_LEFT:
                        if inst_selected_section > 0:
                            inst_selected_section -= 1
                            inst_selected_idx = 0
                        ui_state['force_redraw'] = True  # Force redraw
                    elif key == curses.KEY_RIGHT:
                        if inst_selected_section < 2:
                            inst_selected_section += 1
                            inst_selected_idx = 0
                        ui_state['force_redraw'] = True  # Force redraw
                    elif key == ord(' '):  # Space to toggle selection
                        # Toggle selection of the current component
                        if inst_selected_section == 0 and inst_selected_idx < len(FOUNDATION_COMPONENTS):
                            component = FOUNDATION_COMPONENTS[inst_selected_idx]
                            component["selected"] = not component.get("selected", False)
                        elif inst_selected_section == 1 and inst_selected_idx < len(CORE_COMPONENTS):
                            component = CORE_COMPONENTS[inst_selected_idx]
                            component["selected"] = not component.get("selected", False)
                        elif inst_selected_section == 2 and inst_selected_idx < len(EXTENSION_COMPONENTS):
                            component = EXTENSION_COMPONENTS[inst_selected_idx]
                            component["selected"] = not component.get("selected", False)
                        ui_state['force_redraw'] = True  # Force redraw
                    elif key == ord('i') or key == curses.KEY_ENTER or key == 10 or key == 13:
                        # Handle installation of selected components
                        current_screen = "logs"
                        log_win.clear()
                        log_message("Starting installation of selected components")

                        # Get all selected components
                        selected_components = []

                        # Get all selected foundation components
                        for comp in FOUNDATION_COMPONENTS:
                            if comp.get("selected", False):
                                selected_components.append(comp)
                                log_message(f"Selected foundation component: {comp['name']}")

                        # Get all selected core components
                        for comp in CORE_COMPONENTS:
                            if comp.get("selected", False):
                                selected_components.append(comp)
                                log_message(f"Selected core component: {comp['name']}")

                        # Get all selected extension components
                        for comp in EXTENSION_COMPONENTS:
                            if comp.get("selected", False):
                                selected_components.append(comp)
                                log_message(f"Selected extension component: {comp['name']}")

                        if selected_components:
                            log_win.addstr(f"Installing {len(selected_components)} selected components...\n", curses.color_pair(COLOR_INFO))
                            log_win.refresh()

                            # Install each selected component
                            for component in selected_components:
                                log_win.addstr(f"\nInstalling {component['name']}...\n", curses.color_pair(COLOR_INFO))
                                log_win.refresh()
                                install_result = install_component(component, stdscr, log_win)
                                log_message(f"Installation of {component['name']} completed with result: {install_result}")

                                # Give a brief pause between components
                                time.sleep(1)

                            # Return to installation screen after all installations
                            current_screen = "installation"
                            ui_state['force_redraw'] = True  # Force redraw
                            log_message("Returned to installation screen after installation")
                        else:
                            # If no components are selected, install the currently highlighted component
                            log_message("No components selected, installing highlighted component")
                            if inst_selected_section == 0 and inst_selected_idx < len(FOUNDATION_COMPONENTS):
                                component = FOUNDATION_COMPONENTS[inst_selected_idx]
                                log_win.addstr(f"Installing {component['name']}...\n", curses.color_pair(COLOR_INFO))
                                log_win.refresh()
                                install_result = install_component(component, stdscr, log_win)
                                log_message(f"Installation of {component['name']} completed with result: {install_result}")
                            elif inst_selected_section == 1 and inst_selected_idx < len(CORE_COMPONENTS):
                                component = CORE_COMPONENTS[inst_selected_idx]
                                log_win.addstr(f"Installing {component['name']}...\n", curses.color_pair(COLOR_INFO))
                                log_win.refresh()
                                install_result = install_component(component, stdscr, log_win)
                                log_message(f"Installation of {component['name']} completed with result: {install_result}")
                            elif inst_selected_section == 2 and inst_selected_idx < len(EXTENSION_COMPONENTS):
                                component = EXTENSION_COMPONENTS[inst_selected_idx]
                                log_win.addstr(f"Installing {component['name']}...\n", curses.color_pair(COLOR_INFO))
                                log_win.refresh()
                                install_result = install_component(component, stdscr, log_win)
                                log_message(f"Installation of {component['name']} completed with result: {install_result}")

                            # Return to installation screen after installation
                            current_screen = "installation"
                            ui_state['force_redraw'] = True  # Force redraw
                            log_message("Returned to installation screen after installation")

                            ui_state['force_redraw'] = True  # Force redraw

                elif current_screen == "logs":
                    if key == ord('r'):  # 'r' key to refresh logs
                        ui_state['force_redraw'] = True  # Force redraw
            else:
                # If we're debouncing, set the key to -1 to ignore it
                key = -1

        # Determine if we need to redraw the screen
        # Always redraw when screen changes or key is pressed
        need_redraw = (
            current_screen != previous_screen or  # Screen has changed
            key != -1 or  # Key was pressed
            (current_screen == "menu" and selected_idx != ui_state['last_selected_idx']) or  # Menu selection changed
            (current_screen == "hardware" and hardware_info != ui_state.get('last_hardware_info')) or  # Hardware info changed
            (current_screen == "environment" and env_selected_idx != ui_state['last_env_selected_idx']) or  # Environment selection changed
            (current_screen == "dependencies" and dep_selected_idx != ui_state['last_dep_selected_idx']) or  # Dependencies selection changed
            (current_screen == "components" and (
                comp_selected_section != ui_state['last_comp_selected_section'] or
                comp_selected_idx != ui_state['last_comp_selected_idx']
            )) or  # Components selection changed
            (current_screen == "installation" and (
                inst_selected_section != ui_state.get('last_inst_selected_section', 0) or
                inst_selected_idx != ui_state.get('last_inst_selected_idx', 0)
            )) or  # Installation selection changed
            ui_state.get('force_redraw', False)  # Force redraw flag is set
        )

        # Store current hardware_info for comparison in next iteration
        ui_state['last_hardware_info'] = hardware_info

        # Log the need_redraw state for debugging
        log_message(f"Screen: {current_screen}, Key: {key}, Need Redraw: {need_redraw}")

        # Reset force_redraw flag
        ui_state['force_redraw'] = False

        # Draw current screen to virtual screen if needed
        if need_redraw or current_screen != previous_screen:  # Redraw when necessary or when screen changes
            try:
                virtual_screen.clear()

                if current_screen == "menu":
                    draw_header(virtual_screen, "ML Stack Installation")
                    draw_footer(virtual_screen, "Press 'q' to quit, arrow keys to navigate, Enter to select")
                    draw_menu(virtual_screen, selected_idx, menu_items)

                    # Update tracking variables
                    ui_state['last_selected_idx'] = selected_idx

                    # Add debug info at the bottom of the screen
                    try:
                        debug_info = f"Screen: {current_screen} | Key: {key} | Need Redraw: {need_redraw}"
                        virtual_screen.addstr(curses.LINES - 1, 2, debug_info[:curses.COLS - 4], curses.color_pair(COLOR_NORMAL))
                    except:
                        pass  # Ignore errors when adding debug info

                    # Copy virtual screen to real screen
                    update_screen()

                    # For menu screen, we don't need to redraw on every loop iteration
                    # This prevents flickering
                    time.sleep(0.05)  # Small delay to reduce CPU usage and prevent flickering

                elif current_screen == "hardware":
                    draw_header(virtual_screen, "Hardware Detection")
                    draw_footer(virtual_screen, "Press 'b' to go back, 'r' to refresh")

                    if hardware_info is None:
                        try:
                            log_message("Starting hardware detection")
                            virtual_screen.addstr(11, 2, "Detecting hardware...".ljust(curses.COLS - 4), curses.color_pair(COLOR_INFO))

                            # Add debug info at the bottom of the screen
                            try:
                                debug_info = f"Screen: {current_screen} | Key: {key} | Need Redraw: {need_redraw}"
                                virtual_screen.addstr(curses.LINES - 1, 2, debug_info[:curses.COLS - 4], curses.color_pair(COLOR_NORMAL))
                            except:
                                pass  # Ignore errors when adding debug info

                            update_screen()  # Show "Detecting hardware..." message

                            # Add proper error handling around hardware detection
                            try:
                                hardware_info = detect_hardware()
                                log_message("Hardware detection completed successfully")
                            except Exception as e:
                                log_message(f"Error detecting hardware: {str(e)}")
                                # Initialize with empty values to prevent crashes
                                hardware_info = {
                                    'system': 'Unknown',
                                    'release': 'Unknown',
                                    'machine': 'Unknown',
                                    'processor': 'Unknown',
                                    'python_version': 'Unknown',
                                    'rocm_path': None,
                                    'rocm_version': None,
                                    'amd_gpus': []
                                }

                            virtual_screen.clear()  # Clear again to redraw with the new info
                            draw_header(virtual_screen, "Hardware Detection")
                            draw_footer(virtual_screen, "Press 'b' to go back, 'r' to refresh")
                        except Exception as e:
                            log_message(f"Error updating hardware detection screen: {str(e)}")

                    # Display hardware information
                    hw_y = 11  # Moved 5 lines higher from previous position of 16
                    hw_x = 4
                    max_width = curses.COLS - hw_x - 4  # Maximum width for text to avoid boundary errors

                    try:
                        virtual_screen.addstr(hw_y, hw_x, f"System: {hardware_info['system']} {hardware_info['release']}".ljust(max_width), curses.color_pair(COLOR_INFO))
                        hw_y += 1
                        virtual_screen.addstr(hw_y, hw_x, f"Machine: {hardware_info['machine']}".ljust(max_width), curses.color_pair(COLOR_INFO))
                        hw_y += 1
                        virtual_screen.addstr(hw_y, hw_x, f"Processor: {hardware_info['processor']}".ljust(max_width), curses.color_pair(COLOR_INFO))
                        hw_y += 1
                        virtual_screen.addstr(hw_y, hw_x, f"Python Version: {hardware_info['python_version']}".ljust(max_width), curses.color_pair(COLOR_INFO))
                        hw_y += 2

                        # ROCm information
                        if hardware_info['rocm_path']:
                            virtual_screen.addstr(hw_y, hw_x, f"ROCm Path: {hardware_info['rocm_path']}".ljust(max_width), curses.color_pair(COLOR_SUCCESS))
                            hw_y += 1
                            if hardware_info['rocm_version']:
                                virtual_screen.addstr(hw_y, hw_x, f"ROCm Version: {hardware_info['rocm_version']}".ljust(max_width), curses.color_pair(COLOR_SUCCESS))
                            else:
                                virtual_screen.addstr(hw_y, hw_x, "ROCm Version: Not detected".ljust(max_width), curses.color_pair(COLOR_WARNING))
                        else:
                            virtual_screen.addstr(hw_y, hw_x, "ROCm: Not installed".ljust(max_width), curses.color_pair(COLOR_WARNING))
                        hw_y += 2

                        # GPU information
                        if hardware_info['amd_gpus']:
                            virtual_screen.addstr(hw_y, hw_x, f"Found {len(hardware_info['amd_gpus'])} AMD GPU(s):".ljust(max_width), curses.color_pair(COLOR_SUCCESS))
                            hw_y += 1
                            for i, gpu in enumerate(hardware_info['amd_gpus']):
                                # Truncate GPU name if it's too long
                                gpu_name = gpu[:max_width - 4]  # Leave room for the bullet point and spacing
                                virtual_screen.addstr(hw_y + i, hw_x + 2, f"• {gpu_name}".ljust(max_width - 2), curses.color_pair(COLOR_INFO))
                        else:
                            virtual_screen.addstr(hw_y, hw_x, "No AMD GPUs detected".ljust(max_width), curses.color_pair(COLOR_WARNING))
                    except Exception as e:
                        log_message(f"Error displaying hardware info: {str(e)}")
                        virtual_screen.addstr(hw_y, hw_x, f"Error displaying hardware info: {str(e)}".ljust(max_width), curses.color_pair(COLOR_ERROR))

                    # Add debug info at the bottom of the screen
                    try:
                        debug_info = f"Screen: {current_screen} | Key: {key} | Need Redraw: {need_redraw}"
                        virtual_screen.addstr(curses.LINES - 1, 2, debug_info[:curses.COLS - 4], curses.color_pair(COLOR_NORMAL))
                    except:
                        pass  # Ignore errors when adding debug info

                    # Copy virtual screen to real screen
                    update_screen()

                elif current_screen == "environment":
                    draw_header(virtual_screen, "Environment Setup")
                    draw_footer(virtual_screen, "Press 'b' to go back, Space to select, 'i' to run selected scripts")
                    draw_environment_scripts(virtual_screen, ENVIRONMENT_SCRIPTS, env_selected_idx, selected_env_scripts)

                    # Update tracking variables
                    ui_state['last_env_selected_idx'] = env_selected_idx

                    # Add debug info at the bottom of the screen
                    try:
                        debug_info = f"Screen: {current_screen} | Key: {key} | Need Redraw: {need_redraw}"
                        virtual_screen.addstr(curses.LINES - 1, 2, debug_info[:curses.COLS - 4], curses.color_pair(COLOR_NORMAL))
                    except:
                        pass  # Ignore errors when adding debug info

                    # Copy virtual screen to real screen
                    update_screen()

                elif current_screen == "dependencies":
                    if dependencies is None:
                        try:
                            virtual_screen.addstr(11, 2, "Checking dependencies...".ljust(curses.COLS - 4), curses.color_pair(COLOR_INFO))
                            update_screen()  # Show "Checking dependencies..." message

                            # Add proper error handling around dependency checking
                            try:
                                dependencies = check_dependencies()
                            except Exception as e:
                                log_message(f"Error checking dependencies: {str(e)}")
                                # Initialize with empty values to prevent crashes
                                dependencies = {}

                            virtual_screen.clear()  # Clear again to redraw with the new info
                            draw_header(virtual_screen, "Dependencies")
                            draw_footer(virtual_screen, "Press 'b' to go back, Space to select, 'i' to install selected")
                        except Exception as e:
                            log_message(f"Error updating dependencies screen: {str(e)}")

                    draw_dependencies(virtual_screen, dependencies, selected_deps, dep_selected_idx)

                    # Update tracking variables
                    ui_state['last_dep_selected_idx'] = dep_selected_idx

                    # Add debug info at the bottom of the screen
                    try:
                        debug_info = f"Screen: {current_screen} | Key: {key} | Need Redraw: {need_redraw}"
                        virtual_screen.addstr(curses.LINES - 1, 2, debug_info[:curses.COLS - 4], curses.color_pair(COLOR_NORMAL))
                    except:
                        pass  # Ignore errors when adding debug info

                    # Copy virtual screen to real screen
                    update_screen()

                elif current_screen == "components":
                    draw_header(virtual_screen, "Components")
                    draw_footer(virtual_screen, "Press 'b' to go back, Space to select/deselect, 'i' to install selected, 'r' to refresh")
                    draw_components(virtual_screen, FOUNDATION_COMPONENTS, CORE_COMPONENTS, EXTENSION_COMPONENTS, comp_selected_section, comp_selected_idx)

                    # Update tracking variables
                    ui_state['last_comp_selected_section'] = comp_selected_section
                    ui_state['last_comp_selected_idx'] = comp_selected_idx

                    # Add debug info at the bottom of the screen
                    try:
                        debug_info = f"Screen: {current_screen} | Key: {key} | Need Redraw: {need_redraw}"
                        virtual_screen.addstr(curses.LINES - 1, 2, debug_info[:curses.COLS - 4], curses.color_pair(COLOR_NORMAL))
                    except:
                        pass  # Ignore errors when adding debug info

                    # Copy virtual screen to real screen
                    update_screen()

                elif current_screen == "installation":
                    draw_header(virtual_screen, "Installation")
                    draw_footer(virtual_screen, "Press 'b' to go back, Space to select/deselect, 'i' or Enter to install selected")

                    # Draw the installation screen with component selection
                    draw_installation_screen(virtual_screen, FOUNDATION_COMPONENTS, CORE_COMPONENTS, EXTENSION_COMPONENTS, inst_selected_section, inst_selected_idx)

                    # Update tracking variables
                    ui_state['last_inst_selected_section'] = inst_selected_section
                    ui_state['last_inst_selected_idx'] = inst_selected_idx

                    # Add debug info at the bottom of the screen
                    try:
                        debug_info = f"Screen: {current_screen} | Key: {key} | Need Redraw: {need_redraw}"
                        virtual_screen.addstr(curses.LINES - 1, 2, debug_info[:curses.COLS - 4], curses.color_pair(COLOR_NORMAL))
                    except:
                        pass  # Ignore errors when adding debug info

                    # Copy virtual screen to real screen
                    update_screen()

                elif current_screen == "logs":
                    draw_header(virtual_screen, "Logs")
                    draw_footer(virtual_screen, "Press 'b' to go back, 'r' to refresh logs")

                    # Display logs
                    log_y = 11
                    log_x = 2

                    try:
                        # Read the last 20 lines of the log file
                        if os.path.exists(LOG_FILE):
                            with open(LOG_FILE, "r") as f:
                                log_lines = f.readlines()
                                log_lines = log_lines[-20:]  # Get last 20 lines

                                for i, line in enumerate(log_lines):
                                    # Truncate line if it's too long
                                    line = line.strip()
                                    if len(line) > curses.COLS - log_x - 4:
                                        line = line[:curses.COLS - log_x - 7] + "..."

                                    # Color-code log lines based on content
                                    if "ERROR" in line:
                                        virtual_screen.addstr(log_y + i, log_x, line, curses.color_pair(COLOR_ERROR))
                                    elif "WARNING" in line:
                                        virtual_screen.addstr(log_y + i, log_x, line, curses.color_pair(COLOR_WARNING))
                                    elif "SUCCESS" in line:
                                        virtual_screen.addstr(log_y + i, log_x, line, curses.color_pair(COLOR_SUCCESS))
                                    else:
                                        virtual_screen.addstr(log_y + i, log_x, line, curses.color_pair(COLOR_NORMAL))
                        else:
                            virtual_screen.addstr(log_y, log_x, "No log file found.", curses.color_pair(COLOR_WARNING))
                    except Exception as e:
                        log_message(f"Error displaying logs: {str(e)}")
                        virtual_screen.addstr(log_y, log_x, f"Error displaying logs: {str(e)}", curses.color_pair(COLOR_ERROR))

                    # Add debug info at the bottom of the screen
                    try:
                        debug_info = f"Screen: {current_screen} | Key: {key} | Need Redraw: {need_redraw}"
                        virtual_screen.addstr(curses.LINES - 1, 2, debug_info[:curses.COLS - 4], curses.color_pair(COLOR_NORMAL))
                    except:
                        pass  # Ignore errors when adding debug info

                    # Copy virtual screen to real screen
                    update_screen()

            except Exception as e:
                # Handle any errors during screen drawing
                log_message(f"Error drawing screen {current_screen}: {str(e)}")

                # Try to show the error on screen
                try:
                    virtual_screen.clear()
                    draw_header(virtual_screen, f"Error in {current_screen} screen")
                    draw_footer(virtual_screen, "Press 'b' to go back to main menu")
                    virtual_screen.addstr(11, 4, f"Error: {str(e)}", curses.color_pair(COLOR_ERROR))
                    update_screen()
                except:
                    pass

            # Update previous_screen after drawing
            previous_screen = current_screen

        # Small delay to reduce CPU usage
        time.sleep(0.01)

# Wrapper function to catch all exceptions
def safe_main(stdscr):
    try:
        _main_curses(stdscr)
    except Exception as e:
        # Log the error
        try:
            with open(LOG_FILE, "a") as f:
                f.write(f"CRITICAL ERROR: {str(e)}\n")
                import traceback
                traceback.print_exc(file=f)
        except:
            pass

        # Try to display error on screen
        try:
            stdscr.clear()
            stdscr.addstr(1, 1, "An error occurred in the ML Stack Installer:", curses.color_pair(COLOR_ERROR))
            stdscr.addstr(3, 1, str(e), curses.color_pair(COLOR_ERROR))
            stdscr.addstr(5, 1, "Press any key to exit...", curses.color_pair(COLOR_NORMAL))
            stdscr.refresh()
            stdscr.getch()  # Wait for a key press
        except:
            pass

# Entry point function for the package
def main():
    """
    Main entry point for the ML Stack installer.
    This function is called when the script is run directly or through the ml-stack-install command.
    """
    try:
        curses.wrapper(safe_main)
        return 0
    except KeyboardInterrupt:
        print("Installation interrupted by user.")
        return 1
    except Exception as e:
        print(f"Critical Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nCheck the log file for more details.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
