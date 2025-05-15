#!/usr/bin/env python3
"""
Enhanced ML Stack Installation Script with Textual UI

Author: Stanley Chisango (Scooter Lacroix)
Email: scooterlacroix@gmail.com
GitHub: https://github.com/scooter-lacroix
X: https://x.com/scooter_lacroix
Patreon: https://patreon.com/ScooterLacroix

If this code saved you time, consider buying me a coffee! ☕
"Code is like humor. When you have to explain it, it's bad!" - Cory House
"""

# Debug mode - set to True to enable debug mode
DEBUG_MODE = True

import os
import sys
import time
import subprocess
import platform
import shutil
import shlex
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable

# Global variable to store sudo password
SUDO_PASSWORD = None

try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical
    from textual.widgets import Header, Footer, Static, Button, Label, ProgressBar, Log, Input
    from textual.screen import Screen
    from textual import events
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.syntax import Syntax
    from rich.table import Table
except ImportError:
    print("Installing required packages using uv...")
    try:
        # First try using uv as specified in the instructions
        subprocess.run(["uv", "pip", "install", "textual", "rich"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to pip if uv is not available
        print("uv not found, falling back to pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "textual", "rich"], check=True)

    # Now import the modules
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical
    from textual.widgets import Header, Footer, Static, Button, Label, ProgressBar, Log
    from textual.screen import Screen
    from textual import events
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.syntax import Syntax
    from rich.table import Table

# ASCII Art Banner
BANNER = """
  ██████╗████████╗ █████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗         ███████╗████████╗ █████╗  ██████╗██╗  ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝    ████╗ ████║██║         ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
 ╚█████╗    ██║   ███████║██╔██╗ ██║███████╗    ██╔████╔██║██║         ███████╗   ██║   ███████║██║     █████╔╝
  ╚═══██╗   ██║   ██╔══██║██║╚██╗██║╚════██║    ██║╚██╔╝██║██║         ╚════██║   ██║   ██╔══██║██║     ██╔═██╗
 ██████╔╝   ██║   ██║  ██║██║ ╚████║███████║    ██║ ╚═╝ ██║███████╗    ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
"""

# Constants
HOME_DIR = os.path.expanduser("~")
MLSTACK_DIR = os.path.join(HOME_DIR, "Prod", "Stan-s-ML-Stack")
SCRIPTS_DIR = os.path.join(MLSTACK_DIR, "scripts")
LOGS_DIR = os.path.join(MLSTACK_DIR, "logs")
LOG_FILE = os.path.join(LOGS_DIR, f"ml_stack_install_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Console for rich output
console = Console()

# Components to install
CORE_COMPONENTS = [
    {
        "name": "ML Stack Core",
        "description": "Core ML Stack components",
        "script": "install_ml_stack.sh",
        "required": True,
        "status": "pending"
    },
    {
        "name": "Flash Attention",
        "description": "Efficient attention computation",
        "script": "install_flash_attention_ck.sh",
        "required": True,
        "status": "pending"
    }
]

EXTENSION_COMPONENTS = [
    {
        "name": "Triton",
        "description": "Compiler for parallel programming",
        "script": "install_triton.sh",
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
        "name": "vLLM",
        "description": "High-throughput inference engine for LLMs",
        "script": "install_vllm.sh",
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

# Helper functions
def log_message(message: str, level: str = "INFO") -> None:
    """Log a message to the log file and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{level}] {message}"

    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")

    if level == "ERROR":
        console.print(f"[bold red]{log_line}[/bold red]")
    elif level == "WARNING":
        console.print(f"[bold yellow]{log_line}[/bold yellow]")
    elif level == "SUCCESS":
        console.print(f"[bold green]{log_line}[/bold green]")
    else:
        console.print(log_line)

def run_command(command: str, cwd: Optional[str] = None, timeout: Optional[int] = 300, use_sudo: bool = False) -> Tuple[int, str, str]:
    """Run a shell command and return the return code, stdout, and stderr with a timeout."""
    global SUDO_PASSWORD

    # If the command starts with sudo and we have a sudo password, use it
    if command.strip().startswith("sudo") and SUDO_PASSWORD:
        log_message(f"Running sudo command with cached credentials: {command}")
        # Modify the command to use the stored sudo password
        command = f"echo {shlex.quote(SUDO_PASSWORD)} | sudo -S {command[5:]}"
    elif use_sudo and SUDO_PASSWORD:
        log_message(f"Running command with sudo: {command}")
        # Add sudo with password to the command
        command = f"echo {shlex.quote(SUDO_PASSWORD)} | sudo -S {command}"
    else:
        log_message(f"Running command: {command}")

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            log_message(f"Command timed out after {timeout} seconds: {command}", "WARNING")
            return 1, "", f"Command timed out after {timeout} seconds"

        if return_code != 0:
            log_message(f"Command failed with return code {return_code}", "ERROR")
            log_message(f"stderr: {stderr}", "ERROR")

        return return_code, stdout, stderr
    except Exception as e:
        log_message(f"Error executing command: {str(e)}", "ERROR")
        return 1, "", f"Error: {str(e)}"

def detect_hardware(progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Detect hardware information with progress updates via callback."""
    def update_progress(message: str):
        """Update progress if callback is provided."""
        if progress_callback:
            progress_callback(message)
        log_message(message)

    update_progress("Gathering basic system information...")
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

    # Simplified hardware detection to avoid hanging
    update_progress("Detecting ROCm installation...")

    # Check if ROCm is installed
    return_code, stdout, stderr = run_command("which rocminfo", timeout=5)
    if return_code == 0 and stdout.strip():
        update_progress("ROCm is installed")
        hardware_info["rocm_path"] = os.path.dirname(os.path.dirname(stdout.strip()))

        # Try to get ROCm version
        return_code, stdout, stderr = run_command("ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\\+\\.[0-9]\\+\\.[0-9]\\+' | head -n 1", timeout=5)
        if return_code == 0 and stdout.strip():
            hardware_info["rocm_version"] = stdout.strip()
            update_progress(f"ROCm version: {hardware_info['rocm_version']}")
    else:
        update_progress("ROCm is not installed or not in PATH")

        # Check if ROCm is installed in common locations
        return_code, stdout, stderr = run_command("ls -d /opt/rocm* 2>/dev/null", timeout=5)
        if return_code == 0 and stdout.strip():
            hardware_info["rocm_path"] = stdout.strip().split("\n")[0]
            update_progress(f"Found potential ROCm installation at {hardware_info['rocm_path']}")

            # Try to get ROCm version
            rocm_version = hardware_info["rocm_path"].split("-")[-1] if "-" in hardware_info["rocm_path"] else None
            if rocm_version:
                hardware_info["rocm_version"] = rocm_version
                update_progress(f"ROCm version: {hardware_info['rocm_version']}")

    # Detect AMD GPUs using lspci
    update_progress("Detecting AMD GPUs...")
    return_code, stdout, stderr = run_command("lspci | grep -i 'amd\\|radeon\\|advanced micro devices' | grep -i 'vga\\|3d\\|display'", timeout=5)
    if return_code == 0 and stdout.strip():
        hardware_info["amd_gpus"] = [line.strip() for line in stdout.strip().split("\n")]
        update_progress(f"Found {len(hardware_info['amd_gpus'])} AMD GPU(s)")
        for gpu in hardware_info["amd_gpus"]:
            update_progress(f"  - {gpu}")
    else:
        update_progress("No AMD GPUs detected using lspci or command failed")

    # Hardware detection completed
    update_progress("Hardware detection completed")

    # Return the hardware info
    return hardware_info

def detect_hardware_original(progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Original hardware detection method as fallback."""
    def update_progress(message: str):
        """Update progress if callback is provided."""
        if progress_callback:
            progress_callback(message)
        log_message(message)

    update_progress("Gathering basic system information...")
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

    # Detect AMD GPUs
    try:
        if hardware_info["system"] == "Linux":
            # Try using lspci with a short timeout
            update_progress("Detecting AMD GPUs using lspci...")
            return_code, stdout, stderr = run_command(
                "lspci | grep -i 'amd\\|radeon\\|advanced micro devices' | grep -i 'vga\\|3d\\|display'",
                timeout=5
            )
            if return_code == 0 and stdout.strip():
                hardware_info["amd_gpus"] = [line.strip() for line in stdout.strip().split("\n")]
                update_progress(f"Found {len(hardware_info['amd_gpus'])} AMD GPU(s) using lspci")
            else:
                update_progress("No AMD GPUs detected using lspci or command failed")

            # Check if ROCm is installed
            update_progress("Checking for ROCm installation...")
            return_code, stdout, stderr = run_command("which rocminfo", timeout=5)

            if return_code == 0 and stdout.strip():
                rocminfo_path = stdout.strip()
                hardware_info["rocm_path"] = os.path.dirname(os.path.dirname(rocminfo_path))
                update_progress(f"ROCm found at: {hardware_info['rocm_path']}")

                # Get GPU information from rocminfo with a timeout
                update_progress("Getting GPU details from rocminfo...")
                return_code, stdout, stderr = run_command(
                    "rocminfo | grep -A 10 'Device Type:.*GPU' | grep 'Marketing Name'",
                    timeout=10
                )

                if return_code == 0 and stdout.strip():
                    try:
                        gpu_names = [line.split(":", 1)[1].strip() for line in stdout.strip().split("\n")]
                        hardware_info["amd_gpus"] = gpu_names
                        update_progress(f"Found {len(gpu_names)} AMD GPU(s) using rocminfo")
                    except Exception as e:
                        update_progress(f"Error parsing rocminfo output: {str(e)}")
                else:
                    update_progress("Could not get GPU details from rocminfo")

                # Get ROCm version with a timeout
                update_progress("Getting ROCm version...")
                return_code, stdout, stderr = run_command(
                    "rocminfo | grep -i 'ROCm Version'",
                    timeout=5
                )

                if return_code == 0 and stdout.strip():
                    try:
                        hardware_info["rocm_version"] = stdout.split(":", 1)[1].strip()
                        update_progress(f"ROCm version: {hardware_info['rocm_version']}")
                    except Exception as e:
                        update_progress(f"Error parsing ROCm version: {str(e)}")
                else:
                    # Try alternative method to get ROCm version
                    update_progress("Trying alternative method to get ROCm version...")
                    return_code, stdout, stderr = run_command(
                        "ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\\+\\.[0-9]\\+\\.[0-9]\\+' | head -n 1",
                        timeout=5
                    )
                    if return_code == 0 and stdout.strip():
                        hardware_info["rocm_version"] = stdout.strip()
                        update_progress(f"ROCm version (from path): {hardware_info['rocm_version']}")
                    else:
                        update_progress("Could not determine ROCm version")
            else:
                update_progress("ROCm is not installed or not in PATH")
                # Check if ROCm is installed in common locations
                return_code, stdout, stderr = run_command("ls -d /opt/rocm* 2>/dev/null", timeout=5)
                if return_code == 0 and stdout.strip():
                    hardware_info["rocm_path"] = stdout.strip().split("\n")[0]
                    update_progress(f"Found potential ROCm installation at {hardware_info['rocm_path']}")
                else:
                    update_progress("Could not find ROCm installation in common locations")
    except Exception as e:
        update_progress(f"Error detecting hardware: {str(e)}")

    update_progress("Hardware detection completed")
    return hardware_info

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed."""
    log_message("Starting dependency check")
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
        try:
            # Use a short timeout to prevent hanging
            return_code, stdout, stderr = run_command(f"dpkg -l | grep -q 'ii  {dep} '", timeout=5)
            dependencies[dep] = return_code == 0
            log_message(f"Dependency {dep}: {'Installed' if dependencies[dep] else 'Not installed'}")
        except Exception as e:
            log_message(f"Error checking dependency {dep}: {str(e)}", "ERROR")
            dependencies[dep] = False

    log_message("Dependency check completed")
    return dependencies

def install_dependencies(missing_deps: List[str]) -> bool:
    """Install missing dependencies."""
    if not missing_deps:
        log_message("No missing dependencies to install")
        return True

    log_message(f"Installing missing dependencies: {', '.join(missing_deps)}")

    try:
        # Use sudo with the cached password
        log_message("Running apt-get update")
        update_cmd = "apt-get update"
        update_return_code, _, update_stderr = run_command(update_cmd, use_sudo=True, timeout=60)
        if update_return_code != 0:
            log_message(f"Failed to update package lists: {update_stderr}", "ERROR")
            return False

        # Install each dependency separately to avoid hanging if one fails
        for dep in missing_deps:
            log_message(f"Installing {dep}")
            install_cmd = f"apt-get install -y {dep}"
            install_return_code, _, install_stderr = run_command(install_cmd, use_sudo=True, timeout=60)
            if install_return_code != 0:
                log_message(f"Failed to install {dep}: {install_stderr}", "ERROR")
                continue

            # Verify installation - add a small delay to allow package database to update
            log_message(f"Waiting for package database to update...")
            time.sleep(2)  # Add a short delay

            # Use a more reliable verification method
            verify_cmd = f"dpkg-query -W -f='${{Status}}' {dep} 2>/dev/null | grep -q 'install ok installed'"
            verify_return_code, _, _ = run_command(verify_cmd, timeout=5)
            if verify_return_code != 0:
                log_message(f"Failed to verify installation of {dep}", "ERROR")
                continue

            log_message(f"Successfully installed {dep}")

        # Final verification
        log_message("Verifying all dependencies")
        all_installed = True
        for dep in missing_deps:
            # Use the more reliable verification method
            verify_cmd = f"dpkg-query -W -f='${{Status}}' {dep} 2>/dev/null | grep -q 'install ok installed'"
            return_code, _, _ = run_command(verify_cmd, timeout=5)
            if return_code != 0:
                log_message(f"Dependency {dep} is still missing", "ERROR")
                all_installed = False

        if all_installed:
            log_message("All dependencies installed successfully")
            return True
        else:
            log_message("Some dependencies failed to install", "WARNING")
            return False
    except Exception as e:
        log_message(f"Error installing dependencies: {str(e)}", "ERROR")
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}", "ERROR")
        return False

def setup_environment() -> bool:
    """Set up the environment for ML Stack."""
    log_message("Setting up environment...")

    # Create a simple environment setup script that checks for uv
    script_content = """#!/bin/bash
echo "Setting up environment for ML Stack..."
echo "Checking for required tools..."

# Check for git
if ! command -v git &> /dev/null; then
    echo "git is not installed. Please install git and try again."
    exit 1
fi

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "python3 is not installed. Please install python3 and try again."
    exit 1
fi

# Check for uv (preferred package manager)
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Checking for pip as fallback..."

    # Check for pip as fallback
    if ! command -v pip3 &> /dev/null; then
        echo "Neither uv nor pip3 is installed. Please install uv or pip3 and try again."
        exit 1
    fi

    echo "Using pip3 as fallback package manager."
else
    echo "Using uv as package manager."
fi

echo "Environment setup completed successfully."
exit 0
"""

    # Create a temporary script
    script_path = os.path.join(SCRIPTS_DIR, "temp_setup_environment.sh")
    with open(script_path, "w") as f:
        f.write(script_content)

    # Make the script executable
    os.chmod(script_path, 0o755)

    # Run the script
    return_code, stdout, stderr = run_command(f"bash {script_path}")
    if return_code != 0:
        log_message(f"Environment setup failed: {stderr}", "ERROR")
        return False

    log_message("Environment setup completed successfully", "SUCCESS")
    return True

def install_component(component: Dict[str, Any], log_widget: Optional[Any] = None) -> bool:
    """Install a component."""
    name = component["name"]
    script = component["script"]
    script_path = os.path.join(SCRIPTS_DIR, script)

    if not os.path.exists(script_path):
        log_message(f"Installation script not found for {name}: {script_path}", "ERROR")
        if log_widget:
            log_widget.write(f"[bold red]ERROR: Installation script not found for {name}[/bold red]")
        return False

    # Make the script executable
    os.chmod(script_path, 0o755)

    # Update component status
    component["status"] = "installing"
    if log_widget:
        log_widget.write(f"[bold blue]Installing {name}...[/bold blue]")

    # Modify the script to use uv for Python package installations
    # Create a temporary modified script that uses uv
    temp_script_path = os.path.join(SCRIPTS_DIR, f"temp_{script}")
    try:
        # Read the original script
        with open(script_path, 'r') as f:
            script_content = f.read()

        # Replace pip install commands with uv pip install
        modified_script = script_content.replace('pip install', 'uv pip install')
        modified_script = modified_script.replace('pip3 install', 'uv pip install')
        modified_script = modified_script.replace('python -m pip install', 'uv pip install')
        modified_script = modified_script.replace('python3 -m pip install', 'uv pip install')

        # Replace python3 -m uv pip install with direct uv command
        modified_script = modified_script.replace('python3 -m uv pip install', 'uv pip install')
        modified_script = modified_script.replace('python -m uv pip install', 'uv pip install')

        # Add fallback to pip if uv is not available
        uv_check = """
# Check if uv is available, otherwise fall back to pip
if ! command -v uv &> /dev/null; then
    echo "uv not found, falling back to pip"
    alias uv="pip"
fi

# Create a function to handle uv commands properly
uv_pip_install() {
    # Check if uv is available as a command
    if command -v uv &> /dev/null; then
        # Use uv directly as a command
        uv pip install "$@"
    else
        # Fall back to pip
        python3 -m pip install "$@"
    fi
}

# Export the uv wrapper function
export -f uv_pip_install
"""
        # Insert the uv check after the shebang line
        if modified_script.startswith('#!/'):
            shebang_end = modified_script.find('\n') + 1
            modified_script = modified_script[:shebang_end] + uv_check + modified_script[shebang_end:]
        else:
            modified_script = "#!/bin/bash\n" + uv_check + modified_script

        # Write the modified script
        with open(temp_script_path, 'w') as f:
            f.write(modified_script)

        # Make the temporary script executable
        os.chmod(temp_script_path, 0o755)

        # Run the modified installation script
        log_message(f"Installing {name} using uv package manager...")
        return_code, stdout, stderr = run_command(f"bash {temp_script_path}", timeout=600)  # 10 minutes timeout

        # Clean up the temporary script
        try:
            os.unlink(temp_script_path)
        except:
            pass
    except Exception as e:
        log_message(f"Error modifying script to use uv: {str(e)}", "ERROR")
        # Fall back to the original script
        log_message(f"Falling back to original script for {name}...")
        return_code, stdout, stderr = run_command(f"bash {script_path}", timeout=600)  # 10 minutes timeout

    if return_code != 0:
        log_message(f"Installation of {name} failed: {stderr}", "ERROR")
        component["status"] = "failed"
        if log_widget:
            log_widget.write(f"[bold red]ERROR: Installation of {name} failed[/bold red]")
            log_widget.write(stderr)
        return False

    # Update component status
    component["status"] = "installed"
    log_message(f"Installation of {name} completed successfully", "SUCCESS")
    if log_widget:
        log_widget.write(f"[bold green]SUCCESS: {name} installed successfully[/bold green]")

    return True

def check_ml_stack_components() -> Dict[str, bool]:
    """Check which ML Stack components are installed using the component detector script."""
    components = {
        "pytorch": False,
        "onnxruntime": False,
        "migraphx": False,
        "flash_attention": False,
        "rccl": False,
        "mpi": False,
        "megatron": False,
        "triton": False,
        "bitsandbytes": False,
        "vllm": False,
        "rocm_smi": False,
        "pytorch_profiler": False,
        "wandb": False
    }

    # Create a temporary script to capture the output of the component detector
    temp_script = os.path.join(SCRIPTS_DIR, "temp_component_check.sh")
    with open(temp_script, "w") as f:
        f.write("""#!/bin/bash
# Source the component detector library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
DETECTOR_SCRIPT="$PARENT_DIR/scripts/ml_stack_component_detector.sh"

if [ -f "$DETECTOR_SCRIPT" ]; then
    source "$DETECTOR_SCRIPT"
else
    echo "Error: Component detector script not found at $DETECTOR_SCRIPT"
    exit 1
fi

# Check components
check_pytorch > /dev/null
pytorch_installed=$?

check_onnxruntime > /dev/null
onnxruntime_installed=$?

check_migraphx > /dev/null
migraphx_installed=$?

check_flash_attention > /dev/null
flash_attention_installed=$?

check_rccl > /dev/null
rccl_installed=$?

check_mpi > /dev/null
mpi_installed=$?

check_megatron > /dev/null
megatron_installed=$?

check_triton > /dev/null
triton_installed=$?

check_bitsandbytes > /dev/null
bitsandbytes_installed=$?

check_vllm > /dev/null
vllm_installed=$?

check_rocm_smi > /dev/null
rocm_smi_installed=$?

check_pytorch_profiler > /dev/null
pytorch_profiler_installed=$?

check_wandb > /dev/null
wandb_installed=$?

# Output component status in JSON format
echo "{"
echo "  \\"pytorch\\": $([ $pytorch_installed -eq 0 ] && echo "true" || echo "false"),"
echo "  \\"onnxruntime\\": $([ $onnxruntime_installed -eq 0 ] && echo "true" || echo "false"),"
echo "  \\"migraphx\\": $([ $migraphx_installed -eq 0 ] && echo "true" || echo "false"),"
echo "  \\"flash_attention\\": $([ $flash_attention_installed -eq 0 ] && echo "true" || echo "false"),"
echo "  \\"rccl\\": $([ $rccl_installed -eq 0 ] && echo "true" || echo "false"),"
echo "  \\"mpi\\": $([ $mpi_installed -eq 0 ] && echo "true" || echo "false"),"
echo "  \\"megatron\\": $([ $megatron_installed -eq 0 ] && echo "true" || echo "false"),"
echo "  \\"triton\\": $([ $triton_installed -eq 0 ] && echo "true" || echo "false"),"
echo "  \\"bitsandbytes\\": $([ $bitsandbytes_installed -eq 0 ] && echo "true" || echo "false"),"
echo "  \\"vllm\\": $([ $vllm_installed -eq 0 ] && echo "true" || echo "false"),"
echo "  \\"rocm_smi\\": $([ $rocm_smi_installed -eq 0 ] && echo "true" || echo "false"),"
echo "  \\"pytorch_profiler\\": $([ $pytorch_profiler_installed -eq 0 ] && echo "true" || echo "false"),"
echo "  \\"wandb\\": $([ $wandb_installed -eq 0 ] && echo "true" || echo "false")"
echo "}"
""")

    # Make the script executable
    os.chmod(temp_script, 0o755)

    # Run the script
    log_message("Running component detection script...")
    return_code, stdout, stderr = run_command(f"bash {temp_script}", timeout=30)

    # Clean up
    os.remove(temp_script)

    if return_code == 0 and stdout.strip():
        try:
            # Parse the JSON output
            import json
            component_data = json.loads(stdout.strip())

            # Update components with the detected values
            for key in components:
                if key in component_data:
                    components[key] = component_data[key]

            log_message(f"Component detection completed: {json.dumps(components, indent=2)}")
        except Exception as e:
            log_message(f"Error parsing component detection output: {str(e)}", "ERROR")

            # Fallback to the original detection method
            log_message("Falling back to original component detection method...")
            return check_ml_stack_components_original()
    else:
        log_message(f"Component detection script failed: {stderr}", "ERROR")

        # Fallback to the original detection method
        log_message("Falling back to original component detection method...")
        return check_ml_stack_components_original()

    return components

def check_ml_stack_components_original() -> Dict[str, bool]:
    """Original method to check which ML Stack components are installed."""
    components = {
        "pytorch": False,
        "onnxruntime": False,
        "migraphx": False,
        "flash_attention": False,
        "rccl": False,
        "mpi": False,
        "megatron": False,
        "triton": False,
        "bitsandbytes": False,
        "vllm": False,
        "rocm_smi": False,
        "pytorch_profiler": False,
        "wandb": False
    }

    # Define search paths for components
    search_paths = [
        "/home/stan/pytorch",
        "/home/stan/ml_stack/flash_attn_amd_direct",
        "/home/stan/ml_stack/flash_attn_amd",
        "/home/stan/ml_stack/flash_attn_amd/build/lib.linux-x86_64-cpython-313",
        "/home/stan/.local/lib/python3.13/site-packages",
        "/home/stan/rocm_venv/lib/python3.13/site-packages",
        "/home/stan/megatron/Megatron-LM",
        "/home/stan/onnxruntime_build",
        "/home/stan/migraphx_build",
        "/home/stan/migraphx_package",
        "/home/stan/vllm_build",
        "/home/stan/vllm_py313",
        "/home/stan/ml_stack/bitsandbytes/bitsandbytes"
    ]

    # Helper function to check if a component is installed by directory
    def check_component_dir(component_name, paths):
        for path in paths:
            if os.path.exists(os.path.join(path, component_name)):
                return True
        return False

    # Try import first, then fall back to directory check

    # Check PyTorch
    try:
        import torch
        components["pytorch"] = True
    except ImportError:
        components["pytorch"] = check_component_dir("torch", search_paths) or os.path.exists("/home/stan/pytorch")

    # Check PyTorch Profiler
    try:
        from torch.profiler import profile
        components["pytorch_profiler"] = True
    except ImportError:
        for path in search_paths:
            if os.path.exists(os.path.join(path, "torch/profiler")):
                components["pytorch_profiler"] = True
                break

    # Check ONNX Runtime
    try:
        import onnxruntime
        components["onnxruntime"] = True
    except ImportError:
        components["onnxruntime"] = check_component_dir("onnxruntime", search_paths) or os.path.exists("/home/stan/onnxruntime_build")

    # Check MIGraphX
    try:
        import migraphx
        components["migraphx"] = True
    except ImportError:
        components["migraphx"] = check_component_dir("migraphx", search_paths) or os.path.exists("/home/stan/migraphx_build") or os.path.exists("/home/stan/migraphx_package")

    # Check Flash Attention
    try:
        import flash_attention_amd
        components["flash_attention"] = True
    except ImportError:
        try:
            import flash_attn
            components["flash_attention"] = True
        except ImportError:
            components["flash_attention"] = check_component_dir("flash_attention_amd", search_paths) or os.path.exists("/home/stan/ml_stack/flash_attn_amd") or os.path.exists("/home/stan/ml_stack/flash_attn_amd_direct")

    # Check RCCL
    if os.path.exists("/opt/rocm/lib/librccl.so"):
        components["rccl"] = True

    # Check MPI
    try:
        return_code, stdout, stderr = run_command("which mpirun", timeout=5)
        if return_code == 0 and stdout.strip():
            components["mpi"] = True
    except Exception:
        pass

    # Check Megatron-LM
    try:
        import megatron
        components["megatron"] = True
    except ImportError:
        components["megatron"] = check_component_dir("megatron", search_paths) or os.path.exists("/home/stan/megatron/Megatron-LM")

    # Check Triton
    try:
        import triton
        components["triton"] = True
    except ImportError:
        components["triton"] = check_component_dir("triton", search_paths)

    # Check BITSANDBYTES
    try:
        import bitsandbytes
        components["bitsandbytes"] = True
    except ImportError:
        components["bitsandbytes"] = check_component_dir("bitsandbytes", search_paths) or os.path.exists("/home/stan/ml_stack/bitsandbytes")

    # Check vLLM
    try:
        import vllm
        components["vllm"] = True
    except ImportError:
        components["vllm"] = check_component_dir("vllm", search_paths) or os.path.exists("/home/stan/vllm_build") or os.path.exists("/home/stan/vllm_py313")

    # Check ROCm SMI
    try:
        from rocm_smi_lib import rsmi
        components["rocm_smi"] = True
    except ImportError:
        components["rocm_smi"] = os.path.exists("/opt/rocm/bin/rocm-smi")

    # Check Weights & Biases
    try:
        import wandb
        components["wandb"] = True
    except ImportError:
        components["wandb"] = check_component_dir("wandb", search_paths)

    return components

def verify_installation() -> Dict[str, bool]:
    """Verify the installation of components."""
    verification_results = {}

    # Use the component detector to check installed components
    components = check_ml_stack_components_original()

    # Map component names to verification results
    component_mapping = {
        "ROCm Platform": "rocm",
        "PyTorch": "pytorch",
        "ONNX Runtime": "onnxruntime",
        "MIGraphX": "migraphx",
        "Flash Attention": "flash_attention",
        "RCCL": "rccl",
        "MPI": "mpi",
        "Megatron-LM": "megatron",
        "Triton": "triton",
        "BITSANDBYTES": "bitsandbytes",
        "vLLM": "vllm",
        "ROCm SMI": "rocm_smi",
        "PyTorch Profiler": "pytorch_profiler",
        "Weights & Biases": "wandb"
    }

    # Map component detection results to verification results
    for component in CORE_COMPONENTS + EXTENSION_COMPONENTS:
        name = component["name"]
        if name in component_mapping and component_mapping[name] in components:
            verification_results[name] = components[component_mapping[name]]
        else:
            # Fallback to checking if the component is in the output
            verification_results[name] = False

    # If the component detector didn't work, fall back to the original method
    if not verification_results:
        # Run the verification script
        script_path = os.path.join(SCRIPTS_DIR, "verify_installation.sh")
        if not os.path.exists(script_path):
            log_message(f"Verification script not found: {script_path}", "ERROR")
            return verification_results

        # Make the script executable
        os.chmod(script_path, 0o755)

        # Run the script
        log_message("Verifying installation...")
        return_code, stdout, stderr = run_command(f"bash {script_path}")

        # Parse verification results
        for component in CORE_COMPONENTS + EXTENSION_COMPONENTS:
            name = component["name"]
            if f"{name}: Installed" in stdout:
                verification_results[name] = True
            else:
                verification_results[name] = False

    return verification_results

# Textual UI Classes
class SudoPasswordInput(Screen):
    """Screen to get sudo password from the user."""

    def compose(self) -> ComposeResult:
        """Compose the sudo password input screen."""
        yield Header(show_clock=True)

        with Container(id="sudo-container"):
            yield Static(BANNER, id="banner")
            yield Static("ML Stack Installation - Sudo Access Required", id="welcome-title")
            yield Static("This installation requires sudo access to install dependencies and components.", id="welcome-description")
            yield Static("Please enter your sudo password to continue:", id="password-prompt")
            yield Input(placeholder="Enter sudo password", password=True, id="sudo-password")
            yield Button("Continue", id="sudo-continue-button")
            yield Static("", id="sudo-status")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Event handler called when a button is pressed."""
        if event.button.id == "sudo-continue-button":
            self.verify_sudo_password()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Event handler called when an input is submitted."""
        if event.input.id == "sudo-password":
            self.verify_sudo_password()

    def verify_sudo_password(self) -> None:
        """Verify the sudo password and proceed if valid."""
        try:
            log_message("Starting sudo password verification")
            password = self.query_one("#sudo-password").value
            if not password:
                self.query_one("#sudo-status").update("[bold red]Please enter your sudo password[/bold red]")
                log_message("No password entered")
                return

            # Update status
            self.query_one("#sudo-status").update("[bold blue]Verifying sudo access...[/bold blue]")
            log_message("Verifying sudo access")

            # Create a temporary script to verify sudo access
            temp_script = os.path.join(SCRIPTS_DIR, "temp_sudo_verify.sh")
            with open(temp_script, "w") as f:
                f.write("""#!/bin/bash
echo "$1" | sudo -S echo "Success" 2>/dev/null
if [ $? -eq 0 ]; then
    # Cache sudo credentials for a while
    echo "$1" | sudo -S echo "Caching sudo credentials" > /dev/null 2>&1
    exit 0
else
    exit 1
fi
""")

            # Make the script executable
            os.chmod(temp_script, 0o755)
            log_message(f"Created verification script at {temp_script}")

            # Run the script
            log_message("Running sudo verification script")
            return_code, stdout, stderr = run_command(f"bash {temp_script} {shlex.quote(password)}")
            log_message(f"Verification script returned: {return_code}")

            # Clean up
            os.remove(temp_script)
            log_message("Removed verification script")

            if return_code == 0:
                # Store the password in a global variable for later use
                global SUDO_PASSWORD
                SUDO_PASSWORD = password
                log_message("Sudo password verified successfully")

                # Create the welcome screen first
                welcome_screen = WelcomeScreen()
                log_message("Welcome screen created")

                # Proceed to the welcome screen
                log_message("Pushing welcome screen")
                try:
                    self.app.push_screen(welcome_screen)
                    log_message("Welcome screen pushed successfully")
                except Exception as e:
                    log_message(f"Error pushing welcome screen: {str(e)}", "ERROR")
                    self.query_one("#sudo-status").update(f"[bold red]Error: {str(e)}[/bold red]")
            else:
                log_message("Incorrect sudo password")
                self.query_one("#sudo-status").update("[bold red]Incorrect sudo password. Please try again.[/bold red]")
        except Exception as e:
            log_message(f"Error in verify_sudo_password: {str(e)}", "ERROR")
            import traceback
            log_message(f"Traceback: {traceback.format_exc()}", "ERROR")
            try:
                self.query_one("#sudo-status").update(f"[bold red]Error: {str(e)}[/bold red]")
            except Exception as inner_e:
                log_message(f"Error updating status: {str(inner_e)}", "ERROR")


class WelcomeScreen(Screen):
    """Welcome screen with hardware detection and dependency check."""

    def compose(self) -> ComposeResult:
        """Compose the welcome screen."""
        log_message("WelcomeScreen compose method called")
        yield Header(show_clock=True)

        with Container(id="welcome-container"):
            # Use the PNG logo image if available, otherwise fall back to ASCII banner
            logo_content = self._get_logo_image()
            if not logo_content:
                logo_content = BANNER
            yield Static(logo_content, id="banner")
            yield Static("Welcome to the ML Stack Installation", id="welcome-title")
            yield Static("This wizard will guide you through the installation of the ML Stack for AMD GPUs.", id="welcome-description")

            yield Static("Detecting hardware...", id="hardware-status")
            yield Static("", id="hardware-info")

            yield Static("Checking dependencies...", id="dependencies-status")
            yield Static("", id="dependencies-info")

            yield Button("Continue", id="continue-button", disabled=True)

        yield Footer()

    def _get_logo_image(self) -> str:
        """Get the ML Stack logo image with colored ASCII art."""
        import os
        import subprocess

        # Path to the logo image
        logo_image_path = os.path.join(MLSTACK_DIR, "assets/ml_stack_logo.png")
        log_message(f"Looking for logo at: {logo_image_path}")

        # Check if the image exists and is a valid image file
        if os.path.exists(logo_image_path) and os.path.getsize(logo_image_path) > 100:
            try:
                # Use jp2a to convert the PNG to colored ASCII art
                # Capture the output directly - Textual/Rich can handle ANSI color codes
                jp2a_cmd = f"jp2a --width=80 --height=40 --colors {logo_image_path}"
                result = subprocess.run(jp2a_cmd, shell=True, capture_output=True, text=True)

                if result.returncode == 0 and result.stdout:
                    log_message("Successfully generated colored ASCII art with jp2a")
                    # Return the colored ASCII art directly - Textual can handle ANSI color codes
                    return result.stdout
                else:
                    log_message("Failed to generate colored ASCII art with jp2a, falling back to banner")
            except Exception as e:
                log_message(f"Error using jp2a: {str(e)}", "ERROR")

        # If we can't display the image, return the ASCII banner
        return BANNER

    def _update_hardware_progress(self, message: str) -> None:
        """Update the hardware info widget with progress messages."""
        try:
            # Escape any square brackets in the message to prevent markup interpretation
            escaped_message = message.replace("[", "\\[").replace("]", "\\]")
            hardware_info_widget = self.query_one("#hardware-info")
            current_text = hardware_info_widget.renderable
            hardware_info_widget.update(f"{current_text}{escaped_message}\n")
        except Exception as e:
            log_message(f"Error in _update_hardware_progress: {str(e)}", "ERROR")

    def on_mount(self) -> None:
        """Event handler called when the screen is mounted."""
        try:
            log_message("WelcomeScreen mounted")
            # Start hardware detection in the background after a short delay
            log_message("Scheduling hardware detection")
            self.app.set_timer(0.5, self.detect_hardware)
            log_message("Hardware detection scheduled")
        except Exception as e:
            log_message(f"Error in WelcomeScreen.on_mount: {str(e)}", "ERROR")
            import traceback
            log_message(f"Traceback: {traceback.format_exc()}", "ERROR")

    def detect_hardware(self) -> None:
        """Detect hardware and check dependencies."""
        try:
            log_message("Starting hardware detection")
            # Detect hardware
            self.query_one("#hardware-status").update("Detecting hardware...")

            # Create a progress indicator
            hardware_info_widget = self.query_one("#hardware-info")
            hardware_info_widget.update("Starting hardware detection...\n")

            # Define progress update function
            def update_progress(message: str) -> None:
                log_message(f"Progress update: {message}")
                self._update_hardware_progress(message)

            # Run hardware detection directly (not in a worker thread)
            log_message("Running hardware detection")
            try:
                hardware_info = detect_hardware(update_progress)
                log_message("Hardware detection completed")
            except Exception as e:
                log_message(f"Error in hardware detection: {str(e)}", "ERROR")
                hardware_info = {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "python_version": platform.python_version(),
                    "amd_gpus": [],
                    "rocm_version": None,
                    "rocm_path": None,
                    "error": str(e)
                }

            # Update hardware info with a summary
            log_message("Updating hardware info display")

            # Escape any square brackets in the hardware info to prevent markup interpretation
            def escape_text(text):
                return str(text).replace("[", "\\[").replace("]", "\\]")

            hardware_text = f"System: {escape_text(hardware_info['system'])} {escape_text(hardware_info['release'])}\n"
            hardware_text += f"Processor: {escape_text(hardware_info['processor'])}\n"
            hardware_text += f"Python Version: {escape_text(hardware_info['python_version'])}\n"

            if "error" in hardware_info and hardware_info["error"]:
                hardware_text += f"Error during detection: {escape_text(hardware_info['error'])}\n"

            if hardware_info["amd_gpus"]:
                hardware_text += f"AMD GPUs detected: {len(hardware_info['amd_gpus'])}\n"
                for i, gpu in enumerate(hardware_info["amd_gpus"]):
                    hardware_text += f"  - GPU {i}: {escape_text(gpu)}\n"
            else:
                hardware_text += "No AMD GPUs detected. The ML Stack requires AMD GPUs.\n"

            if hardware_info["rocm_version"]:
                hardware_text += f"ROCm Version: {escape_text(hardware_info['rocm_version'])}\n"
                hardware_text += f"ROCm Path: {escape_text(hardware_info['rocm_path'])}\n"
            else:
                hardware_text += "ROCm not detected. The ML Stack requires ROCm.\n"

            # Update the hardware info widget with the detected information
            hardware_info_widget.update(hardware_text)

            # Update the hardware status based on detection results
            if hardware_info["amd_gpus"] and hardware_info["rocm_version"]:
                self.query_one("#hardware-status").update("Hardware detection completed successfully.")
            else:
                self.query_one("#hardware-status").update("Hardware detection completed with warnings.")

            # Check dependencies
            log_message("Checking dependencies")
            self.query_one("#dependencies-status").update("Checking dependencies...")

            # Run dependency check directly (not in a worker thread)
            try:
                dependencies = check_dependencies()
                log_message("Dependency check completed")
            except Exception as e:
                log_message(f"Error checking dependencies: {str(e)}", "ERROR")
                dependencies = {}

            # Update dependencies info
            log_message("Updating dependencies info display")
            dependencies_text = "Dependencies:\n"
            missing_deps = []

            for dep, installed in dependencies.items():
                # Use plain text checkmarks instead of special characters to avoid rendering issues
                status = "+" if installed else "-"
                # Escape any square brackets in the dependency name
                escaped_dep = str(dep).replace("[", "\\[").replace("]", "\\]")
                dependencies_text += f"  {status} {escaped_dep}\n"

                if not installed:
                    missing_deps.append(dep)

            self.query_one("#dependencies-info").update(dependencies_text)

            if missing_deps:
                log_message(f"Installing missing dependencies: {missing_deps}")
                self.query_one("#dependencies-status").update("Installing missing dependencies...")

                # Run dependency installation directly (not in a worker thread)
                try:
                    install_success = install_dependencies(missing_deps)
                    log_message("Dependency installation completed")
                except Exception as e:
                    log_message(f"Error installing dependencies: {str(e)}", "ERROR")
                    install_success = False

                if install_success:
                    self.query_one("#dependencies-status").update("Dependencies installed successfully.")
                    # Update the dependencies text with the new status
                    dependencies_text = "Dependencies:\n"
                    for dep in dependencies:
                        # Use plain text checkmarks
                        status = "+"  # All should be installed now
                        # Escape any square brackets in the dependency name
                        escaped_dep = str(dep).replace("[", "\\[").replace("]", "\\]")
                        dependencies_text += f"  {status} {escaped_dep}\n"
                    self.query_one("#dependencies-info").update(dependencies_text)

                    # Automatically proceed to the component selection screen after a short delay
                    log_message("All dependencies installed. Proceeding to component selection...")
                    self.app.set_timer(1.0, self.proceed_to_component_selection)
                else:
                    self.query_one("#dependencies-status").update("Failed to install some dependencies.")
            else:
                self.query_one("#dependencies-status").update("All dependencies are installed.")

            # Enable continue button
            log_message("Enabling continue button")
            self.query_one("#continue-button").disabled = False

        except Exception as e:
            log_message(f"Error in detect_hardware: {str(e)}", "ERROR")
            try:
                self.query_one("#hardware-status").update(f"Hardware detection failed: {str(e)}")
                self.query_one("#dependencies-status").update("Dependency check skipped due to hardware detection failure.")
                self.query_one("#continue-button").disabled = False
            except Exception as inner_e:
                log_message(f"Error updating UI after hardware detection failure: {str(inner_e)}", "ERROR")

    def proceed_to_component_selection(self) -> None:
        """Proceed to the component selection screen."""
        log_message("Proceeding to component selection screen")
        self.app.push_screen(ComponentSelectionScreen())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Event handler called when a button is pressed."""
        if event.button.id == "continue-button":
            self.app.push_screen(ComponentSelectionScreen())

class ComponentSelectionScreen(Screen):
    """Component selection screen."""

    def compose(self) -> ComposeResult:
        """Compose the component selection screen."""
        yield Header(show_clock=True)

        with Container(id="selection-container"):
            yield Static(BANNER, id="banner")
            yield Static("Select Components to Install", id="selection-title")

            yield Static("Core Components (Required)", id="core-title")
            with Container(id="core-components"):
                for i, component in enumerate(CORE_COMPONENTS):
                    with Horizontal(classes="component-row"):
                        yield Button("✓", id=f"core-{i}", classes="component-button", disabled=True)
                        yield Static(f"{component['name']}: {component['description']}", classes="component-description")

            yield Static("Extension Components (Optional)", id="extension-title")
            with Container(id="extension-components"):
                for i, component in enumerate(EXTENSION_COMPONENTS):
                    with Horizontal(classes="component-row"):
                        yield Button("✓", id=f"extension-{i}", classes="component-button")
                        yield Static(f"{component['name']}: {component['description']}", classes="component-description")

            with Horizontal(id="selection-buttons"):
                yield Button("Select All", id="select-all-button")
                yield Button("Deselect All", id="deselect-all-button")
                yield Button("Install", id="install-button")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Event handler called when a button is pressed."""
        try:
            log_message(f"Button pressed: {event.button.id}")

            if event.button.id == "select-all-button":
                for i in range(len(EXTENSION_COMPONENTS)):
                    button = self.query_one(f"#extension-{i}")
                    button.label = "✓"
                    EXTENSION_COMPONENTS[i]["selected"] = True
                log_message("Selected all extension components")

            elif event.button.id == "deselect-all-button":
                for i in range(len(EXTENSION_COMPONENTS)):
                    button = self.query_one(f"#extension-{i}")
                    button.label = "□"
                    EXTENSION_COMPONENTS[i]["selected"] = False
                log_message("Deselected all extension components")

            elif event.button.id == "install-button":
                # Mark selected components
                log_message("Install button pressed")
                try:
                    # Mark selected components
                    for i in range(len(EXTENSION_COMPONENTS)):
                        button = self.query_one(f"#extension-{i}")
                        EXTENSION_COMPONENTS[i]["selected"] = button.label == "✓"
                        log_message(f"Extension component {i} selected: {EXTENSION_COMPONENTS[i]['selected']}")

                    # Create the installation screen
                    log_message("Creating InstallationScreen")
                    installation_screen = InstallationScreen()
                    log_message("InstallationScreen created")

                    # Push the installation screen
                    log_message("Pushing InstallationScreen")
                    self.app.push_screen(installation_screen)
                    log_message("InstallationScreen pushed")
                except Exception as e:
                    log_message(f"Error in install button handler: {str(e)}", "ERROR")
                    import traceback
                    log_message(f"Traceback: {traceback.format_exc()}", "ERROR")

            elif event.button.id.startswith("extension-"):
                # Toggle extension component selection
                log_message(f"Extension component button pressed: {event.button.id}")
                index = int(event.button.id.split("-")[1])

                # Toggle extension component selection
                if event.button.label == "✓":
                    log_message(f"Deselecting extension component {index}")
                    event.button.label = "□"
                    EXTENSION_COMPONENTS[index]["selected"] = False
                else:
                    log_message(f"Selecting extension component {index}")
                    event.button.label = "✓"
                    EXTENSION_COMPONENTS[index]["selected"] = True
        except Exception as e:
            log_message(f"Error in ComponentSelectionScreen.on_button_pressed: {str(e)}", "ERROR")
            import traceback
            log_message(f"Traceback: {traceback.format_exc()}", "ERROR")

class InstallationScreen(Screen):
    """Installation screen."""

    def compose(self) -> ComposeResult:
        """Compose the installation screen."""
        log_message("InstallationScreen.compose called")
        try:
            yield Header(show_clock=True)

            with Container(id="installation-container"):
                yield Static(BANNER, id="banner")
                yield Static("Installing ML Stack Components", id="installation-title")

                yield Static("Progress", id="progress-title")
                yield ProgressBar(id="progress-bar", total=100)

                yield Static("Installation Log", id="log-title")
                yield Log(id="installation-log", highlight=True)

                yield Button("Close", id="cancel-button")

            yield Footer()
            log_message("InstallationScreen.compose completed")
        except Exception as e:
            log_message(f"Error in InstallationScreen.compose: {str(e)}", "ERROR")
            import traceback
            log_message(f"Traceback: {traceback.format_exc()}", "ERROR")
            raise

    def on_mount(self) -> None:
        """Event handler called when the screen is mounted."""
        log_message("InstallationScreen.on_mount called")
        try:
            # Display a simple message
            log_widget = self.query_one("#installation-log")
            log_widget.write("[bold blue]Installation screen mounted[/bold blue]")
            log_message("Installation screen mounted successfully")

            # Start the installation process in a separate thread
            log_message("Starting installation process in a separate thread")
            installation_thread = threading.Thread(target=self._start_installation)
            installation_thread.daemon = True
            installation_thread.start()
        except Exception as e:
            log_message(f"Error in InstallationScreen.on_mount: {str(e)}", "ERROR")
            import traceback
            log_message(f"Traceback: {traceback.format_exc()}", "ERROR")

    def _update_ui(self, widget_id, action, **kwargs):
        """Update the UI from a separate thread."""
        def update():
            try:
                widget = self.query_one(widget_id)
                if action == "write" and hasattr(widget, "write"):
                    widget.write(kwargs.get("text", ""))
                elif action == "update_progress" and hasattr(widget, "progress"):
                    widget.progress = kwargs.get("progress", 0)
                elif action == "update_total" and hasattr(widget, "total"):
                    widget.total = kwargs.get("total", 0)
                elif action == "update_label" and hasattr(widget, "label"):
                    widget.label = kwargs.get("label", "")
                self.app.refresh()
            except Exception as e:
                log_message(f"Error updating UI: {str(e)}", "ERROR")

        # Schedule the update to run in the main thread
        self.app.call_from_thread(update)

    def _start_installation(self) -> None:
        """Start the installation process."""
        try:
            log_message("Starting installation process")
            log_message("Getting UI widgets")

            # Update the UI
            self._update_ui("#installation-log", "write", text="[bold blue]Starting installation...[/bold blue]")
            log_message("Starting installation process")

            # Check if installation scripts exist
            self._update_ui("#installation-log", "write", text="[bold blue]Checking installation scripts...[/bold blue]")
            missing_scripts = []

            for component in CORE_COMPONENTS:
                script_path = os.path.join(SCRIPTS_DIR, component["script"])
                if not os.path.exists(script_path):
                    missing_scripts.append(f"{component['name']} ({component['script']})")

            for component in EXTENSION_COMPONENTS:
                if component.get("selected", False):
                    script_path = os.path.join(SCRIPTS_DIR, component["script"])
                    if not os.path.exists(script_path):
                        missing_scripts.append(f"{component['name']} ({component['script']})")

            if missing_scripts:
                self._update_ui("#installation-log", "write", text="[bold red]ERROR: The following installation scripts are missing:[/bold red]")
                for script in missing_scripts:
                    self._update_ui("#installation-log", "write", text=f"[bold red]- {script}[/bold red]")
                self._update_ui("#installation-log", "write", text="[bold yellow]Please check the component definitions in the UI script.[/bold yellow]")
                self._update_ui("#installation-log", "write", text="[bold yellow]Installation cannot proceed with missing scripts.[/bold yellow]")

                # Update button
                self._update_ui("#cancel-button", "update_label", label="Close")
                return

            # Calculate total components to install
            total_components = len(CORE_COMPONENTS)
            for component in EXTENSION_COMPONENTS:
                if component.get("selected", False):
                    total_components += 1

            # Set up progress bar
            self._update_ui("#progress-bar", "update_total", total=total_components)
            self._update_ui("#progress-bar", "update_progress", progress=0)

            # Set up environment
            self._update_ui("#installation-log", "write", text="[bold blue]Setting up environment...[/bold blue]")
            if setup_environment():
                self._update_ui("#installation-log", "write", text="[bold green]Environment setup completed successfully[/bold green]")
                # Update progress bar
                self._update_ui("#progress-bar", "update_progress", progress=1)
            else:
                self._update_ui("#installation-log", "write", text="[bold red]Environment setup failed[/bold red]")
                return

            # Install core components
            self._update_ui("#installation-log", "write", text="[bold blue]Installing core components...[/bold blue]")
            current_progress = 1  # Start at 1 because environment setup is complete

            for i, component in enumerate(CORE_COMPONENTS):
                self._update_ui("#installation-log", "write", text=f"[bold blue]Installing {component['name']}...[/bold blue]")

                # Double-check that the script exists
                script_path = os.path.join(SCRIPTS_DIR, component["script"])
                if not os.path.exists(script_path):
                    self._update_ui("#installation-log", "write", text=f"[bold red]ERROR: Installation script not found for {component['name']}: {component['script']}[/bold red]")
                    if component["required"]:
                        self._update_ui("#installation-log", "write", text=f"[bold red]ERROR: Required component {component['name']} failed to install[/bold red]")
                        return
                    continue

                # Create a custom log widget that updates the UI
                class ThreadSafeLogWidget:
                    def __init__(self, parent):
                        self.parent = parent

                    def write(self, text):
                        self.parent._update_ui("#installation-log", "write", text=text)

                thread_safe_log_widget = ThreadSafeLogWidget(self)

                if install_component(component, thread_safe_log_widget):
                    current_progress += 1
                    self._update_ui("#progress-bar", "update_progress", progress=current_progress)
                elif component["required"]:
                    self._update_ui("#installation-log", "write", text=f"[bold red]ERROR: Required component {component['name']} failed to install[/bold red]")
                    return

            # Install selected extension components
            self._update_ui("#installation-log", "write", text="[bold blue]Installing extension components...[/bold blue]")
            for component in EXTENSION_COMPONENTS:
                if component.get("selected", False):
                    self._update_ui("#installation-log", "write", text=f"[bold blue]Installing {component['name']}...[/bold blue]")

                    # Double-check that the script exists
                    script_path = os.path.join(SCRIPTS_DIR, component["script"])
                    if not os.path.exists(script_path):
                        self._update_ui("#installation-log", "write", text=f"[bold red]ERROR: Installation script not found for {component['name']}: {component['script']}[/bold red]")
                        continue

                    if install_component(component, thread_safe_log_widget):
                        current_progress += 1
                        self._update_ui("#progress-bar", "update_progress", progress=current_progress)

            # Verify installation
            self._update_ui("#installation-log", "write", text="[bold blue]Verifying installation...[/bold blue]")
            verification_results = verify_installation()

            # Show verification results
            self._update_ui("#installation-log", "write", text="[bold blue]Verification Results:[/bold blue]")
            for component in CORE_COMPONENTS + EXTENSION_COMPONENTS:
                name = component["name"]
                if name in verification_results:
                    status = "✓" if verification_results[name] else "✗"
                    status_color = "green" if verification_results[name] else "red"
                    self._update_ui("#installation-log", "write", text=f"[bold {status_color}]{status} {name}[/bold {status_color}]")

            # Installation complete
            self._update_ui("#installation-log", "write", text="[bold green]Installation complete![/bold green]")
            self._update_ui("#installation-log", "write", text=f"[bold blue]Log file: {LOG_FILE}[/bold blue]")

            # Update button
            self._update_ui("#cancel-button", "update_label", label="Finish")

        except Exception as e:
            log_message(f"Error in installation process: {str(e)}", "ERROR")
            import traceback
            log_message(f"Traceback: {traceback.format_exc()}", "ERROR")
            try:
                self._update_ui("#installation-log", "write", text=f"[bold red]ERROR: {str(e)}[/bold red]")
            except Exception as inner_e:
                log_message(f"Error updating log widget: {str(inner_e)}", "ERROR")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Event handler called when a button is pressed."""
        log_message(f"InstallationScreen.on_button_pressed called: {event.button.id}")
        try:
            if event.button.id == "cancel-button":
                log_message("Cancel button pressed, exiting app")
                self.app.exit()
        except Exception as e:
            log_message(f"Error in InstallationScreen.on_button_pressed: {str(e)}", "ERROR")
            import traceback
            log_message(f"Traceback: {traceback.format_exc()}", "ERROR")

class MLStackInstallerApp(App):
    """ML Stack Installer App."""

    CSS = """
    #banner {
        width: 100%;
        content-align: center middle;
        color: $accent;
    }

    #welcome-title, #selection-title, #installation-title {
        width: 100%;
        content-align: center middle;
        text-style: bold;
        margin: 1 0;
    }

    #welcome-description {
        width: 100%;
        content-align: center middle;
        margin: 1 0 2 0;
    }

    #hardware-status, #dependencies-status, #progress-title, #log-title {
        text-style: bold;
        margin: 1 0;
    }

    #hardware-info, #dependencies-info {
        margin: 0 0 1 2;
    }

    #core-title, #extension-title {
        text-style: bold;
        margin: 1 0;
    }

    .component-row {
        height: 1;
        margin: 0 0 0 2;
    }

    .component-button {
        width: 3;
        height: 1;
    }

    .component-description {
        width: 100%;
        height: 1;
        margin: 0 0 0 1;
    }

    #selection-buttons {
        width: 100%;
        content-align: center middle;
        margin: 2 0 0 0;
    }

    #continue-button, #install-button, #cancel-button {
        margin: 2 0 0 0;
    }

    #progress-bar {
        width: 100%;
        margin: 0 0 1 0;
    }

    #installation-log {
        height: 20;
        width: 100%;
    }
    """

    def on_mount(self) -> None:
        """Event handler called when the app is mounted."""
        try:
            log_message("MLStackInstallerApp mounted")
            log_message("Pushing SudoPasswordInput screen")
            self.push_screen(SudoPasswordInput())
            log_message("SudoPasswordInput screen pushed")
        except Exception as e:
            log_message(f"Error in MLStackInstallerApp.on_mount: {str(e)}", "ERROR")

    def on_screen_resume(self, event: events.ScreenResume) -> None:
        """Event handler called when a screen is resumed."""
        try:
            log_message(f"Screen resumed: {event.screen.__class__.__name__}")
        except Exception as e:
            log_message(f"Error in on_screen_resume: {str(e)}", "ERROR")

    def on_screen_suspend(self, event: events.ScreenSuspend) -> None:
        """Event handler called when a screen is suspended."""
        try:
            log_message(f"Screen suspended: {event.screen.__class__.__name__}")
        except Exception as e:
            log_message(f"Error in on_screen_suspend: {str(e)}", "ERROR")

    def on_error(self, error: Exception) -> None:
        """Event handler called when an error occurs."""
        log_message(f"Error in app: {str(error)}", "ERROR")
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}", "ERROR")

# Main function
def main():
    """Main function."""
    # Create log file
    with open(LOG_FILE, "w") as f:
        f.write(f"ML Stack Installation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"System: {platform.system()} {platform.release()}\n")
        f.write(f"Python: {platform.python_version()}\n")
        f.write("-" * 80 + "\n\n")

    if DEBUG_MODE:
        log_message("DEBUG MODE ENABLED")
        try:
            # Test sudo password verification
            log_message("Testing sudo password verification")
            password = "test_password"  # This is just a test password
            temp_script = os.path.join(SCRIPTS_DIR, "temp_sudo_verify.sh")
            with open(temp_script, "w") as f:
                f.write("""#!/bin/bash
echo "This is a test script"
exit 0
""")
            os.chmod(temp_script, 0o755)
            return_code, stdout, stderr = run_command(f"bash {temp_script}")
            log_message(f"Test script returned: {return_code}, stdout: {stdout}, stderr: {stderr}")
            os.remove(temp_script)

            # Test hardware detection
            log_message("Testing hardware detection")
            def update_progress(message):
                log_message(f"Progress: {message}")
            hardware_info = detect_hardware(update_progress)
            log_message(f"Hardware detection result: {hardware_info}")

            # Test dependency checking
            log_message("Testing dependency checking")
            dependencies = check_dependencies()
            log_message(f"Dependency check result: {dependencies}")

            log_message("All tests completed successfully")
            log_message("Starting UI in debug mode")
        except Exception as e:
            log_message(f"Error in debug mode: {str(e)}", "ERROR")
            import traceback
            log_message(f"Traceback: {traceback.format_exc()}", "ERROR")

    # Start the app
    app = MLStackInstallerApp()
    app.run()

if __name__ == "__main__":
    main()
