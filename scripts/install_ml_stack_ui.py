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

import os
import sys
import time
import subprocess
import platform
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable

try:
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
except ImportError:
    print("Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "textual", "rich"], check=True)
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
MLSTACK_DIR = os.path.join(HOME_DIR, "Desktop", "Stans_MLStack")
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
        "name": "ROCm Platform",
        "description": "Foundation for GPU computing on AMD hardware",
        "script": "install_rocm.sh",
        "required": True,
        "status": "pending"
    },
    {
        "name": "PyTorch",
        "description": "Deep learning framework with ROCm support",
        "script": "install_pytorch.sh",
        "required": True,
        "status": "pending"
    },
    {
        "name": "ONNX Runtime",
        "description": "Optimized inference for ONNX models",
        "script": "build_onnxruntime.sh",
        "required": True,
        "status": "pending"
    },
    {
        "name": "MIGraphX",
        "description": "AMD's graph optimization library",
        "script": "install_migraphx.sh",
        "required": True,
        "status": "pending"
    },
    {
        "name": "Flash Attention",
        "description": "Efficient attention computation",
        "script": "build_flash_attn_amd.sh",
        "required": True,
        "status": "pending"
    },
    {
        "name": "RCCL",
        "description": "ROCm Collective Communication Library",
        "script": "install_rccl.sh",
        "required": True,
        "status": "pending"
    },
    {
        "name": "MPI",
        "description": "Message Passing Interface for distributed computing",
        "script": "install_mpi.sh",
        "required": True,
        "status": "pending"
    },
    {
        "name": "Megatron-LM",
        "description": "Framework for training large language models",
        "script": "install_megatron.sh",
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

def run_command(command: str, cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """Run a shell command and return the return code, stdout, and stderr."""
    log_message(f"Running command: {command}")
    
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd
    )
    
    stdout, stderr = process.communicate()
    return_code = process.returncode
    
    if return_code != 0:
        log_message(f"Command failed with return code {return_code}", "ERROR")
        log_message(f"stderr: {stderr}", "ERROR")
    
    return return_code, stdout, stderr

def detect_hardware() -> Dict[str, Any]:
    """Detect hardware information."""
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
            # Try using lspci
            return_code, stdout, stderr = run_command("lspci | grep -i 'amd\\|radeon\\|advanced micro devices' | grep -i 'vga\\|3d\\|display'")
            if return_code == 0 and stdout.strip():
                hardware_info["amd_gpus"] = [line.strip() for line in stdout.strip().split("\n")]
            
            # Try using rocminfo if available
            return_code, stdout, stderr = run_command("which rocminfo")
            if return_code == 0:
                return_code, stdout, stderr = run_command("rocminfo | grep -A 1 'GPU ID' | grep 'Marketing Name'")
                if return_code == 0 and stdout.strip():
                    hardware_info["amd_gpus"] = [line.split(":", 1)[1].strip() for line in stdout.strip().split("\n")]
                
                # Get ROCm version
                return_code, stdout, stderr = run_command("rocminfo | grep -i 'ROCm Version'")
                if return_code == 0 and stdout.strip():
                    hardware_info["rocm_version"] = stdout.split(":", 1)[1].strip()
                
                # Get ROCm path
                return_code, stdout, stderr = run_command("which rocminfo")
                if return_code == 0:
                    rocminfo_path = stdout.strip()
                    hardware_info["rocm_path"] = os.path.dirname(os.path.dirname(rocminfo_path))
    except Exception as e:
        log_message(f"Error detecting hardware: {str(e)}", "ERROR")
    
    return hardware_info

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed."""
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
        return_code, stdout, stderr = run_command(f"dpkg -l | grep -q 'ii  {dep} '")
        dependencies[dep] = return_code == 0
    
    return dependencies

def install_dependencies(missing_deps: List[str]) -> bool:
    """Install missing dependencies."""
    if not missing_deps:
        return True
    
    log_message(f"Installing missing dependencies: {', '.join(missing_deps)}")
    
    try:
        return_code, stdout, stderr = run_command(f"sudo apt-get update && sudo apt-get install -y {' '.join(missing_deps)}")
        if return_code != 0:
            log_message(f"Failed to install dependencies: {stderr}", "ERROR")
            return False
        
        # Verify installation
        for dep in missing_deps:
            return_code, stdout, stderr = run_command(f"dpkg -l | grep -q 'ii  {dep} '")
            if return_code != 0:
                log_message(f"Failed to install {dep}", "ERROR")
                return False
        
        return True
    except Exception as e:
        log_message(f"Error installing dependencies: {str(e)}", "ERROR")
        return False

def setup_environment() -> bool:
    """Set up the environment for ML Stack."""
    log_message("Setting up environment...")
    
    # Run the enhanced environment setup script
    script_path = os.path.join(SCRIPTS_DIR, "enhanced_setup_environment.sh")
    if not os.path.exists(script_path):
        log_message(f"Environment setup script not found: {script_path}", "ERROR")
        return False
    
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
    
    # Run the installation script
    log_message(f"Installing {name}...")
    return_code, stdout, stderr = run_command(f"bash {script_path}")
    
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

def verify_installation() -> Dict[str, bool]:
    """Verify the installation of components."""
    verification_results = {}
    
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
class WelcomeScreen(Screen):
    """Welcome screen with hardware detection and dependency check."""
    
    def compose(self) -> ComposeResult:
        """Compose the welcome screen."""
        yield Header(show_clock=True)
        
        with Container(id="welcome-container"):
            yield Static(BANNER, id="banner")
            yield Static("Welcome to the ML Stack Installation", id="welcome-title")
            yield Static("This wizard will guide you through the installation of the ML Stack for AMD GPUs.", id="welcome-description")
            
            yield Static("Detecting hardware...", id="hardware-status")
            yield Static("", id="hardware-info")
            
            yield Static("Checking dependencies...", id="dependencies-status")
            yield Static("", id="dependencies-info")
            
            yield Button("Continue", id="continue-button", disabled=True)
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Event handler called when the screen is mounted."""
        self.detect_hardware()
    
    async def detect_hardware(self) -> None:
        """Detect hardware and check dependencies."""
        # Detect hardware
        self.query_one("#hardware-status").update("Detecting hardware...")
        hardware_info = detect_hardware()
        
        # Update hardware info
        hardware_text = f"System: {hardware_info['system']} {hardware_info['release']}\n"
        hardware_text += f"Processor: {hardware_info['processor']}\n"
        hardware_text += f"Python Version: {hardware_info['python_version']}\n"
        
        if hardware_info["amd_gpus"]:
            hardware_text += f"AMD GPUs detected: {len(hardware_info['amd_gpus'])}\n"
            for i, gpu in enumerate(hardware_info["amd_gpus"]):
                hardware_text += f"  - GPU {i}: {gpu}\n"
        else:
            hardware_text += "No AMD GPUs detected. The ML Stack requires AMD GPUs.\n"
        
        if hardware_info["rocm_version"]:
            hardware_text += f"ROCm Version: {hardware_info['rocm_version']}\n"
            hardware_text += f"ROCm Path: {hardware_info['rocm_path']}\n"
        else:
            hardware_text += "ROCm not detected. The ML Stack requires ROCm.\n"
        
        self.query_one("#hardware-info").update(hardware_text)
        self.query_one("#hardware-status").update("Hardware detection completed.")
        
        # Check dependencies
        self.query_one("#dependencies-status").update("Checking dependencies...")
        dependencies = check_dependencies()
        
        # Update dependencies info
        dependencies_text = "Dependencies:\n"
        missing_deps = []
        
        for dep, installed in dependencies.items():
            status = "✓" if installed else "✗"
            dependencies_text += f"  {status} {dep}\n"
            
            if not installed:
                missing_deps.append(dep)
        
        self.query_one("#dependencies-info").update(dependencies_text)
        
        if missing_deps:
            self.query_one("#dependencies-status").update("Installing missing dependencies...")
            if install_dependencies(missing_deps):
                self.query_one("#dependencies-status").update("Dependencies installed successfully.")
                self.query_one("#dependencies-info").update(dependencies_text.replace("✗", "✓"))
            else:
                self.query_one("#dependencies-status").update("Failed to install some dependencies.")
        else:
            self.query_one("#dependencies-status").update("All dependencies are installed.")
        
        # Enable continue button
        self.query_one("#continue-button").disabled = False
    
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
        if event.button.id == "select-all-button":
            for i in range(len(EXTENSION_COMPONENTS)):
                self.query_one(f"#extension-{i}").update("✓")
                EXTENSION_COMPONENTS[i]["selected"] = True
        
        elif event.button.id == "deselect-all-button":
            for i in range(len(EXTENSION_COMPONENTS)):
                self.query_one(f"#extension-{i}").update("□")
                EXTENSION_COMPONENTS[i]["selected"] = False
        
        elif event.button.id == "install-button":
            # Mark selected components
            for i in range(len(EXTENSION_COMPONENTS)):
                button = self.query_one(f"#extension-{i}")
                EXTENSION_COMPONENTS[i]["selected"] = button.renderable == "✓"
            
            self.app.push_screen(InstallationScreen())
        
        elif event.button.id.startswith("extension-"):
            # Toggle extension component selection
            index = int(event.button.id.split("-")[1])
            if event.button.renderable == "✓":
                event.button.update("□")
                EXTENSION_COMPONENTS[index]["selected"] = False
            else:
                event.button.update("✓")
                EXTENSION_COMPONENTS[index]["selected"] = True

class InstallationScreen(Screen):
    """Installation screen."""
    
    def compose(self) -> ComposeResult:
        """Compose the installation screen."""
        yield Header(show_clock=True)
        
        with Container(id="installation-container"):
            yield Static(BANNER, id="banner")
            yield Static("Installing ML Stack Components", id="installation-title")
            
            yield Static("Progress", id="progress-title")
            yield ProgressBar(id="progress-bar", total=100)
            
            yield Static("Installation Log", id="log-title")
            yield Log(id="installation-log", highlight=True, markup=True)
            
            yield Button("Cancel", id="cancel-button")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Event handler called when the screen is mounted."""
        self.start_installation()
    
    async def start_installation(self) -> None:
        """Start the installation process."""
        log_widget = self.query_one("#installation-log")
        progress_bar = self.query_one("#progress-bar")
        
        # Calculate total components to install
        total_components = len(CORE_COMPONENTS)
        for component in EXTENSION_COMPONENTS:
            if component.get("selected", False):
                total_components += 1
        
        # Set up progress bar
        progress_bar.total = total_components
        progress_bar.progress = 0
        
        # Set up environment
        log_widget.write("[bold blue]Setting up environment...[/bold blue]")
        if setup_environment():
            log_widget.write("[bold green]Environment setup completed successfully[/bold green]")
        else:
            log_widget.write("[bold red]Environment setup failed[/bold red]")
            return
        
        # Install core components
        log_widget.write("[bold blue]Installing core components...[/bold blue]")
        for i, component in enumerate(CORE_COMPONENTS):
            log_widget.write(f"[bold blue]Installing {component['name']}...[/bold blue]")
            
            if install_component(component, log_widget):
                progress_bar.progress += 1
            elif component["required"]:
                log_widget.write(f"[bold red]ERROR: Required component {component['name']} failed to install[/bold red]")
                return
        
        # Install selected extension components
        log_widget.write("[bold blue]Installing extension components...[/bold blue]")
        for component in EXTENSION_COMPONENTS:
            if component.get("selected", False):
                log_widget.write(f"[bold blue]Installing {component['name']}...[/bold blue]")
                
                if install_component(component, log_widget):
                    progress_bar.progress += 1
        
        # Verify installation
        log_widget.write("[bold blue]Verifying installation...[/bold blue]")
        verification_results = verify_installation()
        
        # Show verification results
        log_widget.write("[bold blue]Verification Results:[/bold blue]")
        for component in CORE_COMPONENTS + EXTENSION_COMPONENTS:
            name = component["name"]
            if name in verification_results:
                status = "✓" if verification_results[name] else "✗"
                status_color = "green" if verification_results[name] else "red"
                log_widget.write(f"[bold {status_color}]{status} {name}[/bold {status_color}]")
        
        # Installation complete
        log_widget.write("[bold green]Installation complete![/bold green]")
        log_widget.write(f"[bold blue]Log file: {LOG_FILE}[/bold blue]")
        
        # Update button
        self.query_one("#cancel-button").update("Finish")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Event handler called when a button is pressed."""
        if event.button.id == "cancel-button":
            if event.button.renderable == "Finish":
                self.app.exit()
            else:
                # TODO: Implement cancellation
                self.app.exit()

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
        self.push_screen(WelcomeScreen())

# Main function
def main():
    """Main function."""
    # Create log file
    with open(LOG_FILE, "w") as f:
        f.write(f"ML Stack Installation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"System: {platform.system()} {platform.release()}\n")
        f.write(f"Python: {platform.python_version()}\n")
        f.write("-" * 80 + "\n\n")
    
    # Start the app
    app = MLStackInstallerApp()
    app.run()

if __name__ == "__main__":
    main()
