#!/bin/bash
#
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
#
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House
#
# =============================================================================
# ROCm SMI Installation Script for AMD GPUs
# =============================================================================
# This script installs and configures ROCm System Management Interface (SMI)
# for monitoring and profiling AMD GPUs.
#
# Author: User
# Date: $(date +"%Y-%m-%d")
# =============================================================================

set -e  # Exit on error

# Create log directory
LOG_DIR="$HOME/Desktop/ml_stack_extensions/logs"
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/rocm_smi_install_$(date +"%Y%m%d_%H%M%S").log"

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Start installation
log "=== Starting ROCm SMI Installation ==="
log "System: $(uname -a)"
log "ROCm Path: $(which hipcc 2>/dev/null || echo 'Not found')"

# Check if ROCm SMI is already installed
if command_exists rocm-smi; then
    log "ROCm SMI is already installed: $(rocm-smi --version 2>&1)"
    log "Checking for updates..."
else
    log "ROCm SMI is not installed. Installing..."
fi

# Create installation directory
INSTALL_DIR="$HOME/ml_stack/rocm_smi"
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# Clone ROCm SMI repository
if [ ! -d "rocm_smi_lib" ]; then
    log "Cloning ROCm SMI repository..."
    git clone https://github.com/RadeonOpenCompute/rocm_smi_lib.git
    cd rocm_smi_lib
else
    log "ROCm SMI repository already exists, updating..."
    cd rocm_smi_lib
    git pull
fi

# Install Python wrapper
log "Installing ROCm SMI Python wrapper..."
if [ -d "python_smi_tools" ]; then
    cd python_smi_tools
    if [ -f "setup.py" ]; then
        if command_exists uv; then
            log "Using uv to install ROCm SMI Python wrapper..."
            uv pip install -e .
        else
            log "Using pip to install ROCm SMI Python wrapper..."
            pip install -e . --break-system-packages
        fi
    else
        log "No setup.py found in python_smi_tools. Creating a simple wrapper instead."
        mkdir -p $INSTALL_DIR/python_wrapper
        cat > $INSTALL_DIR/python_wrapper/rocm_smi_lib.py << 'EOF'
#!/usr/bin/env python3
"""
Simple wrapper for ROCm SMI command-line tool
"""

import subprocess
import re
import json

class rsmi:
    @staticmethod
    def rsmi_init(flags=0):
        """Initialize ROCm SMI."""
        return 0

    @staticmethod
    def rsmi_shut_down():
        """Shut down ROCm SMI."""
        return 0

    @staticmethod
    def rsmi_num_monitor_devices():
        """Get number of GPU devices."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showallinfo", "--json"], text=True)
            data = json.loads(output)
            return len(data.keys()), 0
        except:
            return 0, 1

    @staticmethod
    def rsmi_dev_name_get(device_id):
        """Get device name."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showallinfo", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            name = data[device_key].get("Card Series", "Unknown")
            return 0, name
        except:
            return 1, "Unknown"

    @staticmethod
    def rsmi_dev_gpu_busy_percent_get(device_id):
        """Get GPU utilization."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showuse", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            util_str = data[device_key].get("GPU use (%)", "0%")
            util = int(re.sub(r'[^0-9]', '', util_str))
            return 0, util
        except:
            return 1, 0

    @staticmethod
    def rsmi_dev_temp_metric_get(device_id, sensor_type, metric_type):
        """Get temperature."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showtemp", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            temp_str = data[device_key].get("Temperature (Sensor edge) (C)", "0.0")
            temp = float(re.sub(r'[^0-9.]', '', temp_str)) * 1000  # Convert to millidegrees
            return 0, int(temp)
        except:
            return 1, 0

    @staticmethod
    def rsmi_dev_memory_usage_get(device_id, memory_type):
        """Get memory usage."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showmemuse", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            used_str = data[device_key].get("GPU Memory Used (B)", "0")
            total_str = data[device_key].get("GPU Memory Total (B)", "0")
            used = int(re.sub(r'[^0-9]', '', used_str))
            total = int(re.sub(r'[^0-9]', '', total_str))
            return 0, used, total
        except:
            return 1, 0, 0

    @staticmethod
    def rsmi_dev_power_ave_get(device_id):
        """Get power consumption."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showpower", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            power_str = data[device_key].get("Average Graphics Package Power (W)", "0.0")
            power = float(re.sub(r'[^0-9.]', '', power_str)) * 1000000  # Convert to microwatts
            return 0, int(power)
        except:
            return 1, 0

    @staticmethod
    def rsmi_dev_gpu_clk_freq_get(device_id, clock_type):
        """Get clock frequency."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showclocks", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]

            if clock_type == 0:  # GPU clock
                clock_str = data[device_key].get("GPU Clock Level", "0 MHz")
            else:  # Memory clock
                clock_str = data[device_key].get("GPU Memory Clock Level", "0 MHz")

            clock = float(re.sub(r'[^0-9.]', '', clock_str)) * 1000000  # Convert to Hz
            return 0, int(clock)
        except:
            return 1, 0
EOF

        # Create setup.py
        cat > $INSTALL_DIR/python_wrapper/setup.py << 'EOF'
from setuptools import setup

setup(
    name="rocm_smi_lib",
    version="0.1.0",
    py_modules=["rocm_smi_lib"],
    description="Simple wrapper for ROCm SMI command-line tool",
)
EOF

        # Install the wrapper
        cd $INSTALL_DIR/python_wrapper
        if command_exists uv; then
            log "Using uv to install ROCm SMI Python wrapper..."
            uv pip install -e .
        else
            log "Using pip to install ROCm SMI Python wrapper..."
            pip install -e . --break-system-packages
        fi
    fi
else
    log "python_smi_tools directory not found. Creating a simple wrapper instead."
    mkdir -p $INSTALL_DIR/python_wrapper
    cat > $INSTALL_DIR/python_wrapper/rocm_smi_lib.py << 'EOF'
#!/usr/bin/env python3
"""
Simple wrapper for ROCm SMI command-line tool
"""

import subprocess
import re
import json

class rsmi:
    @staticmethod
    def rsmi_init(flags=0):
        """Initialize ROCm SMI."""
        return 0

    @staticmethod
    def rsmi_shut_down():
        """Shut down ROCm SMI."""
        return 0

    @staticmethod
    def rsmi_num_monitor_devices():
        """Get number of GPU devices."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showallinfo", "--json"], text=True)
            data = json.loads(output)
            return len(data.keys()), 0
        except:
            return 0, 1

    @staticmethod
    def rsmi_dev_name_get(device_id):
        """Get device name."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showallinfo", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            name = data[device_key].get("Card Series", "Unknown")
            return 0, name
        except:
            return 1, "Unknown"

    @staticmethod
    def rsmi_dev_gpu_busy_percent_get(device_id):
        """Get GPU utilization."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showuse", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            util_str = data[device_key].get("GPU use (%)", "0%")
            util = int(re.sub(r'[^0-9]', '', util_str))
            return 0, util
        except:
            return 1, 0

    @staticmethod
    def rsmi_dev_temp_metric_get(device_id, sensor_type, metric_type):
        """Get temperature."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showtemp", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            temp_str = data[device_key].get("Temperature (Sensor edge) (C)", "0.0")
            temp = float(re.sub(r'[^0-9.]', '', temp_str)) * 1000  # Convert to millidegrees
            return 0, int(temp)
        except:
            return 1, 0

    @staticmethod
    def rsmi_dev_memory_usage_get(device_id, memory_type):
        """Get memory usage."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showmemuse", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            used_str = data[device_key].get("GPU Memory Used (B)", "0")
            total_str = data[device_key].get("GPU Memory Total (B)", "0")
            used = int(re.sub(r'[^0-9]', '', used_str))
            total = int(re.sub(r'[^0-9]', '', total_str))
            return 0, used, total
        except:
            return 1, 0, 0

    @staticmethod
    def rsmi_dev_power_ave_get(device_id):
        """Get power consumption."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showpower", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            power_str = data[device_key].get("Average Graphics Package Power (W)", "0.0")
            power = float(re.sub(r'[^0-9.]', '', power_str)) * 1000000  # Convert to microwatts
            return 0, int(power)
        except:
            return 1, 0

    @staticmethod
    def rsmi_dev_gpu_clk_freq_get(device_id, clock_type):
        """Get clock frequency."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showclocks", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]

            if clock_type == 0:  # GPU clock
                clock_str = data[device_key].get("GPU Clock Level", "0 MHz")
            else:  # Memory clock
                clock_str = data[device_key].get("GPU Memory Clock Level", "0 MHz")

            clock = float(re.sub(r'[^0-9.]', '', clock_str)) * 1000000  # Convert to Hz
            return 0, int(clock)
        except:
            return 1, 0
EOF

    # Create setup.py
    cat > $INSTALL_DIR/python_wrapper/setup.py << 'EOF'
from setuptools import setup

setup(
    name="rocm_smi_lib",
    version="0.1.0",
    py_modules=["rocm_smi_lib"],
    description="Simple wrapper for ROCm SMI command-line tool",
)
EOF

    # Install the wrapper
    cd $INSTALL_DIR/python_wrapper
    if command_exists uv; then
        log "Using uv to install ROCm SMI Python wrapper..."
        uv pip install -e .
    else
        log "Using pip to install ROCm SMI Python wrapper..."
        pip install -e . --break-system-packages
    fi
fi

# Verify installation
log "Verifying ROCm SMI installation..."
python3 -c "from rocm_smi_lib import rsmi; print('ROCm SMI Python wrapper installed successfully')"

if [ $? -eq 0 ]; then
    log "ROCm SMI Python wrapper installation successful!"
else
    log "ROCm SMI Python wrapper installation failed. Please check the logs."
    exit 1
fi

# Create a simple monitoring script
MONITOR_SCRIPT="$INSTALL_DIR/monitor_gpus.py"
cat > $MONITOR_SCRIPT << 'EOF'
#!/usr/bin/env python3
"""
ROCm SMI GPU Monitoring Script

This script provides real-time monitoring of AMD GPUs using the ROCm SMI library.
It displays information such as GPU utilization, temperature, memory usage, and power consumption.
"""

import time
import argparse
import os
import sys
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

try:
    from rocm_smi_lib import rsmi
except ImportError:
    print("Error: ROCm SMI Python wrapper not found.")
    print("Please install it with: pip install -e /path/to/rocm_smi_lib/python_smi_tools")
    sys.exit(1)

class GPUMonitor:
    def __init__(self, interval=1.0, log_file=None, plot=False):
        """Initialize the GPU monitor."""
        self.interval = interval
        self.log_file = log_file
        self.plot = plot
        self.history = {
            'timestamp': [],
            'utilization': {},
            'temperature': {},
            'memory': {},
            'power': {}
        }

        # Initialize ROCm SMI
        rsmi.rsmi_init(0)

        # Get number of devices
        self.num_devices = rsmi.rsmi_num_monitor_devices()
        print(f"Found {self.num_devices} GPU device(s)")

        # Initialize history for each device
        for i in range(self.num_devices):
            self.history['utilization'][i] = []
            self.history['temperature'][i] = []
            self.history['memory'][i] = []
            self.history['power'][i] = []

    def __del__(self):
        """Clean up ROCm SMI."""
        rsmi.rsmi_shut_down()

    def get_device_info(self, device_id):
        """Get information for a specific device."""
        info = {}

        try:
            # Get device name
            name = rsmi.rsmi_dev_name_get(device_id)[1]
            info['name'] = name

            # Get GPU utilization
            util = rsmi.rsmi_dev_gpu_busy_percent_get(device_id)[1]
            info['utilization'] = util

            # Get temperature
            temp = rsmi.rsmi_dev_temp_metric_get(device_id, 0, 0)[1] / 1000.0  # Convert to °C
            info['temperature'] = temp

            # Get memory usage
            mem_info = rsmi.rsmi_dev_memory_usage_get(device_id, 0)
            mem_used = mem_info[1] / (1024 * 1024)  # Convert to MB
            mem_total = mem_info[2] / (1024 * 1024)  # Convert to MB
            info['memory_used'] = mem_used
            info['memory_total'] = mem_total
            info['memory_percent'] = (mem_used / mem_total) * 100 if mem_total > 0 else 0

            # Get power consumption
            power = rsmi.rsmi_dev_power_ave_get(device_id)[1] / 1000000.0  # Convert to W
            info['power'] = power

            # Get clock speeds
            gpu_clock = rsmi.rsmi_dev_gpu_clk_freq_get(device_id, 0)[1] / 1000000.0  # Convert to GHz
            mem_clock = rsmi.rsmi_dev_gpu_clk_freq_get(device_id, 1)[1] / 1000000.0  # Convert to GHz
            info['gpu_clock'] = gpu_clock
            info['memory_clock'] = mem_clock

        except Exception as e:
            print(f"Error getting information for device {device_id}: {e}")

        return info

    def update_history(self, device_id, info):
        """Update history for plotting."""
        if device_id == 0:  # Only add timestamp once
            self.history['timestamp'].append(datetime.now())

        self.history['utilization'][device_id].append(info.get('utilization', 0))
        self.history['temperature'][device_id].append(info.get('temperature', 0))
        self.history['memory'][device_id].append(info.get('memory_percent', 0))
        self.history['power'][device_id].append(info.get('power', 0))

    def plot_history(self):
        """Plot the history of GPU metrics."""
        if not self.plot or len(self.history['timestamp']) < 2:
            return

        # Create figure with subplots
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        # Convert timestamps to relative seconds
        start_time = self.history['timestamp'][0]
        timestamps = [(t - start_time).total_seconds() for t in self.history['timestamp']]

        # Plot utilization
        for device_id in range(self.num_devices):
            axs[0].plot(timestamps, self.history['utilization'][device_id],
                        label=f"GPU {device_id}")
        axs[0].set_ylabel('Utilization (%)')
        axs[0].set_title('GPU Utilization')
        axs[0].grid(True)
        axs[0].legend()

        # Plot temperature
        for device_id in range(self.num_devices):
            axs[1].plot(timestamps, self.history['temperature'][device_id],
                        label=f"GPU {device_id}")
        axs[1].set_ylabel('Temperature (°C)')
        axs[1].set_title('GPU Temperature')
        axs[1].grid(True)
        axs[1].legend()

        # Plot memory usage
        for device_id in range(self.num_devices):
            axs[2].plot(timestamps, self.history['memory'][device_id],
                        label=f"GPU {device_id}")
        axs[2].set_ylabel('Memory Usage (%)')
        axs[2].set_title('GPU Memory Usage')
        axs[2].grid(True)
        axs[2].legend()

        # Plot power consumption
        for device_id in range(self.num_devices):
            axs[3].plot(timestamps, self.history['power'][device_id],
                        label=f"GPU {device_id}")
        axs[3].set_ylabel('Power (W)')
        axs[3].set_title('GPU Power Consumption')
        axs[3].set_xlabel('Time (s)')
        axs[3].grid(True)
        axs[3].legend()

        plt.tight_layout()
        plt.savefig('gpu_monitoring.png')
        plt.close()

    def log_to_file(self, data):
        """Log data to a file."""
        if not self.log_file:
            return

        with open(self.log_file, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"Timestamp: {timestamp}\n")
            for device_id, info in data.items():
                f.write(f"GPU {device_id} ({info.get('name', 'Unknown')}):\n")
                f.write(f"  Utilization: {info.get('utilization', 0):.1f}%\n")
                f.write(f"  Temperature: {info.get('temperature', 0):.1f}°C\n")
                f.write(f"  Memory: {info.get('memory_used', 0):.1f}/{info.get('memory_total', 0):.1f} MB ({info.get('memory_percent', 0):.1f}%)\n")
                f.write(f"  Power: {info.get('power', 0):.1f} W\n")
                f.write(f"  GPU Clock: {info.get('gpu_clock', 0):.2f} GHz\n")
                f.write(f"  Memory Clock: {info.get('memory_clock', 0):.2f} GHz\n")
            f.write("\n")

    def monitor(self, duration=None):
        """Monitor GPUs and display information."""
        try:
            # Initialize log file
            if self.log_file:
                with open(self.log_file, 'w') as f:
                    f.write(f"ROCm SMI GPU Monitoring Log\n")
                    f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            start_time = time.time()
            iteration = 0

            while True:
                # Clear screen
                os.system('clear')

                # Get current time
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"ROCm SMI GPU Monitor - {current_time}")
                print(f"Monitoring interval: {self.interval} seconds")
                if duration:
                    elapsed = time.time() - start_time
                    remaining = duration - elapsed
                    print(f"Duration: {duration:.1f}s (Remaining: {max(0, remaining):.1f}s)")
                print()

                # Get information for all devices
                data = {}
                table_data = []

                for i in range(self.num_devices):
                    info = self.get_device_info(i)
                    data[i] = info

                    # Update history
                    self.update_history(i, info)

                    # Add to table data
                    table_data.append([
                        i,
                        info.get('name', 'Unknown'),
                        f"{info.get('utilization', 0):.1f}%",
                        f"{info.get('temperature', 0):.1f}°C",
                        f"{info.get('memory_used', 0):.1f}/{info.get('memory_total', 0):.1f} MB ({info.get('memory_percent', 0):.1f}%)",
                        f"{info.get('power', 0):.1f} W",
                        f"{info.get('gpu_clock', 0):.2f} GHz",
                        f"{info.get('memory_clock', 0):.2f} GHz"
                    ])

                # Display table
                headers = ["ID", "Name", "Utilization", "Temperature", "Memory", "Power", "GPU Clock", "Memory Clock"]
                print(tabulate(table_data, headers=headers, tablefmt="grid"))

                # Log to file
                self.log_to_file(data)

                # Plot history
                if iteration % 10 == 0:  # Update plot every 10 iterations
                    self.plot_history()

                # Check if duration has elapsed
                if duration and (time.time() - start_time) >= duration:
                    break

                # Wait for next update
                time.sleep(self.interval)
                iteration += 1

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            # Final plot
            self.plot_history()

            if self.log_file:
                print(f"\nLog file saved to: {self.log_file}")

            if self.plot:
                print(f"Plot saved to: gpu_monitoring.png")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Monitor AMD GPUs using ROCm SMI')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                        help='Monitoring interval in seconds (default: 1.0)')
    parser.add_argument('-d', '--duration', type=float,
                        help='Monitoring duration in seconds (default: indefinite)')
    parser.add_argument('-l', '--log', type=str,
                        help='Log file path')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Generate plots of GPU metrics')

    args = parser.parse_args()

    # Install tabulate if not available
    try:
        import tabulate
    except ImportError:
        print("Installing tabulate package...")
        os.system('pip install tabulate --break-system-packages')

    # Install matplotlib if plotting is enabled
    if args.plot:
        try:
            import matplotlib
        except ImportError:
            print("Installing matplotlib package...")
            os.system('pip install matplotlib --break-system-packages')

    # Create monitor and start monitoring
    monitor = GPUMonitor(interval=args.interval, log_file=args.log, plot=args.plot)
    monitor.monitor(duration=args.duration)

if __name__ == '__main__':
    main()
EOF

# Make the script executable
chmod +x $MONITOR_SCRIPT

log "Created GPU monitoring script at $MONITOR_SCRIPT"
log "You can run it with: python3 $MONITOR_SCRIPT"

# Create a simple wrapper script
WRAPPER_SCRIPT="$INSTALL_DIR/rocm_smi_wrapper.py"
cat > $WRAPPER_SCRIPT << 'EOF'
#!/usr/bin/env python3
"""
ROCm SMI Python Wrapper Example

This script demonstrates how to use the ROCm SMI Python wrapper to access
GPU information programmatically.
"""

import sys
from rocm_smi_lib import rsmi

def initialize_rsmi():
    """Initialize ROCm SMI."""
    try:
        rsmi.rsmi_init(0)
        return True
    except Exception as e:
        print(f"Error initializing ROCm SMI: {e}")
        return False

def shutdown_rsmi():
    """Shut down ROCm SMI."""
    try:
        rsmi.rsmi_shut_down()
    except Exception as e:
        print(f"Error shutting down ROCm SMI: {e}")

def get_device_count():
    """Get the number of GPU devices."""
    try:
        return rsmi.rsmi_num_monitor_devices()
    except Exception as e:
        print(f"Error getting device count: {e}")
        return 0

def get_device_info(device_id):
    """Get detailed information for a specific device."""
    info = {}

    try:
        # Device name
        info['name'] = rsmi.rsmi_dev_name_get(device_id)[1]

        # Device ID and vendor ID
        info['device_id'] = rsmi.rsmi_dev_id_get(device_id)[1]
        info['vendor_id'] = rsmi.rsmi_dev_vendor_id_get(device_id)[1]

        # PCI info
        info['pci_bandwidth'] = rsmi.rsmi_dev_pci_bandwidth_get(device_id)[1]
        info['pci_address'] = rsmi.rsmi_dev_pci_id_get(device_id)[1]

        # GPU utilization
        info['gpu_utilization'] = rsmi.rsmi_dev_gpu_busy_percent_get(device_id)[1]
        info['memory_utilization'] = rsmi.rsmi_dev_memory_busy_percent_get(device_id)[1]

        # Temperature
        info['temperature'] = rsmi.rsmi_dev_temp_metric_get(device_id, 0, 0)[1] / 1000.0  # Convert to °C

        # Memory info
        mem_info = rsmi.rsmi_dev_memory_usage_get(device_id, 0)
        info['memory_used'] = mem_info[1] / (1024 * 1024)  # Convert to MB
        info['memory_total'] = mem_info[2] / (1024 * 1024)  # Convert to MB

        # Power consumption
        info['power'] = rsmi.rsmi_dev_power_ave_get(device_id)[1] / 1000000.0  # Convert to W

        # Clock speeds
        info['gpu_clock'] = rsmi.rsmi_dev_gpu_clk_freq_get(device_id, 0)[1] / 1000000.0  # Convert to GHz
        info['memory_clock'] = rsmi.rsmi_dev_gpu_clk_freq_get(device_id, 1)[1] / 1000000.0  # Convert to GHz

        # Fan speed
        try:
            info['fan_speed'] = rsmi.rsmi_dev_fan_speed_get(device_id, 0)[1]
            info['fan_speed_max'] = rsmi.rsmi_dev_fan_speed_max_get(device_id, 0)[1]
            info['fan_speed_percent'] = (info['fan_speed'] / info['fan_speed_max']) * 100 if info['fan_speed_max'] > 0 else 0
        except:
            info['fan_speed'] = 'N/A'
            info['fan_speed_max'] = 'N/A'
            info['fan_speed_percent'] = 'N/A'

        # Voltage
        try:
            info['voltage'] = rsmi.rsmi_dev_volt_metric_get(device_id, 0)[1] / 1000.0  # Convert to V
        except:
            info['voltage'] = 'N/A'

        # Performance level
        try:
            info['performance_level'] = rsmi.rsmi_dev_perf_level_get(device_id)[1]
        except:
            info['performance_level'] = 'N/A'

        # Overdrive level
        try:
            info['overdrive_level'] = rsmi.rsmi_dev_overdrive_level_get(device_id)[1]
        except:
            info['overdrive_level'] = 'N/A'

        # ECC status
        try:
            info['ecc_status'] = rsmi.rsmi_dev_ecc_status_get(device_id, 0)[1]
        except:
            info['ecc_status'] = 'N/A'

    except Exception as e:
        print(f"Error getting information for device {device_id}: {e}")

    return info

def print_device_info(device_id, info):
    """Print device information in a formatted way."""
    print(f"\n{'=' * 50}")
    print(f"GPU {device_id}: {info.get('name', 'Unknown')}")
    print(f"{'=' * 50}")

    print(f"Device ID: {info.get('device_id', 'N/A')}")
    print(f"Vendor ID: {info.get('vendor_id', 'N/A')}")
    print(f"PCI Address: {info.get('pci_address', 'N/A')}")
    print(f"PCI Bandwidth: {info.get('pci_bandwidth', 'N/A')}")

    print(f"\nUtilization:")
    print(f"  GPU: {info.get('gpu_utilization', 'N/A')}%")
    print(f"  Memory: {info.get('memory_utilization', 'N/A')}%")

    print(f"\nMemory:")
    print(f"  Used: {info.get('memory_used', 'N/A'):.2f} MB")
    print(f"  Total: {info.get('memory_total', 'N/A'):.2f} MB")
    print(f"  Utilization: {(info.get('memory_used', 0) / info.get('memory_total', 1)) * 100:.2f}%")

    print(f"\nTemperature: {info.get('temperature', 'N/A'):.2f}°C")
    print(f"Power: {info.get('power', 'N/A'):.2f} W")

    print(f"\nClock Speeds:")
    print(f"  GPU: {info.get('gpu_clock', 'N/A'):.2f} GHz")
    print(f"  Memory: {info.get('memory_clock', 'N/A'):.2f} GHz")

    print(f"\nFan:")
    if info.get('fan_speed', 'N/A') != 'N/A':
        print(f"  Speed: {info.get('fan_speed', 'N/A')}")
        print(f"  Max Speed: {info.get('fan_speed_max', 'N/A')}")
        print(f"  Percentage: {info.get('fan_speed_percent', 'N/A'):.2f}%")
    else:
        print(f"  Speed: N/A (possibly integrated GPU)")

    print(f"\nVoltage: {info.get('voltage', 'N/A')}")
    print(f"Performance Level: {info.get('performance_level', 'N/A')}")
    print(f"Overdrive Level: {info.get('overdrive_level', 'N/A')}")
    print(f"ECC Status: {info.get('ecc_status', 'N/A')}")

def main():
    """Main function."""
    if not initialize_rsmi():
        sys.exit(1)

    try:
        # Get device count
        device_count = get_device_count()
        print(f"Found {device_count} GPU device(s)")

        # Get information for each device
        for i in range(device_count):
            info = get_device_info(i)
            print_device_info(i, info)

    finally:
        shutdown_rsmi()

if __name__ == '__main__':
    main()
EOF

# Make the script executable
chmod +x $WRAPPER_SCRIPT

log "Created ROCm SMI wrapper example at $WRAPPER_SCRIPT"
log "You can run it with: python3 $WRAPPER_SCRIPT"

log "=== ROCm SMI Installation Complete ==="
log "Installation Directory: $INSTALL_DIR"
log "Log File: $LOG_FILE"
log "Documentation: $HOME/Desktop/ml_stack_extensions/docs/rocm_smi_guide.md"
