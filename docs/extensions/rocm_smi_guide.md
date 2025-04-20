# ROCm SMI for AMD GPUs: Comprehensive Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Monitoring GPU Performance](#monitoring-gpu-performance)
5. [Programmatic Access](#programmatic-access)
6. [Advanced Features](#advanced-features)
7. [Integration with ML Workflows](#integration-with-ml-workflows)
8. [Troubleshooting](#troubleshooting)
9. [References](#references)

## Introduction

ROCm System Management Interface (ROCm SMI) is a suite of tools for monitoring and managing AMD GPUs. It provides both command-line utilities and a Python library for accessing GPU information, monitoring performance, and controlling various aspects of GPU operation.

### Key Features

- **GPU Monitoring**: Track utilization, temperature, memory usage, and power consumption
- **Performance Control**: Adjust clock speeds, power limits, and fan speeds
- **System Information**: View detailed hardware information about your GPUs
- **Python API**: Programmatically access GPU metrics and control GPU settings
- **Integration**: Works seamlessly with ROCm and PyTorch

### Benefits for ML Workflows

1. **Resource Optimization**: Monitor GPU utilization to identify bottlenecks
2. **Thermal Management**: Track temperatures during long training runs
3. **Power Efficiency**: Monitor and optimize power consumption
4. **Memory Management**: Track memory usage to avoid out-of-memory errors
5. **Performance Tuning**: Adjust GPU settings for optimal performance

## Installation

### Prerequisites

- ROCm 5.0+ installed
- Python 3.6+
- AMD Radeon GPU with ROCm support

### Automated Installation

We provide an installation script that handles all dependencies and configuration:

```bash
# Make the script executable
chmod +x $HOME/Desktop/ml_stack_extensions/install_rocm_smi.sh

# Run the installation script
$HOME/Desktop/ml_stack_extensions/install_rocm_smi.sh
```

### Manual Installation

If you prefer to install manually:

```bash
# Clone the repository
git clone https://github.com/RadeonOpenCompute/rocm_smi_lib.git
cd rocm_smi_lib

# Install Python wrapper
cd python_smi_tools
pip install -e .

# Verify installation
python -c "from rocm_smi_lib import rsmi; print('ROCm SMI installed successfully')"
```

### Verifying Installation

To verify that ROCm SMI is installed correctly:

```bash
# Check command-line tool
rocm-smi --version

# Check Python wrapper
python -c "from rocm_smi_lib import rsmi; print('ROCm SMI Python wrapper installed successfully')"
```

## Basic Usage

### Command-Line Interface

ROCm SMI provides a command-line tool called `rocm-smi` for quick access to GPU information:

```bash
# Display basic GPU information
rocm-smi

# Show detailed information
rocm-smi --showallinfo

# Show GPU utilization
rocm-smi --showuse

# Show memory usage
rocm-smi --showmemuse

# Show temperature
rocm-smi --showtemp

# Show power consumption
rocm-smi --showpower

# Show clock speeds
rocm-smi --showclocks
```

### Common Commands

Here are some common commands for monitoring and controlling GPUs:

```bash
# Monitor GPU metrics continuously
watch -n 1 rocm-smi

# Show information for a specific GPU
rocm-smi -d 0

# Set GPU clock frequency
rocm-smi --setsclk 7

# Set memory clock frequency
rocm-smi --setmclk 3

# Set power limit
rocm-smi --setpoweroverdrive 180

# Set fan speed
rocm-smi --setfan 70

# Reset all settings to default
rocm-smi --resetclocks
```

## Monitoring GPU Performance

### Using the Monitoring Script

We provide a comprehensive GPU monitoring script that displays real-time information and can generate plots:

```bash
# Basic monitoring
python $HOME/ml_stack/rocm_smi/monitor_gpus.py

# Monitor with 2-second interval
python $HOME/ml_stack/rocm_smi/monitor_gpus.py --interval 2

# Monitor for 5 minutes
python $HOME/ml_stack/rocm_smi/monitor_gpus.py --duration 300

# Save log to file
python $HOME/ml_stack/rocm_smi/monitor_gpus.py --log gpu_log.txt

# Generate performance plots
python $HOME/ml_stack/rocm_smi/monitor_gpus.py --plot
```

### Metrics Explained

The monitoring script provides the following metrics:

1. **GPU Utilization**: Percentage of time the GPU is actively processing
2. **Memory Utilization**: Percentage of GPU memory in use
3. **Temperature**: GPU temperature in degrees Celsius
4. **Power Consumption**: Power draw in watts
5. **GPU Clock**: Current GPU clock frequency in GHz
6. **Memory Clock**: Current memory clock frequency in GHz
7. **Fan Speed**: Fan speed as a percentage of maximum (if applicable)

### Interpreting Results

- **High GPU Utilization (>90%)**: GPU is being fully utilized, which is ideal for ML workloads
- **Low GPU Utilization (<50%)**: Potential bottleneck elsewhere (CPU, data loading, etc.)
- **High Memory Usage (>90%)**: Risk of out-of-memory errors, consider reducing batch size
- **High Temperature (>85°C)**: GPU is running hot, may throttle performance
- **Fluctuating Clocks**: May indicate thermal throttling or power limitations

## Programmatic Access

### Python API Basics

The ROCm SMI Python wrapper provides programmatic access to GPU information:

```python
from rocm_smi_lib import rsmi

# Initialize ROCm SMI
rsmi.rsmi_init(0)

try:
    # Get number of devices
    num_devices = rsmi.rsmi_num_monitor_devices()
    print(f"Found {num_devices} GPU device(s)")
    
    # Get device name
    for i in range(num_devices):
        name = rsmi.rsmi_dev_name_get(i)[1]
        print(f"GPU {i}: {name}")
        
        # Get GPU utilization
        util = rsmi.rsmi_dev_gpu_busy_percent_get(i)[1]
        print(f"  Utilization: {util}%")
        
        # Get temperature
        temp = rsmi.rsmi_dev_temp_metric_get(i, 0, 0)[1] / 1000.0  # Convert to °C
        print(f"  Temperature: {temp}°C")
        
        # Get memory usage
        mem_info = rsmi.rsmi_dev_memory_usage_get(i, 0)
        mem_used = mem_info[1] / (1024 * 1024)  # Convert to MB
        mem_total = mem_info[2] / (1024 * 1024)  # Convert to MB
        print(f"  Memory: {mem_used:.2f}/{mem_total:.2f} MB ({(mem_used/mem_total)*100:.2f}%)")
        
        # Get power consumption
        power = rsmi.rsmi_dev_power_ave_get(i)[1] / 1000000.0  # Convert to W
        print(f"  Power: {power:.2f} W")

finally:
    # Clean up
    rsmi.rsmi_shut_down()
```

### Example: Custom Monitoring

Here's an example of creating a custom monitoring solution:

```python
import time
from rocm_smi_lib import rsmi

def monitor_gpu(device_id, interval=1.0, duration=60):
    """Monitor a specific GPU for a given duration."""
    rsmi.rsmi_init(0)
    
    try:
        name = rsmi.rsmi_dev_name_get(device_id)[1]
        print(f"Monitoring GPU {device_id}: {name}")
        print(f"{'Time':10} {'Util%':6} {'Temp°C':6} {'Mem%':6} {'Power':6}")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            # Get metrics
            util = rsmi.rsmi_dev_gpu_busy_percent_get(device_id)[1]
            temp = rsmi.rsmi_dev_temp_metric_get(device_id, 0, 0)[1] / 1000.0
            
            mem_info = rsmi.rsmi_dev_memory_usage_get(device_id, 0)
            mem_used = mem_info[1]
            mem_total = mem_info[2]
            mem_percent = (mem_used / mem_total) * 100
            
            power = rsmi.rsmi_dev_power_ave_get(device_id)[1] / 1000000.0
            
            # Print metrics
            elapsed = time.time() - start_time
            print(f"{elapsed:10.1f} {util:6.1f} {temp:6.1f} {mem_percent:6.1f} {power:6.1f}")
            
            # Wait for next interval
            time.sleep(interval)
    
    finally:
        rsmi.rsmi_shut_down()

# Monitor GPU 0 for 30 seconds with 2-second interval
monitor_gpu(0, interval=2.0, duration=30)
```

### Example: Performance Logging

Here's an example of logging GPU performance during model training:

```python
import time
import csv
from datetime import datetime
from rocm_smi_lib import rsmi

class GPULogger:
    def __init__(self, log_file, interval=1.0):
        """Initialize GPU logger."""
        self.log_file = log_file
        self.interval = interval
        self.running = False
        
        # Initialize ROCm SMI
        rsmi.rsmi_init(0)
        
        # Get number of devices
        self.num_devices = rsmi.rsmi_num_monitor_devices()
        
        # Create CSV file
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Timestamp']
            for i in range(self.num_devices):
                header.extend([
                    f'GPU{i}_Util',
                    f'GPU{i}_Temp',
                    f'GPU{i}_MemUsed',
                    f'GPU{i}_MemTotal',
                    f'GPU{i}_Power'
                ])
            writer.writerow(header)
    
    def __del__(self):
        """Clean up ROCm SMI."""
        rsmi.rsmi_shut_down()
    
    def log_metrics(self):
        """Log current GPU metrics to file."""
        metrics = [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        
        for i in range(self.num_devices):
            try:
                # Get GPU utilization
                util = rsmi.rsmi_dev_gpu_busy_percent_get(i)[1]
                
                # Get temperature
                temp = rsmi.rsmi_dev_temp_metric_get(i, 0, 0)[1] / 1000.0
                
                # Get memory usage
                mem_info = rsmi.rsmi_dev_memory_usage_get(i, 0)
                mem_used = mem_info[1] / (1024 * 1024)  # MB
                mem_total = mem_info[2] / (1024 * 1024)  # MB
                
                # Get power consumption
                power = rsmi.rsmi_dev_power_ave_get(i)[1] / 1000000.0  # W
                
                metrics.extend([util, temp, mem_used, mem_total, power])
            except:
                metrics.extend([0, 0, 0, 0, 0])  # Default values on error
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metrics)
    
    def start_logging(self):
        """Start logging in a loop."""
        self.running = True
        try:
            while self.running:
                self.log_metrics()
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("Logging stopped by user")
        finally:
            self.running = False
    
    def stop_logging(self):
        """Stop logging."""
        self.running = False

# Usage example
logger = GPULogger('gpu_performance.csv', interval=5.0)
logger.start_logging()  # This will run until interrupted
```

## Advanced Features

### Controlling GPU Settings

ROCm SMI allows you to control various GPU settings:

```python
from rocm_smi_lib import rsmi

# Initialize ROCm SMI
rsmi.rsmi_init(0)

try:
    device_id = 0
    
    # Set GPU clock level (0-7, where 7 is highest)
    rsmi.rsmi_dev_gpu_clk_freq_set(device_id, 0, 7)
    
    # Set memory clock level (0-3, where 3 is highest)
    rsmi.rsmi_dev_gpu_clk_freq_set(device_id, 1, 3)
    
    # Set power limit (in microwatts)
    power_limit = 180 * 1000000  # 180 W
    rsmi.rsmi_dev_power_cap_set(device_id, 0, power_limit)
    
    # Set performance level (auto, low, high, manual)
    rsmi.rsmi_dev_perf_level_set(device_id, rsmi.RSMI_DEV_PERF_LEVEL_AUTO)
    
    # Set fan speed (as percentage)
    rsmi.rsmi_dev_fan_speed_set(device_id, 0, 70)

finally:
    # Clean up
    rsmi.rsmi_shut_down()
```

### Persistent Settings

To make settings persist across reboots, you can create a systemd service:

```bash
# Create a script with your desired settings
cat > $HOME/gpu_settings.sh << 'EOF'
#!/bin/bash
# Set GPU settings
rocm-smi --setsclk 7
rocm-smi --setmclk 3
rocm-smi --setpoweroverdrive 180
EOF

# Make it executable
chmod +x $HOME/gpu_settings.sh

# Create a systemd service file
sudo tee /etc/systemd/system/gpu-settings.service > /dev/null << 'EOF'
[Unit]
Description=Set GPU settings
After=display-manager.service

[Service]
Type=oneshot
ExecStart=$HOME/gpu_settings.sh
User=root

[Install]
WantedBy=multi-user.target
EOF

# Enable the service
sudo systemctl enable gpu-settings.service
```

## Integration with ML Workflows

### PyTorch Integration

Here's an example of integrating ROCm SMI with PyTorch training:

```python
import torch
import time
import threading
from rocm_smi_lib import rsmi

class GPUMonitor:
    def __init__(self, interval=1.0):
        """Initialize GPU monitor."""
        self.interval = interval
        self.running = False
        self.stats = {}
        
        # Initialize ROCm SMI
        rsmi.rsmi_init(0)
        
        # Get number of devices
        self.num_devices = rsmi.rsmi_num_monitor_devices()
        
        # Initialize stats for each device
        for i in range(self.num_devices):
            self.stats[i] = {
                'utilization': [],
                'temperature': [],
                'memory_used': [],
                'memory_total': [],
                'power': []
            }
    
    def __del__(self):
        """Clean up ROCm SMI."""
        rsmi.rsmi_shut_down()
    
    def collect_stats(self):
        """Collect GPU statistics."""
        for i in range(self.num_devices):
            try:
                # Get GPU utilization
                util = rsmi.rsmi_dev_gpu_busy_percent_get(i)[1]
                self.stats[i]['utilization'].append(util)
                
                # Get temperature
                temp = rsmi.rsmi_dev_temp_metric_get(i, 0, 0)[1] / 1000.0
                self.stats[i]['temperature'].append(temp)
                
                # Get memory usage
                mem_info = rsmi.rsmi_dev_memory_usage_get(i, 0)
                mem_used = mem_info[1] / (1024 * 1024)  # MB
                mem_total = mem_info[2] / (1024 * 1024)  # MB
                self.stats[i]['memory_used'].append(mem_used)
                self.stats[i]['memory_total'].append(mem_total)
                
                # Get power consumption
                power = rsmi.rsmi_dev_power_ave_get(i)[1] / 1000000.0  # W
                self.stats[i]['power'].append(power)
            except:
                pass  # Skip on error
    
    def start_monitoring(self):
        """Start monitoring in a separate thread."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def _monitor_loop(self):
        """Monitoring loop."""
        while self.running:
            self.collect_stats()
            time.sleep(self.interval)
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=self.interval*2)
    
    def get_summary(self):
        """Get summary statistics."""
        summary = {}
        for i in range(self.num_devices):
            device_stats = self.stats[i]
            summary[i] = {
                'utilization': {
                    'mean': sum(device_stats['utilization']) / max(1, len(device_stats['utilization'])),
                    'max': max(device_stats['utilization']) if device_stats['utilization'] else 0,
                    'min': min(device_stats['utilization']) if device_stats['utilization'] else 0
                },
                'temperature': {
                    'mean': sum(device_stats['temperature']) / max(1, len(device_stats['temperature'])),
                    'max': max(device_stats['temperature']) if device_stats['temperature'] else 0,
                    'min': min(device_stats['temperature']) if device_stats['temperature'] else 0
                },
                'memory_used': {
                    'mean': sum(device_stats['memory_used']) / max(1, len(device_stats['memory_used'])),
                    'max': max(device_stats['memory_used']) if device_stats['memory_used'] else 0,
                    'min': min(device_stats['memory_used']) if device_stats['memory_used'] else 0
                },
                'power': {
                    'mean': sum(device_stats['power']) / max(1, len(device_stats['power'])),
                    'max': max(device_stats['power']) if device_stats['power'] else 0,
                    'min': min(device_stats['power']) if device_stats['power'] else 0
                }
            }
        return summary

# Example usage with PyTorch training
def train_model():
    # Initialize model and data
    model = torch.nn.Linear(1000, 1000).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Create dummy data
    x = torch.randn(100, 1000).to('cuda')
    y = torch.randn(100, 1000).to('cuda')
    
    # Initialize GPU monitor
    monitor = GPUMonitor(interval=0.5)
    monitor.start_monitoring()
    
    # Training loop
    try:
        for epoch in range(10):
            start_time = time.time()
            
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}, Time: {epoch_time:.2f}s")
    
    finally:
        # Stop monitoring and print summary
        monitor.stop_monitoring()
        summary = monitor.get_summary()
        
        print("\nGPU Statistics Summary:")
        for device_id, device_summary in summary.items():
            print(f"\nGPU {device_id}:")
            print(f"  Utilization: {device_summary['utilization']['mean']:.1f}% (Min: {device_summary['utilization']['min']:.1f}%, Max: {device_summary['utilization']['max']:.1f}%)")
            print(f"  Temperature: {device_summary['temperature']['mean']:.1f}°C (Min: {device_summary['temperature']['min']:.1f}°C, Max: {device_summary['temperature']['max']:.1f}°C)")
            print(f"  Memory Used: {device_summary['memory_used']['mean']:.1f} MB (Min: {device_summary['memory_used']['min']:.1f} MB, Max: {device_summary['memory_used']['max']:.1f} MB)")
            print(f"  Power: {device_summary['power']['mean']:.1f} W (Min: {device_summary['power']['min']:.1f} W, Max: {device_summary['power']['max']:.1f} W)")

# Run training with monitoring
train_model()
```

### Optimizing GPU Performance

Here are some tips for optimizing GPU performance based on ROCm SMI metrics:

1. **Memory Management**:
   - Monitor memory usage to avoid OOM errors
   - Adjust batch size based on available memory
   - Use gradient accumulation for large models

2. **Thermal Management**:
   - Monitor temperature to avoid thermal throttling
   - Adjust fan speed or power limit if temperatures are too high
   - Ensure proper cooling for long training runs

3. **Power Efficiency**:
   - Monitor power consumption to optimize efficiency
   - Find the sweet spot between performance and power usage
   - Consider undervolting for better efficiency

4. **Clock Speed Optimization**:
   - Monitor GPU and memory clock speeds
   - Adjust clock speeds based on workload requirements
   - Find optimal settings for your specific model

## Troubleshooting

### Common Issues

1. **Permission Denied**:
   ```
   Error: Permission denied
   ```
   Solution: Run with sudo or add user to the 'video' group:
   ```bash
   sudo usermod -a -G video $USER
   # Log out and log back in
   ```

2. **Device Not Found**:
   ```
   Error: No AMD GPU devices found
   ```
   Solution: Verify ROCm installation and GPU support:
   ```bash
   rocminfo
   ```

3. **API Initialization Failed**:
   ```
   Error: RSMI initialization failed
   ```
   Solution: Check ROCm installation and try reinstalling:
   ```bash
   sudo apt reinstall rocm-smi
   ```

4. **Python Wrapper Not Found**:
   ```
   ImportError: No module named 'rocm_smi_lib'
   ```
   Solution: Reinstall the Python wrapper:
   ```bash
   cd /path/to/rocm_smi_lib/python_smi_tools
   pip install -e .
   ```

### Debugging Tips

1. **Check ROCm Installation**:
   ```bash
   rocminfo
   ```

2. **Verify GPU Detection**:
   ```bash
   lspci | grep -i amd
   ```

3. **Check Driver Status**:
   ```bash
   sudo dmesg | grep -i amdgpu
   ```

4. **Test Basic Functionality**:
   ```bash
   rocm-smi --showallinfo
   ```

5. **Enable Verbose Logging**:
   ```bash
   export RSMI_LOGGING=1
   rocm-smi
   ```

## References

1. [ROCm SMI GitHub Repository](https://github.com/RadeonOpenCompute/rocm_smi_lib)
2. [ROCm Documentation](https://rocm.docs.amd.com/)
3. [ROCm SMI Command-Line Reference](https://github.com/RadeonOpenCompute/rocm_smi_lib/blob/master/README.md)
4. [ROCm SMI Python API Reference](https://github.com/RadeonOpenCompute/rocm_smi_lib/blob/master/python_smi_tools/README.md)
5. [AMD GPU Architecture Guide](https://www.amd.com/en/technologies/rdna-2)
6. [PyTorch ROCm Support](https://pytorch.org/docs/stable/notes/hip.html)


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

