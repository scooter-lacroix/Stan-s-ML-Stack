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
# PyTorch Profiler Installation Script for AMD GPUs
# =============================================================================
# This script installs and configures PyTorch Profiler for performance analysis
# of PyTorch models on AMD GPUs.
#
# Author: User
# Date: $(date +"%Y-%m-%d")
# =============================================================================

set -e  # Exit on error

# Create log directory
LOG_DIR="$HOME/Desktop/ml_stack_extensions/logs"
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/pytorch_profiler_install_$(date +"%Y%m%d_%H%M%S").log"

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Start installation
log "=== Starting PyTorch Profiler Installation ==="
log "System: $(uname -a)"
log "ROCm Path: $(which hipcc 2>/dev/null || echo 'Not found')"
log "Python Version: $(python3 --version)"
log "PyTorch Version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

# Check if PyTorch is installed
if ! python3 -c "import torch" &>/dev/null; then
    log "Error: PyTorch is not installed. Please install PyTorch with ROCm support first."
    exit 1
fi

# Create installation directory
INSTALL_DIR="$HOME/ml_stack/pytorch_profiler"
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# Install dependencies
log "Installing dependencies..."
pip install tensorboard pandas matplotlib --break-system-packages

# Verify installation
log "Verifying PyTorch Profiler installation..."
python3 -c "from torch.profiler import profile, record_function, ProfilerActivity; print('PyTorch Profiler is available')"

if [ $? -eq 0 ]; then
    log "PyTorch Profiler installation successful!"
else
    log "PyTorch Profiler installation failed. Please check the logs."
    exit 1
fi

# Create example scripts
log "Creating example scripts..."

# Basic profiling example
BASIC_EXAMPLE="$INSTALL_DIR/basic_profiling_example.py"
cat > $BASIC_EXAMPLE << 'EOF'
#!/usr/bin/env python3
"""
Basic PyTorch Profiler Example

This script demonstrates how to use PyTorch Profiler to analyze the performance
of a simple neural network on AMD GPUs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        with record_function("fc1"):
            x = self.fc1(x)
        with record_function("relu"):
            x = self.relu(x)
        with record_function("fc2"):
            x = self.fc2(x)
        return x

# Create model and move to GPU
input_size = 1000
hidden_size = 2000
output_size = 500
model = SimpleModel(input_size, hidden_size, output_size).to(device)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create dummy data
batch_size = 64
x = torch.randn(batch_size, input_size, device=device)
y = torch.randn(batch_size, output_size, device=device)

# Loss function
criterion = nn.MSELoss()

# Warm-up
for _ in range(5):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Profile with PyTorch Profiler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for _ in range(10):
        with record_function("training_batch"):
            optimizer.zero_grad()
            with record_function("forward"):
                output = model(x)
                loss = criterion(output, y)
            with record_function("backward"):
                loss.backward()
            with record_function("optimizer_step"):
                optimizer.step()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export trace
prof.export_chrome_trace("trace.json")
print("Trace exported to trace.json")

# Export stack trace
prof.export_stacks("stacks.txt", "self_cuda_time_total")
print("Stack trace exported to stacks.txt")

print("\nTo view the trace in Chrome:")
print("1. Open Chrome and navigate to chrome://tracing")
print("2. Click 'Load' and select the trace.json file")
EOF

# Advanced profiling example
ADVANCED_EXAMPLE="$INSTALL_DIR/advanced_profiling_example.py"
cat > $ADVANCED_EXAMPLE << 'EOF'
#!/usr/bin/env python3
"""
Advanced PyTorch Profiler Example

This script demonstrates advanced usage of PyTorch Profiler, including
TensorBoard integration, memory profiling, and custom events.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import schedule, tensorboard_trace_handler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# Define a more complex model
class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 10)
    
    def forward(self, x):
        with record_function("conv_block1"):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.pool1(x)
        
        with record_function("conv_block2"):
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.pool2(x)
        
        with record_function("conv_block3"):
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.pool3(x)
        
        with record_function("classifier"):
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu4(x)
            x = self.dropout(x)
            x = self.fc2(x)
        
        return x

# Create model and move to GPU
model = ComplexModel().to(device)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create dummy data (simulating images)
batch_size = 32
x = torch.randn(batch_size, 3, 32, 32, device=device)
y = torch.randint(0, 10, (batch_size,), device=device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Create directory for TensorBoard logs
log_dir = "tb_logs"
os.makedirs(log_dir, exist_ok=True)

# Define profiler schedule
def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace(f"trace_{p.step_num}.json")

# Custom profiling schedule
my_schedule = schedule(
    skip_first=5,    # Skip first 5 steps (warm-up)
    wait=1,          # Wait 1 step
    warmup=1,        # Warmup for 1 step
    active=3,        # Profile for 3 steps
    repeat=1         # Repeat the cycle once
)

# Training function
def train(num_epochs=5):
    # Warm-up
    print("Warming up...")
    for _ in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    # Profile with PyTorch Profiler
    print("Starting profiling...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=my_schedule,
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as prof:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Simulate multiple batches
            for batch in range(5):
                with record_function(f"epoch_{epoch}_batch_{batch}"):
                    # Data loading simulation
                    with record_function("data_loading"):
                        time.sleep(0.01)  # Simulate data loading
                        batch_x = torch.randn(batch_size, 3, 32, 32, device=device)
                        batch_y = torch.randint(0, 10, (batch_size,), device=device)
                    
                    # Training step
                    optimizer.zero_grad()
                    
                    with record_function("forward"):
                        output = model(batch_x)
                        loss = criterion(output, batch_y)
                    
                    with record_function("backward"):
                        loss.backward()
                    
                    with record_function("optimizer_step"):
                        optimizer.step()
                
                # Step the profiler
                prof.step()
    
    print("Profiling complete!")
    print(f"TensorBoard logs saved to {log_dir}")
    print("\nTo view in TensorBoard, run:")
    print(f"tensorboard --logdir={log_dir}")

if __name__ == "__main__":
    train()
EOF

# Memory profiling example
MEMORY_EXAMPLE="$INSTALL_DIR/memory_profiling_example.py"
cat > $MEMORY_EXAMPLE << 'EOF'
#!/usr/bin/env python3
"""
Memory Profiling Example

This script demonstrates how to use PyTorch Profiler to analyze memory usage
of PyTorch models on AMD GPUs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torch.profiler import profile, record_function, ProfilerActivity

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# Define a model with memory-intensive operations
class MemoryIntensiveModel(nn.Module):
    def __init__(self):
        super(MemoryIntensiveModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 1 * 1, 4096)
        self.fc2 = nn.Linear(4096, 1000)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Store intermediate activations
        activations = []
        
        with record_function("conv_block1"):
            x = self.conv1(x)
            x = self.relu(x)
            activations.append(x.clone())  # Store activation
            x = self.pool(x)
        
        with record_function("conv_block2"):
            x = self.conv2(x)
            x = self.relu(x)
            activations.append(x.clone())  # Store activation
            x = self.pool(x)
        
        with record_function("conv_block3"):
            x = self.conv3(x)
            x = self.relu(x)
            activations.append(x.clone())  # Store activation
            x = self.pool(x)
        
        with record_function("conv_block4"):
            x = self.conv4(x)
            x = self.relu(x)
            activations.append(x.clone())  # Store activation
            x = self.pool(x)
        
        with record_function("conv_block5"):
            x = self.conv5(x)
            x = self.relu(x)
            activations.append(x.clone())  # Store activation
            x = self.pool(x)
        
        with record_function("classifier"):
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            activations.append(x.clone())  # Store activation
            x = self.fc2(x)
        
        # Concatenate activations to simulate memory pressure
        with record_function("memory_pressure"):
            concat = torch.cat([a.flatten(start_dim=1) for a in activations], dim=1)
            result = torch.matmul(concat, concat.transpose(0, 1))
            x = x + result.mean(dim=1, keepdim=True)
        
        return x

# Create model and move to GPU
model = MemoryIntensiveModel().to(device)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create dummy data (simulating images)
batch_size = 16
x = torch.randn(batch_size, 3, 32, 32, device=device)
y = torch.randint(0, 1000, (batch_size,), device=device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Warm-up
print("Warming up...")
for _ in range(3):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Profile with PyTorch Profiler
print("Starting memory profiling...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    optimizer.zero_grad()
    with record_function("forward"):
        output = model(x)
        loss = criterion(output, y)
    with record_function("backward"):
        loss.backward()
    with record_function("optimizer_step"):
        optimizer.step()

# Print memory stats
print("\nMemory Stats by Operator:")
memory_stats = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
print(memory_stats)

# Export trace
prof.export_chrome_trace("memory_trace.json")
print("Memory trace exported to memory_trace.json")

# Analyze and visualize memory usage
events = prof.events()
memory_events = []

for evt in events:
    if evt.cuda_memory_usage != 0:
        memory_events.append({
            'name': evt.name,
            'memory_usage': evt.cuda_memory_usage / (1024 * 1024),  # Convert to MB
            'self_memory_usage': evt.self_cuda_memory_usage / (1024 * 1024),  # Convert to MB
            'device': evt.device
        })

if memory_events:
    # Convert to DataFrame
    df = pd.DataFrame(memory_events)
    
    # Plot memory usage by operator
    plt.figure(figsize=(12, 6))
    top_memory_ops = df.nlargest(10, 'self_memory_usage')
    plt.barh(top_memory_ops['name'], top_memory_ops['self_memory_usage'])
    plt.xlabel('Self Memory Usage (MB)')
    plt.title('Top 10 Memory-Intensive Operations')
    plt.tight_layout()
    plt.savefig('memory_usage_by_op.png')
    print("Memory usage plot saved to memory_usage_by_op.png")
else:
    print("No memory events recorded")

print("\nMemory Profiling Tips:")
print("1. Look for large memory allocations in the 'self_cuda_memory_usage' column")
print("2. Check for memory leaks by monitoring memory usage over time")
print("3. Use smaller batch sizes or gradient accumulation for large models")
print("4. Consider using checkpointing for memory-intensive models")
print("5. Use mixed precision training to reduce memory usage")
EOF

# Make the example scripts executable
chmod +x $BASIC_EXAMPLE $ADVANCED_EXAMPLE $MEMORY_EXAMPLE

log "Created example scripts:"
log "- Basic profiling: $BASIC_EXAMPLE"
log "- Advanced profiling: $ADVANCED_EXAMPLE"
log "- Memory profiling: $MEMORY_EXAMPLE"

# Create a README file
README_FILE="$INSTALL_DIR/README.md"
cat > $README_FILE << 'EOF'
# PyTorch Profiler Examples

This directory contains examples of using PyTorch Profiler with AMD GPUs.

## Basic Profiling Example

```bash
python basic_profiling_example.py
```

This example demonstrates:
- Basic profiling setup
- Recording function calls
- Analyzing CPU and GPU performance
- Exporting Chrome traces

## Advanced Profiling Example

```bash
python advanced_profiling_example.py
```

This example demonstrates:
- TensorBoard integration
- Custom profiling schedules
- Memory profiling
- FLOP counting
- Custom events

## Memory Profiling Example

```bash
python memory_profiling_example.py
```

This example demonstrates:
- Memory usage analysis
- Identifying memory-intensive operations
- Visualizing memory usage
- Memory optimization tips

## Viewing Traces

To view Chrome traces:
1. Open Chrome and navigate to `chrome://tracing`
2. Click "Load" and select the generated trace file

To view TensorBoard traces:
```bash
tensorboard --logdir=tb_logs
```
EOF

log "Created README file: $README_FILE"

log "=== PyTorch Profiler Installation Complete ==="
log "Installation Directory: $INSTALL_DIR"
log "Log File: $LOG_FILE"
log "Documentation: $HOME/Desktop/ml_stack_extensions/docs/pytorch_profiler_guide.md"
