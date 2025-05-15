#!/bin/bash
#
# WandB Installation Script for ML Stack
#
# This script installs Weights & Biases (WandB), a tool for experiment tracking,
# model visualization, and collaboration.
#

set -e  # Exit on error

# Create log directory
LOG_DIR="$HOME/Desktop/ml_stack_extensions/logs"
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/wandb_install_$(date +"%Y%m%d_%H%M%S").log"

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Start installation
log "=== Starting WandB Installation ==="
log "System: $(uname -a)"
log "Python Version: $(python3 --version)"
log "PyTorch Version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

# Check for required dependencies
log "Checking dependencies..."
DEPS=("python3" "pip")
MISSING_DEPS=()

for dep in "${DEPS[@]}"; do
    if ! command_exists $dep; then
        MISSING_DEPS+=("$dep")
    fi
done

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    log "Missing dependencies: ${MISSING_DEPS[*]}"
    log "Please install them and run this script again."
    exit 1
fi

# Create installation directory
INSTALL_DIR="$HOME/ml_stack/wandb"
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# Install WandB
log "Installing WandB..."
if command_exists uv; then
    log "Using uv to install wandb..."
    uv pip install wandb
else
    log "Using pip to install wandb..."
    pip install wandb --break-system-packages
fi

# Verify installation
log "Verifying WandB installation..."
python3 -c "import wandb; print('WandB version:', wandb.__version__)"

# If verification failed, try installing with pip directly
if [ $? -ne 0 ]; then
    log "First installation attempt failed, trying alternative method..."
    pip install --upgrade pip
    pip install --upgrade setuptools wheel
    pip install wandb --no-cache-dir

    # Verify again
    log "Verifying WandB installation (second attempt)..."
    python3 -c "import wandb; print('WandB version:', wandb.__version__)"
fi

if [ $? -eq 0 ]; then
    log "WandB installation successful!"
else
    log "WandB installation failed. Please check the logs."
    exit 1
fi

# Create a simple test script
TEST_SCRIPT="$INSTALL_DIR/test_wandb.py"
cat > $TEST_SCRIPT << 'EOF'
#!/usr/bin/env python3
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

def test_wandb_logging():
    """Test basic WandB logging."""
    print("=== Testing WandB Logging ===")

    # Initialize WandB
    run = wandb.init(project="ml_stack_test", name="test_run", mode="offline")

    # Log some metrics
    for i in range(10):
        wandb.log({
            "loss": 1.0 - i * 0.1,
            "accuracy": i * 0.1,
            "step": i
        })

    # Log a table
    data = [[i, i**2, i**3] for i in range(10)]
    table = wandb.Table(data=data, columns=["x", "x^2", "x^3"])
    wandb.log({"table": table})

    # Log a plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    data = [[x[i], y[i]] for i in range(len(x))]
    table = wandb.Table(data=data, columns=["x", "sin(x)"])
    wandb.log({"plot": wandb.plot.line(table, "x", "sin(x)", title="Sin Wave")})

    # Finish the run
    run.finish()

    print("WandB logging test completed successfully!")
    return True

def test_wandb_model_tracking():
    """Test WandB model tracking."""
    print("\n=== Testing WandB Model Tracking ===")

    # Initialize WandB
    run = wandb.init(project="ml_stack_test", name="model_tracking", mode="offline")

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(10, 5)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(5, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    model = SimpleModel()

    # Watch the model
    wandb.watch(model, log="all")

    # Create some data
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)

    # Train the model
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        wandb.log({
            "epoch": epoch,
            "loss": loss.item()
        })

    # Save the model
    torch.save(model.state_dict(), "simple_model.pt")
    wandb.save("simple_model.pt")

    # Finish the run
    run.finish()

    print("WandB model tracking test completed successfully!")
    return True

def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test WandB")
    parser.add_argument("--test-logging", action="store_true", help="Run logging test")
    parser.add_argument("--test-model", action="store_true", help="Run model tracking test")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    # If no specific test is selected, run all tests
    if not (args.test_logging or args.test_model) or args.all:
        args.test_logging = args.test_model = True

    # Run tests
    results = []

    if args.test_logging:
        try:
            result = test_wandb_logging()
            results.append(("Logging Test", result))
        except Exception as e:
            print(f"Error in logging test: {e}")
            results.append(("Logging Test", False))

    if args.test_model:
        try:
            result = test_wandb_model_tracking()
            results.append(("Model Tracking Test", result))
        except Exception as e:
            print(f"Error in model tracking test: {e}")
            results.append(("Model Tracking Test", False))

    # Print summary
    print("\n=== Test Summary ===")
    all_passed = True
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")
        all_passed = all_passed and result

    if all_passed:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Please check the logs.")

if __name__ == "__main__":
    main()
EOF

log "Created test script at $TEST_SCRIPT"
log "You can run it with: python3 $TEST_SCRIPT"

# Create documentation directory
DOCS_DIR="$HOME/Desktop/ml_stack_extensions/docs"
mkdir -p $DOCS_DIR

# Create documentation
DOCS_FILE="$DOCS_DIR/wandb_guide.md"
cat > $DOCS_FILE << 'EOF'
# Weights & Biases (WandB) Guide

## Overview

Weights & Biases (WandB) is a tool for experiment tracking, model visualization, and collaboration. It helps you track your machine learning experiments, visualize model performance, and share your results with your team.

## Features

- **Experiment Tracking**: Track hyperparameters, metrics, and artifacts for your machine learning experiments.
- **Visualization**: Visualize model performance with interactive plots and dashboards.
- **Collaboration**: Share your results with your team and collaborate on machine learning projects.
- **Model Management**: Version and manage your models and datasets.
- **Reports**: Create and share reports with your team.

## Usage

### Basic Logging

```python
import wandb

# Initialize WandB
wandb.init(project="my_project", name="my_run")

# Log metrics
for i in range(10):
    wandb.log({
        "loss": 1.0 - i * 0.1,
        "accuracy": i * 0.1,
        "step": i
    })

# Finish the run
wandb.finish()
```

### Model Tracking

```python
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize WandB
wandb.init(project="my_project", name="model_tracking")

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleModel()

# Watch the model
wandb.watch(model, log="all")

# Train the model
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    wandb.log({
        "epoch": epoch,
        "loss": loss.item()
    })

# Save the model
torch.save(model.state_dict(), "simple_model.pt")
wandb.save("simple_model.pt")

# Finish the run
wandb.finish()
```

## Resources

- [WandB Documentation](https://docs.wandb.ai/)
- [WandB GitHub Repository](https://github.com/wandb/wandb)
- [WandB Python API Reference](https://docs.wandb.ai/ref/python)
EOF

log "Created documentation at $DOCS_FILE"

log "=== WandB Installation Complete ==="
log "Installation Directory: $INSTALL_DIR"
log "Log File: $LOG_FILE"
log "Documentation: $DOCS_FILE"
