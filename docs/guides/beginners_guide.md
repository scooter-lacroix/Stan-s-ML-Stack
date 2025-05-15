# Stan's ML Stack: Beginner's Guide

Welcome to Stan's ML Stack! This guide is designed to help absolute beginners get started with machine learning on AMD GPUs. We'll walk through everything step by step, using simple language and plenty of examples.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [UnderStanding the Components](#underStanding-the-components)
- [Your First ML Project](#your-first-ml-project)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Glossary](#glossary)
- [Learning Resources](#learning-resources)
- [Frequently Asked Questions](#frequently-asked-questions)

## Introduction

### What is Stan's ML Stack?

Stan's ML Stack is a collection of software tools that work together to help you build and run machine learning (ML) models on AMD graphics cards (GPUs). Think of it as a complete toolkit that has everything you need to create AI applications.

### Why Use AMD GPUs for Machine Learning?

AMD GPUs like the RX 7900 XTX and RX 7800 XT are powerful and cost-effective for machine learning. They can process lots of data in parallel, which makes training and running ML models much faster than using just a regular computer processor (CPU).

### What Can You Do With This Stack?

With Stan's ML Stack, you can:
- Train neural networks (a type of AI model)
- Run large language models (like ChatGPT)
- Process images, text, and other data
- Create your own AI applications
- Experiment with cutting-edge ML techniques

## Prerequisites

Before you begin, you'll need:

### Hardware Requirements

- **AMD GPU**: You need an AMD graphics card. This guide is optimized for:
  - RX 7900 XTX (best performance)
  - RX 7800 XT (also great)
  - Other AMD GPUs may work too

- **Computer Specifications**:
  - At least 16GB of RAM (32GB or more recommended)
  - A decent CPU (any modern processor will do)
  - At least 100GB of free disk space
  - Power supply that can handle your GPU (750W+ recommended)

### Software Requirements

- **Operating System**: Ubuntu 25.04 (or another Linux distribution)
- **Basic Knowledge**: 
  - How to use the terminal/command line
  - Basic underStanding of what machine learning is

Don't worry if you're not an expert! This guide explains everything in simple terms.

## Installation

### Step 1: Prepare Your System

First, make sure your system is up to date:

```bash
sudo apt update
sudo apt upgrade
```

### Step 2: Get the ML Stack

Download Stan's ML Stack:

```bash
cd ~/Desktop
git clone https://github.com/Stan/ml-stack.git Stans_MLStack
cd Stans_MLStack
```

### Step 3: Run the Installation Script

The installation script will guide you through the process:

```bash
chmod +x scripts/install_ml_stack.sh
./scripts/install_ml_stack.sh
```

You'll see a colorful menu like this:

```
  ██████╗████████╗ █████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗         ███████╗████████╗ █████╗  ██████╗██╗  ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝    ████╗ ████║██║         ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
 ╚█████╗    ██║   ███████║██╔██╗ ██║███████╗    ██╔████╔██║██║         ███████╗   ██║   ███████║██║     █████╔╝ 
  ╚═══██╗   ██║   ██╔══██║██║╚██╗██║╚════██║    ██║╚██╔╝██║██║         ╚════██║   ██║   ██╔══██║██║     ██╔═██╗ 
 ██████╔╝   ██║   ██║  ██║██║ ╚████║███████║    ██║ ╚═╝ ██║███████╗    ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

A Complete Machine Learning Stack for AMD GPUs
https://github.com/Stan/ml-stack

Which components would you like to install?
1. Core Components
   a. ROCm Configuration - Set up environment variables for optimal GPU performance
   b. PyTorch with ROCm - Deep learning framework with AMD GPU support
   c. ONNX Runtime - Cross-platform inference engine for ONNX models
   d. MIGraphX - AMD's graph optimization library for deep learning
   e. Megatron-LM - Framework for training large language models
   f. Flash Attention - Efficient attention computation for transformers
   g. RCCL - ROCm Collective Communication Library for multi-GPU training
   h. MPI - Message Passing Interface for distributed computing
2. Extension Components
   a. Triton - Compiler for parallel programming with GPU kernels
   b. BITSANDBYTES - Efficient quantization for deep learning models
   c. vLLM - High-throughput inference engine for LLMs
   d. ROCm SMI - System monitoring and management for AMD GPUs
   e. PyTorch Profiler - Performance analysis for PyTorch models
   f. Weights & Biases - Experiment tracking and visualization
3. All Components - Install everything (may take several hours)
4. Exit

Enter your choice (1, 2, 3, or 4): 
```

For beginners, we recommend:

1. Choose option `1` (Core Components)
2. Then choose option `i` (All Core Components)

This will install the essential components you need to get started.

The installation will take some time (1-2 hours). You'll see progress bars for each component.

### Step 4: Verify the Installation

After installation completes, verify that everything is working:

```bash
./scripts/verify_installation.sh
```

This will check all components and show you a summary:

```
Stan's ML Stack Verification Summary:

Components Verified: 12
Components Installed: 12
Components Failed: 0
Installation Percentage: 100%

Progress: [##################################################] 100%

Component Status:
Core Components:
  ROCm: ✓ Installed
  PyTorch: ✓ Installed
  ONNX Runtime: ✓ Installed
  MIGraphX: ✓ Installed
  Flash Attention: ✓ Installed
  MPI: ✓ Installed

Extension Components:
  Triton: ✓ Installed
  BITSANDBYTES: ✓ Installed
  vLLM: ✓ Installed
  ROCm SMI: ✓ Installed
  PyTorch Profiler: ✓ Installed
  Weights & Biases: ✓ Installed
```

If any component shows as "Not installed," you can run the installation script again and select just that component.

## UnderStanding the Components

Let's break down what each part of the ML stack does in simple terms:

### Core Components

#### ROCm
**What it is**: The software that lets your computer talk to your AMD GPU.  
**Why it matters**: Without this, your GPU would just sit there doing nothing!  
**Think of it as**: The translator between your computer and your GPU.

#### PyTorch
**What it is**: A popular framework for building and training ML models.  
**Why it matters**: Makes it easy to create neural networks and other ML models.  
**Think of it as**: Your ML construction kit with building blocks for AI.

#### ONNX Runtime
**What it is**: A tool that helps run ML models efficiently.  
**Why it matters**: Makes your models run faster and work on different devices.  
**Think of it as**: A performance booster for your ML models.

#### MIGraphX
**What it is**: AMD's tool for optimizing ML models.  
**Why it matters**: Makes models run even faster on AMD GPUs.  
**Think of it as**: A mechanic that tunes your model for maximum speed.

#### Flash Attention
**What it is**: A faster way to process attention in transformer models.  
**Why it matters**: Makes large language models run much faster.  
**Think of it as**: A supercharger for AI that processes text and images.

### Extension Components

#### Triton
**What it is**: A tool for writing custom GPU operations.  
**Why it matters**: Lets you create specialized, fast code for your GPU.  
**Think of it as**: A workshop for building custom GPU tools.

#### BITSANDBYTES
**What it is**: A tool for making models smaller and faster.  
**Why it matters**: Lets you run larger models with less memory.  
**Think of it as**: A compression tool for your ML models.

#### vLLM
**What it is**: A specialized tool for running large language models.  
**Why it matters**: Makes text generation models run much faster.  
**Think of it as**: A race car engine specifically for text AI.

#### ROCm SMI
**What it is**: A tool for monitoring your GPU.  
**Why it matters**: Helps you see how your GPU is performing.  
**Think of it as**: The dashboard for your GPU.

#### PyTorch Profiler
**What it is**: A tool for finding performance bottlenecks.  
**Why it matters**: Helps you figure out why your model might be slow.  
**Think of it as**: A detective that finds performance problems.

#### Weights & Biases
**What it is**: A tool for tracking your ML experiments.  
**Why it matters**: Helps you keep track of your models and results.  
**Think of it as**: A lab notebook for your AI experiments.

## Your First ML Project

Let's create a simple project to recognize handwritten digits using the MNIST dataset. This is the "Hello World" of machine learning!

### Step 1: Create a Project Folder

```bash
mkdir ~/Desktop/my_first_ml_project
cd ~/Desktop/my_first_ml_project
```

### Step 2: Create the Python Script

Create a file called `mnist_classifier.py` with this content:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and transform the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download the training dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

# Create a data loader
train_loader = DataLoader(
    train_dataset, 
    batch_size=64, 
    shuffle=True
)

# Download the test dataset
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

# Create a data loader for test data
test_loader = DataLoader(
    test_dataset, 
    batch_size=1000, 
    shuffle=False
)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

# Create the model and move it to the GPU
model = SimpleNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        # Move data to GPU
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Print statistics every 100 batches
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0
    
    # Test the model after each epoch
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy: {100 * correct / total:.2f}%')

# Save the model
torch.save(model.state_dict(), 'mnist_model.pth')
print('Training complete! Model saved as mnist_model.pth')
```

### Step 3: Run the Script

```bash
python mnist_classifier.py
```

This script will:
1. Download the MNIST dataset (handwritten digits)
2. Create a simple neural network
3. Train the model on your GPU
4. Test the model's accuracy
5. Save the trained model

You should see output like this:

```
Using device: cuda
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Processing...
Done!
Epoch [1/5], Step [100/938], Loss: 0.3301
Epoch [1/5], Step [200/938], Loss: 0.1539
...
Epoch [1/5], Accuracy: 96.23%
...
Epoch [5/5], Accuracy: 98.17%
Training complete! Model saved as mnist_model.pth
```

Congratulations! You've just trained your first neural network on an AMD GPU!

### Step 4: Test Your Model with a Single Image

Create a file called `test_single_image.py`:

```python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(28 * 28, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10)
).to(device)
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# Load a test image
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transforms.ToTensor()
)

# Get a random test image
index = np.random.randint(0, len(test_dataset))
image, label = test_dataset[index]

# Display the image
plt.figure(figsize=(3, 3))
plt.imshow(image.squeeze().numpy(), cmap='gray')
plt.title(f"True Label: {label}")
plt.axis('off')
plt.savefig('test_image.png')

# Prepare the image for the model
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
normalized_image = transform(image.squeeze().numpy())
input_tensor = normalized_image.unsqueeze(0).to(device)

# Make a prediction
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    probability = torch.nn.functional.softmax(output, dim=1)[0]

print(f"True label: {label}")
print(f"Predicted label: {predicted.item()}")
print(f"Confidence: {probability[predicted.item()].item()*100:.2f}%")

# Print probabilities for all digits
for i in range(10):
    print(f"Probability of being {i}: {probability[i].item()*100:.2f}%")
```

Run it:

```bash
python test_single_image.py
```

This will:
1. Load your trained model
2. Select a random test image
3. Make a prediction
4. Show the probabilities for each digit

You should see output like:

```
True label: 7
Predicted label: 7
Confidence: 99.87%
Probability of being 0: 0.00%
Probability of being 1: 0.00%
...
Probability of being 7: 99.87%
...
Probability of being 9: 0.00%
```

It will also save the test image as `test_image.png` so you can see what digit was being classified.

## Common Issues and Solutions

### "No CUDA GPUs are available"

**Problem**: Your code can't find your AMD GPU.

**Solution**:
1. Check that ROCm is installed: `rocminfo`
2. Set environment variables:
   ```bash
   export HIP_VISIBLE_DEVICES=0
   export CUDA_VISIBLE_DEVICES=0
   export PYTORCH_ROCM_DEVICE=0
   ```
3. Add these to your `~/.bashrc` file for permanent effect

### "CUDA out of memory"

**Problem**: Your model is too large for your GPU's memory.

**Solution**:
1. Reduce batch size
2. Use mixed precision training:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   with autocast():
       outputs = model(inputs)
       loss = criterion(outputs, targets)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

### Slow Training

**Problem**: Training is slower than expected.

**Solution**:
1. Check GPU utilization: `rocm-smi`
2. Optimize data loading:
   ```python
   train_loader = DataLoader(
       train_dataset, 
       batch_size=64, 
       shuffle=True,
       num_workers=4,  # Use multiple CPU cores
       pin_memory=True  # Faster data transfer to GPU
   )
   ```
3. Set PyTorch to benchmark mode:
   ```python
   torch.backends.cudnn.benchmark = True
   ```

### Installation Errors

**Problem**: Component installation fails.

**Solution**:
1. Check log files in the `logs` directory
2. Make sure you have all prerequisites installed
3. Try installing the specific component manually
4. Check for permission issues (you might need `sudo` for some operations)

## Glossary

- **GPU**: Graphics Processing Unit - specialized hardware for parallel processing
- **CPU**: Central Processing Unit - the main processor in your computer
- **ML**: Machine Learning - a type of AI that learns from data
- **Neural Network**: A computing system inspired by biological neural networks
- **ROCm**: Radeon Open Compute - AMD's platform for GPU computing
- **PyTorch**: A popular deep learning framework
- **CUDA**: NVIDIA's platform for GPU computing (ROCm provides compatibility)
- **Tensor**: A multi-dimensional array used in deep learning
- **Batch Size**: Number of samples processed before model update
- **Epoch**: One complete pass through the entire training dataset
- **Inference**: Using a trained model to make predictions
- **Quantization**: Reducing precision of model weights to save memory

## Learning Resources

### Beginner Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [3Blue1Brown Neural Network Videos](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

### Intermediate Resources

- [Dive into Deep Learning](https://d2l.ai/)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Stanford CS231n: Convolutional Neural Networks](http://cs231n.Stanford.edu/)

### Advanced Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Papers With Code](https://paperswithcode.com/)
- [Distill.pub](https://distill.pub/)

## Frequently Asked Questions

### General Questions

**Q: Do I need to be a programmer to use this?**  
A: Basic Python knowledge is helpful, but you can start with the examples and learn as you go.

**Q: How much math do I need to know?**  
A: Basic underStanding of algebra and statistics helps, but you can start without it and learn the concepts as needed.

**Q: Can I use this for commercial projects?**  
A: Yes, the stack is open-source and can be used for commercial projects.

### Hardware Questions

**Q: Which AMD GPU is best for machine learning?**  
A: The RX 7900 XTX offers the best performance, but the RX 7800 XT is also excellent and more affordable.

**Q: Can I use multiple GPUs?**  
A: Yes, the stack supports multi-GPU training with RCCL and MPI.

**Q: Do I need a special cooling system?**  
A: Standard GPU cooling is sufficient, but ensure good airflow in your case for long training sessions.

### Software Questions

**Q: Can I use this on Windows?**  
A: The stack is designed for Linux. While some components might work on Windows, we recommend Ubuntu for the best experience.

**Q: How often should I update the stack?**  
A: Check for updates monthly, or whenever you start a new project.

**Q: Can I use this with Jupyter notebooks?**  
A: Yes! Install Jupyter with `pip install jupyter` and run notebooks with `jupyter notebook`.

### Learning Questions

**Q: What should I learn first?**  
A: Start with basic PyTorch tutorials, then try the MNIST example in this guide.

**Q: How long does it take to become proficient?**  
A: You can build basic models within a few days, but becoming proficient might take a few months of regular practice.

**Q: What's the best way to learn?**  
A: Hands-on practice! Start with tutorials, then modify them, then build your own projects.

---

We hope this guide helps you get started with machine learning on AMD GPUs! If you have any questions or need help, check the documentation in the `docs` directory or reach out to the community.

Happy learning!


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

