## Usage Examples

These examples demonstrate how to use the ML stack for common tasks.

### Basic PyTorch Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Initialize WandB
wandb.init(project="amd-gpu-example")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(64 * 16 * 16, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 16 * 16)
        x = self.fc(x)
        return x

# Create model and move to GPU
model = SimpleModel().to(device)

# Log model architecture
wandb.watch(model, log="all")

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create dummy data
batch_size = 64
x = torch.randn(batch_size, 3, 32, 32, device=device)
y = torch.randint(0, 10, (batch_size,), device=device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == y).sum().item()
    accuracy = correct / batch_size
    
    # Log metrics
    wandb.log({
        "epoch": epoch,
        "loss": loss.item(),
        "accuracy": accuracy
    })
    
    # Print progress
    print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

# Save model
torch.save(model.state_dict(), "model.pt")
wandb.save("model.pt")

# Finish WandB run
wandb.finish()
```

### Distributed Training with RCCL and MPI

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    # Setup process group
    setup(rank, world_size)
    
    # Set device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Create model
    model = SimpleModel().to(device)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create optimizer
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    
    # Create dummy data
    batch_size = 64
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(10):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = ddp_model(x)
        loss = criterion(outputs, y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print progress on rank 0
        if rank == 0:
            print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")
    
    # Cleanup
    cleanup()

# Run distributed training
world_size = torch.cuda.device_count()
mp.spawn(train, args=(world_size,), nprocs=world_size)
```

### Quantization with BITSANDBYTES

```python
import torch
import torch.nn as nn
import bitsandbytes as bnb

# Create a standard model
model = nn.Sequential(
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024)
).to("cuda")

# Convert to 8-bit model
model_8bit = nn.Sequential(
    bnb.nn.Linear8bitLt(1024, 1024, has_fp16_weights=False),
    nn.ReLU(),
    bnb.nn.Linear8bitLt(1024, 1024, has_fp16_weights=False),
    nn.ReLU(),
    bnb.nn.Linear8bitLt(1024, 1024, has_fp16_weights=False)
).to("cuda")

# Copy weights from original model to 8-bit model
for i in range(0, len(model), 2):
    model_8bit[i].weight.data = model[i].weight.data
    model_8bit[i].bias.data = model[i].bias.data

# Create input data
x = torch.randn(32, 1024, device="cuda")

# Compare outputs
with torch.no_grad():
    output_fp32 = model(x)
    output_int8 = model_8bit(x)
    
    # Check error
    error = torch.abs(output_fp32 - output_int8).mean()
    print(f"Mean absolute error: {error.item()}")
    
    # Check memory usage
    print(f"FP32 model size: {sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024:.2f} MB")
    print(f"8-bit model size: {sum(p.numel() * (1 if '8bit' in p.__class__.__name__ else 4) for p in model_8bit.parameters()) / 1024 / 1024:.2f} MB")
```

### LLM Inference with vLLM

```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(model="facebook/opt-1.3b")

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# Generate text
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The best programming language is"
]

# Generate completions
outputs = llm.generate(prompts, sampling_params)

# Print results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print()
```

### Custom Kernels with Triton

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    # Block start
    block_start = pid * BLOCK_SIZE
    # Offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle case where the block size doesn't divide the number of elements
    mask = offsets < n_elements
    # Load x and y
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Add x and y
    output = x + y
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def add_vectors(x, y):
    # Check input dimensions
    assert x.shape == y.shape, "Input shapes must match"
    assert x.is_cuda and y.is_cuda, "Inputs must be on GPU"
    
    # Output tensor
    output = torch.empty_like(x)
    
    # Get tensor dimensions
    n_elements = output.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    add_kernel[grid, BLOCK_SIZE](
        x, y, output, n_elements, BLOCK_SIZE
    )
    
    return output

# Test the kernel
x = torch.randn(1024, 1024, device='cuda')
y = torch.randn(1024, 1024, device='cuda')
output = add_vectors(x, y)
print(f"Max error: {torch.max(torch.abs(output - (x + y)))}")
```

### Profiling with PyTorch Profiler

```python
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# Create model
model = nn.Sequential(
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024)
).to("cuda")

# Create input data
x = torch.randn(32, 1024, device="cuda")

# Warm-up
for _ in range(5):
    model(x)

# Profile with PyTorch Profiler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("model_inference"):
        model(x)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export trace
prof.export_chrome_trace("trace.json")
```

### GPU Monitoring with ROCm SMI

```python
from rocm_smi_lib import rsmi
import time

# Initialize ROCm SMI
rsmi.rsmi_init(0)

try:
    # Get number of devices
    num_devices = rsmi.rsmi_num_monitor_devices()
    print(f"Found {num_devices} GPU device(s)")
    
    # Monitor GPUs
    for i in range(5):  # Monitor for 5 iterations
        print(f"\nIteration {i+1}/5")
        
        for device_id in range(num_devices):
            # Get device name
            name = rsmi.rsmi_dev_name_get(device_id)[1]
            print(f"\nGPU {device_id}: {name}")
            
            # Get GPU utilization
            util = rsmi.rsmi_dev_gpu_busy_percent_get(device_id)[1]
            print(f"  Utilization: {util}%")
            
            # Get temperature
            temp = rsmi.rsmi_dev_temp_metric_get(device_id, 0, 0)[1] / 1000.0  # Convert to °C
            print(f"  Temperature: {temp}°C")
            
            # Get memory usage
            mem_info = rsmi.rsmi_dev_memory_usage_get(device_id, 0)
            mem_used = mem_info[1] / (1024 * 1024)  # Convert to MB
            mem_total = mem_info[2] / (1024 * 1024)  # Convert to MB
            print(f"  Memory: {mem_used:.2f}/{mem_total:.2f} MB ({(mem_used/mem_total)*100:.2f}%)")
            
            # Get power consumption
            power = rsmi.rsmi_dev_power_ave_get(device_id)[1] / 1000000.0  # Convert to W
            print(f"  Power: {power:.2f} W")
        
        # Wait before next iteration
        time.sleep(1)

finally:
    # Clean up
    rsmi.rsmi_shut_down()
```


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

