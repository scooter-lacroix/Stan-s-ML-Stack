# RCCL Build Guide for AMD GPUs

## Introduction

This guide provides detailed instructions for building and installing RCCL (ROCm Collective Communication Library) for AMD GPUs. RCCL is a library of multi-GPU collective communication primitives optimized for AMD GPUs, similar to NVIDIA's NCCL.

### Purpose of this Guide

The purpose of this guide is to:

1. Provide step-by-step instructions for building RCCL from source
2. Document the configuration options for optimal performance
3. Offer usage examples for common scenarios
4. Share troubleshooting tips for common issues

### Overview of RCCL

RCCL (ROCm Collective Communication Library) is AMD's implementation of collective communication primitives for multi-GPU training. It provides the following key features:

- **Collective Operations**: AllReduce, Broadcast, Reduce, AllGather, and ReduceScatter
- **Multi-GPU Support**: Efficient communication between multiple GPUs
- **Multi-Node Support**: Communication across multiple nodes in a cluster
- **NCCL API Compatibility**: API compatible with NVIDIA's NCCL for easy porting

### Importance for Distributed Training

RCCL is essential for distributed training of deep learning models on AMD GPUs for several reasons:

1. **Efficient Communication**: Optimized for high-bandwidth, low-latency communication between GPUs
2. **Scalability**: Enables scaling to multiple GPUs and nodes
3. **Performance**: Critical for achieving good performance in distributed training
4. **Framework Integration**: Used by PyTorch, TensorFlow, and other frameworks for distributed training

## Prerequisites

Before building RCCL, ensure you have the following:

### Required Hardware

- **AMD GPU**: RX 7900 XTX, RX 7800 XT, or other supported AMD GPU
- **System Memory**: At least 16GB of RAM
- **Storage**: At least 2GB of free disk space

### Required Software

- **ROCm**: Version 6.3/6.4 or later
- **CMake**: Version 3.5 or later
- **GCC**: Version 7.0 or later
- **Python**: Version 3.6 or later (for testing)
- **PyTorch with ROCm**: For integration testing

### Environment Setup

Before proceeding, ensure your environment is properly set up:

1. **ROCm Installation**: Follow the [ROCm Installation Guide](/docs/core/rocm_installation_guide.md)
2. **Environment Variables**:
   ```bash
   # ROCm Path
   export ROCM_PATH=/opt/rocm
   export PATH=$PATH:$ROCM_PATH/bin:$ROCM_PATH/hip/bin
   
   # GPU Selection
   export HIP_VISIBLE_DEVICES=0,1  # Adjust based on your GPU configuration
   ```
## Installation Options

There are two ways to install RCCL:

1. **Install from ROCm Repository**: Easier but may not have the latest version
2. **Build from Source**: More complex but provides the latest version and customization options

### Install from ROCm Repository

The easiest way to install RCCL is from the ROCm repository:

```bash
# Update package lists
sudo apt update

# Install RCCL
sudo apt install rccl
```

Verify the installation:

```bash
# Check if RCCL is installed
ls -la /opt/rocm/lib/librccl*
```

You should see output similar to:

```
-rwxr-xr-x 1 root root 12345678 Jan 1 2023 /opt/rocm/lib/librccl.so
-rwxr-xr-x 1 root root 12345678 Jan 1 2023 /opt/rocm/lib/librccl.so.2
-rwxr-xr-x 1 root root 12345678 Jan 1 2023 /opt/rocm/lib/librccl.so.2.0.0
```

## Building from Source

Building RCCL from source provides the latest features and optimizations.

### Clone the Repository

First, clone the RCCL repository:

```bash
# Create a directory for the build
mkdir -p $HOME/rccl-build
cd $HOME/rccl-build

# Clone the repository
git clone https://github.com/ROCmSoftwarePlatform/rccl.git
cd rccl
```

### Configure the Build

Configure the build with CMake:

```bash
# Create a build directory
mkdir -p build
cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$ROCM_PATH \
      -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/hipcc \
      -DCMAKE_C_COMPILER=$ROCM_PATH/bin/hipcc \
      ..
```

### Compile and Install

Compile and install RCCL:

```bash
# Compile
make -j$(nproc)

# Install (requires sudo)
sudo make install
```

### Verify Installation

Verify that RCCL is installed correctly:

```bash
# Check if RCCL is installed
ls -la $ROCM_PATH/lib/librccl*
```

You should see output similar to:

```
-rwxr-xr-x 1 root root 12345678 Jan 1 2023 /opt/rocm/lib/librccl.so
-rwxr-xr-x 1 root root 12345678 Jan 1 2023 /opt/rocm/lib/librccl.so.2
-rwxr-xr-x 1 root root 12345678 Jan 1 2023 /opt/rocm/lib/librccl.so.2.0.0
```

### Build and Run Tests

Build and run the RCCL tests to verify functionality:

```bash
# Build tests
cd $HOME/rccl-build/rccl/build
make test

# Run tests
cd test
./all_reduce_test
```

You should see output indicating that the tests passed.
## Configuration

Proper configuration is essential for optimal performance of RCCL on AMD GPUs.

### Environment Variables

Set these environment variables for optimal performance:

```bash
# RCCL Debug and Logging
export RCCL_DEBUG=INFO  # Set debug level (INFO, WARN, TRACE)
export RCCL_DEBUG_FILE=/tmp/rccl.log  # Log file for debugging

# Network Configuration
export RCCL_IB_DISABLE=0  # Enable InfiniBand (if available)
export RCCL_SOCKET_IFNAME=eth0  # Network interface to use
export RCCL_TOPO_DUMP_FILE=/tmp/rccl_topo.xml  # Topology dump file

# Performance Tuning
export RCCL_ALLREDUCE_ALGO=ring  # Algorithm for AllReduce (ring, tree)
export RCCL_ALLREDUCE_ELEMENTS_THRESHOLD=16384  # Threshold for algorithm selection
export RCCL_BUFFSIZE=33554432  # Buffer size in bytes
```

### Performance Tuning

Tune RCCL for optimal performance:

1. **Buffer Size**: Adjust `RCCL_BUFFSIZE` based on your workload
   ```bash
   # For large messages
   export RCCL_BUFFSIZE=67108864  # 64MB
   
   # For small messages
   export RCCL_BUFFSIZE=16777216  # 16MB
   ```

2. **Algorithm Selection**: Choose the best algorithm for your workload
   ```bash
   # For large models
   export RCCL_ALLREDUCE_ALGO=ring
   
   # For small models
   export RCCL_ALLREDUCE_ALGO=tree
   ```

3. **Network Interface**: Select the best network interface
   ```bash
   # For InfiniBand
   export RCCL_SOCKET_IFNAME=ib0
   
   # For Ethernet
   export RCCL_SOCKET_IFNAME=eth0
   ```

### Multi-GPU Setup

Configure RCCL for multi-GPU setups:

```bash
# Set visible devices
export HIP_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0, 1, 2, and 3

# Set RCCL to use all visible devices
export RCCL_DEVICES=0,1,2,3
```

## Usage Examples

Here are some examples of how to use RCCL:

### Basic Collective Operations

Here's a simple example of using RCCL for AllReduce:

```cpp
#include <stdio.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

int main(int argc, char* argv[]) {
    // Initialize HIP
    hipSetDevice(0);
    
    // Allocate memory
    float *sendbuff, *recvbuff;
    hipMalloc(&sendbuff, 1024 * sizeof(float));
    hipMalloc(&recvbuff, 1024 * sizeof(float));
    
    // Initialize data
    float *hostbuff = (float*)malloc(1024 * sizeof(float));
    for (int i = 0; i < 1024; i++) {
        hostbuff[i] = i;
    }
    hipMemcpy(sendbuff, hostbuff, 1024 * sizeof(float), hipMemcpyHostToDevice);
    
    // Create RCCL communicator
    ncclComm_t comm;
    ncclCommInitRank(&comm, 1, NULL, 0);
    
    // Perform AllReduce
    ncclAllReduce(sendbuff, recvbuff, 1024, ncclFloat, ncclSum, comm, 0);
    
    // Copy result back to host
    hipMemcpy(hostbuff, recvbuff, 1024 * sizeof(float), hipMemcpyDeviceToHost);
    
    // Print result
    printf("Result: %f\n", hostbuff[0]);
    
    // Clean up
    free(hostbuff);
    hipFree(sendbuff);
    hipFree(recvbuff);
    ncclCommDestroy(comm);
    
    return 0;
}
```

Compile and run:

```bash
hipcc -o allreduce_example allreduce_example.cpp -lrccl
./allreduce_example
```
### Integration with PyTorch

Here's an example of using RCCL with PyTorch for distributed training:

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net = nn.Linear(10, 10)
        
    def forward(self, x):
        return self.net(x)

def demo_basic(rank, world_size):
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create optimizer
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    # Forward pass
    loss_fn = nn.MSELoss()
    inputs = torch.randn(20, 10).to(rank)
    labels = torch.randn(20, 10).to(rank)
    
    # Training loop
    for _ in range(10):
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"Rank {rank}, Loss: {loss.item()}")
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(demo_basic, args=(world_size,), nprocs=world_size)
```

Run the example:

```bash
python pytorch_rccl_example.py
```

### Multi-Node Setup

For multi-node training, you need to set up the environment variables correctly:

```bash
# On the master node
export MASTER_ADDR=<master-node-ip>
export MASTER_PORT=12355
export WORLD_SIZE=<total-number-of-gpus>
export RANK=<node-rank>
export LOCAL_RANK=<local-gpu-rank>

# On worker nodes
export MASTER_ADDR=<master-node-ip>
export MASTER_PORT=12355
export WORLD_SIZE=<total-number-of-gpus>
export RANK=<node-rank>
export LOCAL_RANK=<local-gpu-rank>
```

Then run your distributed training script:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=<gpus-per-node> \
    --nnodes=<number-of-nodes> \
    --node_rank=<node-rank> \
    --master_addr=<master-node-ip> \
    --master_port=12355 \
    your_training_script.py
```

## Performance Optimization

Optimizing RCCL performance is crucial for efficient distributed training.

### Bandwidth Optimization

1. **Buffer Size**: Adjust buffer size for optimal bandwidth
   ```bash
   export RCCL_BUFFSIZE=67108864  # 64MB for high bandwidth
   ```

2. **Algorithm Selection**: Choose the right algorithm for your workload
   ```bash
   export RCCL_ALLREDUCE_ALGO=ring  # Ring algorithm for high bandwidth
   ```

3. **Network Interface**: Use the fastest available network interface
   ```bash
   export RCCL_SOCKET_IFNAME=ib0  # Use InfiniBand for high bandwidth
   ```

### Latency Optimization

1. **Small Message Optimization**: Optimize for small messages
   ```bash
   export RCCL_ALLREDUCE_ALGO=tree  # Tree algorithm for low latency
   ```

2. **Threshold Adjustment**: Adjust thresholds for algorithm selection
   ```bash
   export RCCL_ALLREDUCE_ELEMENTS_THRESHOLD=8192  # Lower threshold for small messages
   ```

3. **Direct Communication**: Enable direct communication for low latency
   ```bash
   export RCCL_DIRECT=1  # Enable direct communication
   ```

### Memory Optimization

1. **Buffer Management**: Optimize buffer management
   ```bash
   export RCCL_BUFFSIZE=16777216  # 16MB for lower memory usage
   ```

2. **Shared Memory**: Use shared memory for intra-node communication
   ```bash
   export RCCL_SHM_DISABLE=0  # Enable shared memory
   ```

3. **Memory Pool**: Configure memory pool for efficient allocation
   ```bash
   export RCCL_MEMPOOL_DISABLE=0  # Enable memory pool
   ```
## Troubleshooting

### Common Issues

1. **Library Not Found**:
   ```
   error while loading shared libraries: librccl.so: cannot open shared object file: No such file or directory
   ```
   
   Solutions:
   - Check if RCCL is installed: `ls -la $ROCM_PATH/lib/librccl*`
   - Add RCCL to library path: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROCM_PATH/lib`
   - Reinstall RCCL: `sudo apt install --reinstall rccl`

2. **Initialization Failure**:
   ```
   NCCL WARN Bootstrap : no socket interface found
   NCCL WARN Bootstrap : using internal network for interface lo
   ```
   
   Solutions:
   - Set network interface: `export RCCL_SOCKET_IFNAME=eth0`
   - Check network configuration: `ifconfig`
   - Verify firewall settings: `sudo ufw status`

3. **Performance Issues**:
   ```
   NCCL WARN Trees/rings/graphs are incompatible
   ```
   
   Solutions:
   - Check GPU topology: `rocm-smi --showtoponuma`
   - Optimize environment variables: `export RCCL_ALLREDUCE_ALGO=ring`
   - Use debug mode to identify bottlenecks: `export RCCL_DEBUG=INFO`

4. **Multi-Node Issues**:
   ```
   NCCL WARN Connect to rank X failed
   ```
   
   Solutions:
   - Check network connectivity: `ping <other-node-ip>`
   - Verify firewall settings: `sudo ufw status`
   - Set correct environment variables: `export MASTER_ADDR=<master-node-ip>`

### Debugging Tips

1. **Enable Debug Logging**:
   ```bash
   export RCCL_DEBUG=INFO
   export RCCL_DEBUG_FILE=/tmp/rccl.log
   ```

2. **Check Topology**:
   ```bash
   rocm-smi --showtoponuma
   ```

3. **Test Bandwidth**:
   ```bash
   cd $HOME/rccl-build/rccl/build/test
   ./all_reduce_perf -b 8 -e 128M -f 2 -g 2
   ```

4. **Check Network**:
   ```bash
   ifconfig
   ping <other-node-ip>
   ```

## References

### Documentation Links

- [RCCL GitHub Repository](https://github.com/ROCmSoftwarePlatform/rccl)
- [RCCL API Documentation](https://github.com/ROCmSoftwarePlatform/rccl/blob/develop/docs/API.md)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)

### Community Resources

- [ROCm GitHub Issues](https://github.com/RadeonOpenCompute/ROCm/issues)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [AMD Developer Forums](https://community.amd.com/t5/AMD-ROCm/bd-p/amd-rocm)

### Papers and Articles

- [NCCL: Optimized Primitives for Collective Multi-GPU Communication](https://arxiv.org/abs/2006.02327)
- [Scaling Distributed Training with RCCL](https://developer.amd.com/blog/scaling-distributed-training-with-rccl/)
- [Optimizing Multi-GPU Communication with RCCL](https://developer.amd.com/blog/optimizing-multi-gpu-communication-with-rccl/)

## Conclusion

RCCL is a critical component for distributed training on AMD GPUs. By building from source and optimizing the configuration, you can achieve optimal performance for your distributed training workloads.

The key points to remember are:

1. **Build from Source**: For the latest features and optimizations
2. **Configure Properly**: Set the right environment variables for your workload
3. **Optimize Performance**: Tune buffer sizes, algorithms, and network settings
4. **Debug Effectively**: Use debug logging and performance tools to identify issues

With these optimizations, you can efficiently scale your distributed training to multiple GPUs and nodes.


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

