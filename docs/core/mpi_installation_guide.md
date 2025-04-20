# MPI Installation Guide for AMD GPUs

## Introduction

This guide provides detailed instructions for installing and configuring MPI (Message Passing Interface) for use with AMD GPUs. MPI is a standardized and portable message-passing system designed for parallel computing, essential for distributed training of machine learning models across multiple nodes.

### Purpose of this Guide

The purpose of this guide is to:

1. Provide step-by-step instructions for installing MPI with ROCm support
2. Document the configuration options for optimal performance
3. Offer usage examples for common scenarios
4. Share troubleshooting tips for common issues

### Overview of MPI

MPI (Message Passing Interface) is a standardized and portable message-passing system designed for parallel computing. Key features include:

- **Standardized API**: Well-defined interface for message passing
- **Portability**: Works across different hardware and operating systems
- **Scalability**: Scales from a few processes to thousands
- **Performance**: Optimized for high-performance computing
- **Flexibility**: Supports various communication patterns

The most common implementations of MPI are:

- **OpenMPI**: Open-source implementation with broad hardware support
- **MPICH**: Another popular open-source implementation
- **Intel MPI**: Optimized for Intel hardware
- **Microsoft MPI**: Windows-specific implementation

For AMD GPUs, OpenMPI is the recommended implementation due to its good support for ROCm.

### Importance for Distributed Training

MPI is essential for distributed training of machine learning models for several reasons:

1. **Process Communication**: Enables communication between processes across multiple nodes
2. **Data Parallelism**: Facilitates data-parallel training across multiple GPUs and nodes
3. **Model Parallelism**: Supports model-parallel training for large models
4. **Scalability**: Allows scaling to hundreds or thousands of GPUs
5. **Integration**: Works well with other distributed training libraries like Horovod

## Prerequisites

Before installing MPI, ensure you have the following:

### Required Hardware

- **AMD GPU**: RX 7900 XTX, RX 7800 XT, or other supported AMD GPU
- **System Memory**: At least 16GB of RAM
- **Network**: High-speed network connection (preferably InfiniBand or high-speed Ethernet)

### Required Software

- **ROCm**: Version 6.3/6.4 or later
- **GCC**: Version 7.0 or later
- **Python**: Version 3.6 or later (for integration with PyTorch)
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

There are two ways to install MPI:

1. **Install from Package Manager**: Easier but may not have optimal ROCm support
2. **Build from Source**: More complex but provides better ROCm integration and customization options

### Install from Package Manager

The easiest way to install OpenMPI is from the package manager:

```bash
# Update package lists
sudo apt update

# Install OpenMPI
sudo apt install openmpi-bin libopenmpi-dev
```

Verify the installation:

```bash
# Check if OpenMPI is installed
mpirun --version
```

You should see output similar to:

```
mpirun (Open MPI) 4.1.2
```

## Installing OpenMPI

### Install from Package Manager

As mentioned above, you can install OpenMPI from the package manager:

```bash
sudo apt install openmpi-bin libopenmpi-dev
```

### Configure with ROCm Support

To ensure OpenMPI works well with ROCm, you may need to set some environment variables:

```bash
# Set ROCm path for OpenMPI
export OMPI_MCA_opal_cuda_support=true
export OMPI_MCA_pml_ucx_opal_cuda_support=true
```

Add these to your `~/.bashrc` file for permanent effect:

```bash
echo 'export OMPI_MCA_opal_cuda_support=true' >> ~/.bashrc
echo 'export OMPI_MCA_pml_ucx_opal_cuda_support=true' >> ~/.bashrc
source ~/.bashrc
```

### Build from Source (Optional)

For better ROCm integration, you can build OpenMPI from source:

```bash
# Install dependencies
sudo apt install build-essential gfortran

# Download OpenMPI
cd $HOME
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
tar -xzf openmpi-4.1.5.tar.gz
cd openmpi-4.1.5

# Configure with ROCm support
./configure --prefix=/opt/openmpi \
            --with-cuda=$ROCM_PATH \
            --enable-mpi-cxx \
            --enable-mpi-fortran \
            --enable-orterun-prefix-by-default \
            --enable-mca-no-build=btl-uct

# Build and install
make -j$(nproc)
sudo make install

# Update environment variables
echo 'export PATH=$PATH:/opt/openmpi/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/openmpi/lib' >> ~/.bashrc
source ~/.bashrc
```

### Verify Installation

Verify that OpenMPI is installed correctly:

```bash
# Check OpenMPI version
mpirun --version

# Run a simple MPI program
cat > mpi_hello.c << 'EOF'
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    printf("Hello from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);

    MPI_Finalize();
    return 0;
}
EOF

# Compile and run
mpicc -o mpi_hello mpi_hello.c
mpirun -np 4 ./mpi_hello
```

You should see output from 4 MPI processes.
## Configuration

Proper configuration is essential for optimal performance of MPI with AMD GPUs.

### Environment Variables

Set these environment variables for optimal performance:

```bash
# OpenMPI Configuration
export OMPI_MCA_opal_cuda_support=true  # Enable CUDA (ROCm) support
export OMPI_MCA_pml_ucx_opal_cuda_support=true  # Enable UCX CUDA support
export OMPI_MCA_btl_openib_allow_ib=true  # Enable InfiniBand support
export OMPI_MCA_btl_openib_warn_no_device_params_found=0  # Suppress warnings

# Performance Tuning
export OMPI_MCA_coll_hcoll_enable=0  # Disable HCOLL for compatibility
export OMPI_MCA_pml=ucx  # Use UCX for point-to-point communication
export OMPI_MCA_osc=ucx  # Use UCX for one-sided communication
export OMPI_MCA_btl=^openib,uct  # Exclude OpenIB and UCT BTLs
```

### Performance Tuning

Tune OpenMPI for optimal performance:

1. **Communication Layer**: Choose the best communication layer
   ```bash
   # Use UCX for best performance
   export OMPI_MCA_pml=ucx
   export OMPI_MCA_osc=ucx
   ```

2. **Collective Operations**: Optimize collective operations
   ```bash
   # Use tuned collective operations
   export OMPI_MCA_coll=tuned
   ```

3. **Buffer Size**: Adjust buffer size for optimal performance
   ```bash
   # Set buffer size
   export OMPI_MCA_btl_sm_eager_limit=8192
   export OMPI_MCA_btl_openib_eager_limit=8192
   ```

### Multi-GPU Setup

Configure OpenMPI for multi-GPU setups:

```bash
# Set visible devices
export HIP_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0, 1, 2, and 3

# Map processes to GPUs
mpirun -np 4 --map-by ppr:1:gpu ./your_program
```

## Usage Examples

Here are some examples of how to use MPI with AMD GPUs:

### Basic MPI Operations

Here's a simple example of using MPI for distributed computation:

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    // Print a hello message
    printf("Hello from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);
    
    // Perform a collective operation (sum)
    int local_value = world_rank + 1;  // Each process has a different value
    int global_sum;
    
    MPI_Reduce(&local_value, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Process 0 prints the result
    if (world_rank == 0) {
        printf("Sum of all values: %d\n", global_sum);
    }
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}
```

Compile and run:

```bash
mpicc -o mpi_example mpi_example.c
mpirun -np 4 ./mpi_example
```
### Integration with PyTorch

Here's an example of using MPI with PyTorch for distributed training:

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from mpi4py import MPI

def setup_mpi():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    # Set PyTorch environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    return rank, world_size, comm

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net = nn.Linear(10, 10)
        
    def forward(self, x):
        return self.net(x)

def main():
    # Setup MPI
    rank, world_size, comm = setup_mpi()
    
    # Set device
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    
    # Create model and move to GPU
    model = ToyModel().to(device)
    ddp_model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    
    # Create optimizer
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    # Forward pass
    loss_fn = nn.MSELoss()
    inputs = torch.randn(20, 10).to(device)
    labels = torch.randn(20, 10).to(device)
    
    # Training loop
    for _ in range(10):
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"Rank {rank}, Loss: {loss.item()}")
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Save this as `pytorch_mpi_example.py` and run:

```bash
mpirun -np 4 python pytorch_mpi_example.py
```

### Multi-Node Setup

For multi-node training, you need to set up the environment variables correctly:

```bash
# On the master node
export MASTER_ADDR=<master-node-ip>
export MASTER_PORT=12355

# On all nodes
mpirun -np <processes-per-node> \
       -H <host1>,<host2>,<host3>,<host4> \
       -bind-to none -map-by slot \
       -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
       -x MASTER_ADDR -x MASTER_PORT \
       python pytorch_mpi_example.py
```

For example, with 4 nodes and 4 GPUs per node:

```bash
mpirun -np 4 \
       -H node1:4,node2:4,node3:4,node4:4 \
       -bind-to none -map-by slot \
       -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
       -x MASTER_ADDR=node1 -x MASTER_PORT=12355 \
       python pytorch_mpi_example.py
```

## Performance Optimization

Optimizing MPI performance is crucial for efficient distributed training.

### Process Placement

1. **GPU Affinity**: Bind processes to GPUs
   ```bash
   mpirun -np 4 --map-by ppr:1:gpu ./your_program
   ```

2. **NUMA Awareness**: Consider NUMA topology
   ```bash
   mpirun -np 4 --map-by ppr:1:gpu --bind-to core ./your_program
   ```

3. **Process Distribution**: Distribute processes across nodes
   ```bash
   mpirun -np 8 -H node1:4,node2:4 --map-by ppr:4:node ./your_program
   ```

### Communication Optimization

1. **Communication Backend**: Choose the best backend
   ```bash
   export OMPI_MCA_pml=ucx  # Use UCX for point-to-point communication
   ```

2. **Network Interface**: Select the best network interface
   ```bash
   export OMPI_MCA_btl_tcp_if_include=eth0  # Use eth0 for TCP communication
   ```

3. **Collective Operations**: Optimize collective operations
   ```bash
   export OMPI_MCA_coll=tuned  # Use tuned collective operations
   ```

### Memory Optimization

1. **Shared Memory**: Use shared memory for intra-node communication
   ```bash
   export OMPI_MCA_btl_sm_use_knem=1  # Use KNEM for shared memory
   ```

2. **Buffer Size**: Adjust buffer size for optimal performance
   ```bash
   export OMPI_MCA_btl_sm_eager_limit=8192  # Set shared memory eager limit
   ```

3. **Memory Binding**: Bind memory to local NUMA node
   ```bash
   mpirun -np 4 --map-by ppr:1:gpu --bind-to core --mca hwloc_base_binding_policy numa ./your_program
   ```
## Troubleshooting

### Common Issues

1. **MPI Initialization Failure**:
   ```
   MPI_Init: Error: Other MPI error
   ```
   
   Solutions:
   - Check OpenMPI installation: `mpirun --version`
   - Verify environment variables: `env | grep OMPI`
   - Check network configuration: `ifconfig`

2. **Process Binding Issues**:
   ```
   Error: Error: could not find a valid binding for process
   ```
   
   Solutions:
   - Use simpler binding: `--bind-to none`
   - Check available resources: `hwloc-ls`
   - Verify GPU visibility: `rocm-smi`

3. **Communication Errors**:
   ```
   MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD
   ```
   
   Solutions:
   - Check network connectivity: `ping <other-node-ip>`
   - Verify firewall settings: `sudo ufw status`
   - Use verbose output: `mpirun --verbose`

4. **Performance Issues**:
   ```
   Slow communication between processes
   ```
   
   Solutions:
   - Check network performance: `iperf3 -s` and `iperf3 -c <server-ip>`
   - Optimize MCA parameters: `export OMPI_MCA_pml=ucx`
   - Use performance tools: `mpirun -np 4 --map-by ppr:1:gpu --report-bindings ./your_program`

### Debugging Tips

1. **Enable Verbose Output**:
   ```bash
   mpirun --verbose -np 4 ./your_program
   ```

2. **Check Process Binding**:
   ```bash
   mpirun -np 4 --map-by ppr:1:gpu --report-bindings ./your_program
   ```

3. **Debug with GDB**:
   ```bash
   mpirun -np 4 xterm -e gdb -ex run --args ./your_program
   ```

4. **Check MCA Parameters**:
   ```bash
   ompi_info --all
   ```

## References

### Documentation Links

- [OpenMPI Documentation](https://www.open-mpi.org/doc/)
- [MPI Standard](https://www.mpi-forum.org/docs/)
- [mpi4py Documentation](https://mpi4py.readthedocs.io/)
- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)

### Community Resources

- [OpenMPI Mailing Lists](https://www.open-mpi.org/community/lists/)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [AMD Developer Forums](https://community.amd.com/t5/AMD-ROCm/bd-p/amd-rocm)

### Papers and Articles

- [Efficient Distributed Training with MPI](https://arxiv.org/abs/1811.02084)
- [Scaling Deep Learning on Multiple GPUs](https://arxiv.org/abs/1810.08313)
- [Performance Analysis of MPI on AMD GPUs](https://www.open-mpi.org/papers/sc-2019/sc19-amd-gpus.pdf)

## Conclusion

MPI is a critical component for distributed training on AMD GPUs. By installing and configuring OpenMPI with ROCm support, you can efficiently scale your distributed training to multiple GPUs and nodes.

The key points to remember are:

1. **Install OpenMPI**: Either from package manager or build from source with ROCm support
2. **Configure Properly**: Set the right environment variables for your workload
3. **Optimize Performance**: Tune process placement, communication, and memory settings
4. **Debug Effectively**: Use verbose output and debugging tools to identify issues

With these optimizations, you can efficiently scale your distributed training to multiple GPUs and nodes, leveraging the full power of your AMD GPU cluster.


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

