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


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

