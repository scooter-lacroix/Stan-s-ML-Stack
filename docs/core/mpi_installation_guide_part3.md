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


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

