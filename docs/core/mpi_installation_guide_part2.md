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


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

