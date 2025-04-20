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
install_megatron() {
    print_section "Installing Megatron-LM"
    
    # Check if Megatron-LM is already installed
    if python3 -c "import megatron" &> /dev/null; then
        print_warning "Megatron-LM is already installed."
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping Megatron-LM installation."
            return 0
        fi
    fi
    
    # Clone Megatron-LM repository
    print_step "Cloning Megatron-LM repository..."
    cd $HOME
    if [ -d "Megatron-LM" ]; then
        print_warning "Megatron-LM repository already exists. Updating..."
        cd Megatron-LM
        git pull
        cd ..
    else
        git clone https://github.com/NVIDIA/Megatron-LM.git
        cd Megatron-LM
    fi
    
    # Create patch file to remove NVIDIA-specific dependencies
    print_step "Creating patch file to remove NVIDIA-specific dependencies..."
    cat > remove_nvidia_deps.patch << 'EOF'
diff --git a/megatron/model/fused_softmax.py b/megatron/model/fused_softmax.py
index 7a5b2e5..3e5c2e5 100644
--- a/megatron/model/fused_softmax.py
+++ b/megatron/model/fused_softmax.py
@@ -15,7 +15,7 @@
 """Fused softmax."""
 
 import torch
-import torch.nn.functional as F
+import torch.nn.functional as F  # Use PyTorch's native implementation
 
 
 class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
@@ -24,8 +24,7 @@ class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
     @staticmethod
     def forward(ctx, inputs, scale):
         """Forward pass.
-        Args:
-            inputs (Tensor): input tensor (b, np, sq, sk)
+        Args: inputs (Tensor): input tensor (b, np, sq, sk)
             scale (float): scaling factor
 
         Returns:
@@ -33,10 +32,10 @@ class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
         """
         # Use the native pytorch implementation for compatibility
         import torch.nn.functional as F
-        scale_t = torch.tensor([scale])
+        scale_t = torch.tensor([scale], device=inputs.device)
         ctx.scale_t = scale_t
         input_scaled = inputs * scale_t
-        # Create a mask for the upper triangular part (including the diagonal)
+        # Create a mask for the upper triangular part
         seq_len = inputs.size(2)
         mask = torch.triu(
             torch.ones(seq_len, seq_len, device=inputs.device, dtype=torch.bool),
@@ -59,7 +58,7 @@ class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
         """
         # Use the native pytorch implementation for compatibility
         import torch.nn.functional as F
-        output_grads_scaled = grad_output * ctx.softmax_results
+        output_grads_scaled = grad_output * ctx.softmax_results  # Element-wise multiplication
         input_grads = output_grads_scaled - torch.sum(
             output_grads_scaled * ctx.softmax_results, dim=-1, keepdim=True
         ) * ctx.softmax_results
@@ -77,8 +76,7 @@ class ScaledMaskedSoftmax(torch.autograd.Function):
     @staticmethod
     def forward(ctx, inputs, mask, scale):
         """Forward pass.
-        Args:
-            inputs (Tensor): input tensor (b, np, sq, sk)
+        Args: inputs (Tensor): input tensor (b, np, sq, sk)
             mask (Tensor): attention mask (b, 1, sq, sk)
             scale (float): scaling factor
 
@@ -87,7 +85,7 @@ class ScaledMaskedSoftmax(torch.autograd.Function):
         """
         # Use the native pytorch implementation for compatibility
         import torch.nn.functional as F
-        scale_t = torch.tensor([scale])
+        scale_t = torch.tensor([scale], device=inputs.device)
         ctx.scale_t = scale_t
         input_scaled = inputs * scale_t
         # Apply the mask
@@ -110,7 +108,7 @@ class ScaledMaskedSoftmax(torch.autograd.Function):
         """
         # Use the native pytorch implementation for compatibility
         import torch.nn.functional as F
-        output_grads_scaled = grad_output * ctx.softmax_results
+        output_grads_scaled = grad_output * ctx.softmax_results  # Element-wise multiplication
         input_grads = output_grads_scaled - torch.sum(
             output_grads_scaled * ctx.softmax_results, dim=-1, keepdim=True
         ) * ctx.softmax_results
diff --git a/megatron/training.py b/megatron/training.py
index 9a5b2e5..3e5c2e5 100644
--- a/megatron/training.py
+++ b/megatron/training.py
@@ -30,7 +30,7 @@ import torch
 from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
 
 from megatron import get_args
-from megatron import get_timers
+from megatron import get_timers  # Timing utilities
 from megatron import get_tensorboard_writer
 from megatron import mpu
 from megatron import print_rank_0
@@ -38,7 +38,7 @@ from megatron.checkpointing import load_checkpoint
 from megatron.checkpointing import save_checkpoint
 from megatron.model import DistributedDataParallel as LocalDDP
 from megatron.model import Float16Module
-from megatron.model.realm_model import ICTBertModel
+from megatron.model.realm_model import ICTBertModel  # Import model
 from megatron.utils import check_adlr_autoresume_termination
 from megatron.utils import unwrap_model
 from megatron.data.data_samplers import build_pretraining_data_loader
@@ -46,7 +46,7 @@ from megatron.utils import report_memory
 
 
 def pretrain(train_valid_test_dataset_provider, model_provider,
-             forward_step_func, extra_args_provider=None, args_defaults={}):
+             forward_step_func, extra_args_provider=None, args_defaults=None):
     """Main training program.
 
     This function will run the followings in the order provided:
@@ -59,6 +59,9 @@ def pretrain(train_valid_test_dataset_provider, model_provider,
         5) validation
 
     """
+    if args_defaults is None:
+        args_defaults = {}
+        
     # Initalize and get arguments, timers, and Tensorboard writer.
     initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
diff --git a/requirements.txt b/requirements.txt
index 9a5b2e5..3e5c2e5 100644
--- a/requirements.txt
+++ b/requirements.txt
@@ -1,6 +1,5 @@
 torch>=1.7
 numpy
-apex
 pybind11
 regex
 nltk
EOF
    
    # Apply patch
    print_step "Applying patch..."
    git apply remove_nvidia_deps.patch
    
    # Install dependencies
    print_step "Installing dependencies..."
    pip install -r requirements.txt
    pip install tensorboard scipy
    
    # Install Megatron-LM
    print_step "Installing Megatron-LM..."
    pip install -e .
    
    # Verify installation
    if python3 -c "import megatron; print('Megatron-LM imported successfully')" &> /dev/null; then
        print_success "Megatron-LM installed successfully"
    else
        print_error "Megatron-LM installation failed."
        return 1
    fi
    
    return 0
}

install_flash_attention() {
    print_section "Installing Flash Attention with AMD GPU support"
    
    # Check if Flash Attention is already installed
    if python3 -c "import flash_attention_amd" &> /dev/null; then
        print_warning "Flash Attention with AMD GPU support is already installed."
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping Flash Attention installation."
            return 0
        fi
    fi
    
    # Build and install Flash Attention with AMD GPU support
    print_step "Building Flash Attention with AMD GPU support..."
    
    # Run the build script
    $HOME/Desktop/Stans_MLStack/scripts/build_flash_attn_amd.sh
    
    # Check if installation was successful
    if [ $? -ne 0 ]; then
        print_error "Flash Attention installation failed."
        return 1
    fi
    
    print_success "Flash Attention with AMD GPU support installed successfully"
    return 0
}

install_rccl() {
    print_section "Installing RCCL"
    
    # Check if RCCL is already installed
    if [ -f "/opt/rocm/lib/librccl.so" ]; then
        print_warning "RCCL is already installed."
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping RCCL installation."
            return 0
        fi
    fi
    
    # Install RCCL from ROCm repository
    print_step "Installing RCCL from ROCm repository..."
    sudo apt-get update
    sudo apt-get install -y rccl
    
    # Verify installation
    if [ -f "/opt/rocm/lib/librccl.so" ]; then
        print_success "RCCL installed successfully"
    else
        print_error "RCCL installation failed."
        return 1
    fi
    
    return 0
}

install_mpi() {
    print_section "Installing MPI"
    
    # Check if MPI is already installed
    if command -v mpirun &> /dev/null; then
        mpi_version=$(mpirun --version | head -n 1)
        print_warning "MPI is already installed ($mpi_version)."
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping MPI installation."
            return 0
        fi
    fi
    
    # Install OpenMPI
    print_step "Installing OpenMPI..."
    sudo apt-get update
    sudo apt-get install -y openmpi-bin libopenmpi-dev
    
    # Configure OpenMPI for ROCm
    print_step "Configuring OpenMPI for ROCm..."
    
    # Create MPI configuration file
    cat > $HOME/.mpirc << 'EOF'
# MPI Configuration File
# Created by ML Stack Installation Script

# OpenMPI Configuration
export OMPI_MCA_opal_cuda_support=true
export OMPI_MCA_pml_ucx_opal_cuda_support=true
export OMPI_MCA_btl_openib_allow_ib=true
export OMPI_MCA_btl_openib_warn_no_device_params_found=0

# Performance Tuning
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_btl=^openib,uct
EOF
    
    # Add source to .bashrc if not already there
    if ! grep -q "source \$HOME/.mpirc" $HOME/.bashrc; then
        echo -e "\n# Source MPI configuration" >> $HOME/.bashrc
        echo "source \$HOME/.mpirc" >> $HOME/.bashrc
    fi
    
    # Source the file
    source $HOME/.mpirc
    
    # Install mpi4py
    print_step "Installing mpi4py..."
    pip install mpi4py
    
    # Verify installation
    if command -v mpirun &> /dev/null && python3 -c "import mpi4py; print('mpi4py imported successfully')" &> /dev/null; then
        mpi_version=$(mpirun --version | head -n 1)
        print_success "MPI installed successfully ($mpi_version)"
    else
        print_error "MPI installation failed."
        return 1
    fi
    
    return 0
}
