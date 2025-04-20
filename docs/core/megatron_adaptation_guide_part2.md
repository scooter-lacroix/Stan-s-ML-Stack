## Adaptation Process

The process of adapting Megatron-LM to AMD GPUs involves several steps:

### Forking the Repository

First, fork the Megatron-LM repository to make your own version:

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
```

### Removing NVIDIA-Specific Dependencies

The main challenge is removing or replacing NVIDIA-specific dependencies. Create a patch file to remove these dependencies:

```bash
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
```

### Patching the Code

Apply the patch to remove NVIDIA-specific dependencies:

```bash
git apply remove_nvidia_deps.patch
```

### Additional Modifications

Some additional modifications may be needed:

1. **Replace NCCL with RCCL**: Update the distributed communication code to use RCCL instead of NCCL
2. **Remove CUDA Extensions**: Replace CUDA extensions with PyTorch native implementations
3. **Update Optimizer**: Replace NVIDIA's Apex optimizer with PyTorch native optimizers

### Testing the Adaptation

Test the adaptation to ensure it works with AMD GPUs:

```bash
# Set up environment
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ROCM_DEVICE=0

# Run a simple test
python -c "import torch; import megatron; print('Megatron-LM imported successfully')"
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

