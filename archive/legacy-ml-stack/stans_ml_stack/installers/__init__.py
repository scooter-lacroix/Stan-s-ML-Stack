"""
Installation modules for Stan's ML Stack components.

This module contains Python implementations of installation scripts for various
ML components optimized for AMD GPUs.
"""

from . import rocm
from . import pytorch
from . import flash_attention
from . import megatron
from . import triton
from . import vllm
from . import wandb

__all__ = ["rocm", "pytorch", "flash_attention", "megatron", "triton", "vllm", "wandb"]