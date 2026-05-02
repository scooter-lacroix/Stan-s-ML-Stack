"""
Core ML components for Stan's ML Stack.

This module contains core machine learning functionality including
distributed training utilities, GPU optimizations, and ML framework adaptations.
"""

from . import pytorch
from . import extensions

__all__ = ["pytorch", "extensions"]