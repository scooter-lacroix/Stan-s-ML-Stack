"""
Stan's ML Stack - A comprehensive machine learning environment optimized for AMD GPUs.

This package provides tools and utilities for setting up and managing ML environments
on AMD hardware, including installation scripts, core ML components, and CLI tools.
"""

__version__ = "0.1.3"
__author__ = "Stanley Chisango (Scooter Lacroix)"
__email__ = "scooterlacroix@gmail.com"

from . import installers
from . import core
from . import utils
from . import cli

__all__ = ["installers", "core", "utils", "cli"]