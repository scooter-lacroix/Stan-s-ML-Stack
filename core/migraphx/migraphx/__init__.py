# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
# 
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House

import os
import sys
import importlib.util

# Try to load the MIGraphX module
try:
    # First try to load from the current directory
    from . import migraphx
except ImportError:
    # If that fails, try to load from the ROCm installation
    rocm_lib_path = "/opt/rocm-6.4.0/lib"
    module_path = os.path.join(rocm_lib_path, "migraphx.cpython-312-x86_64-linux-gnu.so")
    
    if os.path.exists(module_path):
        spec = importlib.util.spec_from_file_location("migraphx.migraphx", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["migraphx.migraphx"] = module
        spec.loader.exec_module(module)
        
        # Import all attributes from the module
        from migraphx.migraphx import *
    else:
        raise ImportError(f"Could not find MIGraphX module at {module_path}")

# Define version
__version__ = "2.12.0"
