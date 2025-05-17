#!/usr/bin/env python3

import sys
import os
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")
print(f"Megatron-LM directory exists: {os.path.exists('/home/stan/Megatron-LM')}")
print(f"Megatron-LM in sys.path: {any('Megatron-LM' in p for p in sys.path)}")

print("\nAttempting to import megatron:")
try:
    import megatron
    print("Megatron-LM imported successfully")
    print(f"Megatron-LM version: {megatron.__version__ if hasattr(megatron, '__version__') else 'Unknown'}")
    print(f"Megatron-LM path: {megatron.__file__}")
except Exception as e:
    print(f"Error importing Megatron-LM: {e}")
    import traceback
    traceback.print_exc()

print("\nChecking dependencies:")
dependencies = [
    "torch",
    "numpy",
    "einops",
    "tensorstore",
    "nvidia_modelopt",
    "triton"
]

for dep in dependencies:
    try:
        module = __import__(dep)
        print(f"✓ {dep} imported successfully")
        if hasattr(module, "__version__"):
            print(f"  Version: {module.__version__}")
        elif hasattr(module, "version"):
            print(f"  Version: {module.version}")
        if hasattr(module, "__file__"):
            print(f"  Path: {module.__file__}")
    except ImportError as e:
        print(f"✗ {dep} import failed: {e}")

# Try to manually add Megatron-LM to path and import
print("\nTrying manual import:")
megatron_path = "/home/stan/Megatron-LM"
if megatron_path not in sys.path:
    sys.path.append(megatron_path)
    print(f"Added {megatron_path} to sys.path")

try:
    import megatron
    print("Megatron-LM imported successfully after path addition")
    print(f"Megatron-LM path: {megatron.__file__}")
except Exception as e:
    print(f"Error importing Megatron-LM after path addition: {e}")
    import traceback
    traceback.print_exc()
