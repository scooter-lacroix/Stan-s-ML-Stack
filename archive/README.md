# ML Stack Archive Directory

This directory contains archived components of the ML Stack project that have been organized according to the scripts analysis report. Files and directories are grouped by their purpose and lifecycle stage.

## Archive Structure

### Part 1: Core Installation (Keep Active)
**Location:** `scripts/` (not archived)
- Core installation scripts that remain actively used
- Files: install_rocm.sh, install_pytorch_rocm.sh, install_ml_stack.sh, install_ml_stack_ui.py

### Part 2: Build Scripts (Archive After Use)
**Location:** `archive/part2/`
- Build scripts used during development, now archived
- Contains scripts for building Flash Attention AMD and ONNX Runtime components

### Part 3: Extension Installers (Keep Recent)
**Location:** `scripts/` (not archived)
- Recent extension installation scripts that remain active
- Files: install_triton.sh, install_vllm.sh, install_flash_attention_ck.sh, install_bitsandbytes.sh, install_rocm_smi.sh, install_pytorch_profiler.sh, install_wandb.sh

### Part 4: Test Suite (Archive After Validation)
**Location:** `archive/part4/`
- Comprehensive test suite used for validation, now archived
- Contains scripts for testing environment consistency, package management, and integration

### Part 5: Utilities (Keep Essential)
**Location:** `scripts/` (not archived)
- Essential utility scripts that remain in active use
- Files: package_manager_utils.sh, ml_stack_component_detector.sh, setup_environment.sh, enhanced_setup_environment.sh, create_persistent_env.sh

### Part 6: Runners and UI (Keep Active)
**Location:** `scripts/` (not archived)
- Active runner scripts and user interfaces
- Files: run_benchmarks.sh, run_tests.sh, run_vllm.sh, run_ml_stack_ui.sh, install_ml_stack_curses.py

### Part 7: Documentation (Archive)
**Location:** `archive/part7/`
- Historical documentation and outline files
- Contains planning documents and process outlines

### Part 8: Archives and Backups (Archive)
**Location:** `archive/part8/`
- Backup files and historical reports
- Contains .bak files and integration test reports

### Part 9: Virtual Environments (Move to Separate Location)
**Location:** `archive/part9/`
- Virtual environment directories (none found to archive)
- Intended for isolating testing environments

## Essential Files Retained in scripts/

The following critical files remain in the `scripts/` directory:

- `verify_installation.sh` - Main verification script
- `install_mpi4py.sh` - MPI4Py installation
- `install_migraphx_python.sh` - MIGraphX Python installation
- `install_megatron.sh` - Megatron installation
- `install_aiter.sh` - AITER installation
- `install_ml_stack_extensions.sh` - Extensions installer
- `check_components.sh` - Component checking
- `comprehensive_rebuild.sh` - Comprehensive rebuild
- `repair_ml_stack.sh` - Repair utilities
- `verify_and_build.sh` - Combined verification and build

## Usage Guidelines

- **Active files** remain in `scripts/` for current use
- **Archived files** are stored in respective `archive/partX/` directories
- Each archive part contains a README.md explaining its contents
- Archived files can be restored if needed for maintenance or debugging
- Virtual environments are archived separately to reduce repository size

## Maintenance

- Review archived content periodically for cleanup
- Update README files when archive contents change
- Consider gitignore patterns for large archived files if needed