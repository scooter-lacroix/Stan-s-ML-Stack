#!/bin/bash
#
# Script to check for ML Stack components in specific locations
#

# Source the component detector library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
DETECTOR_SCRIPT="$PARENT_DIR/scripts/ml_stack_component_detector.sh"

if [ -f "$DETECTOR_SCRIPT" ]; then
    source "$DETECTOR_SCRIPT"
else
    echo "Error: Component detector script not found at $DETECTOR_SCRIPT"
    exit 1
fi

# Print header
print_header "ML Stack Component Checker"

# Check all components
check_all_components

# Check environment variables
print_section "Checking Environment Variables"
if [ -n "$HIP_VISIBLE_DEVICES" ]; then
    print_success "HIP_VISIBLE_DEVICES is set: $HIP_VISIBLE_DEVICES"
else
    print_warning "HIP_VISIBLE_DEVICES is not set"
fi

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    print_success "CUDA_VISIBLE_DEVICES is set: $CUDA_VISIBLE_DEVICES"
else
    print_warning "CUDA_VISIBLE_DEVICES is not set"
fi

if [ -n "$PYTORCH_ROCM_DEVICE" ]; then
    print_success "PYTORCH_ROCM_DEVICE is set: $PYTORCH_ROCM_DEVICE"
else
    print_warning "PYTORCH_ROCM_DEVICE is not set"
fi

if [ -n "$ROCM_HOME" ]; then
    print_success "ROCM_HOME is set: $ROCM_HOME"
else
    print_warning "ROCM_HOME is not set"
fi

if [ -n "$CUDA_HOME" ]; then
    print_success "CUDA_HOME is set: $CUDA_HOME"
else
    print_warning "CUDA_HOME is not set"
fi

if [ -n "$HSA_OVERRIDE_GFX_VERSION" ]; then
    print_success "HSA_OVERRIDE_GFX_VERSION is set: $HSA_OVERRIDE_GFX_VERSION"
else
    print_warning "HSA_OVERRIDE_GFX_VERSION is not set"
fi

if [ -n "$HSA_TOOLS_LIB" ]; then
    print_success "HSA_TOOLS_LIB is set: $HSA_TOOLS_LIB"
else
    print_warning "HSA_TOOLS_LIB is not set"
fi

# Check Python path
print_section "Checking Python Path"
$PYTHON_INTERPRETER -c "import sys; print('\n'.join(sys.path))" | grep -E "flash|rccl|megatron|onnx"

# Generate summary
generate_component_summary
