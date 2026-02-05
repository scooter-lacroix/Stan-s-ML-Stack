#!/bin/bash
# Validation script for ML Stack component installers

scripts=(
    "./scripts/install_rocm.sh"
    "./scripts/install_pytorch_multi.sh"
    "./scripts/install_vllm_multi.sh"
    "./scripts/install_bitsandbytes_multi.sh"
    "./scripts/install_triton_multi.sh"
    "./scripts/install_migraphx_multi.sh"
)

echo "=== Starting Dry-Run Validation of All Core Scripts ==="

for script in "${scripts[@]}"; do
    echo "--------------------------------------------------------"
    echo "Validating $script..."
    if $script --dry-run > /dev/null 2>&1; then
        echo "✓ $script PASSED dry-run validation"
    else
        echo "✗ $script FAILED dry-run validation"
        # Run again without redirection to show error
        $script --dry-run
    fi
done

echo "--------------------------------------------------------"
echo "=== Validation Complete ==="
