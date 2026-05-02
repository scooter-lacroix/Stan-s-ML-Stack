#!/bin/bash
#
# Dependency Conflict Detection Test Script
# Tests for dependency conflicts between ML Stack components
#

# Check if terminal supports colors
if [ -t 1 ]; then
    if [ -z "$NO_COLOR" ]; then
        RED='\033[0;31m'
        GREEN='\033[0;32m'
        YELLOW='\033[0;33m'
        BLUE='\033[0;34m'
        MAGENTA='\033[0;35m'
        CYAN='\033[0;36m'
        BOLD='\033[1m'
        RESET='\033[0m'
    fi
fi

print_header() {
    echo
    echo -e "${CYAN}${BOLD}╔═════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${CYAN}${BOLD}║                                                         ║${RESET}"
    echo -e "${CYAN}${BOLD}║        Dependency Conflict Detection Test             ║${RESET}"
    echo -e "${CYAN}${BOLD}║                                                         ║${RESET}"
    echo -e "${CYAN}${BOLD}╚═════════════════════════════════════════════════════════╝${RESET}"
    echo
}

print_section() {
    echo
    echo -e "${BLUE}${BOLD}┌─────────────────────────────────────────────────────────┐${RESET}"
    echo -e "${BLUE}${BOLD}│ $1${RESET}"
    echo -e "${BLUE}${BOLD}└─────────────────────────────────────────────────────────┘${RESET}"
}

print_success() {
    echo -e "${GREEN}✓ $1${RESET}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${RESET}"
}

print_error() {
    echo -e "${RED}✗ $1${RESET}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${RESET}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get package version
get_package_version() {
    python3 -c "import $1; print(getattr($1, '__version__', 'Unknown'))" 2>/dev/null || echo "Not installed"
}

# Function to check if package is installed
package_installed() {
    python3 -c "import $1" &>/dev/null
}

# Function to analyze requirements files
analyze_requirements() {
    print_info "Analyzing requirements files..."

    local req_files=("requirements.txt" "../requirements.txt")

    for req_file in "${req_files[@]}"; do
        if [ -f "$req_file" ]; then
            print_info "Found requirements file: $req_file"
            local package_count=$(grep -c "^[a-zA-Z]" "$req_file" || echo "0")
            print_info "  - Contains $package_count package specifications"

            # Check for version conflicts in requirements
            local conflicts=$(grep -E "^[a-zA-Z].*[<>=].*[<>]" "$req_file" | wc -l)
            if [ "$conflicts" -gt 0 ]; then
                print_warning "  - Found $conflicts potential version conflicts in $req_file"
            fi
        fi
    done
}

# Function to check core ML component dependencies
check_core_dependencies() {
    print_section "Core ML Component Dependencies"

    local components=("torch" "triton" "vllm" "onnxruntime" "transformers" "numpy" "scipy")

    for component in "${components[@]}"; do
        if package_installed "$component"; then
            local version=$(get_package_version "$component")
            print_success "$component is installed (version: $version)"
        else
            print_warning "$component is not installed"
        fi
    done
}

# Function to check for version conflicts between components
check_version_conflicts() {
    print_section "Version Conflict Detection"

    print_info "Checking for common dependency conflicts..."

    # Check PyTorch and related packages
    if package_installed "torch" && package_installed "torchvision"; then
        local torch_version=$(get_package_version "torch")
        local torchvision_version=$(get_package_version "torchvision")

        # Check if versions are compatible (should have same base version)
        local torch_base=$(echo "$torch_version" | cut -d'+' -f1 | cut -d'.' -f1-2)
        local torchvision_base=$(echo "$torchvision_version" | cut -d'+' -f1 | cut -d'.' -f1-2)

        if [ "$torch_base" != "$torchvision_base" ]; then
            print_warning "PyTorch version mismatch: torch=$torch_version, torchvision=$torchvision_version"
        else
            print_success "PyTorch and torchvision versions are compatible"
        fi
    fi

    # Check transformers and tokenizers compatibility
    if package_installed "transformers" && package_installed "tokenizers"; then
        local transformers_version=$(get_package_version "transformers")
        local tokenizers_version=$(get_package_version "tokenizers")

        print_info "Transformers: $transformers_version, Tokenizers: $tokenizers_version"

        # Basic compatibility check (transformers 4.x should work with tokenizers 0.1x)
        local transformers_major=$(echo "$transformers_version" | cut -d'.' -f1)
        local tokenizers_major=$(echo "$tokenizers_version" | cut -d'.' -f1)

        if [ "$transformers_major" -ge 4 ] && [ "$tokenizers_major" -ge 0 ]; then
            print_success "Transformers and tokenizers versions are compatible"
        else
            print_warning "Potential compatibility issue between transformers and tokenizers"
        fi
    fi

    # Check numpy version compatibility
    if package_installed "numpy"; then
        local numpy_version=$(get_package_version "numpy")
        local numpy_major=$(echo "$numpy_version" | cut -d'.' -f1)
        local numpy_minor=$(echo "$numpy_version" | cut -d'.' -f2)

        if [ "$numpy_major" -eq 1 ] && [ "$numpy_minor" -ge 24 ]; then
            print_success "NumPy version ($numpy_version) is compatible with modern ML packages"
        elif [ "$numpy_major" -eq 2 ]; then
            print_success "NumPy 2.x ($numpy_version) is compatible"
        else
            print_warning "NumPy version ($numpy_version) may be outdated for some ML packages"
        fi
    fi
}

# Function to check import conflicts
check_import_conflicts() {
    print_section "Import Conflict Detection"

    print_info "Testing component imports for conflicts..."

    # Create test script for import testing
    local test_script=$(mktemp)
    cat > "$test_script" << 'EOF'
import sys
import importlib

def test_import(component_name, import_name=None):
    """Test importing a component and return success/failure."""
    if import_name is None:
        import_name = component_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        return f"SUCCESS:{version}"
    except ImportError as e:
        return f"IMPORT_ERROR:{str(e)}"
    except Exception as e:
        return f"ERROR:{str(e)}"

# Test core components
components = [
    ('torch', 'torch'),
    ('triton', 'triton'),
    ('vllm', 'vllm'),
    ('onnxruntime', 'onnxruntime'),
    ('transformers', 'transformers'),
    ('numpy', 'numpy'),
    ('scipy', 'scipy'),
    ('PIL', 'PIL'),
    ('matplotlib', 'matplotlib'),
    ('pandas', 'pandas'),
    ('sklearn', 'sklearn'),
    ('tensorboard', 'tensorboard'),
]

results = {}
for name, import_name in components:
    result = test_import(name, import_name)
    results[name] = result
    print(f"{name}:{result}")
EOF

    local output=$("$test_script" 2>&1)
    rm -f "$test_script"

    # Parse results
    local import_errors=0
    local import_warnings=0

    while IFS= read -r line; do
        local component=$(echo "$line" | cut -d: -f1)
        local status=$(echo "$line" | cut -d: -f2)

        if [[ "$status" == IMPORT_ERROR* ]]; then
            print_error "$component import failed: $(echo "$status" | cut -d: -f2-)"
            import_errors=$((import_errors + 1))
        elif [[ "$status" == ERROR* ]]; then
            print_warning "$component import has issues: $(echo "$status" | cut -d: -f2-)"
            import_warnings=$((import_warnings + 1))
        elif [[ "$status" == SUCCESS* ]]; then
            local version=$(echo "$status" | cut -d: -f2)
            print_success "$component imports successfully (version: $version)"
        fi
    done <<< "$output"

    if [ $import_errors -gt 0 ]; then
        print_error "Found $import_errors import errors"
    elif [ $import_warnings -gt 0 ]; then
        print_warning "Found $import_warnings import warnings"
    else
        print_success "All tested components import successfully"
    fi
}

# Function to check pip dependency tree
check_dependency_tree() {
    print_section "Dependency Tree Analysis"

    if command_exists pip; then
        print_info "Checking pip dependency tree..."

        # Check for pipdeptree
        if ! command_exists pipdeptree; then
            print_info "Installing pipdeptree for dependency analysis..."
            python3 -m pip install pipdeptree >/dev/null 2>&1
        fi

        if command_exists pipdeptree; then
            print_info "Analyzing dependency tree for conflicts..."

            # Get dependency tree and check for conflicts
            local tree_output=$(pipdeptree --warn 2>&1)
            local conflict_count=$(echo "$tree_output" | grep -c "conflict\|incompatible\|error" || echo "0")

            if [ "$conflict_count" -gt 0 ]; then
                print_warning "Found $conflict_count potential dependency conflicts"
                echo "$tree_output" | grep -A 2 -B 2 "conflict\|incompatible\|error" | head -20
            else
                print_success "No dependency conflicts detected in pip tree"
            fi
        else
            print_warning "Could not install pipdeptree, skipping dependency tree analysis"
        fi
    else
        print_warning "pip not available, skipping dependency tree analysis"
    fi
}

# Function to check for CUDA/ROCm conflicts
check_accelerator_conflicts() {
    print_section "Accelerator Compatibility Check"

    print_info "Checking for CUDA/ROCm compatibility issues..."

    # Check if both CUDA and ROCm packages are installed
    local cuda_packages=$(python3 -c "import torch; print('CUDA' if torch.cuda.is_available() else 'No CUDA')" 2>/dev/null || echo "PyTorch not available")
    local rocm_packages=$(python3 -c "import torch; print('ROCm' if hasattr(torch.version, 'hip') and torch.version.hip else 'No ROCm')" 2>/dev/null || echo "PyTorch not available")

    print_info "CUDA support: $cuda_packages"
    print_info "ROCm support: $rocm_packages"

    if [[ "$cuda_packages" == *"CUDA"* ]] && [[ "$rocm_packages" == *"ROCm"* ]]; then
        print_warning "Both CUDA and ROCm support detected - this may cause conflicts"
        print_info "Recommendation: Use ROCm-only PyTorch for AMD GPUs"
    elif [[ "$cuda_packages" == *"CUDA"* ]]; then
        print_info "CUDA-only setup detected"
    elif [[ "$rocm_packages" == *"ROCm"* ]]; then
        print_success "ROCm-only setup detected (optimal for AMD GPUs)"
    else
        print_warning "No GPU acceleration detected"
    fi

    # Check for conflicting environment variables
    if [ -n "$CUDA_VISIBLE_DEVICES" ] && [ -n "$HIP_VISIBLE_DEVICES" ]; then
        if [ "$CUDA_VISIBLE_DEVICES" != "$HIP_VISIBLE_DEVICES" ]; then
            print_warning "CUDA_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES have different values"
            print_info "This may cause GPU device conflicts"
        fi
    fi
}

# Function to check component integration
check_component_integration() {
    print_section "Component Integration Test"

    print_info "Testing integration between ML components..."

    # Test PyTorch + Triton integration
    if package_installed "torch" && package_installed "triton"; then
        print_info "Testing PyTorch + Triton integration..."

        local integration_test=$(python3 -c "
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add_vectors(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    add_kernel[grid, BLOCK_SIZE](x, y, output, n_elements, BLOCK_SIZE)
    return output

try:
    x = torch.randn(1024, device='cuda' if torch.cuda.is_available() else 'cpu')
    y = torch.randn(1024, device='cuda' if torch.cuda.is_available() else 'cpu')
    result = add_vectors(x, y)
    print('SUCCESS: PyTorch + Triton integration works')
except Exception as e:
    print(f'ERROR: PyTorch + Triton integration failed: {e}')
" 2>/dev/null)

        if [[ "$integration_test" == SUCCESS* ]]; then
            print_success "PyTorch + Triton integration works"
        else
            print_warning "PyTorch + Triton integration has issues"
        fi
    fi

    # Test basic component coexistence
    local coexistence_test=$(python3 -c "
try:
    import torch
    import transformers
    import numpy as np
    print('SUCCESS: Core ML components coexist without conflicts')
except ImportError as e:
    print(f'IMPORT_ERROR: {e}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null)

    if [[ "$coexistence_test" == SUCCESS* ]]; then
        print_success "Core ML components can be imported together"
    else
        print_warning "Core ML components have coexistence issues"
    fi
}

# Main test function
test_dependency_conflicts() {
    print_header

    # Analyze requirements files
    analyze_requirements

    # Check core dependencies
    check_core_dependencies

    # Check for version conflicts
    check_version_conflicts

    # Check import conflicts
    check_import_conflicts

    # Check dependency tree
    check_dependency_tree

    # Check accelerator conflicts
    check_accelerator_conflicts

    # Check component integration
    check_component_integration

    print_section "Summary"

    print_success "Dependency conflict analysis completed"
    echo
    print_info "Recommendations for conflict-free ML Stack:"
    echo "  - Use ROCm-compatible PyTorch for AMD GPUs"
    echo "  - Keep transformers and tokenizers versions synchronized"
    echo "  - Use NumPy 1.24+ or 2.x for best compatibility"
    echo "  - Avoid mixing CUDA and ROCm PyTorch installations"
    echo "  - Regularly update pip and use virtual environments"
}

# Run the test
test_dependency_conflicts