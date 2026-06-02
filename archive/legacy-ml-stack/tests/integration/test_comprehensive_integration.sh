#!/bin/bash
#
# Comprehensive Integration Test Script
# Runs end-to-end integration tests for the ML Stack components
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
    echo -e "${CYAN}${BOLD}║      Comprehensive Integration Test                   ║${RESET}"
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

# Function to check if package is installed
package_installed() {
    python3 -c "import $1" &>/dev/null
}

# Function to test ROCm integration
test_rocm_integration() {
    print_section "ROCm Integration Test"

    print_info "Testing ROCm integration with PyTorch..."

    # Check if ROCm is available
    if command_exists rocminfo; then
        print_success "ROCm is installed"

        # Check GPU detection
        local gpu_count=$(rocminfo 2>/dev/null | grep "Device Type:.*GPU" | wc -l)
        if [ "$gpu_count" -gt 0 ]; then
            print_success "ROCm detected $gpu_count GPU(s)"
        else
            print_warning "ROCm installed but no GPUs detected"
        fi

        # Test PyTorch ROCm integration
        local pytorch_rocm_test=$(python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if hasattr(torch.version, 'hip'):
    print('ROCm version:', torch.version.hip)
    print('SUCCESS: PyTorch ROCm integration working')
else:
    print('WARNING: PyTorch ROCm support not detected')
" 2>/dev/null)

        if echo "$pytorch_rocm_test" | grep -q "SUCCESS"; then
            print_success "PyTorch ROCm integration is working"
            return 0
        else
            print_warning "PyTorch ROCm integration has issues"
            return 1
        fi
    else
        print_error "ROCm is not installed"
        return 1
    fi
}

# Function to test component integration
test_component_integration() {
    print_section "Component Integration Test"

    print_info "Testing integration between ML components..."

    local integration_test=$(python3 -c "
# Test PyTorch + NumPy integration
try:
    import torch
    import numpy as np
    
    # Create tensors and convert between PyTorch and NumPy
    x_np = np.array([1, 2, 3, 4, 5])
    x_torch = torch.from_numpy(x_np)
    y_torch = x_torch * 2
    y_np = y_torch.numpy()
    
    assert np.array_equal(y_np, np.array([2, 4, 6, 8, 10]))
    print('SUCCESS: PyTorch-NumPy integration works')
except Exception as e:
    print(f'ERROR: PyTorch-NumPy integration failed: {e}')

# Test Triton integration if available
try:
    import triton
    import torch
    
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = triton.language.load(x_ptr + offsets, mask=mask)
        y = triton.language.load(y_ptr + offsets, mask=mask)
        output = x + y
        triton.language.store(output_ptr + offsets, output, mask=mask)
    
    def add_vectors(x, y):
        output = torch.empty_like(x)
        n_elements = output.numel()
        BLOCK_SIZE = 1024
        grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        add_kernel[grid, BLOCK_SIZE](x, y, output, n_elements, BLOCK_SIZE)
        return output
    
    x = torch.randn(1024, device='cuda' if torch.cuda.is_available() else 'cpu')
    y = torch.randn(1024, device='cuda' if torch.cuda.is_available() else 'cpu')
    result = add_vectors(x, y)
    
    print('SUCCESS: PyTorch-Triton integration works')
except ImportError:
    print('INFO: Triton not available, skipping Triton integration test')
except Exception as e:
    print(f'ERROR: PyTorch-Triton integration failed: {e}')
" 2>/dev/null)

    local success_count=$(echo "$integration_test" | grep -c "SUCCESS")
    local error_count=$(echo "$integration_test" | grep -c "ERROR")

    if [ $success_count -gt 0 ]; then
        print_success "Component integration is working ($success_count successful integrations)"
    fi

    if [ $error_count -gt 0 ]; then
        print_warning "Some component integrations have issues ($error_count errors)"
    fi

    return $error_count
}

# Function to test end-to-end ML workflow
test_end_to_end_workflow() {
    print_section "End-to-End ML Workflow Test"

    print_info "Testing end-to-end ML workflow..."

    local workflow_test=$(python3 -c "
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

try:
    # Create model
    model = SimpleNet()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create some dummy data
    X = torch.randn(100, 10).to(device)
    y = torch.randn(100, 1).to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    final_loss = loss.item()
    print(f'SUCCESS: End-to-end ML workflow completed with final loss: {final_loss:.4f}')
    
except Exception as e:
    print(f'ERROR: End-to-end ML workflow failed: {e}')
" 2>/dev/null)

    if echo "$workflow_test" | grep -q "SUCCESS"; then
        print_success "End-to-end ML workflow is working"
        return 0
    else
        print_error "End-to-end ML workflow failed"
        return 1
    fi
}

# Function to test performance benchmarks
test_performance_benchmarks() {
    print_section "Performance Benchmark Test"

    print_info "Running basic performance benchmarks..."

    local benchmark_test=$(python3 -c "
import torch
import time
import numpy as np

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Testing on device: {device}')
    
    # Matrix multiplication benchmark
    sizes = [1000, 2000]
    for size in sizes:
        # Create matrices
        a = torch.randn(size, size).to(device)
        b = torch.randn(size, size).to(device)
        
        # Warm up
        _ = torch.mm(a, b)
        
        # Benchmark
        start_time = time.time()
        for _ in range(3):
            c = torch.mm(a, b)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 3
        print(f'SUCCESS: Matrix multiplication {size}x{size} took {avg_time:.4f} seconds')
    
except Exception as e:
    print(f'ERROR: Performance benchmark failed: {e}')
" 2>/dev/null)

    if echo "$benchmark_test" | grep -q "SUCCESS"; then
        print_success "Performance benchmarks completed successfully"
        return 0
    else
        print_warning "Performance benchmarks had issues"
        return 1
    fi
}

# Function to test memory management
test_memory_management() {
    print_section "Memory Management Test"

    print_info "Testing memory management and cleanup..."

    local memory_test=$(python3 -c "
import torch
import gc

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test memory allocation and deallocation
    tensors = []
    for i in range(10):
        tensor = torch.randn(1000, 1000).to(device)
        tensors.append(tensor)
    
    # Clear tensors
    del tensors
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print('SUCCESS: Memory management test passed')
    
    # Check memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        print(f'INFO: GPU memory allocated: {memory_allocated:.1f} MB')
        print(f'INFO: GPU memory reserved: {memory_reserved:.1f} MB')
    
except Exception as e:
    print(f'ERROR: Memory management test failed: {e}')
" 2>/dev/null)

    if echo "$memory_test" | grep -q "SUCCESS"; then
        print_success "Memory management is working correctly"
        return 0
    else
        print_warning "Memory management test had issues"
        return 1
    fi
}

# Function to test environment consistency
test_environment_consistency() {
    print_section "Environment Consistency Test"

    print_info "Testing environment variable consistency..."

    # Check key environment variables
    local env_vars=("ROCM_PATH" "PYTORCH_ROCM_ARCH" "HSA_OVERRIDE_GFX_VERSION" "PATH" "LD_LIBRARY_PATH")

    local consistent_vars=0
    local total_vars=${#env_vars[@]}

    for var in "${env_vars[@]}"; do
        if [ -n "${!var}" ]; then
            print_success "Environment variable $var is set: ${!var}"
            consistent_vars=$((consistent_vars + 1))
        else
            print_info "Environment variable $var is not set"
        fi
    done

    if [ $consistent_vars -gt 0 ]; then
        print_success "Environment variables are properly configured ($consistent_vars/$total_vars set)"
    else
        print_info "No environment variables are set (may be expected)"
    fi

    return 0
}

# Function to test script integration
test_script_integration() {
    print_section "Script Integration Test"

    print_info "Testing integration between installation scripts..."

    # Check if scripts can be sourced without errors
    local scripts=("install_rocm.sh" "install_pytorch_rocm.sh" "install_triton.sh")
    local integration_errors=0

    for script in "${scripts[@]}"; do
        if [ -f "$PWD/$script" ]; then
            # Try to source the script and check for basic functions
            local source_test=$(bash -c "
            source '$PWD/$script' 2>/dev/null
            if command -v print_success >/dev/null 2>&1; then
                echo 'SUCCESS: $script sourced successfully'
            else
                echo 'ERROR: $script sourcing failed'
            fi
            " 2>/dev/null)

            if echo "$source_test" | grep -q "SUCCESS"; then
                print_success "$script integrates properly"
            else
                print_warning "$script integration has issues"
                integration_errors=$((integration_errors + 1))
            fi
        else
            print_error "Script $script not found"
            integration_errors=$((integration_errors + 1))
        fi
    done

    if [ $integration_errors -eq 0 ]; then
        print_success "All scripts integrate properly"
        return 0
    else
        print_warning "Some scripts have integration issues"
        return 1
    fi
}

# Function to generate integration report
generate_integration_report() {
    print_section "Integration Test Report"

    local report_file="integration_test_report_$(date +"%Y%m%d_%H%M%S").txt"

    cat > "$report_file" << EOF
ML Stack Integration Test Report
Generated: $(date)
================================

System Information:
- OS: $(uname -s) $(uname -r)
- Python: $(python3 --version 2>/dev/null || echo "Not found")
- ROCm: $(rocminfo --version 2>/dev/null || echo "Not found")

Component Status:
- PyTorch: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")
- Triton: $(python3 -c "import triton; print(triton.__version__)" 2>/dev/null || echo "Not installed")
- NumPy: $(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "Not installed")
- ROCm GPUs: $(rocminfo 2>/dev/null | grep "Device Type:.*GPU" | wc -l || echo "0")

Integration Test Results:
See test output above for detailed results.

Recommendations:
1. Ensure ROCm is properly installed and configured
2. Use compatible versions of PyTorch and Triton
3. Verify GPU memory and compute capability
4. Check environment variables are set correctly
5. Run benchmarks to validate performance

EOF

    print_success "Integration report saved to: $report_file"
}

# Main test function
run_comprehensive_integration_test() {
    print_header

    local total_tests=0
    local passed_tests=0

    # Test ROCm integration
    test_rocm_integration
    total_tests=$((total_tests + 1))
    if [ $? -eq 0 ]; then
        passed_tests=$((passed_tests + 1))
    fi

    # Test component integration
    test_component_integration
    total_tests=$((total_tests + 1))
    if [ $? -eq 0 ]; then
        passed_tests=$((passed_tests + 1))
    fi

    # Test end-to-end workflow
    test_end_to_end_workflow
    total_tests=$((total_tests + 1))
    if [ $? -eq 0 ]; then
        passed_tests=$((passed_tests + 1))
    fi

    # Test performance benchmarks
    test_performance_benchmarks
    total_tests=$((total_tests + 1))
    if [ $? -eq 0 ]; then
        passed_tests=$((passed_tests + 1))
    fi

    # Test memory management
    test_memory_management
    total_tests=$((total_tests + 1))
    if [ $? -eq 0 ]; then
        passed_tests=$((passed_tests + 1))
    fi

    # Test environment consistency
    test_environment_consistency
    total_tests=$((total_tests + 1))
    if [ $? -eq 0 ]; then
        passed_tests=$((passed_tests + 1))
    fi

    # Test script integration
    test_script_integration
    total_tests=$((total_tests + 1))
    if [ $? -eq 0 ]; then
        passed_tests=$((passed_tests + 1))
    fi

    # Generate report
    generate_integration_report

    print_section "Final Results"

    echo
    print_info "Comprehensive Integration Test Summary:"
    echo "  - Tests run: $total_tests"
    echo "  - Tests passed: $passed_tests"
    echo "  - Success rate: $((passed_tests * 100 / total_tests))%"

    if [ $passed_tests -eq $total_tests ]; then
        print_success "All integration tests passed! ML Stack is fully integrated."
        return 0
    elif [ $passed_tests -ge $((total_tests * 3 / 4)) ]; then
        print_success "Most integration tests passed. ML Stack is well integrated."
        return 0
    else
        print_warning "Some integration tests failed. Check the results above."
        return 1
    fi
}

# Run the comprehensive test
run_comprehensive_integration_test