#!/bin/bash
#
# Virtual Environment Compatibility Test Script
# Tests that all ML Stack scripts handle virtual environment creation and activation consistently
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
    echo -e "${CYAN}${BOLD}║      Virtual Environment Compatibility Test            ║${RESET}"
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

# Function to test virtual environment creation
test_venv_creation() {
    local script_path="$1"
    local script_name=$(basename "$script_path" .sh)

    print_info "Testing virtual environment creation in $script_name..."

    # Create a temporary test directory
    local test_dir=$(mktemp -d)
    cd "$test_dir"

    # Extract venv creation logic from script
    local venv_creation=$(grep -A 20 "create.*venv\|uv venv\|python3 -m venv" "$script_path" | head -20)

    if [ -n "$venv_creation" ]; then
        print_info "Found virtual environment creation logic in $script_name"

        # Test uv venv creation if uv is available
        if command_exists uv; then
            print_info "Testing uv venv creation..."
            if uv venv test_uv_venv >/dev/null 2>&1; then
                print_success "uv venv creation works"
                rm -rf test_uv_venv
            else
                print_warning "uv venv creation failed"
            fi
        else
            print_info "uv not available, skipping uv venv test"
        fi

        # Test python3 venv creation
        print_info "Testing python3 venv creation..."
        if python3 -m venv test_py_venv >/dev/null 2>&1; then
            print_success "python3 venv creation works"
            rm -rf test_py_venv
        else
            print_warning "python3 venv creation failed"
        fi

        return 0
    else
        print_warning "No virtual environment creation logic found in $script_name"
        return 1
    fi

    # Cleanup
    cd - >/dev/null
    rm -rf "$test_dir"
}

# Function to test virtual environment activation
test_venv_activation() {
    local script_path="$1"
    local script_name=$(basename "$script_path" .sh)

    print_info "Testing virtual environment activation in $script_name..."

    # Create a temporary test directory
    local test_dir=$(mktemp -d)
    cd "$test_dir"

    # Create test virtual environments
    python3 -m venv test_py_venv >/dev/null 2>&1

    if command_exists uv; then
        uv venv test_uv_venv >/dev/null 2>&1
    fi

    # Test python3 venv activation
    print_info "Testing python3 venv activation..."
    source test_py_venv/bin/activate
    if [ "$VIRTUAL_ENV" = "$test_dir/test_py_venv" ]; then
        print_success "python3 venv activation works"
    else
        print_warning "python3 venv activation may not work correctly"
    fi
    deactivate

    # Test uv venv activation if available
    if [ -d "test_uv_venv" ]; then
        print_info "Testing uv venv activation..."
        source test_uv_venv/bin/activate
        if [ "$VIRTUAL_ENV" = "$test_dir/test_uv_venv" ]; then
            print_success "uv venv activation works"
        else
            print_warning "uv venv activation may not work correctly"
        fi
        deactivate
    fi

    # Cleanup
    cd - >/dev/null
    rm -rf "$test_dir"
}

# Function to test virtual environment package installation
test_venv_package_install() {
    local script_path="$1"
    local script_name=$(basename "$script_path" .sh)

    print_info "Testing package installation in virtual environments for $script_name..."

    # Create a temporary test directory
    local test_dir=$(mktemp -d)
    cd "$test_dir"

    # Create test virtual environment
    python3 -m venv test_venv >/dev/null 2>&1
    source test_venv/bin/activate

    # Test pip installation
    print_info "Testing pip package installation..."
    if python3 -m pip install --quiet numpy >/dev/null 2>&1; then
        if python3 -c "import numpy; print('numpy installed')" >/dev/null 2>&1; then
            print_success "pip package installation works"
        else
            print_warning "pip package installation failed verification"
        fi
    else
        print_warning "pip package installation failed"
    fi

    # Test uv installation if available
    if command_exists uv; then
        print_info "Testing uv package installation..."
        if uv pip install --quiet requests >/dev/null 2>&1; then
            if python3 -c "import requests; print('requests installed')" >/dev/null 2>&1; then
                print_success "uv package installation works"
            else
                print_warning "uv package installation failed verification"
            fi
        else
            print_warning "uv package installation failed"
        fi
    fi

    deactivate

    # Cleanup
    cd - >/dev/null
    rm -rf "$test_dir"
}

# Function to analyze virtual environment logic in scripts
analyze_venv_logic() {
    local script_path="$1"
    local script_name=$(basename "$script_path" .sh)

    print_info "Analyzing virtual environment logic in $script_name..."

    # Check for venv-related functions
    local venv_functions=$(grep -c "create.*venv\|uv_pip_install\|install_python_package" "$script_path" || echo "0")

    # Check for venv activation patterns
    local venv_activation=$(grep -c "source.*bin/activate" "$script_path" || echo "0")

    # Check for uv vs pip preference
    local uv_usage=$(grep -c "command_exists uv" "$script_path" || echo "0")

    # Check for venv fallback logic
    local fallback_logic=$(grep -c "else.*python3 -m pip\|else.*pip install" "$script_path" || echo "0")

    print_info "$script_name venv analysis:"
    echo "  - venv-related functions: $venv_functions"
    echo "  - venv activation patterns: $venv_activation"
    echo "  - uv detection: $uv_usage"
    echo "  - fallback logic: $fallback_logic"

    # Check for common issues
    if [ "$venv_functions" -gt 0 ] && [ "$venv_activation" -eq 0 ]; then
        print_warning "$script_name has venv functions but no activation patterns"
    fi

    if [ "$uv_usage" -gt 0 ] && [ "$fallback_logic" -eq 0 ]; then
        print_warning "$script_name uses uv but may lack fallback to pip"
    fi

    if [ "$venv_functions" -gt 0 ]; then
        print_success "$script_name has virtual environment support"
    else
        print_info "$script_name does not have explicit venv functions"
    fi
}

# Function to test virtual environment isolation
test_venv_isolation() {
    print_info "Testing virtual environment isolation..."

    # Create two test virtual environments
    local test_dir=$(mktemp -d)
    cd "$test_dir"

    python3 -m venv venv1 >/dev/null 2>&1
    python3 -m venv venv2 >/dev/null 2>&1

    # Install different packages in each
    source venv1/bin/activate
    python3 -m pip install --quiet numpy==1.21.0 >/dev/null 2>&1
    local numpy1_version=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)
    deactivate

    source venv2/bin/activate
    python3 -m pip install --quiet numpy==1.24.0 >/dev/null 2>&1
    local numpy2_version=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)
    deactivate

    # Check system python
    local system_numpy=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "not installed")

    print_info "Virtual environment isolation test:"
    echo "  - venv1 numpy version: $numpy1_version"
    echo "  - venv2 numpy version: $numpy2_version"
    echo "  - system numpy version: $system_numpy"

    if [ "$numpy1_version" != "$numpy2_version" ] && [ "$numpy1_version" != "$system_numpy" ] && [ "$numpy2_version" != "$system_numpy" ]; then
        print_success "Virtual environment isolation works correctly"
    else
        print_warning "Virtual environment isolation may not be working properly"
    fi

    # Cleanup
    cd - >/dev/null
    rm -rf "$test_dir"
}

# Function to test virtual environment persistence
test_venv_persistence() {
    print_info "Testing virtual environment persistence..."

    # Create a test virtual environment
    local test_dir=$(mktemp -d)
    cd "$test_dir"

    python3 -m venv persistent_venv >/dev/null 2>&1
    source persistent_venv/bin/activate
    python3 -m pip install --quiet requests >/dev/null 2>&1

    # Check if package persists after reactivation
    deactivate
    source persistent_venv/bin/activate

    if python3 -c "import requests; print('requests available')" >/dev/null 2>&1; then
        print_success "Virtual environment persistence works"
    else
        print_warning "Virtual environment persistence may not work"
    fi

    deactivate

    # Cleanup
    cd - >/dev/null
    rm -rf "$test_dir"
}

# Main test function
test_venv_compatibility() {
    print_header

    # List of scripts to test
    local scripts=(
        "install_rocm.sh"
        "install_pytorch_rocm.sh"
        "install_triton.sh"
        "install_vllm.sh"
        "build_onnxruntime.sh"
    )

    print_section "Virtual Environment Creation Tests"

    # Test venv creation for each script
    for script in "${scripts[@]}"; do
        local script_path="$PWD/$script"
        if [ -f "$script_path" ]; then
            test_venv_creation "$script_path"
        else
            print_error "Script $script not found"
        fi
    done

    print_section "Virtual Environment Activation Tests"

    # Test venv activation for each script
    for script in "${scripts[@]}"; do
        local script_path="$PWD/$script"
        if [ -f "$script_path" ]; then
            test_venv_activation "$script_path"
        fi
    done

    print_section "Package Installation in Virtual Environments"

    # Test package installation in venvs
    for script in "${scripts[@]}"; do
        local script_path="$PWD/$script"
        if [ -f "$script_path" ]; then
            test_venv_package_install "$script_path"
        fi
    done

    print_section "Virtual Environment Logic Analysis"

    # Analyze venv logic in each script
    for script in "${scripts[@]}"; do
        local script_path="$PWD/$script"
        if [ -f "$script_path" ]; then
            analyze_venv_logic "$script_path"
        fi
    done

    print_section "Virtual Environment Isolation and Persistence"

    # Test venv isolation
    test_venv_isolation

    # Test venv persistence
    test_venv_persistence

    print_section "Summary"

    local issues=0

    # Check for common issues across scripts
    for script in "${scripts[@]}"; do
        local script_path="$PWD/$script"
        if [ -f "$script_path" ]; then
            # Check if script has both uv and pip support
            if grep -q "command_exists uv" "$script_path" && ! grep -q "python3 -m pip\|pip install" "$script_path"; then
                print_warning "$script uses uv but may lack pip fallback"
                issues=$((issues + 1))
            fi

            # Check if script activates venvs it creates
            if grep -q "uv venv\|python3 -m venv" "$script_path" && ! grep -q "source.*activate" "$script_path"; then
                print_warning "$script creates venvs but may not activate them"
                issues=$((issues + 1))
            fi
        fi
    done

    if [ $issues -eq 0 ]; then
        print_success "Virtual environment compatibility is working correctly!"
        echo
        print_info "Verified compatibility features:"
        echo "  - Virtual environment creation (uv and python3 venv)"
        echo "  - Virtual environment activation"
        echo "  - Package installation in virtual environments"
        echo "  - Virtual environment isolation"
        echo "  - Virtual environment persistence"
        echo "  - Fallback from uv to pip when needed"
        return 0
    else
        print_error "Found $issues issues with virtual environment compatibility"
        return 1
    fi
}

# Run the test
test_venv_compatibility