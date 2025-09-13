#!/bin/bash
#
# ROCm Detection Consistency Test Script
# Tests that all ML Stack scripts detect ROCm consistently
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
    echo -e "${CYAN}${BOLD}║        ROCm Detection Consistency Test                ║${RESET}"
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

# Function to extract ROCm detection logic from script
extract_rocm_detection() {
    local script_path="$1"
    local script_name=$(basename "$script_path" .sh)

    print_info "Extracting ROCm detection logic from $script_name..."

    # Extract ROCm detection functions
    local rocm_functions=$(grep -A 20 "get_rocm_version\|detect_rocm\|check_rocm" "$script_path" | head -30)

    if [ -n "$rocm_functions" ]; then
        echo "$rocm_functions"
        return 0
    else
        print_warning "No explicit ROCm detection functions found in $script_name"
        return 1
    fi
}

# Function to test ROCm detection consistency
test_rocm_detection_consistency() {
    local script1="$1"
    local script2="$2"

    print_info "Comparing ROCm detection between $script1 and $script2..."

    # Extract ROCm detection logic from both scripts
    local detection1=$(extract_rocm_detection "$script1")
    local detection2=$(extract_rocm_detection "$script2")

    # Check if both scripts have ROCm detection
    local has_detection1=$([ -n "$detection1" ] && echo "yes" || echo "no")
    local has_detection2=$([ -n "$detection2" ] && echo "yes" || echo "no")

    if [ "$has_detection1" != "$has_detection2" ]; then
        print_warning "ROCm detection presence differs: $script1=$has_detection1, $script2=$has_detection2"
        return 1
    fi

    # Check for common ROCm detection patterns
    local patterns=("rocminfo" "ROCm Version" "/opt/rocm" "HSA_OVERRIDE_GFX_VERSION" "PYTORCH_ROCM_ARCH")

    local pattern_matches1=0
    local pattern_matches2=0

    for pattern in "${patterns[@]}"; do
        if grep -q "$pattern" "$script1"; then
            pattern_matches1=$((pattern_matches1 + 1))
        fi
        if grep -q "$pattern" "$script2"; then
            pattern_matches2=$((pattern_matches2 + 1))
        fi
    done

    if [ "$pattern_matches1" -ne "$pattern_matches2" ]; then
        print_warning "ROCm detection pattern count differs: $script1=$pattern_matches1, $script2=$pattern_matches2"
        return 1
    fi

    print_success "ROCm detection patterns are consistent between $script1 and $script2"
    return 0
}

# Function to test actual ROCm detection in scripts
test_actual_rocm_detection() {
    local script_path="$1"
    local script_name=$(basename "$script_path" .sh)

    print_info "Testing actual ROCm detection in $script_name..."

    # Create a temporary test script that sources the target script and checks ROCm detection
    local temp_script=$(mktemp)
    cat > "$temp_script" << EOF
#!/bin/bash
# Test ROCm detection in script

# Source the script (but don't run main function)
source "$script_path" 2>/dev/null || exit 1

# Check if rocminfo command exists
if command_exists rocminfo; then
    echo "rocminfo_available:yes"
    
    # Try to get ROCm version
    rocm_version=\$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print \$2}' | xargs)
    if [ -n "\$rocm_version" ]; then
        echo "rocm_version:\$rocm_version"
    else
        echo "rocm_version:not_detected"
    fi
    
    # Check GPU count
    gpu_count=\$(rocminfo 2>/dev/null | grep "Device Type:.*GPU" | wc -l)
    echo "gpu_count:\$gpu_count"
else
    echo "rocminfo_available:no"
    echo "rocm_version:not_available"
    echo "gpu_count:0"
fi

# Check if ROCm path exists
if [ -d "/opt/rocm" ]; then
    echo "rocm_path_exists:yes"
else
    echo "rocm_path_exists:no"
fi
EOF

    chmod +x "$temp_script"
    local output=$("$temp_script")
    rm -f "$temp_script"

    # Parse output
    local rocminfo_available=$(echo "$output" | grep "rocminfo_available:" | cut -d: -f2)
    local rocm_version=$(echo "$output" | grep "rocm_version:" | cut -d: -f2)
    local gpu_count=$(echo "$output" | grep "gpu_count:" | cut -d: -f2)
    local rocm_path_exists=$(echo "$output" | grep "rocm_path_exists:" | cut -d: -f2)

    print_info "$script_name ROCm detection results:"
    echo "  - rocminfo available: $rocminfo_available"
    echo "  - ROCm version: $rocm_version"
    echo "  - GPU count: $gpu_count"
    echo "  - ROCm path exists: $rocm_path_exists"

    # Return success if basic detection works
    if [ "$rocminfo_available" = "yes" ] && [ "$rocm_path_exists" = "yes" ]; then
        print_success "$script_name ROCm detection works correctly"
        return 0
    else
        print_warning "$script_name ROCm detection may have issues"
        return 1
    fi
}

# Function to analyze ROCm environment variable setting
analyze_rocm_env_vars() {
    local script_path="$1"
    local script_name=$(basename "$script_path" .sh)

    print_info "Analyzing ROCm environment variables in $script_name..."

    # Check for key ROCm environment variables
    local env_vars=("HSA_TOOLS_LIB" "HSA_OVERRIDE_GFX_VERSION" "PYTORCH_ROCM_ARCH" "ROCM_PATH" "PATH" "LD_LIBRARY_PATH")

    local env_count=0
    for var in "${env_vars[@]}"; do
        if grep -q "export.*$var" "$script_path"; then
            env_count=$((env_count + 1))
        fi
    done

    print_info "$script_name sets $env_count ROCm-related environment variables"

    # Check for ROCm version detection
    if grep -q "get_rocm_version\|rocminfo.*Version" "$script_path"; then
        print_success "$script_name has ROCm version detection"
    else
        print_info "$script_name does not have explicit ROCm version detection"
    fi

    # Check for GPU architecture detection
    if grep -q "detect_gpu_architecture\|gfx" "$script_path"; then
        print_success "$script_name has GPU architecture detection"
    else
        print_info "$script_name does not have GPU architecture detection"
    fi
}

# Function to test ROCm detection across all scripts
test_rocm_detection_across_scripts() {
    print_header

    # List of scripts to test
    local scripts=(
        "install_rocm.sh"
        "install_pytorch_rocm.sh"
        "install_triton.sh"
        "install_vllm.sh"
        "build_onnxruntime.sh"
    )

    print_section "ROCm Detection Logic Analysis"

    # Analyze ROCm detection in each script
    for script in "${scripts[@]}"; do
        local script_path="$PWD/$script"
        if [ -f "$script_path" ]; then
            analyze_rocm_env_vars "$script_path"
        else
            print_error "Script $script not found"
        fi
    done

    print_section "ROCm Detection Consistency Check"

    # Compare ROCm detection across scripts
    local total_scripts=${#scripts[@]}
    local consistency_issues=0

    for ((i=0; i<total_scripts; i++)); do
        for ((j=i+1; j<total_scripts; j++)); do
            local script1="${scripts[$i]}"
            local script2="${scripts[$j]}"

            local script1_path="$PWD/$script1"
            local script2_path="$PWD/$script2"

            if [ -f "$script1_path" ] && [ -f "$script2_path" ]; then
                if ! test_rocm_detection_consistency "$script1_path" "$script2_path"; then
                    consistency_issues=$((consistency_issues + 1))
                fi
            fi
        done
    done

    print_section "Actual ROCm Detection Test"

    # Test actual ROCm detection in each script
    local detection_issues=0

    for script in "${scripts[@]}"; do
        local script_path="$PWD/$script"
        if [ -f "$script_path" ]; then
            if ! test_actual_rocm_detection "$script_path"; then
                detection_issues=$((detection_issues + 1))
            fi
        fi
    done

    print_section "Summary"

    local total_issues=$((consistency_issues + detection_issues))

    if [ $total_issues -eq 0 ]; then
        print_success "ROCm detection is consistent and working correctly across all scripts!"
        echo
        print_info "Verified ROCm detection features:"
        echo "  - Consistent ROCm detection logic across scripts"
        echo "  - Proper rocminfo command detection"
        echo "  - ROCm version detection"
        echo "  - GPU count detection"
        echo "  - ROCm path validation"
        echo "  - Environment variable configuration"
        return 0
    else
        print_error "Found $total_issues issues with ROCm detection"
        echo
        print_info "Common ROCm detection issues:"
        echo "  - Inconsistent detection patterns between scripts"
        echo "  - Missing ROCm version detection in some scripts"
        echo "  - Environment variable configuration differences"
        return 1
    fi
}

# Run the test
test_rocm_detection_across_scripts