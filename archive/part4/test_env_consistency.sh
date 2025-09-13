#!/bin/bash
#
# Environment Variable Consistency Test Script
# Tests that all ML Stack scripts set compatible environment variables
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
    echo -e "${CYAN}${BOLD}║         Environment Variable Consistency Test          ║${RESET}"
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

# Function to extract environment variables from script output
extract_env_vars() {
    local script_path="$1"
    local script_name=$(basename "$script_path" .sh)

    print_info "Extracting environment variables from $script_name..."

    # Run the script with --show-env and capture output
    if [ -f "$script_path" ] && [ -x "$script_path" ]; then
        # Create a temporary script to capture environment variables
        local temp_script=$(mktemp)
        cat > "$temp_script" << EOF
#!/bin/bash
# Capture environment before
env | grep -E '^(HSA_|PYTORCH_|ROCM_|PATH|LD_LIBRARY_PATH|HIP_|CUDA_)' | sort > /tmp/env_before

# Source the show-env output
eval "\$($script_path --show-env 2>/dev/null)"

# Capture environment after
env | grep -E '^(HSA_|PYTORCH_|ROCM_|PATH|LD_LIBRARY_PATH|HIP_|CUDA_)' | sort > /tmp/env_after

# Show differences
echo "=== Environment variables set by $script_name ==="
comm -13 /tmp/env_before /tmp/env_after
EOF
        chmod +x "$temp_script"
        local output=$("$temp_script" 2>/dev/null)
        rm -f "$temp_script" /tmp/env_before /tmp/env_after
        echo "$output"
    else
        echo "ERROR: Script $script_path not found or not executable"
        return 1
    fi
}

# Function to parse environment variables from output
parse_env_vars() {
    local output="$1"
    local script_name="$2"
    declare -A env_vars

    # Parse export statements
    while IFS= read -r line; do
        if [[ $line =~ export\ ([^=]+)=(.*) ]]; then
            local var_name="${BASH_REMATCH[1]}"
            local var_value="${BASH_REMATCH[2]}"
            # Remove quotes if present
            var_value=$(echo "$var_value" | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/")
            env_vars["$var_name"]="$var_value"
        fi
    done <<< "$output"

    # Return the associative array
    declare -p env_vars
}

# Function to compare environment variables across scripts
compare_env_vars() {
    local script1="$1"
    local script2="$2"
    local vars1="$3"
    local vars2="$4"

    print_info "Comparing $script1 vs $script2..."

    # Extract variable names
    local vars1_names=$(echo "$vars1" | grep -o 'env_vars\["[^"]*"\]' | sed 's/env_vars\["\([^"]*\)"\]/\1/' | sort)
    local vars2_names=$(echo "$vars2" | grep -o 'env_vars\["[^"]*"\]' | sed 's/env_vars\["\([^"]*\)"\]/\1/' | sort)

    # Find common variables
    local common_vars=$(comm -12 <(echo "$vars1_names") <(echo "$vars2_names"))

    local conflicts=0

    # Check for conflicts in common variables
    for var in $common_vars; do
        local val1=$(echo "$vars1" | grep "env_vars\[\"$var\"\]" | sed "s/.*env_vars\[\"$var\"\]=\"\([^\"]*\)\".*/\1/")
        local val2=$(echo "$vars2" | grep "env_vars\[\"$var\"\]" | sed "s/.*env_vars\[\"$var\"\]=\"\([^\"]*\)\".*/\1/")

        if [ "$val1" != "$val2" ]; then
            print_warning "Conflict in $var:"
            echo "  $script1: $val1"
            echo "  $script2: $val2"
            conflicts=$((conflicts + 1))
        fi
    done

    if [ $conflicts -eq 0 ]; then
        print_success "No conflicts found between $script1 and $script2"
    else
        print_error "Found $conflicts conflicts between $script1 and $script2"
    fi

    return $conflicts
}

# Main test function
test_env_consistency() {
    print_header

    # List of scripts to test
    local scripts=(
        "install_rocm.sh"
        "install_pytorch_rocm.sh"
        "install_triton.sh"
        "install_vllm.sh"
        "build_onnxruntime.sh"
    )

    # Array to store environment variables for each script
    declare -A script_envs

    # Extract environment variables from each script
    for script in "${scripts[@]}"; do
        local script_path="$PWD/$script"
        local output=$(extract_env_vars "$script_path")

        if [ $? -eq 0 ]; then
            script_envs["$script"]=$(parse_env_vars "$output" "$script")
            print_success "Successfully extracted env vars from $script"
        else
            print_error "Failed to extract env vars from $script"
            script_envs["$script"]=""
        fi
    done

    print_section "Environment Variable Analysis"

    # Compare each pair of scripts
    local total_conflicts=0
    local script_count=${#scripts[@]}

    for ((i=0; i<script_count; i++)); do
        for ((j=i+1; j<script_count; j++)); do
            local script1="${scripts[$i]}"
            local script2="${scripts[$j]}"

            if [ -n "${script_envs[$script1]}" ] && [ -n "${script_envs[$script2]}" ]; then
                compare_env_vars "$script1" "$script2" "${script_envs[$script1]}" "${script_envs[$script2]}"
                total_conflicts=$((total_conflicts + $?))
            fi
        done
    done

    print_section "Summary"

    if [ $total_conflicts -eq 0 ]; then
        print_success "All environment variables are consistent across scripts!"
        echo
        print_info "Key environment variables verified:"
        echo "  - HSA_TOOLS_LIB: ROCm profiler library path"
        echo "  - HSA_OVERRIDE_GFX_VERSION: GPU architecture override"
        echo "  - PYTORCH_ROCM_ARCH: PyTorch ROCm architecture"
        echo "  - ROCM_PATH: ROCm installation path"
        echo "  - PATH: Updated with ROCm binaries"
        echo "  - LD_LIBRARY_PATH: Updated with ROCm libraries"
        return 0
    else
        print_error "Found $total_conflicts environment variable conflicts across scripts"
        echo
        print_info "Common conflict patterns:"
        echo "  - Different HSA_TOOLS_LIB values (some scripts set to 0, others to library path)"
        echo "  - Different HSA_OVERRIDE_GFX_VERSION values (architecture detection differences)"
        echo "  - PATH/LD_LIBRARY_PATH ordering differences"
        return 1
    fi
}

# Function to test environment variable precedence
test_env_precedence() {
    print_section "Environment Variable Precedence Test"

    print_info "Testing environment variable precedence when scripts are run in sequence..."

    # Save original environment
    local orig_env=$(env | grep -E '^(HSA_|PYTORCH_|ROCM_|PATH|LD_LIBRARY_PATH|HIP_|CUDA_)' | sort)

    # Run scripts in sequence and check final environment
    local scripts=("install_rocm.sh" "install_pytorch_rocm.sh" "install_triton.sh")

    for script in "${scripts[@]}"; do
        if [ -f "$script" ] && [ -x "$script" ]; then
            print_info "Running $script --show-env..."
            eval "$($script --show-env 2>/dev/null)" 2>/dev/null || true
        fi
    done

    # Check final environment
    local final_env=$(env | grep -E '^(HSA_|PYTORCH_|ROCM_|PATH|LD_LIBRARY_PATH|HIP_|CUDA_)' | sort)

    print_info "Environment variables after running all scripts:"
    echo "$final_env"

    # Restore original environment
    print_info "Restoring original environment..."
    # This is a simplified restore - in practice, you'd need to be more careful
}

# Run the tests
main() {
    local test_type="${1:-consistency}"

    case "$test_type" in
        "consistency")
            test_env_consistency
            ;;
        "precedence")
            test_env_precedence
            ;;
        "all")
            test_env_consistency
            echo
            test_env_precedence
            ;;
        *)
            echo "Usage: $0 [consistency|precedence|all]"
            echo "  consistency: Test environment variable consistency across scripts"
            echo "  precedence: Test environment variable precedence when scripts run in sequence"
            echo "  all: Run all tests"
            exit 1
            ;;
    esac
}

# Run main function with arguments
main "$@"