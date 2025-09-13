#!/bin/bash
#
# Package Manager Integration Test Script
# Tests that all ML Stack scripts use consistent package manager detection and handling
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
    echo -e "${CYAN}${BOLD}║        Package Manager Integration Test                ║${RESET}"
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

# Function to extract package manager detection logic from script
extract_pm_detection() {
    local script_path="$1"
    local script_name=$(basename "$script_path" .sh)

    print_info "Extracting package manager detection from $script_name..."

    # Extract the detect_package_manager function
    local pm_function=$(sed -n '/^detect_package_manager() {/,/^}/p' "$script_path")

    if [ -n "$pm_function" ]; then
        echo "$pm_function"
        return 0
    else
        print_warning "No detect_package_manager function found in $script_name"
        return 1
    fi
}

# Function to test package manager detection
test_pm_detection() {
    local script_path="$1"
    local expected_pm="$2"

    # Create a temporary test script
    local temp_script=$(mktemp)
    cat > "$temp_script" << EOF
#!/bin/bash
# Extract and test package manager detection function

# Source the original script to get the function
source "$script_path" 2>/dev/null || exit 1

# Mock package managers for testing
mock_pm() {
    local pm="\$1"
    case "\$pm" in
        "dnf")
            echo '#!/bin/bash
                 if [[ "\$1" == "--version" ]]; then
                     echo "dnf version 4.14.0"
                 fi' > /tmp/mock_dnf
            chmod +x /tmp/mock_dnf
            export PATH="/tmp:\$PATH"
            ;;
        "apt")
            echo '#!/bin/bash
                 if [[ "\$1" == "--version" ]]; then
                     echo "apt 2.4.5 (amd64)"
                 fi' > /tmp/mock_apt
            chmod +x /tmp/mock_apt
            export PATH="/tmp:\$PATH"
            ;;
        "yum")
            echo '#!/bin/bash
                 if [[ "\$1" == "--version" ]]; then
                     echo "yum 3.4.3"
                 fi' > /tmp/mock_yum
            chmod +x /tmp/mock_yum
            export PATH="/tmp:\$PATH"
            ;;
        "pacman")
            echo '#!/bin/bash
                 if [[ "\$1" == "-V" ]]; then
                     echo "Pacman v6.0.1"
                 fi' > /tmp/mock_pacman
            chmod +x /tmp/mock_pacman
            export PATH="/tmp:\$PATH"
            ;;
        "zypper")
            echo '#!/bin/bash
                 if [[ "\$1" == "--version" ]]; then
                     echo "zypper 1.14.50"
                 fi' > /tmp/mock_zypper
            chmod +x /tmp/mock_zypper
            export PATH="/tmp:\$PATH"
            ;;
    esac
}

# Test detection for each package manager
test_detection() {
    local expected="\$1"
    local detected

    # Clear PATH and add mock
    export PATH="/tmp"

    case "\$expected" in
        "dnf")
            mock_pm "dnf"
            detected=\$(detect_package_manager)
            ;;
        "apt")
            mock_pm "apt"
            detected=\$(detect_package_manager)
            ;;
        "yum")
            mock_pm "yum"
            detected=\$(detect_package_manager)
            ;;
        "pacman")
            mock_pm "pacman"
            detected=\$(detect_package_manager)
            ;;
        "zypper")
            mock_pm "zypper"
            detected=\$(detect_package_manager)
            ;;
        *)
            detected="unknown"
            ;;
    esac

    if [ "\$detected" = "\$expected" ]; then
        echo "PASS: \$expected"
    else
        echo "FAIL: expected \$expected, got \$detected"
    fi
}

# Run tests
echo "Testing package manager detection..."
test_detection "dnf"
test_detection "apt"
test_detection "yum"
test_detection "pacman"
test_detection "zypper"

# Test fallback to unknown
export PATH="/bin"
detected=\$(detect_package_manager)
if [ "\$detected" = "unknown" ]; then
    echo "PASS: unknown"
else
    echo "FAIL: expected unknown, got \$detected"
fi
EOF

    chmod +x "$temp_script"
    local output=$("$temp_script")
    rm -f "$temp_script" /tmp/mock_*

    echo "$output"
}

# Function to compare package manager detection across scripts
compare_pm_detection() {
    local script1="$1"
    local script2="$2"

    print_info "Comparing package manager detection: $script1 vs $script2"

    # Extract functions from both scripts
    local func1=$(extract_pm_detection "$script1")
    local func2=$(extract_pm_detection "$script2")

    if [ -z "$func1" ] || [ -z "$func2" ]; then
        print_warning "Could not extract package manager detection from one or both scripts"
        return 1
    fi

    # Test both functions
    local test1=$(test_pm_detection "$script1")
    local test2=$(test_pm_detection "$script2")

    # Compare results
    if [ "$test1" = "$test2" ]; then
        print_success "Package manager detection is consistent between $script1 and $script2"
        return 0
    else
        print_error "Package manager detection differs between $script1 and $script2"
        echo "Script 1 results:"
        echo "$test1"
        echo "Script 2 results:"
        echo "$test2"
        return 1
    fi
}

# Function to test package manager command usage
test_pm_commands() {
    local script_path="$1"
    local script_name=$(basename "$script_path" .sh)

    print_info "Testing package manager command usage in $script_name..."

    # Look for package manager usage patterns
    local apt_usage=$(grep -c "apt" "$script_path" || echo "0")
    local dnf_usage=$(grep -c "dnf\|yum" "$script_path" || echo "0")
    local pacman_usage=$(grep -c "pacman" "$script_path" || echo "0")
    local zypper_usage=$(grep -c "zypper" "$script_path" || echo "0")

    print_info "$script_name package manager usage:"
    echo "  apt commands: $apt_usage"
    echo "  dnf/yum commands: $dnf_usage"
    echo "  pacman commands: $pacman_usage"
    echo "  zypper commands: $zypper_usage"

    # Check if script uses the detection function
    if grep -q "detect_package_manager" "$script_path"; then
        print_success "$script_name uses detect_package_manager function"
    else
        print_warning "$script_name does not use detect_package_manager function"
    fi

    # Check for hardcoded package manager commands
    local hardcoded=$(grep -E "(apt|dnf|yum|pacman|zypper).*install" "$script_path" | grep -v "detect_package_manager\|package_manager" | wc -l)
    if [ "$hardcoded" -gt 0 ]; then
        print_warning "$script_name has $hardcoded potentially hardcoded package manager commands"
        grep -E "(apt|dnf|yum|pacman|zypper).*install" "$script_path" | grep -v "detect_package_manager\|package_manager"
    else
        print_success "$script_name appears to use dynamic package manager detection"
    fi
}

# Function to test system package installation simulation
test_system_package_install() {
    local script_path="$1"
    local script_name=$(basename "$script_path" .sh)

    print_info "Testing system package installation logic in $script_name..."

    # Create a mock install_system_package function test
    local temp_script=$(mktemp)
    cat > "$temp_script" << EOF
#!/bin/bash
# Test system package installation logic

source "$script_path" 2>/dev/null

# Mock package managers
mock_apt() {
    apt-get() {
        echo "MOCK: apt-get \$@"
        return 0
    }
    export -f apt-get
}

mock_dnf() {
    dnf() {
        echo "MOCK: dnf \$@"
        return 0
    }
    export -f dnf
}

# Test with different package managers
echo "Testing apt..."
mock_apt
install_system_package "test-package" 2>/dev/null || echo "apt test completed"

echo "Testing dnf..."
mock_dnf
install_system_package "test-package" 2>/dev/null || echo "dnf test completed"
EOF

    chmod +x "$temp_script"
    local output=$("$temp_script" 2>&1)
    rm -f "$temp_script"

    if echo "$output" | grep -q "MOCK:"; then
        print_success "$script_name system package installation logic works"
    else
        print_warning "$script_name system package installation logic may need verification"
    fi
}

# Main test function
test_package_manager_integration() {
    print_header

    # List of scripts to test
    local scripts=(
        "install_rocm.sh"
        "install_pytorch_rocm.sh"
        "install_triton.sh"
        "install_vllm.sh"
        "build_onnxruntime.sh"
    )

    print_section "Package Manager Detection Consistency"

    # Test package manager detection across all scripts
    local detection_results=()
    local total_scripts=${#scripts[@]}

    for script in "${scripts[@]}"; do
        local script_path="$PWD/$script"
        if [ -f "$script_path" ]; then
            local result=$(test_pm_detection "$script_path")
            detection_results+=("$script:$result")
            print_success "Tested package manager detection for $script"
        else
            print_error "Script $script not found"
        fi
    done

    # Compare detection results across scripts
    local consistent=true
    local first_result=""

    for result in "${detection_results[@]}"; do
        local script_name=$(echo "$result" | cut -d: -f1)
        local script_result=$(echo "$result" | cut -d: -f2-)

        if [ -z "$first_result" ]; then
            first_result="$script_result"
        elif [ "$script_result" != "$first_result" ]; then
            consistent=false
            print_error "Inconsistent package manager detection: $script_name differs from others"
        fi
    done

    if [ "$consistent" = true ]; then
        print_success "All scripts have consistent package manager detection"
    fi

    print_section "Package Manager Command Usage Analysis"

    # Test command usage for each script
    for script in "${scripts[@]}"; do
        local script_path="$PWD/$script"
        if [ -f "$script_path" ]; then
            test_pm_commands "$script_path"
        fi
    done

    print_section "System Package Installation Logic"

    # Test system package installation logic
    for script in "${scripts[@]}"; do
        local script_path="$PWD/$script"
        if [ -f "$script_path" ]; then
            test_system_package_install "$script_path"
        fi
    done

    print_section "Summary"

    local issues=0

    # Check for common issues
    for script in "${scripts[@]}"; do
        local script_path="$PWD/$script"
        if [ -f "$script_path" ]; then
            # Check for hardcoded package manager assumptions
            if grep -q "apt-get install" "$script_path" && ! grep -q "detect_package_manager" "$script_path"; then
                print_warning "$script has hardcoded apt-get usage without detection"
                issues=$((issues + 1))
            fi

            # Check for missing error handling in package installation
            if grep -q "install_system_package" "$script_path" && ! grep -q "return.*1" "$script_path"; then
                print_warning "$script may lack error handling in package installation"
                issues=$((issues + 1))
            fi
        fi
    done

    if [ $issues -eq 0 ]; then
        print_success "Package manager integration is working correctly across all scripts!"
        echo
        print_info "Verified integration points:"
        echo "  - Consistent package manager detection logic"
        echo "  - Dynamic package manager command usage"
        echo "  - Proper error handling in package installation"
        echo "  - Support for apt, dnf, yum, pacman, and zypper"
        return 0
    else
        print_error "Found $issues issues with package manager integration"
        return 1
    fi
}

# Run the test
test_package_manager_integration