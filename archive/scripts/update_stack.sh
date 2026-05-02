#!/usr/bin/env bash
# update_stack.sh - Interactive component update script for Stan's ML Stack
#
# Supports interactive menu, --all, --list, --help, and specific component IDs.
#
# Usage:
#   ./scripts/update_stack.sh              # Interactive menu
#   ./scripts/update_stack.sh --all        # Update all components
#   ./scripts/update_stack.sh --list       # List installed components
#   ./scripts/update_stack.sh pytorch vllm # Update specific components
#   ./scripts/update_stack.sh --help       # Show usage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source update helper if available
if [[ -f "$SCRIPT_DIR/lib/update_helper.sh" ]]; then
    # shellcheck source=lib/update_helper.sh
    source "$SCRIPT_DIR/lib/update_helper.sh"
else
    echo "Error: update_helper.sh not found at $SCRIPT_DIR/lib/update_helper.sh" >&2
    exit 1
fi

# Source common utilities if available
if [ -f "$SCRIPT_DIR/common_utils.sh" ]; then
    source "$SCRIPT_DIR/common_utils.sh"
fi

PYTHON_BIN="${MLSTACK_PYTHON_BIN:-python3}"

# --- Help ---
show_help() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [COMPONENT...]

Update installed ML Stack components.

Options:
  -a, --all       Update all installed components
  -l, --list      List installed components and their versions
  -h, --help      Show this help message

Components:
  $(up_detect_installed "$PYTHON_BIN" | while read -r id; do
      printf "  %-18s %s\n" "$id" "$(up_display_name "$id")"
  done 2>/dev/null || echo "  (none detected)")

Examples:
  $(basename "$0")                # Interactive menu
  $(basename "$0") --all          # Update everything
  $(basename "$0") pytorch vllm   # Update specific components
  $(basename "$0") --list         # Show what's installed
EOF
}

# --- List installed components ---
show_list() {
    local installed
    installed=$(up_detect_installed "$PYTHON_BIN")

    if [[ -z "$installed" ]]; then
        echo "No ML Stack components detected."
        return 0
    fi

    # Collect component IDs
    local -a comp_ids=()
    while IFS= read -r comp_id; do
        comp_ids+=("$comp_id")
    done <<< "$installed"

    # Batch query Python module versions in single process (O2)
    local -A ver_cache=()
    local -a py_batch=()
    for cid in "${comp_ids[@]}"; do
        case "$cid" in
            pytorch|triton|deepspeed|vllm|aiter|onnx|bitsandbytes|migraphx|flash-attn|mpi4py|wandb)
                py_batch+=("$cid")
                ;;
        esac
    done
    if [[ ${#py_batch[@]} -gt 0 ]] && declare -f up_get_versions_batch &>/dev/null; then
        while IFS='=' read -r key ver; do
            ver_cache["$key"]="$ver"
        done < <(up_get_versions_batch "$PYTHON_BIN" "${py_batch[@]}")
    fi

    printf "%-18s %-25s %s\n" "COMPONENT" "NAME" "VERSION"
    printf "%-18s %-25s %s\n" "--------" "----" "-------"

    for comp_id in "${comp_ids[@]}"; do
        local ver
        if [[ -n "${ver_cache[$comp_id]+isset}" ]]; then
            ver="${ver_cache[$comp_id]}"
        else
            ver=$(up_get_version "$comp_id" "$PYTHON_BIN")
        fi
        printf "%-18s %-25s %s\n" "$comp_id" "$(up_display_name "$comp_id")" "$ver"
    done
}

# --- Interactive menu ---
show_menu() {
    local installed
    installed=$(up_detect_installed "$PYTHON_BIN")

    if [[ -z "$installed" ]]; then
        echo "No ML Stack components detected."
        echo "Run the installer first to set up your ML stack."
        return 1
    fi

    # Collect component IDs
    local -a comp_ids=()
    while IFS= read -r comp_id; do
        comp_ids+=("$comp_id")
    done <<< "$installed"

    # Batch query Python module versions in single process (O2)
    local -A ver_cache=()
    local -a py_batch=()
    for cid in "${comp_ids[@]}"; do
        case "$cid" in
            pytorch|triton|deepspeed|vllm|aiter|onnx|bitsandbytes|migraphx|flash-attn|mpi4py|wandb)
                py_batch+=("$cid")
                ;;
        esac
    done
    if [[ ${#py_batch[@]} -gt 0 ]] && declare -f up_get_versions_batch &>/dev/null; then
        while IFS='=' read -r key ver; do
            ver_cache["$key"]="$ver"
        done < <(up_get_versions_batch "$PYTHON_BIN" "${py_batch[@]}")
    fi

    # Display menu
    echo ""
    echo "=== Stan's ML Stack - Component Update ==="
    echo ""
    printf "  #  %-18s %-25s %s\n" "ID" "NAME" "VERSION"
    printf "  -  %-18s %-25s %s\n" "--" "----" "-------"

    local idx=1
    for comp_id in "${comp_ids[@]}"; do
        local ver
        if [[ -n "${ver_cache[$comp_id]+isset}" ]]; then
            ver="${ver_cache[$comp_id]}"
        else
            ver=$(up_get_version "$comp_id" "$PYTHON_BIN")
        fi
        printf "  %d) %-18s %-25s %s\n" "$idx" "$comp_id" "$(up_display_name "$comp_id")" "$ver"
        ((idx++))
    done

    echo ""
    echo "  a) Update all"
    echo "  q) Quit"
    echo ""
    read -rp "Enter selection (comma-separated numbers, 'a' for all, 'q' to quit): " selection

    case "$selection" in
        q|Q|"")
            echo "Cancelled."
            return 0
            ;;
        a|A)
            # E6: Guard against empty comp_ids
            if [[ ${#comp_ids[@]} -eq 0 ]]; then
                echo "No components to update."
                return 1
            fi
            update_components "${comp_ids[@]}"
            ;;
        *)
            # Parse comma-separated numbers
            local -a selected=()
            local saved_ifs="$IFS"
            IFS=',' read -ra nums <<< "$selection"
            IFS="$saved_ifs"
            for num in "${nums[@]}"; do
                num=$(echo "$num" | tr -d ' ')
                if [[ "$num" =~ ^[0-9]+$ ]] && (( num >= 1 && num <= ${#comp_ids[@]} )); then
                    selected+=("${comp_ids[$((num-1))]}")
                else
                    echo "Warning: Invalid selection '$num', skipping" >&2
                fi
            done

            if [[ ${#selected[@]} -eq 0 ]]; then
                echo "No valid components selected."
                return 1
            fi

            echo ""
            echo "Selected components:"
            for comp in "${selected[@]}"; do
                echo "  - $(up_display_name "$comp")"
            done
            echo ""
            read -rp "Proceed with update? [y/N]: " confirm
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                update_components "${selected[@]}"
            else
                echo "Cancelled."
            fi
            ;;
    esac
}

# --- Update components ---
update_components() {
    local -a components=("$@")
    local -a failed=()
    local -a succeeded=()

    echo ""
    echo "=== Starting Updates ==="
    echo ""

    for comp in "${components[@]}"; do
        echo "--- Updating $(up_display_name "$comp") ---"
        if up_update_component "$comp" "$SCRIPT_DIR" "$PYTHON_BIN"; then
            succeeded+=("$comp")
        else
            failed+=("$comp")
        fi
        echo ""
    done

    echo "=== Update Summary ==="
    if [[ ${#succeeded[@]} -gt 0 ]]; then
        echo "Succeeded (${#succeeded[@]}):"
        for comp in "${succeeded[@]}"; do
            echo "  + $(up_display_name "$comp")"
        done
    fi

    if [[ ${#failed[@]} -gt 0 ]]; then
        echo "Failed (${#failed[@]}):"
        for comp in "${failed[@]}"; do
            echo "  - $(up_display_name "$comp")"
        done
        return 1
    fi

    echo "All updates completed successfully."
    return 0
}

# --- Main ---
main() {
    if [[ $# -eq 0 ]]; then
        show_menu
        return $?
    fi

    case "$1" in
        -h|--help)
            show_help
            return 0
            ;;
        -l|--list)
            show_list
            return 0
            ;;
        -a|--all)
            shift
            local installed
            installed=$(up_detect_installed "$PYTHON_BIN")
            if [[ -z "$installed" ]]; then
                echo "No ML Stack components detected."
                return 1
            fi
            # Convert newline-separated to array
            local -a all_comps=()
            while IFS= read -r comp_id; do
                all_comps+=("$comp_id")
            done <<< "$installed"
            update_components "${all_comps[@]}"
            return $?
            ;;
        *)
            # I7: Validate component IDs before dispatching
            local -a valid_comps=()
            local -a invalid_args=()
            for arg in "$@"; do
                if declare -f up_is_known_component &>/dev/null && up_is_known_component "$arg"; then
                    valid_comps+=("$arg")
                else
                    invalid_args+=("$arg")
                fi
            done
            if [[ ${#invalid_args[@]} -gt 0 ]]; then
                echo "Error: Unknown component(s): ${invalid_args[*]}" >&2
                echo "Run '$(basename "$0") --list' to see available components." >&2
                return 1
            fi
            if [[ ${#valid_comps[@]} -eq 0 ]]; then
                echo "No valid components specified." >&2
                return 1
            fi
            update_components "${valid_comps[@]}"
            return $?
            ;;
    esac
}

main "$@"
