#!/usr/bin/env bash
# ui_installer_helper.sh - Shared UI installer functions
#
# Provides reusable functions for UI component installers (ComfyUI, vLLM Studio,
# text-generation-webui). All callers MUST use declare -f guards with inline
# fallbacks to ensure installers work even if this library is missing.
#
# Usage:
#   if [[ -f "$MLSTACK_SCRIPT_DIR/lib/ui_installer_helper.sh" ]]; then
#       source "$MLSTACK_SCRIPT_DIR/lib/ui_installer_helper.sh"
#   fi

# --- Function: ui_parse_common_args ---
# Parses --dry-run, --dir <path>, --force, --help using bash namerefs (4.3+).
# Sets the caller's variables by nameref.
# Usage:
#   DRY_RUN=false; INSTALL_DIR=""
#   ui_parse_common_args DRY_RUN INSTALL_DIR "$@"
ui_parse_common_args() {
    local -n _dry_run_ref="$1"
    local -n _dir_ref="$2"
    shift 2

    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                _dry_run_ref=true
                shift
                ;;
            --force)
                # Accepted for compatibility with update workflow; UI installers
                # always reinstall so --force is a no-op but should not error.
                shift
                ;;
            --dir)
                if [[ $# -lt 2 ]]; then
                    echo "Error: --dir requires a path argument" >&2
                    return 1
                fi
                # Validate path is absolute and does not target sensitive system directories
                if [[ "$2" != /* ]]; then
                    echo "Error: --dir requires an absolute path" >&2
                    return 1
                fi
                local resolved_dir
                resolved_dir="$(cd "$2" 2>/dev/null && pwd)" || resolved_dir="$2"
                # Block sensitive system paths
                case "$resolved_dir" in
                    /|/usr|/bin|/sbin|/etc|/var|/boot|/dev|/proc|/sys|/opt/rocm)
                        echo "Error: --dir targets a system directory: $resolved_dir" >&2
                        return 1
                        ;;
                esac
                _dir_ref="$resolved_dir"
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [--dry-run] [--dir <path>] [--help]"
                return 2
                ;;
            *)
                shift
                ;;
        esac
    done
}

# --- Function: ui_git_clone_or_update ---
# Clones a repo if it doesn't exist, or updates it if it does.
# Optionally stashes and restores preserve directories during git reset --hard.
# Usage:
#   ui_git_clone_or_update "$INSTALL_DIR" "$REPO_URL" "models" "input"
ui_git_clone_or_update() {
    local install_dir="$1"
    local repo_url="$2"
    shift 2
    local -a preserve_dirs=("$@")

    if [[ -d "$install_dir/.git" ]]; then
        # Ensure remote HEAD is set and fetch latest
        execute_command "git -C \"$install_dir\" remote set-head origin -a || true" "Setting remote HEAD"
        execute_command "git -C \"$install_dir\" fetch --all" "Fetching latest changes"

        # Derive target branch: prefer origin/HEAD, fall back to current branch, then "main"
        local target_branch
        target_branch=$(git -C "$install_dir" symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@') || true
        if [[ -z "$target_branch" ]]; then
            target_branch=$(git -C "$install_dir" rev-parse --abbrev-ref HEAD 2>/dev/null) || true
        fi
        target_branch="${target_branch:-main}"

        # Check for user data in preserve directories
        local has_preserve=false
        local -a dirs_to_preserve=()
        for d in "${preserve_dirs[@]}"; do
            if [[ -d "$install_dir/$d" ]] && [[ -n "$(ls -A "$install_dir/$d" 2>/dev/null)" ]]; then
                dirs_to_preserve+=("$d")
                has_preserve=true
            fi
        done

        if [[ "$has_preserve" == true ]]; then
            # Stash user data before reset
            execute_command "git -C \"$install_dir\" stash push -u -m \"rusty-stack-preserve-user-data\" -- ${dirs_to_preserve[*]}" "Stashing user data before update"
            execute_command "git -C \"$install_dir\" reset --hard \"origin/$target_branch\"" "Resetting to latest"
            execute_command "git -C \"$install_dir\" stash pop" "Restoring user data"
        else
            execute_command "git -C \"$install_dir\" reset --hard \"origin/$target_branch\"" "Resetting to latest"
        fi
    else
        execute_command "git clone \"$repo_url\" \"$install_dir\"" "Cloning repository"
    fi
}

# --- Function: ui_fix_ownership ---
# Fixes file ownership when run with sudo.
# Usage:
#   ui_fix_ownership "$INSTALL_DIR"
ui_fix_ownership() {
    local install_dir="$1"

    if [ "$EUID" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
        chown -R "$SUDO_USER:$SUDO_USER" "$install_dir"
    fi
}

# --- Function: ui_create_launcher_shim ---
# Creates an executable launcher script in ~/.local/bin/.
# Callers are responsible for providing a safe launch_command.
# Usage:
#   ui_create_launcher_shim "myapp" "$HOME/.local/bin" "/path/to/app/start.sh" "My App description"
ui_create_launcher_shim() {
    local shim_name="$1"
    local launcher_dir="$2"
    local launch_command="$3"
    local description="${4:-$shim_name}"

    # Validate inputs
    if [[ -z "$shim_name" ]] || [[ -z "$launch_command" ]]; then
        return 1
    fi

    mkdir -p "$launcher_dir" 2>/dev/null || true

    local launcher_path="$launcher_dir/$shim_name"
    cat > "$launcher_path" << EOF
#!/bin/bash
# ${description} - Rusty Stack launcher
# Auto-generated by Stan's ML Stack installer
${launch_command} "\$@"
EOF
    chmod +x "$launcher_path"
}

# --- Function: ui_create_systemd_service ---
# Creates a systemd user service file with properly quoted values.
# Usage:
#   ui_create_systemd_service "myapp" "/opt/myapp" "/opt/myapp/start.sh" "KEY=VALUE"
ui_create_systemd_service() {
    local service_name="$1"
    local install_dir="$2"
    local exec_command="$3"
    shift 3
    local -a env_vars=("$@")

    local service_dir="$HOME/.config/systemd/user"
    if [[ ! -d "$service_dir" ]]; then
        return 0
    fi

    # Build environment lines with proper quoting
    local env_lines=""
    for env_var in "${env_vars[@]}"; do
        env_lines+="Environment=\"${env_var}\""$'\n'
    done

    cat > "$service_dir/${service_name}.service" << EOF
[Unit]
Description=${service_name}
After=network.target

[Service]
Type=simple
WorkingDirectory="${install_dir}"
${env_lines}ExecStart=${exec_command}
Restart=on-failure

[Install]
WantedBy=default.target
EOF
}

# --- Function: ui_detect_gpu_devices ---
# Detects AMD GPU count via rocm-smi; returns comma-separated device list.
# Usage:
#   GPU_DEVICES=$(ui_detect_gpu_devices)
ui_detect_gpu_devices() {
    if command -v rocm-smi &>/dev/null; then
        local gpu_count
        gpu_count=$(rocm-smi --showproductname 2>/dev/null | grep -c "GPU\[" || true)
        if [[ "$gpu_count" -gt 0 ]]; then
            seq -s, 0 $((gpu_count - 1))
            return 0
        fi
    fi
    return 1
}

# --- Function: ui_print_summary ---
# Prints a formatted installation summary using COMMANDS_ARRAY.
# Usage:
#   COMMANDS_ARRAY=("cmd1 arg1" "cmd2 arg2" "web: http://localhost:8080")
#   ui_print_summary "MyApp" "$HOME/myapp"
ui_print_summary() {
    local component_name="$1"
    local install_dir="$2"

    print_section "${component_name} Installation Summary"
    print_success "${component_name} installed at: ${install_dir}"
    echo
    echo -e "${BOLD}Run Commands:${RESET}"

    if declare -p COMMANDS_ARRAY &>/dev/null; then
        for entry in "${COMMANDS_ARRAY[@]}"; do
            echo "  ${entry}"
        done
    fi
}
