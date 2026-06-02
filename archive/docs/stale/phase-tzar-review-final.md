# Tzar of Excellence -- Fresh Gate Review (Final)

**Branch:** `rocm-7.2.1-update` (3 commits: `b14cbfc`, `683b6ad`, `c6e7982`, `adda65c`, `1d716a3`)
**Scope:** 19 files changed (excluding docs/phase-tzar-review*.md)
**Reviewer:** Tzar of Excellence (independent, from-scratch review)
**Review Date:** 2026-03-26

---

## Verdict: PASS

No critical or security issues remain. All findings are informational or low-severity. The codebase is sound for merge.

---

## Verification Summary

| Check | Result |
|-------|--------|
| `bash -n` on all 9 changed shell scripts | PASS (all 9) |
| `tests/validation/test_phase2_ui_refactor.sh` | PASS (7/7) |
| `tests/validation/test_phase3_update_cli.sh` | PASS (6/6) |
| `tests/validation/test_version_consistency.sh` | PASS (7/7) |
| `cargo check --bin rusty-stack-update` | PASS (clean) |

---

## Files Reviewed (19 changed files, read in full from disk)

1. `rusty-stack/src/bin/update.rs` (59 lines)
2. `rusty-stack/src/component_status.rs` (1067 lines)
3. `rusty-stack/src/state.rs` (555 lines)
4. `rusty-stack/Cargo.toml` (27 lines)
5. `scripts/update_stack.sh` (309 lines)
6. `scripts/lib/update_helper.sh` (401 lines)
7. `scripts/lib/ui_installer_helper.sh` (228 lines)
8. `scripts/install_rocm.sh` (2497 lines)
9. `scripts/install_rocm_channel.sh` (82 lines)
10. `scripts/install_comfyui.sh` (304 lines)
11. `scripts/install_textgen.sh` (309 lines)
12. `scripts/install_vllm_studio.sh` (204 lines)
13. `tests/validation/test_phase2_ui_refactor.sh` (141 lines)
14. `tests/validation/test_phase3_update_cli.sh` (132 lines)
15. `tests/validation/test_version_consistency.sh` (149 lines)
16. `docs/MULTI_CHANNEL_GUIDE.md` (54 lines)
17. `CLAUDE.md` (153 lines)
18. `README.md` (698 lines)
19. `docs/phase-tzar-review.md` (review doc)

---

## Findings

### I1 -- Informational: `install_rocm.sh` lacks `set -euo pipefail`
- **Severity:** Informational (pre-existing, not introduced by this branch)
- **File:** `scripts/install_rocm.sh`
- **Line:** N/A (entire file)
- **Description:** The primary ROCm installer does not use `set -euo pipefail`. All other changed shell scripts (`update_stack.sh`, `update_helper.sh`, `ui_installer_helper.sh`, `install_rocm_channel.sh`, `install_comfyui.sh`, `install_textgen.sh`, `install_vllm_studio.sh`) use it. This is a pre-existing design choice likely because the script has complex error recovery paths (force purge, multi-pass retry, reboot-resume flow) that would be difficult to express under strict mode.
- **Risk:** Low. The script already has extensive manual error checking (`if [ $? -ne 0 ]` guards, `|| true` on non-critical commands).

### I2 -- Informational: `show_env` / `show_env_clean` use non-local variable assignments
- **Severity:** Informational (pre-existing, mitigated by no strict mode)
- **File:** `scripts/install_rocm.sh`
- **Lines:** 443-448, 476-481
- **Description:** These functions assign to `HSA_TOOLS_LIB`, `HSA_OVERRIDE_GFX_VERSION`, `PYTORCH_ROCM_ARCH`, `ROCM_PATH`, `PATH`, `LD_LIBRARY_PATH` without `local`. Since the file has no `set -euo pipefail`, this leaks into the caller's scope. However, these functions are only called at the top level or in `--show-env` exit paths, so the leak is harmless in practice.
- **Risk:** None in current usage.

### I3 -- Informational: `install_rocm.sh` uses `eval` for command execution
- **Severity:** Informational (pre-existing)
- **File:** `scripts/install_rocm.sh`
- **Lines:** 201, 229
- **Description:** `retry_command()` and `execute_command()` use `eval "$cmd"`. The callers always construct `$cmd` from hardcoded strings and controlled variables (not user-supplied). No injection vector exists in the current code paths.
- **Risk:** None in current usage. Would become a concern if `eval` were used with untrusted input.

### I4 -- Informational: `install_rocm.sh` has `set -euo pipefail` only inside heredoc at line 809
- **Severity:** Informational (not a bug)
- **File:** `scripts/install_rocm.sh`
- **Line:** 809
- **Description:** The string `set -euo pipefail` appears at line 809 but only inside a heredoc that generates the autostart resume launcher script. This is correct -- the generated launcher script should use strict mode.

### I5 -- Informational: `python_has_module` uses string interpolation in Python code
- **Severity:** Informational
- **File:** `rusty-stack/src/component_status.rs`
- **Lines:** 750-757
- **Description:** `python_has_module` constructs Python code with `format!` and a module name: `format!("import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('{module}') else 1)")`. The `module` parameter comes from hardcoded string literals in the source code (e.g., `"torch"`, `"triton"`), never from user input.
- **Risk:** None in current usage.

---

## Detailed Review Notes (No Issues Found)

### Rust Code (`update.rs`, `component_status.rs`, `state.rs`)

- **`update.rs`**: Clean, minimal wrapper. Correctly inherits stdin/stdout/stderr for interactive menu support. Proper error handling with exit code propagation. No issues.
- **`component_status.rs`**: Comprehensive component detection with multi-strategy fallback (version file + rocminfo functional check, Python module import, path existence, git repo detection). The dual-check for ROCm (both version file AND functional rocminfo) prevents false positives from partial installs. Python search path logic is thorough. The `resolve_component_user_home` function has a robust candidate resolution chain (env vars, SUDO_USER, passwd lookup, .mlstack detection).
- **`state.rs`**: Well-structured component definitions with correct category assignments. All UI components (comfyui, vllm-studio, textgen) correctly have `needs_sudo: false`. `textgen` is properly registered with `install_textgen.sh`.

### Shell Scripts

- **`update_stack.sh`**: Proper `set -euo pipefail`. All `local` declarations are inside function bodies. Correct use of `declare -f` guards for function existence checks. Empty `comp_ids` guarded at line 169 (E6 comment). Component ID validation at lines 284-303 (I7 comment). IFS save/restore pattern at lines 178-180 is correct for safe comma-parsing. `((idx++))` has `|| true` equivalents via arithmetic context.
- **`update_helper.sh`**: Clean library with well-documented functions. `up_detect_python_modules` (O1 optimization) and `up_get_versions_batch` (O2 optimization) correctly consolidate multiple Python module checks into single processes. The `up_user_home` function has proper fallback chain (HOME -> SUDO_USER -> USER via getent). `up_is_known_component` whitelist matches `up_update_component` dispatch table. Multi-distro installer fallback logic at lines 379-390 is correct.
- **`ui_installer_helper.sh`**: Seven functions, all present and correctly implemented. `ui_parse_common_args` uses bash 4.3+ namerefs (documented in header comment). `--dir` validation blocks system directories correctly. `ui_git_clone_or_update` properly handles stash/pop for user data preservation. `ui_create_launcher_shim` validates non-empty inputs. `ui_detect_gpu_devices` uses `grep -c ... || true` for pipefail safety. `ui_create_systemd_service` correctly generates environment lines with quoting. `ui_print_summary` uses `declare -p` guard for COMMANDS_ARRAY.
- **`install_comfyui.sh`**: Correct `declare -f` guard pattern for all 6 UI helper functions. Inline fallback for when the library is missing. System directory blocking on `--dir`. Proper ownership fix when run with sudo. `grep -c ... || true` for GPU detection. Filtered requirements exclude torch/torchvision/torchaudio/torchsde/sentencepiece correctly.
- **`install_textgen.sh`**: Same robust patterns as ComfyUI. Additional ROCm build verification (`torch.version.hip` check). Requirements filtering excludes nvidia/cuda/tensorrt/xformers/flash-attn packages correctly. Preserves user data directories (models, loras, embeddings, presets, characters, training).
- **`install_vllm_studio.sh`**: Same patterns. Correctly detects bun/npm and falls back. Has a broken logs page fix for upstream issues. Shim installation uses temp file + sudo mv pattern (safe).
- **`install_rocm_channel.sh`**: Clean channel wrapper. Proper preview channel rejection with documentation link. Correct `INSTALL_ROCM_PRESEEDED_CHOICE` mapping (1=legacy, 2=stable, 3=latest). Argument count validation.
- **`install_rocm.sh`**: (Pre-existing file, version references updated in this branch.) Version variables are consistent: `ROCM_VERSION="7.2"`, `ROCM_PKG_VER="7.2.1.70201-1"`, `ROCM_DIR_PATH="7.2.1"`. Menu text shows "ROCm 7.2.1 (Latest - Recommended)". Preview channel properly removed. Multi-pass purge logic is robust. AUR installation path is sophisticated (sudo keepalive, askpass, repo vs AUR package separation).

### Validation Tests

- **`test_phase2_ui_refactor.sh`**: 7 tests covering helper function presence, source guards, textgen component registration, CUDA filtering, and detection logic. All pass.
- **`test_phase3_update_cli.sh`**: 6 tests covering helper existence, function presence, `local`-outside-functions check, Cargo.toml binary target, and source file existence. All pass. The `local`-outside-functions check uses a correct awk pattern.
- **`test_version_consistency.sh`**: 7 tests covering menu options, version variables, outdated references, guide consistency, preview removal, channel wrapper, and package version. All pass.

### Version Consistency

All version references are consistent across files:
- Legacy: ROCm 6.4.3
- Stable: ROCm 7.1
- Latest: ROCm 7.2.1 (ROCM_VERSION="7.2", ROCM_PKG_VER="7.2.1.70201-1", ROCM_DIR_PATH="7.2.1")
- Preview (7.10.0): Properly removed with documentation links preserved

---

## Exclusions (per review instructions)

- **E1**: Custom install location detection -- deferred to future track (not flagged).
- **Unquoted ExecStart in systemd units**: systemd requires it unquoted (not flagged).
- **Shell metacharacter sanitization in launcher shim generation**: Known limitation (not flagged).
- **Pre-existing issues in `install_rocm.sh`**: `set -euo pipefail` absence, `eval` usage, non-local variable assignments in `show_env` -- all pre-existing patterns not introduced by this branch (flagged as informational only).

---

## What Was Verified as Correct

1. All 9 changed shell scripts pass `bash -n` syntax checking.
2. All 3 validation test suites pass (20/20 tests total).
3. `cargo check --bin rusty-stack-update` compiles cleanly.
4. No `local` keyword used outside function bodies in `update_stack.sh`.
5. All `grep -c` calls protected with `|| true` for pipefail safety.
6. All `declare -f` guards present for UI helper functions in all 3 installer scripts.
7. Component ID whitelist in `up_is_known_component` matches dispatch table in `up_update_component`.
8. Version variables (ROCM_VERSION, ROCM_PKG_VER, ROCM_DIR_PATH) follow the correct patch-release model.
9. System directory blocking works for `--dir` in all 3 UI installers.
10. Ownership fix (`ui_fix_ownership`) correctly uses EUID/SUDO_USER check.
11. Batch Python module detection (O1/O2 optimizations) correctly reduces subprocess overhead.
12. Rust `update.rs` wrapper correctly inherits stdio for interactive menu.
13. `component_status.rs` ROCm detection requires both version file AND functional rocminfo.
14. `state.rs` textgen component correctly registered with `needs_sudo: false`.
15. Documentation (MULTI_CHANNEL_GUIDE.md, CLAUDE.md, README.md) is consistent with code.
