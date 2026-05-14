# Installer Git & CWD Mapping

## Files Retrieved

1. `src/installer.rs` (lines 1828–1928) — `NativeCommand` enum and `execute_native_command`
2. `src/installer.rs` (lines 1900–1930) — `is_up_to_date_output()` and exit-code handling
3. `src/installer.rs` (lines 1990–2150) — `execute_native_command` body: CWD, sudo, streaming, error handling
4. `src/installer.rs` (lines 2260–2390) — `is_existing_git_repo()` and `git_clone_or_pull()`
5. `src/installer.rs` (lines 2390–2460) — per-component dispatch preamble
6. `src/installer.rs` (lines 2610–2680) — triton dispatch
7. `src/installer.rs` (lines 2733–2795) — ml-stack-core dispatch (raw System git clone)
8. `src/installer.rs` (lines 2800–2855) — flash_attention_ck dispatch
9. `src/installer.rs` (lines 2870–2910) — megatron dispatch (only user of `from_shell_cmd_with_dir`)
10. `src/installer.rs` (lines 2930–2990) — aiter dispatch
11. `src/installer.rs` (lines 3000–3060) — vllm-studio dispatch
12. `src/installer.rs` (lines 3060–3110) — comfyui dispatch
13. `src/installer.rs` (lines 3110–3150) — textgen dispatch
14. `src/installer.rs` (lines 3150–3210) — onnx dispatch
15. `src/installer.rs` (lines 3210–3250) — bitsandbytes dispatch
16. `src/installer.rs` (lines 3250–3280) — rocm-smi dispatch
17. `src/installers/components/triton.rs` (lines 64–71, 194–222) — ShellCommand struct (no working_dir), `build_pip_install_command`
18. `src/installers/components/aiter.rs` (lines 73–80, 211–231) — ShellCommand struct (no working_dir), `build_pip_install_command`
19. `src/installers/components/megatron.rs` (lines 88–97, 240–266) — ShellCommand struct (HAS working_dir), `build_pip_install_command` with working_dir
20. `src/installers/components/flash_attention_ck.rs` (lines 112–121, 260–362) — ShellCommand struct (HAS working_dir), cmake/make/setup_install all set working_dir
21. `src/installers/components/onnxruntime.rs` (lines 115–124, 186–237) — ShellCommand struct (HAS working_dir), git checkout/submodule/build all set working_dir
22. `src/installers/components/comfyui.rs` (lines 20–27, 253–258) — ShellCommand struct (no working_dir), pip install uses `-r <path>`
23. `src/installers/components/textgen.rs` (lines 20–27, 285–290) — ShellCommand struct (no working_dir), pip install uses `-r <path>`

---

## Key Code

### NativeCommand enum (installer.rs:1830–1844)

```rust
enum NativeCommand {
    Shell {
        program: String,
        args: Vec<String>,
        env: Vec<(String, String)>,
        working_dir: Option<PathBuf>,   // <-- Only Shell variant has CWD
    },
    Pip { program: String, args: Vec<String> },       // NO working_dir
    System { program: String, args: Vec<String> },     // NO working_dir
    Package { program: String, args: Vec<String> },    // NO working_dir
}
```

### Two constructor variants (installer.rs:1846–1874)

```rust
// Drops working_dir — sets it to None
fn from_shell_cmd(program, args, env) -> NativeCommand::Shell { working_dir: None }

// Preserves working_dir
fn from_shell_cmd_with_dir(program, args, env, working_dir) -> NativeCommand::Shell { working_dir }
```

### CWD applied in execute_native_command (installer.rs:1947–1950)

```rust
// Set working directory if specified (e.g., for pip install -e .)
if let Some(ref dir) = working_dir {
    command.current_dir(dir);
}
```

### is_up_to_date_output (installer.rs:1907–1915)

```rust
fn is_up_to_date_output(output: &str) -> bool {
    let lower = output.to_lowercase();
    lower.contains("up to date -- skipping")
        || lower.contains("up to date, skipping")
        || lower.contains("there is nothing to do")
        || lower.contains("already installed")
        || (lower.contains("warning:") && lower.contains("up to date"))
}
```

Applied at installer.rs:2117 — when `!status.success()`, combined stdout+stderr is checked against `is_up_to_date_output()`. If matched, `return Ok(())`.

### is_existing_git_repo (installer.rs:2260–2277)

```rust
fn is_existing_git_repo(target_dir: &str) -> bool {
    let path = Path::new(target_dir);
    match std::fs::symlink_metadata(target_dir) {
        Ok(meta) if meta.is_dir() => {
            path.join(".git").symlink_metadata()
                .map(|m| m.is_dir() || m.is_file())
                .unwrap_or(false)
        }
        _ => false,
    }
}
```

### git_clone_or_pull (installer.rs:2291–2378)

```rust
fn git_clone_or_pull(repo_url, target_dir, extra_clone_args, sudo_pw, sender, component_name) -> Result<()> {
    if is_existing_git_repo(target_dir) {
        // Try: git -C <dir> pull --ff-only
        // On failure: git -C <dir> fetch --all  +  git -C <dir> reset --hard origin/HEAD
    } else {
        // git clone [extra_args] <repo_url> <target_dir>
    }
}
```

---

## Architecture

### Component → git method matrix

| Component | git_clone_or_pull? | Extra args | Raw System clone? |
|-----------|-------------------|------------|-------------------|
| triton | ✅ Yes | `[]` | No |
| flash_attention_ck | ✅ Yes | `[]` | No |
| megatron | ✅ Yes | `[]` | No |
| aiter | ✅ Yes | `["--recursive"]` | No |
| vllm-studio | ✅ Yes | `[]` | No |
| comfyui | ✅ Yes | `[]` | No |
| textgen | ✅ Yes | `[]` | No |
| onnx | ✅ Yes | `["--recursive"]` | No |
| bitsandbytes | ✅ Yes | `["--recursive"]` | No |
| rocm-smi | ✅ Yes | `[]` | No |
| ml-stack-core → megatron clone | ❌ No | N/A | ✅ `NativeCommand::System` (line 2773) |

### CWD / working_dir propagation status

| Component | ShellCommand has `working_dir` field? | Dispatch uses `from_shell_cmd`? | Dispatch uses `from_shell_cmd_with_dir`? | **CWD BUG?** |
|-----------|--------------------------------------|--------------------------------|------------------------------------------|---------------|
| **megatron** | ✅ Yes | No | ✅ Yes (line 2905) | ✅ OK |
| **flash_attention_ck** | ✅ Yes | ✅ Yes (lines 2833, 2842, 2850) | No | ⚠️ **BUG** — cmake/make/setup_install all need CWD |
| **onnxruntime** | ✅ Yes | ✅ Yes (lines 3164, 3174, 3184, 3192, 3201) | No | ⚠️ **BUG** — git checkout, submodule, build all need CWD |
| **triton** | ❌ No | ✅ Yes (line 2666) | No | ⚠️ **BUG** — `pip install .` runs in wrong dir |
| **aiter** | ❌ No | ✅ Yes (line 2965) | No | ⚠️ **BUG** — `pip install .` runs in wrong dir |
| comfyui | ❌ No | ✅ Yes (line 3078) | N/A | ✅ OK (uses `-r <abs_path>`) |
| textgen | ❌ No | ✅ Yes (line 3118) | N/A | ✅ OK (uses `-r <abs_path>`) |
| vllm-studio | ❌ No | ✅ Yes (lines 3024, 3033) | N/A | ✅ OK (npm/yarn commands, no `.`) |
| bitsandbytes | ❌ No | ✅ Yes (line 3229) | N/A | ✅ OK (pypi install, no `.`) |
| rocm-smi | ❌ No | ✅ Yes (line 3260) | N/A | ✅ OK (distro package manager) |

---

## Bug Analysis

### Bug 1: flash_attention_ck — CWD dropped for cmake/make/setup_install

**Files:**
- `src/installers/components/flash_attention_ck.rs` lines 314–316 (cmake sets `working_dir: Some(install_dir.join("build"))`)
- `src/installers/components/flash_attention_ck.rs` lines 341–343 (make sets `working_dir: Some(install_dir.join("build"))`)
- `src/installers/components/flash_attention_ck.rs` lines 360–362 (setup_install sets `working_dir: Some(install_dir)`)
- `src/installer.rs` lines 2832–2851 — dispatch uses `from_shell_cmd()` which sets `working_dir: None`

**Impact:** cmake, make, and `python setup_flash_attn_amd.py install` all run in the parent process CWD instead of the flash-attention clone/build directory.

### Bug 2: onnxruntime — CWD dropped for git ops and build

**Files:**
- `src/installers/components/onnxruntime.rs` lines 219–222 (checkout sets `working_dir: Some(workdir.join("onnxruntime"))`)
- `src/installers/components/onnxruntime.rs` lines 234–237 (submodule sets `working_dir: Some(workdir.join("onnxruntime"))`)
- `src/installers/components/onnxruntime.rs` lines 357–359 (build sets `working_dir: Some(workdir.join("onnxruntime"))`)
- `src/installer.rs` lines 3163–3201 — dispatch uses `from_shell_cmd()` for all of these

**Impact:** `git checkout v1.20.1`, `git submodule update`, and `./build.sh` all run in wrong directory.

### Bug 3: triton — `pip install .` with no CWD

**Files:**
- `src/installers/components/triton.rs` lines 194–222 — `build_pip_install_command` returns args including `"."` but ShellCommand has no `working_dir` field
- `src/installer.rs` line 2666 — dispatch uses `from_shell_cmd()`

**Impact:** `pip install .` runs in the parent process CWD (likely project root), not `~/.mlstack/triton/triton/python`.

### Bug 4: aiter — `pip install .` with no CWD

**Files:**
- `src/installers/components/aiter.rs` lines 211–231 — `build_pip_install_command` returns args including `"."` but ShellCommand has no `working_dir` field
- `src/installer.rs` line 2965 — dispatch uses `from_shell_cmd()`

**Impact:** `pip install .` runs in wrong directory, not `~/.mlstack/aiter`.

### Bug 5: ml-stack-core — non-idempotent megatron clone

**Files:**
- `src/installer.rs` lines 2773–2779 — uses `NativeCommand::System` with raw `git clone`, no `git_clone_or_pull`

**Impact:** Re-running ml-stack-core install will fail if the megatron clone directory already exists.

---

## Pacman "already up to date" handling

Located at `src/installer.rs` lines 1907–1915 (`is_up_to_date_output`) and 2117–2126.

When `execute_native_command` sees a non-zero exit code, it captures combined stdout+stderr into a string, then calls `is_up_to_date_output()`. If the output matches any of these patterns (case-insensitive):
- `"up to date -- skipping"`
- `"up to date, skipping"`
- `"there is nothing to do"`
- `"already installed"`
- `"warning:"` AND `"up to date"`

Then it logs `"[native] {component} — package already up to date, treating as success"` and returns `Ok(())` instead of `Err`. This converts non-zero exit codes from pacman/yay into successes.

---

## ShellCommand struct inconsistency

There are **three variants** of `ShellCommand` across the component installers:

1. **Basic** (no `working_dir`): Used by triton, aiter, comfyui, textgen, vllm-studio, bitsandbytes, rocm-smi, migraphx_multi, migraphx_python, amdgpu_drivers, permanent_env, repair, wandb, pytorch_profiler, vllm_multi
   ```rust
   pub struct ShellCommand {
       pub program: String,
       pub args: Vec<String>,
       pub env: Vec<(String, String)>,
   }
   ```

2. **Extended** (has `working_dir`): Used by megatron, flash_attention_ck, onnxruntime
   ```rust
   pub struct ShellCommand {
       pub program: String,
       pub args: Vec<String>,
       pub env: Vec<(String, String)>,
       pub working_dir: Option<PathBuf>,
   }
   ```

3. **ml_stack** (separate ShellCommand in `ml_stack.rs`): Not examined but likely follows one of the above patterns.

---

## Start Here

Open `src/installer.rs` at line 1830 (`NativeCommand` enum) to understand the dispatch mechanism, then look at `src/installer.rs` line 2850 (flash_attention_ck) or line 2666 (triton) to see the `from_shell_cmd` calls that drop working_dir. The fix pattern is already established in megatron at line 2905 which uses `from_shell_cmd_with_dir`.
