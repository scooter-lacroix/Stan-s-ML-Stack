# Rusty Stack Rust Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the stack’s orchestration-critical logic into Rust shared crates in controlled waves while preserving validated installer behavior through adapters until equivalent Rust executors exist.

**Architecture:** Establish a Rust workspace inside `rusty-stack/`, extract shared types and runtime/environment logic first, then migrate detection and dispatch behavior before retiring low-risk shell paths.

**Tech Stack:** Rust workspace crates, existing shell installers, existing validation shell tests, targeted Rust integration tests.

---

### Task 1: Turn `rusty-stack/` Into The Shared Workspace Anchor

**Parallelism:** Serial only — workspace layout changes affect every later track.

**Files:**
- Modify: `rusty-stack/Cargo.toml`
- Modify: `rusty-stack/src/lib.rs`
- Test: `rusty-stack/Cargo.toml`

- [ ] **Step 1: Write the failing workspace-members test**

Create `rusty-stack/tests/workspace_members.rs`:

```rust
#[test]
fn workspace_declares_all_shared_crates() {
    let cargo_toml = std::fs::read_to_string("Cargo.toml").unwrap();
    assert!(cargo_toml.contains("crates/rusty-stack-core"));
    assert!(cargo_toml.contains("crates/rusty-stack-orchestrator"));
    assert!(cargo_toml.contains("crates/rusty-stack-platform"));
    assert!(cargo_toml.contains("crates/rusty-stack-telemetry"));
}
```

- [ ] **Step 2: Run the failing workspace test**

Run: `cd rusty-stack && cargo test workspace_declares_all_shared_crates -- --exact`
Expected: FAIL until the workspace stanza exists.

- [ ] **Step 3: Add the workspace members**

Reuse the workspace stanza from the update plan and ensure the crate paths exist on disk.

- [ ] **Step 4: Run the workspace test again**

Run: `cd rusty-stack && cargo test workspace_declares_all_shared_crates -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the workspace conversion**

```bash
git add rusty-stack/Cargo.toml rusty-stack/tests/workspace_members.rs rusty-stack/src/lib.rs
git commit -m "refactor(rust): convert rusty-stack into shared workspace"
```

### Task 2: Extract Runtime Environment Management From `installer.rs`

**Parallelism:** Serial only — runtime env state is shared by install, update, and Windows orchestration.

**Files:**
- Create: `rusty-stack/crates/rusty-stack-platform/src/runtime_env.rs`
- Test: `rusty-stack/tests/runtime_env_normalization.rs`
- Modify: `rusty-stack/src/installer.rs`

- [ ] **Step 1: Write the failing normalization test**

Create `rusty-stack/tests/runtime_env_normalization.rs`:

```rust
use rusty_stack_platform::runtime_env::normalize_env_line;

#[test]
fn normalizes_unquoted_env_assignments() {
    let (line, changed) = normalize_env_line("export MLSTACK_PYTHON_BIN=/usr/bin/python3", "MLSTACK_PYTHON_BIN", "/usr/bin/python3");
    assert!(changed);
    assert_eq!(line, "export MLSTACK_PYTHON_BIN=/usr/bin/python3");
}
```

- [ ] **Step 2: Run the failing test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml normalizes_unquoted_env_assignments -- --exact`
Expected: FAIL because `runtime_env` module does not expose the helper yet.

- [ ] **Step 3: Implement the minimal runtime helper**

Create `rusty-stack/crates/rusty-stack-platform/src/runtime_env.rs`:

```rust
pub fn normalize_env_line(line: &str, key: &str, desired: &str) -> (String, bool) {
    let marker = format!("export {}=", key);
    if let Some(rest) = line.trim_start().strip_prefix(&marker) {
        let current = rest.trim().trim_matches('"');
        if current != desired {
            return (format!("export {}={}", key, desired), true);
        }
    }
    (line.to_string(), false)
}
```

- [ ] **Step 4: Run the targeted test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml normalizes_unquoted_env_assignments -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the runtime extraction slice**

```bash
git add rusty-stack/crates/rusty-stack-platform/src/runtime_env.rs rusty-stack/tests/runtime_env_normalization.rs rusty-stack/src/installer.rs
git commit -m "refactor(rust): extract runtime env normalization helper"
```

### Task 3: Migrate Installed Detection And Version Lookup

**Parallelism:** Serial only — scanner and version lookup shape the planner contract.

**Files:**
- Create: `rusty-stack/crates/rusty-stack-platform/src/version_detect.rs`
- Test: `rusty-stack/tests/version_detect.rs`
- Modify: `scripts/lib/update_helper.sh`

- [ ] **Step 1: Write the failing version-detect test**

Create `rusty-stack/tests/version_detect.rs`:

```rust
use rusty_stack_platform::version_detect::rocm_version_from_contents;

#[test]
fn trims_rocm_version_file_contents() {
    assert_eq!(rocm_version_from_contents("7.2.1\n"), "7.2.1");
}
```

- [ ] **Step 2: Run the failing version-detect test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml trims_rocm_version_file_contents -- --exact`
Expected: FAIL because `version_detect` does not exist yet.

- [ ] **Step 3: Implement the minimal version helper**

Create `rusty-stack/crates/rusty-stack-platform/src/version_detect.rs`:

```rust
pub fn rocm_version_from_contents(contents: &str) -> String {
    contents.trim().to_string()
}
```

- [ ] **Step 4: Run the targeted test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml trims_rocm_version_file_contents -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the version-detection slice**

```bash
git add rusty-stack/crates/rusty-stack-platform/src/version_detect.rs rusty-stack/tests/version_detect.rs scripts/lib/update_helper.sh
git commit -m "refactor(rust): add version detection helper"
```

### Task 4: Add The Adapter Registry

**Parallelism:** Serial only — executor selection touches planner and apply behavior.

**Files:**
- Create: `rusty-stack/crates/rusty-stack-orchestrator/src/adapter_registry.rs`
- Test: `rusty-stack/tests/adapter_registry.rs`

- [ ] **Step 1: Write the failing adapter-registry test**

Create `rusty-stack/tests/adapter_registry.rs`:

```rust
use rusty_stack_orchestrator::adapter_registry::adapter_script_for;

#[test]
fn maps_pytorch_to_existing_shell_installer() {
    assert_eq!(adapter_script_for("pytorch"), Some("install_pytorch_rocm.sh"));
}
```

- [ ] **Step 2: Run the failing adapter-registry test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml maps_pytorch_to_existing_shell_installer -- --exact`
Expected: FAIL because the registry does not exist.

- [ ] **Step 3: Implement the minimal adapter registry**

Create `rusty-stack/crates/rusty-stack-orchestrator/src/adapter_registry.rs`:

```rust
pub fn adapter_script_for(id: &str) -> Option<&'static str> {
    match id {
        "pytorch" => Some("install_pytorch_rocm.sh"),
        "rocm" => Some("install_rocm.sh"),
        "triton" => Some("install_triton_multi.sh"),
        _ => None,
    }
}
```

- [ ] **Step 4: Run the targeted adapter-registry test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml maps_pytorch_to_existing_shell_installer -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the adapter registry**

```bash
git add rusty-stack/crates/rusty-stack-orchestrator/src/adapter_registry.rs rusty-stack/tests/adapter_registry.rs
git commit -m "refactor(rust): add legacy adapter registry"
```

### Task 5: Retire The First Low-Risk Shell Runtime Path

**Parallelism:** Parallel-eligible only after adapter registry is frozen — overlap is limited to a single component and its tests.

**Files:**
- Modify: `scripts/update_stack.sh`
- Modify: `scripts/lib/update_helper.sh`
- Modify: `tests/validation/test_phase3_update_cli.sh`

- [ ] **Step 1: Write the failing validation assertion**

Add this failing check to `tests/validation/test_phase3_update_cli.sh`:

```bash
echo "[8/8] Testing scan-only path no longer requires update_stack.sh"
if cargo run --manifest-path rusty-stack/Cargo.toml --bin rusty-stack-update -- --scan-only >/tmp/rusty-update-scan.txt 2>&1; then
    ! grep -q "update_stack.sh" /tmp/rusty-update-scan.txt
fi
```

- [ ] **Step 2: Run the validation script**

Run: `bash tests/validation/test_phase3_update_cli.sh`
Expected: FAIL until the scan path is fully Rust-owned.

- [ ] **Step 3: Remove the scan-only dependence on `scripts/update_stack.sh`**

Modify `rusty-stack/src/bin/update.rs` so `--scan-only` and pure planning actions never shell out to `update_stack.sh`.

- [ ] **Step 4: Run the validation script again**

Run: `bash tests/validation/test_phase3_update_cli.sh`
Expected: PASS.

- [ ] **Step 5: Commit the first retirement wave**

```bash
git add rusty-stack/src/bin/update.rs scripts/update_stack.sh scripts/lib/update_helper.sh tests/validation/test_phase3_update_cli.sh
git commit -m "refactor(rust): retire shell dependency for scan-only update path"
```

### Task 6: Verify Migration Wave 1

**Parallelism:** Serial only — this is the migration milestone proof.

**Files:**
- Test: `rusty-stack/Cargo.toml`
- Test: `tests/validation/test_phase3_update_cli.sh`

- [ ] **Step 1: Build the workspace**

Run: `cargo build --manifest-path rusty-stack/Cargo.toml --release`
Expected: PASS.

- [ ] **Step 2: Run the migration-focused tests**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml --test workspace_members --test runtime_env_normalization --test version_detect --test adapter_registry`
Expected: PASS.

- [ ] **Step 3: Run the validation script**

Run: `bash tests/validation/test_phase3_update_cli.sh`
Expected: PASS.

- [ ] **Step 4: Record the milestone in the handoff ledger**

Append this exact note to `docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md`:

```md
- Rust migration wave 1 completed locally with workspace, runtime-env, version-detect, and adapter-registry evidence; awaiting Tzar review.
```

- [ ] **Step 5: Commit the handoff update**

```bash
git add docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md
git commit -m "docs(handoff): record Rust migration wave 1 evidence"
```
