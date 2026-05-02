# Rusty Stack Windows Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first Windows-native Rusty Stack control shell that can drive shared update logic, detect WSL-backed execution requirements, expose supported services locally, and keep Linux/WSL complexity hidden from end users.

**Architecture:** A Windows app crate under the Rust workspace reuses shared Rust core/orchestrator/platform contracts and treats WSL as an internal backend mode rather than a manual user workflow.

**Tech Stack:** Rust workspace apps and crates, Windows-specific platform helpers, localhost-only service exposure, shared manifest/planner contracts.

---

### Task 1: Scaffold The Windows App Crate

**Parallelism:** Serial only — app scaffolding defines the file layout for every later Windows task.

**Files:**
- Modify: `rusty-stack/Cargo.toml`
- Create: `rusty-stack/apps/rusty-stack-windows/Cargo.toml`
- Create: `rusty-stack/apps/rusty-stack-windows/src/main.rs`
- Test: `rusty-stack/tests/windows_app_scaffold.rs`

- [ ] **Step 1: Write the failing scaffold test**

Create `rusty-stack/tests/windows_app_scaffold.rs`:

```rust
#[test]
fn workspace_declares_windows_app() {
    let cargo_toml = std::fs::read_to_string("Cargo.toml").unwrap();
    assert!(cargo_toml.contains("apps/rusty-stack-windows"));
}
```

- [ ] **Step 2: Run the failing scaffold test**

Run: `cd rusty-stack && cargo test workspace_declares_windows_app -- --exact`
Expected: FAIL because the app is not in the workspace yet.

- [ ] **Step 3: Add the app crate to the workspace**

Append this member to the workspace list in `rusty-stack/Cargo.toml`:

```toml
"apps/rusty-stack-windows",
```

Create `rusty-stack/apps/rusty-stack-windows/src/main.rs`:

```rust
fn main() {
    println!("Rusty Stack Windows shell bootstrap");
}
```

- [ ] **Step 4: Run the scaffold test again**

Run: `cd rusty-stack && cargo test workspace_declares_windows_app -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the scaffold**

```bash
git add rusty-stack/Cargo.toml rusty-stack/apps/rusty-stack-windows rusty-stack/tests/windows_app_scaffold.rs
git commit -m "feat(windows): scaffold windows control app"
```

### Task 2: Add Backend Mode Contracts

**Parallelism:** Serial only — backend mode names are shared UI/runtime contracts.

**Files:**
- Modify: `rusty-stack/crates/rusty-stack-core/src/platform.rs`
- Test: `rusty-stack/tests/windows_backend_modes.rs`

- [ ] **Step 1: Write the failing backend-mode test**

Create `rusty-stack/tests/windows_backend_modes.rs`:

```rust
use rusty_stack_core::platform::BackendMode;

#[test]
fn supports_native_and_wsl_backed_modes() {
    assert_eq!(BackendMode::WindowsNative.as_str(), "windows-native");
    assert_eq!(BackendMode::WslBackedLinux.as_str(), "wsl-backed-linux");
}
```

- [ ] **Step 2: Run the failing backend-mode test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml supports_native_and_wsl_backed_modes -- --exact`
Expected: FAIL until the enum exists.

- [ ] **Step 3: Implement the backend-mode enum**

Create or replace `rusty-stack/crates/rusty-stack-core/src/platform.rs` with:

```rust
#[derive(Debug, PartialEq, Eq)]
pub enum BackendMode {
    LinuxNative,
    WindowsNative,
    WslBackedLinux,
}

impl BackendMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LinuxNative => "linux-native",
            Self::WindowsNative => "windows-native",
            Self::WslBackedLinux => "wsl-backed-linux",
        }
    }
}
```

- [ ] **Step 4: Run the targeted test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml supports_native_and_wsl_backed_modes -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the backend-mode contract**

```bash
git add rusty-stack/crates/rusty-stack-core/src/platform.rs rusty-stack/tests/windows_backend_modes.rs
git commit -m "feat(windows): add backend mode contract"
```

### Task 3: Add WSL Detection And Localhost Exposure Rules

**Parallelism:** Serial only — exposure rules are a security boundary.

**Files:**
- Create: `rusty-stack/crates/rusty-stack-platform/src/windows.rs`
- Test: `rusty-stack/tests/windows_localhost_exposure.rs`

- [ ] **Step 1: Write the failing exposure test**

Create `rusty-stack/tests/windows_localhost_exposure.rs`:

```rust
use rusty_stack_platform::windows::LocalServiceEndpoint;

#[test]
fn windows_service_exposure_defaults_to_localhost() {
    let endpoint = LocalServiceEndpoint::new(8188);
    assert_eq!(endpoint.url(), "http://127.0.0.1:8188");
}
```

- [ ] **Step 2: Run the failing exposure test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml windows_service_exposure_defaults_to_localhost -- --exact`
Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement the localhost endpoint helper**

Create `rusty-stack/crates/rusty-stack-platform/src/windows.rs`:

```rust
pub struct LocalServiceEndpoint {
    port: u16,
}

impl LocalServiceEndpoint {
    pub fn new(port: u16) -> Self {
        Self { port }
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}
```

- [ ] **Step 4: Run the targeted test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml windows_service_exposure_defaults_to_localhost -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the exposure rule**

```bash
git add rusty-stack/crates/rusty-stack-platform/src/windows.rs rusty-stack/tests/windows_localhost_exposure.rs
git commit -m "feat(windows): add localhost-only service exposure helper"
```

### Task 4: Add The WSL Health Surface

**Parallelism:** Serial only — WSL health state is a shared runtime contract.

**Files:**
- Create: `rusty-stack/crates/rusty-stack-platform/src/wsl.rs`
- Test: `rusty-stack/tests/wsl_health.rs`

- [ ] **Step 1: Write the failing WSL-health test**

Create `rusty-stack/tests/wsl_health.rs`:

```rust
use rusty_stack_platform::wsl::WslHealth;

#[test]
fn missing_distribution_is_reported_cleanly() {
    let health = WslHealth::missing_distribution();
    assert_eq!(health.status_label(), "missing-distribution");
}
```

- [ ] **Step 2: Run the failing WSL-health test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml missing_distribution_is_reported_cleanly -- --exact`
Expected: FAIL because `wsl` helper types do not exist.

- [ ] **Step 3: Implement the minimal WSL health struct**

Create `rusty-stack/crates/rusty-stack-platform/src/wsl.rs`:

```rust
pub enum WslState {
    Healthy,
    MissingDistribution,
}

pub struct WslHealth {
    state: WslState,
}

impl WslHealth {
    pub fn missing_distribution() -> Self {
        Self {
            state: WslState::MissingDistribution,
        }
    }

    pub fn status_label(&self) -> &'static str {
        match self.state {
            WslState::Healthy => "healthy",
            WslState::MissingDistribution => "missing-distribution",
        }
    }
}
```

- [ ] **Step 4: Run the targeted test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml missing_distribution_is_reported_cleanly -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the WSL health slice**

```bash
git add rusty-stack/crates/rusty-stack-platform/src/wsl.rs rusty-stack/tests/wsl_health.rs
git commit -m "feat(windows): add WSL health contract"
```

### Task 5: Integrate The Windows Shell With Shared Update Planning

**Parallelism:** Parallel-eligible only after backend-mode and WSL-health contracts are frozen — file overlap is limited to the Windows app crate.

**Files:**
- Modify: `rusty-stack/apps/rusty-stack-windows/src/main.rs`
- Test: `rusty-stack/tests/windows_app_plan_view.rs`

- [ ] **Step 1: Write the failing plan-view test**

Create `rusty-stack/tests/windows_app_plan_view.rs`:

```rust
#[test]
fn windows_shell_can_render_a_plan_summary() {
    let summary = rusty_stack_windows::render_plan_summary_for_tests("2 validated updates");
    assert!(summary.contains("2 validated updates"));
}
```

- [ ] **Step 2: Run the failing plan-view test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml windows_shell_can_render_a_plan_summary -- --exact`
Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement the minimal Windows shell renderer**

Replace `rusty-stack/apps/rusty-stack-windows/src/main.rs` with:

```rust
pub fn render_plan_summary_for_tests(summary: &str) -> String {
    format!("Rusty Stack Windows\n{}", summary)
}

fn main() {
    println!("{}", render_plan_summary_for_tests("No plan loaded"));
}
```

- [ ] **Step 4: Run the targeted test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml windows_shell_can_render_a_plan_summary -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the Windows plan-view slice**

```bash
git add rusty-stack/apps/rusty-stack-windows/src/main.rs rusty-stack/tests/windows_app_plan_view.rs
git commit -m "feat(windows): render shared update plan summary"
```

### Task 6: Verify The Windows Foundation Milestone

**Parallelism:** Serial only — this is the milestone proof.

**Files:**
- Test: `rusty-stack/Cargo.toml`
- Test: `rusty-stack/tests/windows_app_scaffold.rs`
- Test: `rusty-stack/tests/windows_backend_modes.rs`
- Test: `rusty-stack/tests/windows_localhost_exposure.rs`
- Test: `rusty-stack/tests/wsl_health.rs`
- Test: `rusty-stack/tests/windows_app_plan_view.rs`

- [ ] **Step 1: Build the workspace**

Run: `cargo build --manifest-path rusty-stack/Cargo.toml --release`
Expected: PASS.

- [ ] **Step 2: Run the Windows foundation tests**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml --test windows_app_scaffold --test windows_backend_modes --test windows_localhost_exposure --test wsl_health --test windows_app_plan_view`
Expected: PASS.

- [ ] **Step 3: Record the milestone in the handoff ledger**

Append this exact note to `docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md`:

```md
- Windows foundation milestone completed locally with scaffold, backend-mode, localhost exposure, WSL health, and plan-view evidence; awaiting Tzar review.
```

- [ ] **Step 4: Commit the handoff update**

```bash
git add docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md
git commit -m "docs(handoff): record windows foundation evidence"
```
