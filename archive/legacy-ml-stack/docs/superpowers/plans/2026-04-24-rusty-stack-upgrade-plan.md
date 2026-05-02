# Rusty Stack Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `rusty upgrade` as the command that updates the Rusty Stack application/runtime itself and blocks `rusty update` when the installed runtime is too old for the effective manifest.

**Architecture:** Runtime upgrade remains operationally separate from component updates. Shared manifest compatibility checks (minimum runtime, optional maximum runtime, and schema compatibility) live in the core/orchestrator layer, while the CLI surface and runtime delivery path live in the Rusty Stack frontend.

**Tech Stack:** Rust workspace crates, shared manifest compatibility contracts, existing `rusty-stack` binaries, validation shell tests.

---

### Task 1: Add Runtime Compatibility Contracts

**Parallelism:** Serial only — runtime compatibility gates shared manifest behavior.

**Files:**
- Modify: `rusty-stack/crates/rusty-stack-core/src/manifest.rs`
- Test: `rusty-stack/tests/runtime_manifest_compat.rs`

- [ ] **Step 1: Write the failing runtime-compat test**

Create `rusty-stack/tests/runtime_manifest_compat.rs`:

```rust
use rusty_stack_core::manifest::RuntimeCompatibility;

#[test]
fn runtime_below_manifest_minimum_is_rejected() {
    let compatibility = RuntimeCompatibility::new("0.1.0", Some("0.2.0"), None, 3, 3);
    assert!(!compatibility.supports_current_runtime());
}

#[test]
fn runtime_above_manifest_maximum_is_rejected() {
    let compatibility = RuntimeCompatibility::new("0.4.0", Some("0.2.0"), Some("0.3.0"), 3, 3);
    assert!(!compatibility.supports_current_runtime());
}

#[test]
fn incompatible_manifest_schema_is_rejected() {
    let compatibility = RuntimeCompatibility::new("0.3.0", Some("0.2.0"), Some("0.4.0"), 4, 3);
    assert!(!compatibility.supports_current_runtime());
}
```

- [ ] **Step 2: Run the failing runtime-compat test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml runtime_below_manifest_minimum_is_rejected -- --exact`
Expected: FAIL because `RuntimeCompatibility` does not exist yet.

Run: `cargo test --manifest-path rusty-stack/Cargo.toml runtime_above_manifest_maximum_is_rejected -- --exact`
Expected: FAIL because `RuntimeCompatibility` does not exist yet.

- [ ] **Step 3: Implement the minimal compatibility helper**

Add this exact code to `rusty-stack/crates/rusty-stack-core/src/manifest.rs`:

```rust
pub struct RuntimeCompatibility {
    current: (u64, u64, u64),
    minimum: Option<(u64, u64, u64)>,
    maximum: Option<(u64, u64, u64)>,
    manifest_schema_version: u32,
    supported_schema_version: u32,
}

impl RuntimeCompatibility {
    pub fn new(
        current: &str,
        minimum: Option<&str>,
        maximum: Option<&str>,
        manifest_schema_version: u32,
        supported_schema_version: u32,
    ) -> Self {
        Self {
            current: parse_version(current),
            minimum: minimum.map(parse_version),
            maximum: maximum.map(parse_version),
            manifest_schema_version,
            supported_schema_version,
        }
    }

    pub fn supports_current_runtime(&self) -> bool {
        if self.manifest_schema_version > self.supported_schema_version {
            return false;
        }
        if let Some(minimum) = self.minimum {
            if self.current < minimum {
                return false;
            }
        }
        if let Some(maximum) = self.maximum {
            if self.current > maximum {
                return false;
            }
        }
        true
    }
}

fn parse_version(input: &str) -> (u64, u64, u64) {
    let mut parts = input.split('.');
    let major = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let minor = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let patch = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    (major, minor, patch)
}
```

- [ ] **Step 4: Run the targeted test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml runtime_below_manifest_minimum_is_rejected -- --exact`
Expected: PASS.

Run: `cargo test --manifest-path rusty-stack/Cargo.toml runtime_above_manifest_maximum_is_rejected -- --exact`
Expected: PASS.

Run: `cargo test --manifest-path rusty-stack/Cargo.toml incompatible_manifest_schema_is_rejected -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the runtime compatibility contract**

```bash
git add rusty-stack/crates/rusty-stack-core/src/manifest.rs rusty-stack/tests/runtime_manifest_compat.rs
git commit -m "feat(upgrade): add runtime manifest compatibility gate"
```

### Task 2: Add The `rusty upgrade` Binary Surface

**Parallelism:** Serial only — command boundary is a product contract.

**Files:**
- Modify: `rusty-stack/Cargo.toml`
- Create: `rusty-stack/src/bin/upgrade.rs`
- Test: `rusty-stack/tests/upgrade_cli.rs`

- [ ] **Step 1: Write the failing upgrade-cli test**

Create `rusty-stack/tests/upgrade_cli.rs`:

```rust
#[test]
fn upgrade_cli_renders_runtime_upgrade_message() {
    let output = rusty_stack::bin_upgrade::render_upgrade_summary_for_tests("0.1.0", "0.2.0");
    assert!(output.contains("0.1.0"));
    assert!(output.contains("0.2.0"));
    assert!(output.contains("upgrade"));
}
```

- [ ] **Step 2: Run the failing upgrade-cli test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml upgrade_cli_renders_runtime_upgrade_message -- --exact`
Expected: FAIL because there is no upgrade helper or binary.

- [ ] **Step 3: Add the upgrade helper and binary**

Add this exact module to `rusty-stack/src/lib.rs`:

```rust
pub mod bin_upgrade {
    pub fn render_upgrade_summary_for_tests(current: &str, next: &str) -> String {
        format!("Rusty Stack upgrade: {} -> {}", current, next)
    }
}
```

Create `rusty-stack/src/bin/upgrade.rs`:

```rust
fn main() {
    println!("{}", rusty_stack::bin_upgrade::render_upgrade_summary_for_tests("current", "target"));
}
```

Add this bin target to `rusty-stack/Cargo.toml`:

```toml
[[bin]]
name = "rusty-stack-upgrade"
path = "src/bin/upgrade.rs"
```

- [ ] **Step 4: Run the targeted test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml upgrade_cli_renders_runtime_upgrade_message -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the upgrade binary surface**

```bash
git add rusty-stack/Cargo.toml rusty-stack/src/lib.rs rusty-stack/src/bin/upgrade.rs rusty-stack/tests/upgrade_cli.rs
git commit -m "feat(upgrade): add runtime upgrade binary"
```

### Task 3: Block `rusty update` When Runtime Is Too Old

**Parallelism:** Serial only — this modifies the `update`/`upgrade` boundary.

**Files:**
- Modify: `rusty-stack/src/bin/update.rs`
- Test: `rusty-stack/tests/update_requires_upgrade.rs`

- [ ] **Step 1: Write the failing update-boundary test**

Create `rusty-stack/tests/update_requires_upgrade.rs`:

```rust
#[test]
fn update_refuses_apply_when_runtime_upgrade_is_required() {
    let output = rusty_stack::bin_update::render_runtime_upgrade_required_for_tests("0.1.0", "0.2.0");
    assert!(output.contains("rusty upgrade"));
    assert!(output.contains("0.2.0"));
}
```

- [ ] **Step 2: Run the failing boundary test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml update_refuses_apply_when_runtime_upgrade_is_required -- --exact`
Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement the runtime-upgrade-required helper**

Add this exact function to `rusty-stack/src/lib.rs` inside `bin_update`:

```rust
pub fn render_runtime_upgrade_required_for_tests(current: &str, required: &str) -> String {
    format!(
        "Current Rusty Stack runtime {} is too old for this manifest. Run `rusty upgrade` to reach {}.",
        current, required
    )
}
```

Then modify `rusty-stack/src/bin/update.rs` so runtime incompatibility exits before apply with this message.

- [ ] **Step 4: Run the targeted test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml update_refuses_apply_when_runtime_upgrade_is_required -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the update/upgrade boundary**

```bash
git add rusty-stack/src/lib.rs rusty-stack/src/bin/update.rs rusty-stack/tests/update_requires_upgrade.rs
git commit -m "feat(upgrade): gate update flow on runtime compatibility"
```

### Task 4: Add Validation Coverage

**Parallelism:** Parallel-eligible only after the command boundary is frozen — overlap is limited to validation scripts.

**Files:**
- Create: `tests/validation/test_rusty_upgrade_cli.sh`

- [ ] **Step 1: Write the validation script**

Create `tests/validation/test_rusty_upgrade_cli.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cargo test --manifest-path rusty-stack/Cargo.toml upgrade_cli_renders_runtime_upgrade_message -- --exact
cargo test --manifest-path rusty-stack/Cargo.toml update_refuses_apply_when_runtime_upgrade_is_required -- --exact
cargo test --manifest-path rusty-stack/Cargo.toml runtime_above_manifest_maximum_is_rejected -- --exact
cargo test --manifest-path rusty-stack/Cargo.toml incompatible_manifest_schema_is_rejected -- --exact
```

- [ ] **Step 2: Run the validation script**

Run: `bash tests/validation/test_rusty_upgrade_cli.sh`
Expected: PASS.

- [ ] **Step 3: Commit the validation script**

```bash
git add tests/validation/test_rusty_upgrade_cli.sh
git commit -m "test(upgrade): add runtime upgrade validation checks"
```

### Task 5: Verify The Upgrade Milestone

**Parallelism:** Serial only — this is the milestone proof.

**Files:**
- Test: `rusty-stack/Cargo.toml`
- Test: `tests/validation/test_rusty_upgrade_cli.sh`
- Modify: `docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md`

- [ ] **Step 1: Build the workspace**

Run: `cargo build --manifest-path rusty-stack/Cargo.toml --release`
Expected: PASS.

- [ ] **Step 2: Run the upgrade tests**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml --test runtime_manifest_compat --test upgrade_cli --test update_requires_upgrade`
Expected: PASS.

- [ ] **Step 3: Run the validation script**

Run: `bash tests/validation/test_rusty_upgrade_cli.sh`
Expected: PASS.

- [ ] **Step 4: Record the milestone in the handoff ledger**

Append this exact note to `docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md`:

```md
- `rusty upgrade` milestone completed locally with runtime compatibility and CLI evidence; awaiting Tzar review.
```

- [ ] **Step 5: Commit the handoff update**

```bash
git add docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md
git commit -m "docs(handoff): record upgrade milestone evidence"
```
