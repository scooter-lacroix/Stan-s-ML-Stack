# Rusty Stack Update Intelligence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the shell-wrapper update flow with a real Rust `rusty update` command that resolves manifests safely, scans installed state, produces a guardrailed selectable plan, applies eligible updates, and verifies outcomes.

**Architecture:** Introduce shared core manifest/plan types, a Rust scan-plan-apply orchestrator, and a compatibility-aware CLI frontend while keeping legacy installer scripts available behind adapters until equivalent Rust executors exist.

**Tech Stack:** Rust workspace crates under `rusty-stack/`, shell validation tests under `tests/validation/`, existing shell installers under `scripts/`, existing TUI frontend under `rusty-stack/src/`.

---

### Task 1: Create The Shared Manifest And Plan Contracts

**Parallelism:** Serial only — manifest and planner schemas are shared-core contracts.

**Files:**
- Modify: `rusty-stack/Cargo.toml`
- Modify: `rusty-stack/src/lib.rs`
- Create: `rusty-stack/crates/rusty-stack-core/Cargo.toml`
- Create: `rusty-stack/crates/rusty-stack-core/src/lib.rs`
- Create: `rusty-stack/crates/rusty-stack-core/src/component.rs`
- Create: `rusty-stack/crates/rusty-stack-core/src/manifest.rs`
- Create: `rusty-stack/crates/rusty-stack-core/src/plan.rs`
- Create: `rusty-stack/crates/rusty-stack-core/src/platform.rs`
- Test: `rusty-stack/tests/update_manifest_resolution.rs`

- [ ] **Step 1: Write the failing manifest contract test**

Create `rusty-stack/tests/update_manifest_resolution.rs` with this exact test first:

```rust
use rusty_stack_core::manifest::{EffectiveManifest, ManifestOverlay, ManifestTrustError};

#[test]
fn rejects_stale_remote_overlay() {
    let baseline = EffectiveManifest::baseline_for_tests(10);
    let stale = ManifestOverlay::signed_for_tests(9);

    let err = EffectiveManifest::merge_overlay(baseline, stale).unwrap_err();
    assert!(matches!(err, ManifestTrustError::RollbackDetected));
}
```

- [ ] **Step 2: Run the test to prove the contract is missing**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml rejects_stale_remote_overlay -- --exact`
Expected: FAIL because `rusty_stack_core` and manifest types do not exist yet.

- [ ] **Step 3: Create the new workspace members**

Add this exact workspace skeleton to `rusty-stack/Cargo.toml` above the package table:

```toml
[workspace]
members = [
  ".",
  "crates/rusty-stack-core",
  "crates/rusty-stack-orchestrator",
  "crates/rusty-stack-platform",
  "crates/rusty-stack-telemetry",
]
resolver = "2"
```

- [ ] **Step 4: Add the minimal core manifest types**

Create `rusty-stack/crates/rusty-stack-core/src/lib.rs`:

```rust
pub mod component;
pub mod manifest;
pub mod plan;
pub mod platform;
```

Create `rusty-stack/crates/rusty-stack-core/src/manifest.rs`:

```rust
#[derive(Clone, Debug)]
pub struct EffectiveManifest {
    pub sequence: u64,
}

#[derive(Clone, Debug)]
pub struct ManifestOverlay {
    pub sequence: u64,
    pub signature_valid: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ManifestTrustError {
    InvalidSignature,
    RollbackDetected,
}

impl EffectiveManifest {
    pub fn baseline_for_tests(sequence: u64) -> Self {
        Self { sequence }
    }

    pub fn merge_overlay(
        baseline: Self,
        overlay: ManifestOverlay,
    ) -> Result<Self, ManifestTrustError> {
        if !overlay.signature_valid {
            return Err(ManifestTrustError::InvalidSignature);
        }
        if overlay.sequence < baseline.sequence {
            return Err(ManifestTrustError::RollbackDetected);
        }
        Ok(Self {
            sequence: overlay.sequence,
        })
    }
}

impl ManifestOverlay {
    pub fn signed_for_tests(sequence: u64) -> Self {
        Self {
            sequence,
            signature_valid: true,
        }
    }
}
```

- [ ] **Step 5: Run the targeted test again**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml rejects_stale_remote_overlay -- --exact`
Expected: PASS.

- [ ] **Step 6: Refactor the contract names only if the test still passes**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml -p rusty-stack-core`
Expected: PASS.

- [ ] **Step 7: Commit the core contract slice**

```bash
git add rusty-stack/Cargo.toml rusty-stack/src/lib.rs rusty-stack/crates/rusty-stack-core rusty-stack/tests/update_manifest_resolution.rs
git commit -m "feat(update): add core manifest contract crate"
```

### Task 2: Expand Manifest Trust And Validation-Tier Policy Coverage

**Parallelism:** Serial only — trust and validation policy are core safety boundaries.

**Files:**
- Modify: `rusty-stack/crates/rusty-stack-core/src/manifest.rs`
- Modify: `rusty-stack/crates/rusty-stack-core/src/plan.rs`
- Test: `rusty-stack/tests/update_manifest_policy.rs`
- Test: `rusty-stack/tests/update_validation_tier_policy.rs`

- [ ] **Step 1: Write failing manifest-policy tests for schema, expiry, and fallback**

Create `rusty-stack/tests/update_manifest_policy.rs`:

```rust
use rusty_stack_core::manifest::{EffectiveManifest, ManifestOverlay, ManifestTrustError, RemoteResolutionOutcome};

#[test]
fn rejects_overlay_with_unsupported_schema() {
    let baseline = EffectiveManifest::baseline_for_tests(10);
    let overlay = ManifestOverlay::signed_for_tests(11).with_schema_version(99);
    let err = EffectiveManifest::merge_overlay(baseline, overlay).unwrap_err();
    assert!(matches!(err, ManifestTrustError::UnsupportedSchemaVersion));
}

#[test]
fn rejects_expired_overlay() {
    let baseline = EffectiveManifest::baseline_for_tests(10);
    let overlay = ManifestOverlay::signed_for_tests(11).expired_for_tests();
    let err = EffectiveManifest::merge_overlay(baseline, overlay).unwrap_err();
    assert!(matches!(err, ManifestTrustError::ExpiredOverlay));
}

#[test]
fn falls_back_to_cached_overlay_when_fresh_fetch_is_invalid() {
    let baseline = EffectiveManifest::baseline_for_tests(10);
    let cached = ManifestOverlay::signed_for_tests(11);
    let fetched = ManifestOverlay::unsigned_for_tests(12);
    let outcome = EffectiveManifest::resolve_layers_for_tests(baseline, Some(cached), Some(fetched));
    assert!(matches!(outcome, RemoteResolutionOutcome::UsedCachedOverlay));
}
```

- [ ] **Step 2: Write failing validation-tier policy tests**

Create `rusty-stack/tests/update_validation_tier_policy.rs`:

```rust
use rusty_stack_core::plan::{SelectionPolicy, ValidationTier};

#[test]
fn candidate_is_visible_but_not_preselected_or_all_safe() {
    let policy = SelectionPolicy::for_tests(ValidationTier::Candidate, false);
    assert!(policy.visible);
    assert!(!policy.preselected);
    assert!(!policy.included_in_all_safe);
}

#[test]
fn experimental_hidden_without_explicit_opt_in() {
    let policy = SelectionPolicy::for_tests(ValidationTier::Experimental, false);
    assert!(!policy.visible);
}

#[test]
fn blocked_never_actionable() {
    let policy = SelectionPolicy::for_tests(ValidationTier::Blocked, true);
    assert!(!policy.actionable);
}
```

- [ ] **Step 3: Run the failing policy tests**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml rejects_overlay_with_unsupported_schema -- --exact`
Expected: FAIL.

Run: `cargo test --manifest-path rusty-stack/Cargo.toml candidate_is_visible_but_not_preselected_or_all_safe -- --exact`
Expected: FAIL.

- [ ] **Step 4: Implement minimal trust-policy and selection-policy helpers**

Extend `rusty-stack/crates/rusty-stack-core/src/manifest.rs` with:
- schema-version compatibility checks
- expiry rejection checks
- cached-last-valid fallback resolution outcome for invalid fresh fetches

Extend `rusty-stack/crates/rusty-stack-core/src/plan.rs` with:
- `ValidationTier` enum: `Validated`, `Candidate`, `Experimental`, `Blocked`
- `SelectionPolicy` helper that enforces visibility/preselection/all-safe/actionable rules

- [ ] **Step 5: Run targeted tests and commit**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml --test update_manifest_policy --test update_validation_tier_policy`
Expected: PASS.

```bash
git add rusty-stack/crates/rusty-stack-core/src/manifest.rs rusty-stack/crates/rusty-stack-core/src/plan.rs rusty-stack/tests/update_manifest_policy.rs rusty-stack/tests/update_validation_tier_policy.rs
git commit -m "feat(update): add manifest trust and validation-tier policy coverage"
```

### Task 3: Build The Installed-State Scanner

**Parallelism:** Serial only — scanner output feeds planner semantics.

**Files:**
- Create: `rusty-stack/crates/rusty-stack-platform/Cargo.toml`
- Create: `rusty-stack/crates/rusty-stack-platform/src/lib.rs`
- Create: `rusty-stack/crates/rusty-stack-platform/src/linux_scan.rs`
- Create: `rusty-stack/crates/rusty-stack-platform/src/runtime_env.rs`
- Test: `rusty-stack/tests/update_installed_scan.rs`
- Test: `tests/validation/test_rusty_update_scan_contract.sh`

- [ ] **Step 1: Write the failing installed-scan test**

Create `rusty-stack/tests/update_installed_scan.rs`:

```rust
use rusty_stack_platform::linux_scan::InstalledScan;

#[test]
fn maps_shell_detected_components_into_structured_scan() {
    let scan = InstalledScan::from_detected_ids_for_tests(vec!["rocm", "pytorch", "comfyui"]);

    assert!(scan.has_component("rocm"));
    assert!(scan.has_component("pytorch"));
    assert!(scan.has_component("comfyui"));
    assert!(!scan.has_component("vllm"));
}
```

- [ ] **Step 2: Run the failing test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml maps_shell_detected_components_into_structured_scan -- --exact`
Expected: FAIL because `rusty_stack_platform` and `InstalledScan` do not exist yet.

- [ ] **Step 3: Implement the minimal scanner struct**

Create `rusty-stack/crates/rusty-stack-platform/src/linux_scan.rs`:

```rust
use std::collections::BTreeSet;

#[derive(Debug, Default)]
pub struct InstalledScan {
    ids: BTreeSet<String>,
}

impl InstalledScan {
    pub fn from_detected_ids_for_tests(ids: Vec<&str>) -> Self {
        Self {
            ids: ids.into_iter().map(str::to_string).collect(),
        }
    }

    pub fn has_component(&self, id: &str) -> bool {
        self.ids.contains(id)
    }
}
```

Create `rusty-stack/crates/rusty-stack-platform/src/lib.rs`:

```rust
pub mod linux_scan;
pub mod runtime_env;
```

- [ ] **Step 4: Add a lightweight validation script for the scan contract**

Create `tests/validation/test_rusty_update_scan_contract.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cargo test --manifest-path rusty-stack/Cargo.toml maps_shell_detected_components_into_structured_scan -- --exact
```

- [ ] **Step 5: Run the Rust test and the validation script**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml maps_shell_detected_components_into_structured_scan -- --exact`
Expected: PASS.

Run: `bash tests/validation/test_rusty_update_scan_contract.sh`
Expected: PASS.

- [ ] **Step 6: Commit the scanner slice**

```bash
git add rusty-stack/crates/rusty-stack-platform rusty-stack/tests/update_installed_scan.rs tests/validation/test_rusty_update_scan_contract.sh
git commit -m "feat(update): add installed-state scanner contract"
```

### Task 4: Build The Planner With Untick And Dependency Rules

**Parallelism:** Serial only — planner rules are shared behavior for CLI, TUI, and Windows.

**Files:**
- Create: `rusty-stack/crates/rusty-stack-orchestrator/Cargo.toml`
- Create: `rusty-stack/crates/rusty-stack-orchestrator/src/lib.rs`
- Create: `rusty-stack/crates/rusty-stack-orchestrator/src/planner.rs`
- Test: `rusty-stack/tests/update_planner_selection.rs`

- [ ] **Step 1: Write the failing planner-selection test**

Create `rusty-stack/tests/update_planner_selection.rs`:

```rust
use rusty_stack_orchestrator::planner::{PlannedUpdate, Planner};

#[test]
fn cannot_deselect_required_dependency_while_dependent_is_selected() {
    let mut planner = Planner::for_tests(vec![
        PlannedUpdate::required_dependency("rocm"),
        PlannedUpdate::dependent("pytorch", "rocm"),
    ]);

    let err = planner.deselect("rocm").unwrap_err();
    assert_eq!(err, "cannot deselect required dependency rocm while pytorch is selected");
}
```

- [ ] **Step 2: Run the test to prove the planner is missing**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml cannot_deselect_required_dependency_while_dependent_is_selected -- --exact`
Expected: FAIL because `rusty_stack_orchestrator` does not exist yet.

- [ ] **Step 3: Implement the minimal planner**

Create `rusty-stack/crates/rusty-stack-orchestrator/src/lib.rs`:

```rust
pub mod planner;
```

Create `rusty-stack/crates/rusty-stack-orchestrator/src/planner.rs`:

```rust
#[derive(Clone, Debug)]
pub struct PlannedUpdate {
    pub id: String,
    pub selected: bool,
    pub depends_on: Option<String>,
}

impl PlannedUpdate {
    pub fn required_dependency(id: &str) -> Self {
        Self { id: id.to_string(), selected: true, depends_on: None }
    }

    pub fn dependent(id: &str, depends_on: &str) -> Self {
        Self { id: id.to_string(), selected: true, depends_on: Some(depends_on.to_string()) }
    }
}

pub struct Planner {
    updates: Vec<PlannedUpdate>,
}

impl Planner {
    pub fn for_tests(updates: Vec<PlannedUpdate>) -> Self {
        Self { updates }
    }

    pub fn deselect(&mut self, id: &str) -> Result<(), String> {
        let selected_dependent = self.updates.iter().find(|update| {
            update.selected && update.depends_on.as_deref() == Some(id)
        });
        if let Some(dependent) = selected_dependent {
            return Err(format!(
                "cannot deselect required dependency {} while {} is selected",
                id, dependent.id
            ));
        }
        if let Some(update) = self.updates.iter_mut().find(|update| update.id == id) {
            update.selected = false;
        }
        Ok(())
    }
}
```

- [ ] **Step 4: Run the targeted planner test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml cannot_deselect_required_dependency_while_dependent_is_selected -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the planner slice**

```bash
git add rusty-stack/crates/rusty-stack-orchestrator rusty-stack/tests/update_planner_selection.rs
git commit -m "feat(update): add planner selection contract"
```

### Task 5: Replace The Wrapper Binary With A Real Rust Scan/Plan CLI

**Parallelism:** Serial only — this task binds planner contracts to the user-facing binary.

**Files:**
- Modify: `rusty-stack/src/bin/update.rs`
- Create: `rusty-stack/tests/update_cli_scan_only.rs`
- Modify: `tests/validation/test_phase3_update_cli.sh`

- [ ] **Step 1: Write the failing CLI scan-only test**

Create `rusty-stack/tests/update_cli_scan_only.rs`:

```rust
#[test]
fn scan_only_mode_prints_plan_without_shelling_to_update_stack_sh() {
    let output = rusty_stack::bin_update::render_scan_only_output_for_tests(vec!["rocm", "pytorch"]);
    assert!(output.contains("Plan:"));
    assert!(output.contains("rocm"));
    assert!(output.contains("pytorch"));
    assert!(!output.contains("update_stack.sh"));
}
```

- [ ] **Step 2: Run the failing CLI test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml scan_only_mode_prints_plan_without_shelling_to_update_stack_sh -- --exact`
Expected: FAIL because the helper function does not exist and the binary is still a wrapper.

- [ ] **Step 3: Implement a minimal render helper and use it from the binary**

Add this to `rusty-stack/src/lib.rs`:

```rust
pub mod bin_update {
    pub fn render_scan_only_output_for_tests(ids: Vec<&str>) -> String {
        let mut out = String::from("Plan:\n");
        for id in ids {
            out.push_str(" - ");
            out.push_str(id);
            out.push('\n');
        }
        out
    }
}
```

Then modify `rusty-stack/src/bin/update.rs` so `--scan-only` prints a Rust-generated plan instead of shelling out.

- [ ] **Step 4: Extend the validation shell test**

Add this assertion to `tests/validation/test_phase3_update_cli.sh`:

```bash
echo "[7/7] Testing rusty-stack-update scan-only mode"
if cargo run --manifest-path rusty-stack/Cargo.toml --bin rusty-stack-update -- --scan-only >/tmp/rusty-update-scan.txt 2>&1; then
    grep -q "Plan:" /tmp/rusty-update-scan.txt
fi
```

- [ ] **Step 5: Run the Rust test and validation script**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml scan_only_mode_prints_plan_without_shelling_to_update_stack_sh -- --exact`
Expected: PASS.

Run: `bash tests/validation/test_phase3_update_cli.sh`
Expected: PASS, including the new scan-only check.

- [ ] **Step 6: Commit the CLI slice**

```bash
git add rusty-stack/src/lib.rs rusty-stack/src/bin/update.rs rusty-stack/tests/update_cli_scan_only.rs tests/validation/test_phase3_update_cli.sh
git commit -m "feat(update): replace wrapper with Rust scan-only plan"
```

### Task 6: Add Apply And Verification Wiring

**Parallelism:** Serial only — apply semantics depend on the previous planner contracts.

**Files:**
- Create: `rusty-stack/crates/rusty-stack-orchestrator/src/apply.rs`
- Create: `rusty-stack/tests/update_apply_verification.rs`
- Modify: `rusty-stack/src/bin/update.rs`

- [ ] **Step 1: Write the failing apply-verification test**

Create `rusty-stack/tests/update_apply_verification.rs`:

```rust
use rusty_stack_orchestrator::apply::ApplySummary;

#[test]
fn verification_failure_keeps_component_failed_even_when_executor_succeeds() {
    let summary = ApplySummary::from_test_result("pytorch", true, false);
    assert_eq!(summary.final_success, false);
    assert!(summary.message.contains("verification failed"));
}
```

- [ ] **Step 2: Run the failing test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml verification_failure_keeps_component_failed_even_when_executor_succeeds -- --exact`
Expected: FAIL because `apply` and `ApplySummary` do not exist yet.

- [ ] **Step 3: Implement the minimal apply summary**

Create `rusty-stack/crates/rusty-stack-orchestrator/src/apply.rs`:

```rust
pub struct ApplySummary {
    pub final_success: bool,
    pub message: String,
}

impl ApplySummary {
    pub fn from_test_result(id: &str, executor_success: bool, verification_success: bool) -> Self {
        let final_success = executor_success && verification_success;
        let message = if final_success {
            format!("{} updated successfully", id)
        } else if !verification_success {
            format!("{} update failed because verification failed", id)
        } else {
            format!("{} update failed", id)
        };
        Self { final_success, message }
    }
}
```

- [ ] **Step 4: Run the targeted apply test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml verification_failure_keeps_component_failed_even_when_executor_succeeds -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the apply-verification slice**

```bash
git add rusty-stack/crates/rusty-stack-orchestrator/src/apply.rs rusty-stack/tests/update_apply_verification.rs rusty-stack/src/bin/update.rs
git commit -m "feat(update): add apply verification summary contract"
```

### Task 7: Verify The Update Track End-To-End

**Parallelism:** Serial only — this is the milestone completion proof.

**Files:**
- Test: `rusty-stack/Cargo.toml`
- Test: `tests/validation/test_phase3_update_cli.sh`
- Test: `tests/validation/test_rusty_update_scan_contract.sh`

- [ ] **Step 1: Run the Rust build**

Run: `cargo build --manifest-path rusty-stack/Cargo.toml --release`
Expected: PASS.

- [ ] **Step 2: Run the focused Rust tests**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml --test update_manifest_resolution --test update_manifest_policy --test update_validation_tier_policy --test update_installed_scan --test update_planner_selection --test update_cli_scan_only --test update_apply_verification`
Expected: PASS.

- [ ] **Step 3: Run the validation shell checks**

Run: `bash tests/validation/test_phase3_update_cli.sh`
Expected: PASS.

Run: `bash tests/validation/test_rusty_update_scan_contract.sh`
Expected: PASS.

- [ ] **Step 4: Update the handoff ledger before Tzar review**

Append this line to `docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md` under the relevant milestone notes:

```md
- `rusty update` track implementation completed locally; awaiting Tzar review with build/test evidence attached.
```

- [ ] **Step 5: Commit the completion evidence**

```bash
git add docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md
git commit -m "docs(handoff): record update track completion evidence"
```
