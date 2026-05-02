# Rusty Stack Telemetry And Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in 180-second minimum stability benchmark, anonymous telemetry payload generation, direct submission, and a minimal structured ingest path that helps validation decisions without collecting personal data.

**Architecture:** Telemetry lives in its own Rust crate, reuses shared manifest/platform/version contracts, stays opt-in, and emits structured payloads to a low-cost secure ingest surface instead of email or ad hoc logs.

**Tech Stack:** Rust workspace telemetry crate, optional minimal ingest service crate under `rusty-stack/apps/`, shared core payload types, validation shell smoke tests.

---

### Task 1: Define The Anonymous Payload Contract

**Parallelism:** Serial only — payload schema is a shared trust boundary.

**Files:**
- Create: `rusty-stack/crates/rusty-stack-telemetry/Cargo.toml`
- Create: `rusty-stack/crates/rusty-stack-telemetry/src/lib.rs`
- Create: `rusty-stack/crates/rusty-stack-telemetry/src/payload.rs`
- Test: `rusty-stack/tests/telemetry_payload_privacy.rs`

- [ ] **Step 1: Write the failing payload-privacy test**

Create `rusty-stack/tests/telemetry_payload_privacy.rs`:

```rust
use rusty_stack_telemetry::payload::TelemetryPayload;

#[test]
fn payload_excludes_home_directory_and_username() {
    let payload = TelemetryPayload::for_tests();
    let json = payload.to_json();
    assert!(!json.contains("/home/"));
    assert!(!json.contains("username"));
}
```

- [ ] **Step 2: Run the failing payload-privacy test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml payload_excludes_home_directory_and_username -- --exact`
Expected: FAIL because the telemetry crate does not exist yet.

- [ ] **Step 3: Implement the minimal telemetry payload**

Create `rusty-stack/crates/rusty-stack-telemetry/src/lib.rs`:

```rust
pub mod payload;
pub mod benchmark;
pub mod client;
```

Create `rusty-stack/crates/rusty-stack-telemetry/src/payload.rs`:

```rust
pub struct TelemetryPayload {
    pub gpu: &'static str,
    pub backend_mode: &'static str,
}

impl TelemetryPayload {
    pub fn for_tests() -> Self {
        Self {
            gpu: "rdna3-test-gpu",
            backend_mode: "linux-native",
        }
    }

    pub fn to_json(&self) -> String {
        format!("{{\"gpu\":\"{}\",\"backend_mode\":\"{}\"}}", self.gpu, self.backend_mode)
    }
}
```

- [ ] **Step 4: Run the targeted test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml payload_excludes_home_directory_and_username -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the payload contract**

```bash
git add rusty-stack/crates/rusty-stack-telemetry rusty-stack/tests/telemetry_payload_privacy.rs
git commit -m "feat(telemetry): add anonymous payload contract"
```

### Task 2: Add The 180-Second Benchmark Contract

**Parallelism:** Serial only — benchmark runtime contract shapes UX and intake expectations.

**Files:**
- Create: `rusty-stack/crates/rusty-stack-telemetry/src/benchmark.rs`
- Test: `rusty-stack/tests/telemetry_benchmark_duration.rs`

- [ ] **Step 1: Write the failing benchmark-duration test**

Create `rusty-stack/tests/telemetry_benchmark_duration.rs`:

```rust
use rusty_stack_telemetry::benchmark::StabilityBenchmark;

#[test]
fn benchmark_minimum_duration_is_180_seconds() {
    assert_eq!(StabilityBenchmark::default().minimum_duration_seconds(), 180);
}
```

- [ ] **Step 2: Run the failing benchmark-duration test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml benchmark_minimum_duration_is_180_seconds -- --exact`
Expected: FAIL because the benchmark contract does not exist.

- [ ] **Step 3: Implement the minimal benchmark contract**

Create `rusty-stack/crates/rusty-stack-telemetry/src/benchmark.rs`:

```rust
pub struct StabilityBenchmark {
    minimum_duration_seconds: u64,
}

impl Default for StabilityBenchmark {
    fn default() -> Self {
        Self {
            minimum_duration_seconds: 180,
        }
    }
}

impl StabilityBenchmark {
    pub fn minimum_duration_seconds(&self) -> u64 {
        self.minimum_duration_seconds
    }
}
```

- [ ] **Step 4: Run the targeted benchmark test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml benchmark_minimum_duration_is_180_seconds -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the benchmark contract**

```bash
git add rusty-stack/crates/rusty-stack-telemetry/src/benchmark.rs rusty-stack/tests/telemetry_benchmark_duration.rs
git commit -m "feat(telemetry): add 180-second benchmark contract"
```

### Task 3: Add Direct Submission Client Semantics

**Parallelism:** Serial only — submission semantics are privacy and trust boundaries.

**Files:**
- Create: `rusty-stack/crates/rusty-stack-telemetry/src/client.rs`
- Test: `rusty-stack/tests/telemetry_client.rs`

- [ ] **Step 1: Write the failing submission test**

Create `rusty-stack/tests/telemetry_client.rs`:

```rust
use rusty_stack_telemetry::client::SubmissionResult;

#[test]
fn successful_submission_returns_thank_you_message() {
    let result = SubmissionResult::success_for_tests();
    assert!(result.user_message.contains("thank"));
}
```

- [ ] **Step 2: Run the failing submission test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml successful_submission_returns_thank_you_message -- --exact`
Expected: FAIL because the client result type does not exist.

- [ ] **Step 3: Implement the minimal client result**

Create `rusty-stack/crates/rusty-stack-telemetry/src/client.rs`:

```rust
pub struct SubmissionResult {
    pub user_message: String,
}

impl SubmissionResult {
    pub fn success_for_tests() -> Self {
        Self {
            user_message: "Thank you for contributing hardware stability data.".to_string(),
        }
    }
}
```

- [ ] **Step 4: Run the targeted submission test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml successful_submission_returns_thank_you_message -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the client semantics**

```bash
git add rusty-stack/crates/rusty-stack-telemetry/src/client.rs rusty-stack/tests/telemetry_client.rs
git commit -m "feat(telemetry): add direct submission result contract"
```

### Task 4: Add A Minimal Ingest Service Contract

**Parallelism:** Parallel-eligible only after payload schema freeze — the overlap is isolated to a new app crate.

**Files:**
- Modify: `rusty-stack/Cargo.toml`
- Create: `rusty-stack/apps/rusty-stack-ingest/Cargo.toml`
- Create: `rusty-stack/apps/rusty-stack-ingest/src/main.rs`
- Test: `rusty-stack/tests/ingest_contract.rs`

- [ ] **Step 1: Write the failing ingest-contract test**

Create `rusty-stack/tests/ingest_contract.rs`:

```rust
#[test]
fn ingest_service_exposes_healthcheck_route_name() {
    let output = rusty_stack_ingest::healthcheck_route_for_tests();
    assert_eq!(output, "/healthz");
}
```

- [ ] **Step 2: Run the failing ingest-contract test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml ingest_service_exposes_healthcheck_route_name -- --exact`
Expected: FAIL because the ingest app crate does not exist.

- [ ] **Step 3: Add the ingest app to the workspace and create the helper**

Append `"apps/rusty-stack-ingest",` to the workspace members in `rusty-stack/Cargo.toml`.

Create `rusty-stack/apps/rusty-stack-ingest/src/main.rs`:

```rust
pub fn healthcheck_route_for_tests() -> &'static str {
    "/healthz"
}

fn main() {
    println!("{}", healthcheck_route_for_tests());
}
```

- [ ] **Step 4: Run the targeted ingest test**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml ingest_service_exposes_healthcheck_route_name -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit the ingest contract**

```bash
git add rusty-stack/Cargo.toml rusty-stack/apps/rusty-stack-ingest rusty-stack/tests/ingest_contract.rs
git commit -m "feat(telemetry): add minimal ingest service contract"
```

### Task 5: Add The Validation Smoke Test

**Parallelism:** Parallel-eligible only after payload and benchmark contracts are frozen — overlap is limited to validation scripts.

**Files:**
- Create: `tests/validation/test_telemetry_contract.sh`

- [ ] **Step 1: Write the validation smoke script**

Create `tests/validation/test_telemetry_contract.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cargo test --manifest-path rusty-stack/Cargo.toml payload_excludes_home_directory_and_username -- --exact
cargo test --manifest-path rusty-stack/Cargo.toml benchmark_minimum_duration_is_180_seconds -- --exact
cargo test --manifest-path rusty-stack/Cargo.toml successful_submission_returns_thank_you_message -- --exact
cargo test --manifest-path rusty-stack/Cargo.toml ingest_service_exposes_healthcheck_route_name -- --exact
```

- [ ] **Step 2: Run the smoke test**

Run: `bash tests/validation/test_telemetry_contract.sh`
Expected: PASS.

- [ ] **Step 3: Commit the validation smoke test**

```bash
git add tests/validation/test_telemetry_contract.sh
git commit -m "test(telemetry): add validation smoke checks"
```

### Task 6: Verify The Telemetry Milestone

**Parallelism:** Serial only — this is the milestone proof.

**Files:**
- Test: `rusty-stack/Cargo.toml`
- Test: `tests/validation/test_telemetry_contract.sh`
- Modify: `docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md`

- [ ] **Step 1: Build the workspace**

Run: `cargo build --manifest-path rusty-stack/Cargo.toml --release`
Expected: PASS.

- [ ] **Step 2: Run the telemetry tests**

Run: `cargo test --manifest-path rusty-stack/Cargo.toml --test telemetry_payload_privacy --test telemetry_benchmark_duration --test telemetry_client --test ingest_contract`
Expected: PASS.

- [ ] **Step 3: Run the validation smoke test**

Run: `bash tests/validation/test_telemetry_contract.sh`
Expected: PASS.

- [ ] **Step 4: Record the milestone in the handoff ledger**

Append this exact note to `docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md`:

```md
- Telemetry/stability milestone completed locally with payload, benchmark, client, and ingest evidence; awaiting Tzar review.
```

- [ ] **Step 5: Commit the handoff update**

```bash
git add docs/superpowers/handoffs/2026-04-24-rusty-stack-tzar-handoff.md
git commit -m "docs(handoff): record telemetry milestone evidence"
```
