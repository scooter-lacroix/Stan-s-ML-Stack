# Component Detection & Update Eligibility — Scout Findings

## Files Retrieved

1. `src/core/fixtures/baseline_manifest.json` (entire file) — **ROOT CAUSE**: comfyui/vllm-studio/textgen are `validation_tier: "Experimental"` and megatron is `"Candidate"`
2. `src/orchestrator/planner.rs` (lines 348-410) — `classify_update()` — the classification logic that gates update eligibility
3. `src/orchestrator/planner.rs` (lines 325-340) — `build_plan()` — where `--all-safe` filters
4. `src/platform/registry.rs` (lines 1-400+) — component registry, version detection, installed detection
5. `src/bin/rusty.rs` (lines 1-250) — CLI entry point, scan phase, how `installed_versions` feeds the planner
6. `src/core/manifest.rs` (entire file) — manifest schema, fallback chain, `load_baseline()`
7. `src/core/types.rs` (entire file) — `ValidationTier` enum definitions

## Root Cause Analysis

### Why only ROCm appears with `--all-safe`

There are **two independent filters** that eliminate components:

#### Filter 1: `--all-safe` strips everything non-Safe (planner.rs:337-339)

```rust
// Apply --all-safe: only keep safe items if flag is set
if options.all_safe {
    items.retain(|i| i.classification == UpdateClassification::Safe);
}
```

Only `UpdateClassification::Safe` survives. Safe requires:
- Patch-only version bump AND `ValidationTier::Validated`
- OR exact same version (reinstall)

#### Filter 2: Experimental tier → Experimental classification → hidden by default (planner.rs:358-361)

```rust
// Experimental tier always classifies as experimental
if component.validation_tier == ValidationTier::Experimental {
    return UpdateClassification::Experimental;
}
```

Even with `--include-experimental`, Experimental-classified items are **never** `Safe`, so `--all-safe` still removes them.

### Baseline Manifest Tier Assignments (the smoking gun)

From `src/core/fixtures/baseline_manifest.json`:

| Component | `validation_tier` | Effect |
|-----------|-------------------|--------|
| **rocm** | `Validated` | **Only one that can be Safe** |
| rocm-smi | `Validated` | Could be Safe, but version detection issue (see below) |
| pytorch | `Validated` | Could be Safe, but likely already matches |
| triton | `Validated` | Could be Safe |
| **comfyui** | **`Experimental`** | **Always classified Experimental → blocked by `--all-safe`** |
| **vllm-studio** | **`Experimental`** | **Always classified Experimental → blocked by `--all-safe`** |
| **textgen** | **`Experimental`** | **Always classified Experimental → blocked by `--all-safe`** |
| megatron | `Candidate` | Not Experimental, but also not Validated; minor/patch bumps → Guarded |

### Version Detection Mismatches (second contributing factor)

Even if tier issues were fixed, version comparison would still fail for these components:

#### comfyui: Git hash version detection
- **Installed version**: `v2806163f` (from `git log -1 --format="%h %s"`, `registry.rs:get_version_git()`)
- **Manifest version**: `"latest"` (from baseline_manifest.json)
- **Comparison**: `parse_version_parts("2806163f")` returns `None` → `BumpLevel::Unknown` → `Guarded`
- **Result**: Never `Safe`, even if tier were `Validated`

#### megatron: Git hash version detection
- **Installed version**: `"installed"` (git log likely failed, fallback in `registry.rs:get_version_git()`)
- **Manifest version**: `"25.1"` (non-semver — only 2 parts)
- **Comparison**: `parse_version_parts("installed")` returns `None` → `BumpLevel::Unknown` → `Guarded`

#### rocm-smi: Version string prefix contamination
- **Installed version**: `"ROCM-SMI version: 4.0.0+unknown"` (raw output from `rocm-smi --version`)
- **Manifest version**: `"4.0.0"`
- **Comparison**: `parse_version_parts("ROCM-SMI version: 4.0.0+unknown")` splits on `-`, gets `"ROCM"`, fails to parse → `None` → `BumpLevel::Unknown` → `Guarded`
- **Note**: Even if parse worked, `"4.0.0+unknown"` ≠ `"4.0.0"` → not a reinstall, so it'd need a version delta check

#### textgen / vllm-studio: Same git-based issue as comfyui
- **Installed version**: git hash string
- **Manifest version**: `"latest"`
- **Comparison**: Always `BumpLevel::Unknown`

## Key Code

### Version detection per component type (registry.rs)

```rust
// Git-based: returns "hash subject" or "installed" (registry.rs ~line 320)
fn get_version_git(info: &ComponentInfo, home: &Path) -> String {
    // git log -1 --format="%h %s" → e.g. "2806163f initial commit"
    // fallback: "installed"
}

// Command-based: raw stdout from <cmd> --version (registry.rs ~line 280)
fn get_version_command_based(info: &ComponentInfo) -> String {
    // rocm-smi --version → "ROCM-SMI version: 4.0.0+unknown"
    // Returns the first line, trimmed — no parsing/cleanup
}

// Python module: importlib __version__ (registry.rs ~line 298)
fn get_version_python_single(info: &ComponentInfo) -> String {
    // Returns module.__version__ → e.g. "2.6.0"
}
```

### Classification logic (planner.rs:348-410)

```rust
fn classify_update(&self, component: &ManifestComponent, context: &CompatibilityContext) -> UpdateClassification {
    // 1. Experimental tier → Experimental (always)
    if component.validation_tier == ValidationTier::Experimental { return Experimental; }
    // 2. Blocked tier → Blocked (always)
    if component.validation_tier == ValidationTier::Blocked { return Blocked; }
    // 3. Hardware/executor/ROCm compatibility checks
    // 4. Not installed → Candidate
    if current_version.is_empty() { return Candidate; }
    // 5. Same version → Safe (reinstall)
    if current_version == component.version { return Safe; }
    // 6. Version delta: Patch → Safe (if Validated), Minor → Guarded, Major → Candidate, Unknown → Guarded
}
```

### The `--all-safe` + `--include-experimental` interaction (planner.rs:306-339)

`--include-experimental` controls whether Experimental items are **included in the plan** (line 306):
```rust
if classification == UpdateClassification::Experimental && !options.include_experimental {
    continue; // skip entirely
}
```

But then `--all-safe` removes them (line 337):
```rust
if options.all_safe {
    items.retain(|i| i.classification == UpdateClassification::Safe);
}
```

**So `--all-safe --include-experimental` is contradictory**: Experimental items are included in the plan, then immediately stripped by `--all-safe`. The `--include-experimental` flag has no effect when combined with `--all-safe`.

## Architecture

### Data Flow for `rusty update --all-safe --include-experimental`

```
CLI args (rusty.rs)
  → run_scan() → detect_all_installed() + get_version() per component
  → build_plan()
    → Manifest::load_baseline()  (or remote + overlay)
    → CompatibilityContext { installed_versions, installed_components, ... }
    → UpdatePlanner::build_plan()
      → For each manifest component:
        → classify_update() → tier check → compat checks → version delta
        → --include-experimental: keep Experimental items
        → --all-safe: strip everything not Safe
    → Result: only Safe items remain
  → apply_plan() if not --scan-only
```

### Why ROCm Is the Only One

ROCm's path:
1. `validation_tier: "Validated"` ✓ (passes Experimental guard)
2. Path-based detection → `"/opt/rocm/.info/version"` → e.g. `"7.2.1"` ✓ (clean semver)
3. Manifest version: `"7.2.2"` (patch bump from 7.2.1)
4. `parse_version_parts("7.2.1")` = `Some([7, 2, 1])` ✓
5. `version_bump_level("7.2.1", "7.2.2")` = `BumpLevel::Patch`
6. Patch + Validated → **Safe** ✓
7. `--all-safe` keeps it ✓

## Start Here

Open `src/core/fixtures/baseline_manifest.json` — this is where the fix starts. The `validation_tier` values for comfyui/vllm-studio/textgen ("Experimental") are the primary blocker.

Then look at `src/orchestrator/planner.rs:348-410` (`classify_update`) to understand whether the fix should be:
- (A) Change the manifest tiers from `"Experimental"` to `"Candidate"` or `"Validated"`
- (B) Change the planner logic so `--all-safe --include-experimental` also accepts Experimental items
- (C) Fix version detection for git-based and command-based components so they produce parseable semver

## Recommended Fixes (Ordered by Impact)

### Fix 1: `--all-safe --include-experimental` should include Experimental items
**File**: `src/orchestrator/planner.rs:337-339`
**Problem**: The two flags are contradictory today.
**Fix**: When both flags are set, retain Safe AND Experimental:
```rust
if options.all_safe {
    items.retain(|i| matches!(i.classification, 
        UpdateClassification::Safe | UpdateClassification::Experimental));
}
```

### Fix 2: Clean version strings for command-based components
**File**: `src/platform/registry.rs:get_version_command_based()` (~line 280)
**Problem**: `rocm-smi --version` returns `"ROCM-SMI version: 4.0.0+unknown"` — no parsing.
**Fix**: Extract semver from the output string using regex like `(\d+\.\d+\.\d+)`.

### Fix 3: Git-based components should produce usable version info
**File**: `src/platform/registry.rs:get_version_git()` (~line 310)
**Problem**: Returns `"2806163f commit message"` which can't be compared to manifest `"latest"`.
**Fix**: Either (A) git-based components should use `git describe --tags` for semver, or (B) the planner should treat git-hash ≠ "latest" as "update available" with a special classification.

### Fix 4: Manifest `"latest"` is not semver
**File**: `src/core/fixtures/baseline_manifest.json`
**Problem**: comfyui/vllm-studio/textgen have `"version": "latest"` which can't be compared.
**Fix**: Either use an actual version/tag, or teach the planner to treat `"latest"` as "always eligible for update" when installed version differs.

## Remaining Questions

1. **Is `--all-safe --include-experimental` an intentional user combination?** The user seems to expect it means "update everything that's safe, including experimental components." Should Experimental be treated as a weaker safety signal when explicitly included?

2. **Should megatron's `"installed"` version be treated as "version unknown"?** The fallback `"installed"` string in `get_version_git()` masks the real issue (git log failure).

3. **What is the intended update semantics for git-based components?** These repos don't have versions — they're "latest main" installs. Should the update check be "git pull available" rather than version comparison?
