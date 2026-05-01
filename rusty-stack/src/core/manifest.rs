//! Manifest schema, trust model, and fallback chain.
//!
//! This module implements:
//! - [`Manifest`] — the top-level manifest structure with components and metadata
//! - [`ManifestOverlay`] — a remote overlay that merges onto the baseline
//! - Trust verification — signature, schema version anti-rollback, expiry
//! - Three-tier fallback chain — fresh remote → cached remote → bundled baseline
//!
//! # Trust Model
//!
//! Every remote manifest must pass three trust checks before use:
//! 1. **Signature verification** — HMAC-SHA256 against a known key
//! 2. **Schema version** — must be ≥ `MIN_SCHEMA_VERSION` (anti-rollback)
//! 3. **Expiry** — if `expires_at` is set, must not be in the past
//!
//! # Fallback Chain
//!
//! ```text
//! fresh remote (if trust checks pass)
//!   ↓ fetch failure
//! cached remote (if valid cache exists)
//!   ↓ no cache
//! bundled baseline (always available, compiled into binary)
//! ```

use crate::core::types::{Category, ValidationTier};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Current schema version produced/consumed by this binary.
pub const CURRENT_SCHEMA_VERSION: u32 = 2;

/// Minimum supported schema version. Manifests with a lower version are rejected.
pub const MIN_SCHEMA_VERSION: u32 = 1;

/// HMAC key used for manifest signature verification.
/// In production this would be a proper public-key system.
const MANIFEST_SIGNING_KEY: &[u8] = b"rusty-stack-manifest-signing-key-v1";

// ---------------------------------------------------------------------------
// Manifest component entry
// ---------------------------------------------------------------------------

/// A single component entry within a manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManifestComponent {
    pub id: String,
    pub version: String,
    pub script: String,
    pub category: Category,
    #[serde(default = "default_validation_tier")]
    pub validation_tier: ValidationTier,
}

fn default_validation_tier() -> ValidationTier {
    ValidationTier::Candidate
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/// The top-level manifest structure.
///
/// Contains metadata (schema version, sequence, timestamps) and a list of
/// component entries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub schema_version: u32,
    /// Monotonically increasing sequence number for anti-rollback.
    #[serde(default)]
    pub sequence: u64,
    /// ISO 8601 timestamp of when the manifest was generated.
    #[serde(default)]
    pub generated_at: String,
    /// ISO 8601 timestamp after which the manifest is considered stale.
    /// `None` means the manifest never expires.
    #[serde(default)]
    pub expires_at: Option<String>,
    /// The component entries.
    pub components: Vec<ManifestComponent>,
    /// HMAC-SHA256 signature over the canonical JSON body (excludes this field).
    #[serde(default)]
    pub signature: Option<String>,
}

impl Manifest {
    /// Parse a manifest from a JSON string.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let manifest: Manifest = serde_json::from_str(json)?;
        Ok(manifest)
    }

    /// Serialize the manifest to a canonical JSON string.
    pub fn to_json(&self) -> anyhow::Result<String> {
        let json = serde_json::to_string(self)?;
        Ok(json)
    }

    /// Load the bundled baseline manifest compiled into the binary.
    pub fn load_baseline() -> anyhow::Result<Self> {
        let json = include_str!("fixtures/baseline_manifest.json");
        Self::from_json(json)
    }

    /// Compute the HMAC-SHA256 signature over the manifest body.
    ///
    /// The signature covers all fields *except* the `signature` field itself.
    /// We produce a canonical JSON representation without the signature, then
    /// compute HMAC-SHA256.
    pub fn compute_signature(&self) -> String {
        let body = self.signatureless_json();
        let mut hasher = Sha256::new();
        sha2::digest::Update::update(&mut hasher, MANIFEST_SIGNING_KEY);
        sha2::digest::Update::update(&mut hasher, body.as_bytes());
        let result = hasher.finalize();
        format!("{:x}", result)
    }

    /// Verify the manifest's signature.
    ///
    /// Returns `Ok(())` if the signature is present and valid.
    /// Returns `Err(ManifestTrustError::SignatureVerificationError)` otherwise.
    pub fn verify_signature(&self) -> Result<(), ManifestTrustError> {
        match &self.signature {
            None => Err(ManifestTrustError::SignatureVerificationError(
                "missing signature".to_string(),
            )),
            Some(sig) => {
                let expected = self.compute_signature();
                if sig == &expected {
                    Ok(())
                } else {
                    Err(ManifestTrustError::SignatureVerificationError(
                        "signature mismatch".to_string(),
                    ))
                }
            }
        }
    }

    /// Check the schema version against the minimum supported version.
    ///
    /// Returns `Ok(())` if `schema_version >= MIN_SCHEMA_VERSION`.
    pub fn verify_schema_version(&self) -> Result<(), ManifestTrustError> {
        if self.schema_version < MIN_SCHEMA_VERSION {
            Err(ManifestTrustError::SchemaVersionError {
                version: self.schema_version,
                minimum: MIN_SCHEMA_VERSION,
            })
        } else {
            Ok(())
        }
    }

    /// Check the manifest expiry.
    ///
    /// Returns `Ok(())` if the manifest has no expiry or has not expired.
    /// Returns `Err(ManifestTrustError::ManifestExpiredError)` if expired.
    pub fn verify_expiry(&self) -> Result<(), ManifestTrustError> {
        match &self.expires_at {
            None => Ok(()),
            Some(expiry_str) => {
                let expiry: DateTime<Utc> =
                    expiry_str
                        .parse()
                        .map_err(|e| ManifestTrustError::ManifestExpiredError {
                            expires_at: expiry_str.clone(),
                            reason: format!("invalid timestamp: {e}"),
                        })?;
                if Utc::now() > expiry {
                    Err(ManifestTrustError::ManifestExpiredError {
                        expires_at: expiry_str.clone(),
                        reason: "manifest has expired".to_string(),
                    })
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Run all trust checks: signature, schema version, and expiry.
    ///
    /// Returns the first error encountered, or `Ok(())` if all pass.
    pub fn verify_trust(&self) -> Result<(), ManifestTrustError> {
        self.verify_signature()?;
        self.verify_schema_version()?;
        self.verify_expiry()?;
        Ok(())
    }

    /// Produce the canonical JSON of the manifest body without the signature field.
    fn signatureless_json(&self) -> String {
        // Create a copy without the signature
        let mut copy = self.clone();
        copy.signature = None;
        serde_json::to_string(&copy).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Manifest Overlay
// ---------------------------------------------------------------------------

/// A remote overlay that merges onto a baseline manifest.
///
/// The overlay contains a subset of components. For any component `id` present
/// in both baseline and overlay, the overlay's fields replace the baseline's.
/// For any component `id` present only in the overlay, it is appended.
/// Baseline-only entries remain unchanged.
///
/// The merge is deterministic: the result depends only on the content of
/// baseline and overlay, not on ordering.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManifestOverlay {
    /// Overlay components to merge.
    pub components: Vec<ManifestComponent>,
}

impl ManifestOverlay {
    /// Parse an overlay from a JSON string.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let overlay: ManifestOverlay = serde_json::from_str(json)?;
        Ok(overlay)
    }

    /// Merge this overlay onto a baseline manifest.
    ///
    /// Returns a new [`Manifest`] with the merged component list.
    /// The baseline's metadata (schema_version, sequence, etc.) is preserved.
    pub fn merge_onto(&self, baseline: &Manifest) -> Manifest {
        let mut merged = baseline.clone();

        for overlay_comp in &self.components {
            if let Some(existing) = merged
                .components
                .iter_mut()
                .find(|c| c.id == overlay_comp.id)
            {
                // Replace fields of the existing component with overlay values
                *existing = overlay_comp.clone();
            } else {
                // Append new component
                merged.components.push(overlay_comp.clone());
            }
        }

        merged
    }
}

// ---------------------------------------------------------------------------
// Trust Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during manifest trust verification.
#[derive(Debug, Clone, PartialEq)]
pub enum ManifestTrustError {
    /// The manifest signature is missing or does not match.
    SignatureVerificationError(String),
    /// The schema version is below the minimum supported version.
    SchemaVersionError { version: u32, minimum: u32 },
    /// The manifest has expired.
    ManifestExpiredError { expires_at: String, reason: String },
}

impl std::fmt::Display for ManifestTrustError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifestTrustError::SignatureVerificationError(msg) => {
                write!(f, "signature verification failed: {msg}")
            }
            ManifestTrustError::SchemaVersionError { version, minimum } => {
                write!(
                    f,
                    "schema version {version} is below minimum supported version {minimum}"
                )
            }
            ManifestTrustError::ManifestExpiredError { expires_at, reason } => {
                write!(f, "manifest expired at {expires_at}: {reason}")
            }
        }
    }
}

impl std::error::Error for ManifestTrustError {}

// ---------------------------------------------------------------------------
// Fallback Chain
// ---------------------------------------------------------------------------

/// Source of the manifest after fallback resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestSource {
    /// Freshly fetched from remote URL.
    FreshRemote,
    /// Loaded from local cache (remote was unreachable).
    CachedRemote,
    /// Bundled with the binary (last resort).
    Baseline,
}

/// Result of manifest fallback resolution.
#[derive(Debug, Clone)]
pub struct ResolvedManifest {
    pub manifest: Manifest,
    pub source: ManifestSource,
}

/// Trait for fetching and caching manifests. Implementations provide the
/// actual remote fetch and cache read/write logic.
pub trait ManifestFetcher {
    /// Attempt to fetch a fresh remote manifest.
    /// Returns `None` if the fetch fails for any reason.
    fn fetch_remote(&self) -> Option<Manifest>;

    /// Attempt to load a cached remote manifest.
    /// Returns `None` if no valid cache exists.
    fn load_cached(&self) -> Option<Manifest>;
}

/// Resolve a manifest using the three-tier fallback chain.
///
/// 1. Try to fetch a fresh remote manifest and verify trust.
/// 2. If that fails, try the cached remote manifest.
/// 3. If that also fails, fall back to the bundled baseline.
///
/// The baseline is always available (compiled into the binary), so this
/// function never returns an error.
pub fn resolve_manifest(fetcher: &dyn ManifestFetcher) -> ResolvedManifest {
    // Tier 1: Fresh remote
    if let Some(remote) = fetcher.fetch_remote() {
        if remote.verify_trust().is_ok() {
            return ResolvedManifest {
                manifest: remote,
                source: ManifestSource::FreshRemote,
            };
        }
    }

    // Tier 2: Cached remote
    if let Some(cached) = fetcher.load_cached() {
        return ResolvedManifest {
            manifest: cached,
            source: ManifestSource::CachedRemote,
        };
    }

    // Tier 3: Bundled baseline (always succeeds)
    let baseline = Manifest::load_baseline().expect("bundled baseline must be valid");
    ResolvedManifest {
        manifest: baseline,
        source: ManifestSource::Baseline,
    }
}

// ---------------------------------------------------------------------------
// Helper: create a signed manifest for testing
// ---------------------------------------------------------------------------

/// Create a test manifest with a valid signature.
pub fn create_signed_manifest(
    schema_version: u32,
    sequence: u64,
    expires_at: Option<String>,
    components: Vec<ManifestComponent>,
) -> Manifest {
    let mut manifest = Manifest {
        schema_version,
        sequence,
        generated_at: Utc::now().to_rfc3339(),
        expires_at,
        components,
        signature: None,
    };
    manifest.signature = Some(manifest.compute_signature());
    manifest
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a simple manifest component for testing.
    fn make_component(id: &str, version: &str) -> ManifestComponent {
        ManifestComponent {
            id: id.to_string(),
            version: version.to_string(),
            script: format!("scripts/install_{id}.sh"),
            category: Category::Core,
            validation_tier: ValidationTier::Validated,
        }
    }

    // ---- VAL-CORE-005: Baseline loading ----

    #[test]
    fn test_manifest_baseline_loads_with_non_empty_components() {
        let baseline = Manifest::load_baseline().expect("baseline should load");
        assert!(
            !baseline.components.is_empty(),
            "baseline must have non-empty components"
        );
        assert_eq!(baseline.schema_version, CURRENT_SCHEMA_VERSION);
    }

    #[test]
    fn test_manifest_baseline_every_component_has_valid_fields() {
        let baseline = Manifest::load_baseline().expect("baseline should load");
        for comp in &baseline.components {
            assert!(
                !comp.id.is_empty(),
                "component id must be non-empty, got empty"
            );
            assert!(
                !comp.script.is_empty(),
                "component script must be non-empty for id '{}'",
                comp.id
            );
            // Category should be a valid variant (serde already ensured this)
            assert!(
                Category::from_label(&comp.category.to_string()).is_some(),
                "component '{}' has invalid category",
                comp.id
            );
        }
    }

    #[test]
    fn test_manifest_baseline_schema_version_matches_constant() {
        let baseline = Manifest::load_baseline().expect("baseline should load");
        assert_eq!(baseline.schema_version, CURRENT_SCHEMA_VERSION);
    }

    #[test]
    fn test_manifest_baseline_sequence_is_positive() {
        let baseline = Manifest::load_baseline().expect("baseline should load");
        assert!(baseline.sequence > 0, "baseline sequence must be positive");
    }

    // ---- VAL-CORE-006: Remote overlay merge ----

    #[test]
    fn test_manifest_overlay_merges_onto_baseline_deterministically() {
        let baseline = Manifest {
            schema_version: 2,
            sequence: 1,
            generated_at: "2026-01-01T00:00:00Z".to_string(),
            expires_at: None,
            components: vec![
                make_component("rocm", "7.2.1"),
                make_component("pytorch", "2.6.0"),
                make_component("triton", "3.2.0"),
            ],
            signature: None,
        };

        let overlay = ManifestOverlay {
            components: vec![
                // Overlapping: update pytorch version
                ManifestComponent {
                    id: "pytorch".to_string(),
                    version: "2.7.0".to_string(),
                    script: "scripts/install_pytorch_rocm.sh".to_string(),
                    category: Category::Core,
                    validation_tier: ValidationTier::Validated,
                },
                // New component
                ManifestComponent {
                    id: "vllm".to_string(),
                    version: "0.9.0".to_string(),
                    script: "scripts/install_vllm.sh".to_string(),
                    category: Category::Extension,
                    validation_tier: ValidationTier::Candidate,
                },
            ],
        };

        let merged = overlay.merge_onto(&baseline);

        // Should have 4 components: rocm, pytorch (updated), triton, vllm (new)
        assert_eq!(merged.components.len(), 4);

        // rocm unchanged
        let rocm = merged.components.iter().find(|c| c.id == "rocm").unwrap();
        assert_eq!(rocm.version, "7.2.1");

        // pytorch updated
        let pytorch = merged
            .components
            .iter()
            .find(|c| c.id == "pytorch")
            .unwrap();
        assert_eq!(pytorch.version, "2.7.0");

        // triton unchanged
        let triton = merged.components.iter().find(|c| c.id == "triton").unwrap();
        assert_eq!(triton.version, "3.2.0");

        // vllm appended
        let vllm = merged.components.iter().find(|c| c.id == "vllm").unwrap();
        assert_eq!(vllm.version, "0.9.0");
        assert_eq!(vllm.category, Category::Extension);
    }

    #[test]
    fn test_manifest_overlay_empty_overlay_preserves_baseline() {
        let baseline = Manifest {
            schema_version: 2,
            sequence: 1,
            generated_at: String::new(),
            expires_at: None,
            components: vec![make_component("rocm", "7.2.1")],
            signature: None,
        };

        let overlay = ManifestOverlay { components: vec![] };

        let merged = overlay.merge_onto(&baseline);
        assert_eq!(merged.components.len(), 1);
        assert_eq!(merged.components[0].id, "rocm");
    }

    #[test]
    fn test_manifest_overlay_preserves_baseline_metadata() {
        let baseline = Manifest {
            schema_version: 2,
            sequence: 42,
            generated_at: "2026-01-01T00:00:00Z".to_string(),
            expires_at: Some("2027-01-01T00:00:00Z".to_string()),
            components: vec![make_component("rocm", "7.2.1")],
            signature: None,
        };

        let overlay = ManifestOverlay {
            components: vec![make_component("pytorch", "2.7.0")],
        };

        let merged = overlay.merge_onto(&baseline);
        assert_eq!(merged.schema_version, 2);
        assert_eq!(merged.sequence, 42);
        assert_eq!(merged.generated_at, "2026-01-01T00:00:00Z");
        assert_eq!(merged.expires_at, Some("2027-01-01T00:00:00Z".to_string()));
    }

    // ---- VAL-CORE-007: Signature verification ----

    #[test]
    fn test_manifest_signature_accepts_valid_signature() {
        let manifest = create_signed_manifest(
            CURRENT_SCHEMA_VERSION,
            1,
            None,
            vec![make_component("rocm", "7.2.1")],
        );
        assert!(
            manifest.verify_signature().is_ok(),
            "valid signature should be accepted"
        );
    }

    #[test]
    fn test_manifest_signature_rejects_tampered_manifest() {
        let mut manifest = create_signed_manifest(
            CURRENT_SCHEMA_VERSION,
            1,
            None,
            vec![make_component("rocm", "7.2.1")],
        );

        // Tamper with the manifest content
        manifest.components[0].version = "99.99.99".to_string();

        let result = manifest.verify_signature();
        assert!(result.is_err(), "tampered manifest should be rejected");
        match result {
            Err(ManifestTrustError::SignatureVerificationError(_)) => {}
            other => panic!("expected SignatureVerificationError, got {:?}", other),
        }
    }

    #[test]
    fn test_manifest_signature_rejects_missing_signature() {
        let manifest = Manifest {
            schema_version: CURRENT_SCHEMA_VERSION,
            sequence: 1,
            generated_at: String::new(),
            expires_at: None,
            components: vec![make_component("rocm", "7.2.1")],
            signature: None,
        };

        let result = manifest.verify_signature();
        assert!(result.is_err(), "missing signature should be rejected");
        match result {
            Err(ManifestTrustError::SignatureVerificationError(msg)) => {
                assert!(msg.contains("missing"), "error should mention missing");
            }
            other => panic!("expected SignatureVerificationError, got {:?}", other),
        }
    }

    #[test]
    fn test_manifest_signature_rejects_wrong_signature() {
        let mut manifest = create_signed_manifest(
            CURRENT_SCHEMA_VERSION,
            1,
            None,
            vec![make_component("rocm", "7.2.1")],
        );
        manifest.signature = Some("badsignature".to_string());

        let result = manifest.verify_signature();
        assert!(result.is_err(), "wrong signature should be rejected");
    }

    // ---- VAL-CORE-008: Schema version anti-rollback ----

    #[test]
    fn test_manifest_schema_version_rejects_old_version() {
        let manifest = create_signed_manifest(0, 1, None, vec![make_component("rocm", "7.2.1")]);

        let result = manifest.verify_schema_version();
        assert!(result.is_err(), "schema version 0 should be rejected");
        match result {
            Err(ManifestTrustError::SchemaVersionError { version, minimum }) => {
                assert_eq!(version, 0);
                assert_eq!(minimum, MIN_SCHEMA_VERSION);
            }
            other => panic!("expected SchemaVersionError, got {:?}", other),
        }
    }

    #[test]
    fn test_manifest_schema_version_accepts_current_version() {
        let manifest = create_signed_manifest(
            CURRENT_SCHEMA_VERSION,
            1,
            None,
            vec![make_component("rocm", "7.2.1")],
        );
        assert!(
            manifest.verify_schema_version().is_ok(),
            "current schema version should be accepted"
        );
    }

    #[test]
    fn test_manifest_schema_version_accepts_minimum_version() {
        let manifest = create_signed_manifest(
            MIN_SCHEMA_VERSION,
            1,
            None,
            vec![make_component("rocm", "7.2.1")],
        );
        assert!(
            manifest.verify_schema_version().is_ok(),
            "minimum schema version should be accepted"
        );
    }

    #[test]
    fn test_manifest_schema_version_accepts_future_version() {
        let manifest = create_signed_manifest(
            CURRENT_SCHEMA_VERSION + 10,
            1,
            None,
            vec![make_component("rocm", "7.2.1")],
        );
        assert!(
            manifest.verify_schema_version().is_ok(),
            "future schema version should be accepted"
        );
    }

    // ---- VAL-CORE-009: Expiry rejection ----

    #[test]
    fn test_manifest_expiry_rejects_expired_manifest() {
        // Set expiry to 1 hour ago
        let one_hour_ago = (Utc::now() - chrono::Duration::hours(1)).to_rfc3339();
        let manifest = create_signed_manifest(
            CURRENT_SCHEMA_VERSION,
            1,
            Some(one_hour_ago),
            vec![make_component("rocm", "7.2.1")],
        );

        let result = manifest.verify_expiry();
        assert!(result.is_err(), "expired manifest should be rejected");
        match result {
            Err(ManifestTrustError::ManifestExpiredError { reason, .. }) => {
                assert!(reason.contains("expired"), "reason should mention expired");
            }
            other => panic!("expected ManifestExpiredError, got {:?}", other),
        }
    }

    #[test]
    fn test_manifest_expiry_accepts_future_expiry() {
        // Set expiry to 1 year from now
        let future = (Utc::now() + chrono::Duration::days(365)).to_rfc3339();
        let manifest = create_signed_manifest(
            CURRENT_SCHEMA_VERSION,
            1,
            Some(future),
            vec![make_component("rocm", "7.2.1")],
        );
        assert!(
            manifest.verify_expiry().is_ok(),
            "future expiry should be accepted"
        );
    }

    #[test]
    fn test_manifest_expiry_accepts_no_expiry() {
        let manifest = create_signed_manifest(
            CURRENT_SCHEMA_VERSION,
            1,
            None,
            vec![make_component("rocm", "7.2.1")],
        );
        assert!(
            manifest.verify_expiry().is_ok(),
            "no expiry should be accepted"
        );
    }

    // ---- VAL-CORE-010: Fallback chain ----

    /// A mock fetcher for testing the fallback chain.
    struct MockFetcher {
        remote: Option<Manifest>,
        cached: Option<Manifest>,
    }

    impl ManifestFetcher for MockFetcher {
        fn fetch_remote(&self) -> Option<Manifest> {
            self.remote.clone()
        }

        fn load_cached(&self) -> Option<Manifest> {
            self.cached.clone()
        }
    }

    #[test]
    fn test_manifest_fallback_returns_fresh_remote_when_available() {
        let remote = create_signed_manifest(
            CURRENT_SCHEMA_VERSION,
            10,
            None,
            vec![make_component("rocm", "7.3.0")],
        );

        let fetcher = MockFetcher {
            remote: Some(remote.clone()),
            cached: None,
        };

        let resolved = resolve_manifest(&fetcher);
        assert_eq!(resolved.source, ManifestSource::FreshRemote);
        assert_eq!(resolved.manifest.sequence, 10);
    }

    #[test]
    fn test_manifest_fallback_returns_cached_when_remote_fails() {
        let cached = create_signed_manifest(
            CURRENT_SCHEMA_VERSION,
            5,
            None,
            vec![make_component("rocm", "7.2.1")],
        );

        let fetcher = MockFetcher {
            remote: None,
            cached: Some(cached.clone()),
        };

        let resolved = resolve_manifest(&fetcher);
        assert_eq!(resolved.source, ManifestSource::CachedRemote);
        assert_eq!(resolved.manifest.sequence, 5);
    }

    #[test]
    fn test_manifest_fallback_returns_cached_when_remote_trust_fails() {
        // Remote with bad signature
        let mut bad_remote = create_signed_manifest(
            CURRENT_SCHEMA_VERSION,
            10,
            None,
            vec![make_component("rocm", "7.3.0")],
        );
        bad_remote.signature = Some("bad".to_string());

        let cached = create_signed_manifest(
            CURRENT_SCHEMA_VERSION,
            5,
            None,
            vec![make_component("rocm", "7.2.1")],
        );

        let fetcher = MockFetcher {
            remote: Some(bad_remote),
            cached: Some(cached),
        };

        let resolved = resolve_manifest(&fetcher);
        assert_eq!(resolved.source, ManifestSource::CachedRemote);
    }

    #[test]
    fn test_manifest_fallback_returns_baseline_when_both_fail() {
        let fetcher = MockFetcher {
            remote: None,
            cached: None,
        };

        let resolved = resolve_manifest(&fetcher);
        assert_eq!(resolved.source, ManifestSource::Baseline);
        assert!(
            !resolved.manifest.components.is_empty(),
            "baseline must have components"
        );
    }

    #[test]
    fn test_manifest_fallback_baseline_never_errors() {
        // Even with no remote and no cache, baseline should always work
        let fetcher = MockFetcher {
            remote: None,
            cached: None,
        };

        let resolved = resolve_manifest(&fetcher);
        assert_eq!(resolved.source, ManifestSource::Baseline);
    }

    // ---- Combined trust verification ----

    #[test]
    fn test_manifest_verify_trust_all_pass() {
        let manifest = create_signed_manifest(
            CURRENT_SCHEMA_VERSION,
            1,
            None,
            vec![make_component("rocm", "7.2.1")],
        );
        assert!(manifest.verify_trust().is_ok());
    }

    #[test]
    fn test_manifest_verify_trust_fails_on_bad_signature() {
        let mut manifest = create_signed_manifest(
            CURRENT_SCHEMA_VERSION,
            1,
            None,
            vec![make_component("rocm", "7.2.1")],
        );
        manifest.signature = Some("bad".to_string());
        assert!(manifest.verify_trust().is_err());
    }

    #[test]
    fn test_manifest_verify_trust_fails_on_old_schema() {
        let manifest = create_signed_manifest(0, 1, None, vec![make_component("rocm", "7.2.1")]);
        assert!(manifest.verify_trust().is_err());
    }

    #[test]
    fn test_manifest_verify_trust_fails_on_expired() {
        let one_hour_ago = (Utc::now() - chrono::Duration::hours(1)).to_rfc3339();
        let manifest = create_signed_manifest(
            CURRENT_SCHEMA_VERSION,
            1,
            Some(one_hour_ago),
            vec![make_component("rocm", "7.2.1")],
        );
        assert!(manifest.verify_trust().is_err());
    }

    // ---- Serde roundtrip ----

    #[test]
    fn test_manifest_serde_roundtrip() {
        let manifest = create_signed_manifest(
            CURRENT_SCHEMA_VERSION,
            42,
            Some("2027-01-01T00:00:00Z".to_string()),
            vec![
                make_component("rocm", "7.2.1"),
                ManifestComponent {
                    id: "pytorch".to_string(),
                    version: "2.6.0".to_string(),
                    script: "scripts/install_pytorch_rocm.sh".to_string(),
                    category: Category::Core,
                    validation_tier: ValidationTier::Validated,
                },
            ],
        );

        let json = manifest.to_json().unwrap();
        let back: Manifest = Manifest::from_json(&json).unwrap();
        assert_eq!(manifest, back);
    }

    #[test]
    fn test_manifest_overlay_serde_roundtrip() {
        let overlay = ManifestOverlay {
            components: vec![
                make_component("rocm", "7.3.0"),
                ManifestComponent {
                    id: "new-comp".to_string(),
                    version: "1.0.0".to_string(),
                    script: "scripts/install_new.sh".to_string(),
                    category: Category::Extension,
                    validation_tier: ValidationTier::Candidate,
                },
            ],
        };

        let json = serde_json::to_string(&overlay).unwrap();
        let back: ManifestOverlay = serde_json::from_str(&json).unwrap();
        assert_eq!(overlay, back);
    }

    #[test]
    fn test_manifest_error_display() {
        let err = ManifestTrustError::SignatureVerificationError("test".to_string());
        assert!(err.to_string().contains("test"));

        let err = ManifestTrustError::SchemaVersionError {
            version: 0,
            minimum: 1,
        };
        assert!(err.to_string().contains("0"));
        assert!(err.to_string().contains("1"));

        let err = ManifestTrustError::ManifestExpiredError {
            expires_at: "2026-01-01T00:00:00Z".to_string(),
            reason: "expired".to_string(),
        };
        assert!(err.to_string().contains("2026-01-01"));
    }
}
