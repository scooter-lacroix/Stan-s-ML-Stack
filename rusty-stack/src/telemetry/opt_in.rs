//! Opt-in gate for telemetry collection and submission.
//!
//! Telemetry is **disabled by default** (VAL-TELE-010). The user must
//! explicitly enable it via config file or CLI toggle.
//!
//! # Immediate Effect (VAL-TELE-011)
//!
//! Changes to the opt-in setting take effect immediately without requiring
//! a restart. The setting is persisted to the config file.
//!
//! # No Promotional Language (VAL-TELE-012)
//!
//! UI strings in this module contain no promotional language, upgrade nudges,
//! or paid tier links.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Config file name for telemetry settings.
const TELEMETRY_CONFIG_FILENAME: &str = "telemetry.json";

/// Directory name under .mlstack for config.
const MLSTACK_CONFIG_DIR: &str = ".mlstack/config";

/// Telemetry configuration persisted to disk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TelemetryConfig {
    /// Whether telemetry is enabled. Defaults to false.
    pub enabled: bool,
    /// Timestamp of when telemetry was last toggled.
    #[serde(default)]
    pub last_toggled_at: Option<String>,
}

/// Telemetry opt-in gate. Manages the enabled/disabled state.
///
/// The gate reads from and writes to a config file. Changes are immediate.
#[derive(Debug, Clone)]
pub struct OptInGate {
    config: TelemetryConfig,
    config_path: PathBuf,
}

impl OptInGate {
    /// Create a new opt-in gate with the default config directory.
    ///
    /// Looks for config at `~/.mlstack/config/telemetry.json`.
    /// If not found, defaults to disabled.
    pub fn new() -> Result<Self> {
        let config_path = Self::default_config_path()?;
        let config = Self::load_config(&config_path).unwrap_or_default();
        Ok(Self {
            config,
            config_path,
        })
    }

    /// Create a new opt-in gate with a custom config path (for testing).
    pub fn with_config_path(config_path: PathBuf) -> Self {
        let config = Self::load_config(&config_path).unwrap_or_default();
        Self {
            config,
            config_path,
        }
    }

    /// Check if telemetry is currently enabled.
    ///
    /// VAL-TELE-010: Returns false by default.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Enable telemetry.
    ///
    /// VAL-TELE-011: Takes effect immediately, no restart required.
    pub fn enable(&mut self) -> Result<()> {
        self.config.enabled = true;
        self.config.last_toggled_at = Some(chrono_now_rfc3339());
        self.persist()
    }

    /// Disable telemetry.
    ///
    /// VAL-TELE-011: Takes effect immediately, no restart required.
    pub fn disable(&mut self) -> Result<()> {
        self.config.enabled = false;
        self.config.last_toggled_at = Some(chrono_now_rfc3339());
        self.persist()
    }

    /// Toggle telemetry on/off. Returns the new state.
    pub fn toggle(&mut self) -> Result<bool> {
        if self.config.enabled {
            self.disable()?;
        } else {
            self.enable()?;
        }
        Ok(self.config.enabled)
    }

    /// Get the current config.
    pub fn config(&self) -> &TelemetryConfig {
        &self.config
    }

    /// Get the config file path.
    pub fn config_path(&self) -> &PathBuf {
        &self.config_path
    }

    /// Reload config from disk (for detecting external changes).
    pub fn reload(&mut self) -> Result<()> {
        self.config = Self::load_config(&self.config_path).unwrap_or_default();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn default_config_path() -> Result<PathBuf> {
        let home = dirs::home_dir().context("Cannot determine home directory")?;
        Ok(home
            .join(MLSTACK_CONFIG_DIR)
            .join(TELEMETRY_CONFIG_FILENAME))
    }

    fn load_config(path: &PathBuf) -> Result<TelemetryConfig> {
        let content = fs::read_to_string(path).context("Config file not found")?;
        let config: TelemetryConfig =
            serde_json::from_str(&content).context("Invalid telemetry config")?;
        Ok(config)
    }

    fn persist(&self) -> Result<()> {
        if let Some(parent) = self.config_path.parent() {
            fs::create_dir_all(parent).context("Failed to create config directory")?;
        }
        let json = serde_json::to_string_pretty(&self.config)?;
        fs::write(&self.config_path, json).context("Failed to write telemetry config")?;
        Ok(())
    }
}

/// Get the current time as an RFC 3339 string.
fn chrono_now_rfc3339() -> String {
    chrono::Local::now().to_rfc3339()
}

// ---------------------------------------------------------------------------
// UI Strings (VAL-TELE-012: No promotional language)
// ---------------------------------------------------------------------------

/// Description of telemetry for user-facing display.
/// No promotional language, upgrade nudges, or paid tier links.
pub const TELEMETRY_DESCRIPTION: &str =
    "Share anonymous hardware metrics to help improve GPU support.";

/// Label for the enable toggle.
pub const TELEMETRY_ENABLE_LABEL: &str = "Enable anonymous telemetry";

/// Label for the disable toggle.
pub const TELEMETRY_DISABLE_LABEL: &str = "Disable telemetry";

/// Confirmation message when telemetry is enabled.
pub const TELEMETRY_ENABLED_CONFIRMATION: &str = "Telemetry enabled. Thank you for contributing.";

/// Confirmation message when telemetry is disabled.
pub const TELEMETRY_DISABLED_CONFIRMATION: &str = "Telemetry disabled.";

/// Status text when telemetry is enabled.
pub const TELEMETRY_STATUS_ENABLED: &str = "Telemetry: enabled";

/// Status text when telemetry is disabled.
pub const TELEMETRY_STATUS_DISABLED: &str = "Telemetry: disabled";

/// What data is collected (displayed to user).
pub const TELEMETRY_DATA_DESCRIPTION: &str =
    "Collected data: GPU utilization, VRAM usage, temperature, clock speed, throttling status.";

/// Privacy note for user display.
pub const TELEMETRY_PRIVACY_NOTE: &str =
    "No personal data is collected. Payloads contain only hardware metrics.";

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // =======================================================================
    // VAL-TELE-010: Opt-In Gate — Default Off
    // =======================================================================

    #[test]
    fn test_telemetry_disabled_by_default() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("telemetry.json");
        let gate = OptInGate::with_config_path(config_path);
        assert!(
            !gate.is_enabled(),
            "Telemetry should be disabled by default"
        );
    }

    #[test]
    fn test_telemetry_config_default_disabled() {
        let config = TelemetryConfig::default();
        assert!(!config.enabled, "Default config should have enabled=false");
    }

    #[test]
    fn test_fresh_config_has_no_file() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("telemetry.json");
        // File doesn't exist — should default to disabled
        let gate = OptInGate::with_config_path(config_path);
        assert!(!gate.is_enabled());
    }

    // =======================================================================
    // VAL-TELE-011: Opt-In Gate — Explicit Toggle
    // =======================================================================

    #[test]
    fn test_enable_takes_effect_immediately() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("telemetry.json");
        let mut gate = OptInGate::with_config_path(config_path);

        assert!(!gate.is_enabled());
        gate.enable().unwrap();
        assert!(gate.is_enabled(), "Enable should take effect immediately");
    }

    #[test]
    fn test_disable_takes_effect_immediately() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("telemetry.json");
        let mut gate = OptInGate::with_config_path(config_path);

        gate.enable().unwrap();
        assert!(gate.is_enabled());

        gate.disable().unwrap();
        assert!(!gate.is_enabled(), "Disable should take effect immediately");
    }

    #[test]
    fn test_toggle_cycles_state() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("telemetry.json");
        let mut gate = OptInGate::with_config_path(config_path);

        assert!(!gate.is_enabled());
        let new_state = gate.toggle().unwrap();
        assert!(new_state, "First toggle should enable");
        assert!(gate.is_enabled());

        let new_state = gate.toggle().unwrap();
        assert!(!new_state, "Second toggle should disable");
        assert!(!gate.is_enabled());
    }

    #[test]
    fn test_toggle_persists_to_disk() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("telemetry.json");
        let mut gate = OptInGate::with_config_path(config_path.clone());

        gate.enable().unwrap();

        // Create a new gate reading from the same path
        let gate2 = OptInGate::with_config_path(config_path);
        assert!(
            gate2.is_enabled(),
            "Persisted enable should be read by new gate instance"
        );
    }

    #[test]
    fn test_disable_persists_to_disk() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("telemetry.json");
        let mut gate = OptInGate::with_config_path(config_path.clone());

        gate.enable().unwrap();
        gate.disable().unwrap();

        let gate2 = OptInGate::with_config_path(config_path);
        assert!(
            !gate2.is_enabled(),
            "Persisted disable should be read by new gate instance"
        );
    }

    #[test]
    fn test_reload_detects_external_changes() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("telemetry.json");

        // Create initial disabled gate
        let mut gate = OptInGate::with_config_path(config_path.clone());
        assert!(!gate.is_enabled());

        // Externally write an enabled config
        let enabled_config = TelemetryConfig {
            enabled: true,
            last_toggled_at: None,
        };
        fs::write(
            &config_path,
            serde_json::to_string(&enabled_config).unwrap(),
        )
        .unwrap();

        // Reload should pick up the change
        gate.reload().unwrap();
        assert!(
            gate.is_enabled(),
            "Reload should detect external config change"
        );
    }

    // =======================================================================
    // VAL-TELE-012: Governance — No Direct Promotion
    // =======================================================================

    #[test]
    fn test_ui_strings_no_promotional_language() {
        let promotional_terms = [
            r"\bupgrade\b",
            r"\bpremium\b",
            r"\bpro\b",
            r"\bpaid\b",
            r"\bsubscribe\b",
            r"\btrial\b",
            r"\bdiscount\b",
            r"\boffer\b",
            r"\bdeal\b",
            r"\bfree trial\b",
            r"\blimited time\b",
            r"\bexclusive\b",
            r"\bunlock\b",
            r"\bbuy\b",
            r"\bpurchase\b",
        ];

        let all_strings = [
            TELEMETRY_DESCRIPTION,
            TELEMETRY_ENABLE_LABEL,
            TELEMETRY_DISABLE_LABEL,
            TELEMETRY_ENABLED_CONFIRMATION,
            TELEMETRY_DISABLED_CONFIRMATION,
            TELEMETRY_STATUS_ENABLED,
            TELEMETRY_STATUS_DISABLED,
            TELEMETRY_DATA_DESCRIPTION,
            TELEMETRY_PRIVACY_NOTE,
        ];

        for string in &all_strings {
            let lower = string.to_lowercase();
            for pattern in &promotional_terms {
                let re = regex::Regex::new(pattern).unwrap();
                assert!(
                    !re.is_match(&lower),
                    "UI string '{}' contains promotional term matching '{}'",
                    string,
                    pattern
                );
            }
        }
    }

    // =======================================================================
    // Config serde
    // =======================================================================

    #[test]
    fn test_telemetry_config_serde_roundtrip() {
        let config = TelemetryConfig {
            enabled: true,
            last_toggled_at: Some("2026-05-01T00:00:00+00:00".to_string()),
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: TelemetryConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, back);
    }

    #[test]
    fn test_telemetry_config_default_serde_roundtrip() {
        let config = TelemetryConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let back: TelemetryConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, back);
    }

    #[test]
    fn test_last_toggled_at_set_on_enable() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("telemetry.json");
        let mut gate = OptInGate::with_config_path(config_path);
        gate.enable().unwrap();
        assert!(
            gate.config().last_toggled_at.is_some(),
            "last_toggled_at should be set after enable"
        );
    }

    #[test]
    fn test_last_toggled_at_set_on_disable() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("telemetry.json");
        let mut gate = OptInGate::with_config_path(config_path);
        gate.enable().unwrap();
        gate.disable().unwrap();
        assert!(
            gate.config().last_toggled_at.is_some(),
            "last_toggled_at should be set after disable"
        );
    }
}
