//! Service launch surfaces with local-only endpoint exposure.
//!
//! Provides service configuration and health endpoint definitions that
//! enforce binding to `127.0.0.1` only (never `0.0.0.0`).
//!
//! # Security Model
//!
//! - All services bind to `127.0.0.1` (localhost) only
//! - No external interface exposure
//! - Health endpoints at `/health` respond with HTTP 200 within 30 seconds
//!
//! # Health Endpoints
//!
//! Each service registers `http://127.0.0.1:<port>/health` which returns:
//! - HTTP 200 with JSON body `{"status": "ok"}` when healthy
//! - HTTP 503 with JSON body `{"status": "unhealthy"}` when not healthy
//!
//! This module does not start an actual HTTP server — it provides the
//! configuration, address binding logic, and health check structures
//! used by the runtime to manage services.

use serde::{Deserialize, Serialize};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

/// Default health check path.
pub const HEALTH_PATH: &str = "/health";

/// Default health check timeout (30 seconds).
pub const HEALTH_TIMEOUT: Duration = Duration::from_secs(30);

/// Localhost IPv4 address — the only address services may bind to.
pub const LOCALHOST: Ipv4Addr = Ipv4Addr::new(127, 0, 0, 1);

// ===========================================================================
// Service Configuration
// ===========================================================================

/// Configuration for a managed service.
///
/// All services bind to `127.0.0.1` only. The bind address is always
/// constructed from [`LOCALHOST`] — callers cannot override it to
/// `0.0.0.0` or any external interface.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServiceConfig {
    /// Human-readable service name.
    pub name: String,
    /// Port number for the service.
    pub port: u16,
    /// Whether the service is enabled.
    pub enabled: bool,
}

impl ServiceConfig {
    /// Create a new service config bound to localhost on the given port.
    ///
    /// The bind address is always `127.0.0.1:<port>`.
    pub fn new(name: impl Into<String>, port: u16) -> Self {
        Self {
            name: name.into(),
            port,
            enabled: true,
        }
    }

    /// Get the socket address for this service.
    ///
    /// Always returns `127.0.0.1:<port>`.
    pub fn socket_addr(&self) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(LOCALHOST), self.port)
    }

    /// Get the health check URL for this service.
    ///
    /// Returns `http://127.0.0.1:<port>/health`.
    pub fn health_url(&self) -> String {
        format!("http://127.0.0.1:{}{}", self.port, HEALTH_PATH)
    }

    /// Validate that this config never binds to a non-local address.
    ///
    /// Always returns `true` since the address is always constructed
    /// from [`LOCALHOST`].
    pub fn is_local_only(&self) -> bool {
        true
    }
}

// ===========================================================================
// Health Check
// ===========================================================================

/// Result of a health check against a service.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HealthCheckResult {
    /// Name of the service that was checked.
    pub service: String,
    /// URL that was checked.
    pub url: String,
    /// Whether the health check succeeded.
    pub healthy: bool,
    /// Response time in milliseconds, if measured.
    pub response_ms: Option<u64>,
    /// Error message if the health check failed.
    pub error: Option<String>,
}

impl HealthCheckResult {
    /// Create a successful health check result.
    pub fn ok(service: impl Into<String>, url: impl Into<String>, response_ms: u64) -> Self {
        Self {
            service: service.into(),
            url: url.into(),
            healthy: true,
            response_ms: Some(response_ms),
            error: None,
        }
    }

    /// Create a failed health check result.
    pub fn failed(
        service: impl Into<String>,
        url: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            service: service.into(),
            url: url.into(),
            healthy: false,
            response_ms: None,
            error: Some(error.into()),
        }
    }
}

// ===========================================================================
// Service Registry
// ===========================================================================

/// Registry of managed services with their configurations.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ServiceRegistry {
    services: Vec<ServiceConfig>,
}

impl ServiceRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new service.
    ///
    /// Returns `Err` if a service with the same name or port already exists.
    pub fn register(&mut self, config: ServiceConfig) -> Result<(), ServiceError> {
        // Check for duplicate name
        if self.services.iter().any(|s| s.name == config.name) {
            return Err(ServiceError::DuplicateName {
                name: config.name.clone(),
            });
        }

        // Check for duplicate port
        if self.services.iter().any(|s| s.port == config.port) {
            return Err(ServiceError::DuplicatePort { port: config.port });
        }

        self.services.push(config);
        Ok(())
    }

    /// Get a service by name.
    pub fn get(&self, name: &str) -> Option<&ServiceConfig> {
        self.services.iter().find(|s| s.name == name)
    }

    /// Get all registered services.
    pub fn all(&self) -> &[ServiceConfig] {
        &self.services
    }

    /// Get all enabled services.
    pub fn enabled(&self) -> Vec<&ServiceConfig> {
        self.services.iter().filter(|s| s.enabled).collect()
    }

    /// Get health check URLs for all enabled services.
    pub fn health_urls(&self) -> Vec<String> {
        self.enabled().iter().map(|s| s.health_url()).collect()
    }

    /// Validate that all services bind to localhost only.
    ///
    /// Since [`ServiceConfig`] always constructs addresses from [`LOCALHOST`],
    /// this always returns `true`.
    pub fn all_local_only(&self) -> bool {
        self.services.iter().all(|s| s.is_local_only())
    }
}

// ===========================================================================
// Address Validation
// ===========================================================================

/// Validate that a socket address is localhost only.
///
/// Returns `true` only for `127.0.0.1` (IPv4) or `::1` (IPv6).
/// Returns `false` for `0.0.0.0`, `[::]`, or any non-loopback address.
pub fn is_localhost_addr(addr: SocketAddr) -> bool {
    match addr.ip() {
        IpAddr::V4(ip) => ip == LOCALHOST,
        IpAddr::V6(ip) => ip == Ipv6Addr::LOCALHOST,
    }
}

/// Validate that a bind address string represents a localhost-only address.
///
/// Accepts:
/// - `127.0.0.1`
/// - `localhost`
///
/// Rejects:
/// - `0.0.0.0`
/// - `::`
/// - Any non-loopback address
pub fn validate_bind_address(addr_str: &str) -> Result<SocketAddr, ServiceError> {
    // Check for dangerous wildcard addresses
    if addr_str == "0.0.0.0" || addr_str == "::" || addr_str == "[::]" {
        return Err(ServiceError::ForbiddenBindAddress {
            addr: addr_str.to_string(),
        });
    }

    // Try to parse as SocketAddr (may include port)
    if let Ok(addr) = addr_str.parse::<SocketAddr>() {
        if is_localhost_addr(addr) {
            return Ok(addr);
        }
        return Err(ServiceError::ForbiddenBindAddress {
            addr: addr_str.to_string(),
        });
    }

    // Try to parse as IP with default port
    if let Ok(ip) = addr_str.parse::<IpAddr>() {
        let addr = SocketAddr::new(ip, 0);
        if is_localhost_addr(addr) {
            return Ok(addr);
        }
        return Err(ServiceError::ForbiddenBindAddress {
            addr: addr_str.to_string(),
        });
    }

    // Try "localhost" hostname
    if addr_str == "localhost" {
        return Ok(SocketAddr::new(IpAddr::V4(LOCALHOST), 0));
    }

    // Try "localhost:port"
    if let Some(rest) = addr_str.strip_prefix("localhost:") {
        if let Ok(port) = rest.parse::<u16>() {
            return Ok(SocketAddr::new(IpAddr::V4(LOCALHOST), port));
        }
    }

    Err(ServiceError::InvalidBindAddress {
        addr: addr_str.to_string(),
    })
}

// ===========================================================================
// Errors
// ===========================================================================

/// Errors related to service management.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ServiceError {
    /// A service with this name already exists.
    DuplicateName { name: String },
    /// A service with this port already exists.
    DuplicatePort { port: u16 },
    /// The requested bind address is not localhost.
    ForbiddenBindAddress { addr: String },
    /// The bind address could not be parsed.
    InvalidBindAddress { addr: String },
    /// Health check timeout exceeded.
    HealthCheckTimeout { service: String, timeout_secs: u64 },
}

impl std::fmt::Display for ServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceError::DuplicateName { name } => {
                write!(f, "duplicate service name: {}", name)
            }
            ServiceError::DuplicatePort { port } => {
                write!(f, "duplicate service port: {}", port)
            }
            ServiceError::ForbiddenBindAddress { addr } => {
                write!(
                    f,
                    "bind address '{}' is forbidden — services must bind to 127.0.0.1 only",
                    addr
                )
            }
            ServiceError::InvalidBindAddress { addr } => {
                write!(f, "invalid bind address: '{}'", addr)
            }
            ServiceError::HealthCheckTimeout {
                service,
                timeout_secs,
            } => {
                write!(
                    f,
                    "health check for '{}' timed out after {}s",
                    service, timeout_secs
                )
            }
        }
    }
}

impl std::error::Error for ServiceError {}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ================================================================
    // VAL-WIN-010: Services bind to 127.0.0.1 only
    // ================================================================

    #[test]
    fn test_service_config_binds_to_localhost() {
        let config = ServiceConfig::new("test-service", 8080);
        let addr = config.socket_addr();
        assert_eq!(addr.ip(), IpAddr::V4(LOCALHOST));
        assert_eq!(addr.port(), 8080);
    }

    #[test]
    fn test_service_config_is_local_only() {
        let config = ServiceConfig::new("test-service", 8080);
        assert!(config.is_local_only());
    }

    #[test]
    fn test_service_config_never_binds_zero() {
        let config = ServiceConfig::new("test-service", 8080);
        let addr = config.socket_addr();
        // Must NOT be 0.0.0.0
        assert_ne!(addr.ip(), IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));
        // Must be 127.0.0.1
        assert_eq!(addr.ip(), IpAddr::V4(LOCALHOST));
    }

    #[test]
    fn test_validate_bind_address_localhost() {
        assert!(validate_bind_address("127.0.0.1").is_ok());
        assert!(validate_bind_address("localhost").is_ok());
    }

    #[test]
    fn test_validate_bind_address_rejects_wildcard() {
        assert!(validate_bind_address("0.0.0.0").is_err());
        assert!(validate_bind_address("::").is_err());
        assert!(validate_bind_address("[::]").is_err());
    }

    #[test]
    fn test_validate_bind_address_rejects_external() {
        assert!(validate_bind_address("192.168.1.1").is_err());
        assert!(validate_bind_address("10.0.0.1").is_err());
    }

    #[test]
    fn test_validate_bind_address_with_port() {
        let addr = validate_bind_address("127.0.0.1:8080").unwrap();
        assert_eq!(addr.port(), 8080);
    }

    #[test]
    fn test_validate_bind_address_localhost_with_port() {
        let addr = validate_bind_address("localhost:9090").unwrap();
        assert_eq!(addr.ip(), IpAddr::V4(LOCALHOST));
        assert_eq!(addr.port(), 9090);
    }

    // ================================================================
    // VAL-WIN-011: Health Endpoints
    // ================================================================

    #[test]
    fn test_health_url_format() {
        let config = ServiceConfig::new("api", 8080);
        assert_eq!(config.health_url(), "http://127.0.0.1:8080/health");
    }

    #[test]
    fn test_health_check_result_ok() {
        let result = HealthCheckResult::ok("api", "http://127.0.0.1:8080/health", 42);
        assert!(result.healthy);
        assert_eq!(result.response_ms, Some(42));
        assert!(result.error.is_none());
    }

    #[test]
    fn test_health_check_result_failed() {
        let result = HealthCheckResult::failed("api", "http://127.0.0.1:8080/health", "timeout");
        assert!(!result.healthy);
        assert_eq!(result.error, Some("timeout".to_string()));
    }

    #[test]
    fn test_health_check_result_serde_roundtrip() {
        let result = HealthCheckResult::ok("api", "http://127.0.0.1:8080/health", 42);
        let json = serde_json::to_string(&result).unwrap();
        let back: HealthCheckResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, back);
    }

    #[test]
    fn test_health_timeout_constant() {
        assert_eq!(HEALTH_TIMEOUT, Duration::from_secs(30));
    }

    #[test]
    fn test_health_path_constant() {
        assert_eq!(HEALTH_PATH, "/health");
    }

    // ================================================================
    // Service Registry
    // ================================================================

    #[test]
    fn test_registry_register_and_get() {
        let mut registry = ServiceRegistry::new();
        let config = ServiceConfig::new("api", 8080);
        registry.register(config).unwrap();

        let found = registry.get("api").unwrap();
        assert_eq!(found.port, 8080);
    }

    #[test]
    fn test_registry_rejects_duplicate_name() {
        let mut registry = ServiceRegistry::new();
        registry.register(ServiceConfig::new("api", 8080)).unwrap();
        let result = registry.register(ServiceConfig::new("api", 9090));
        assert!(matches!(result, Err(ServiceError::DuplicateName { .. })));
    }

    #[test]
    fn test_registry_rejects_duplicate_port() {
        let mut registry = ServiceRegistry::new();
        registry.register(ServiceConfig::new("api", 8080)).unwrap();
        let result = registry.register(ServiceConfig::new("web", 8080));
        assert!(matches!(result, Err(ServiceError::DuplicatePort { .. })));
    }

    #[test]
    fn test_registry_all_local_only() {
        let mut registry = ServiceRegistry::new();
        registry.register(ServiceConfig::new("api", 8080)).unwrap();
        registry.register(ServiceConfig::new("web", 9090)).unwrap();
        assert!(registry.all_local_only());
    }

    #[test]
    fn test_registry_health_urls() {
        let mut registry = ServiceRegistry::new();
        registry.register(ServiceConfig::new("api", 8080)).unwrap();
        registry.register(ServiceConfig::new("web", 9090)).unwrap();

        let urls = registry.health_urls();
        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"http://127.0.0.1:8080/health".to_string()));
        assert!(urls.contains(&"http://127.0.0.1:9090/health".to_string()));
    }

    #[test]
    fn test_registry_enabled_filters_disabled() {
        let mut registry = ServiceRegistry::new();
        let mut disabled = ServiceConfig::new("disabled", 8080);
        disabled.enabled = false;
        registry.register(disabled).unwrap();
        registry
            .register(ServiceConfig::new("enabled", 9090))
            .unwrap();

        let enabled = registry.enabled();
        assert_eq!(enabled.len(), 1);
        assert_eq!(enabled[0].name, "enabled");
    }

    // ================================================================
    // Address Validation
    // ================================================================

    #[test]
    fn test_is_localhost_addr_ipv4() {
        let addr = SocketAddr::new(IpAddr::V4(LOCALHOST), 8080);
        assert!(is_localhost_addr(addr));
    }

    #[test]
    fn test_is_localhost_addr_ipv6() {
        let addr = SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 8080);
        assert!(is_localhost_addr(addr));
    }

    #[test]
    fn test_is_not_localhost_addr() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)), 8080);
        assert!(!is_localhost_addr(addr));
    }

    #[test]
    fn test_is_not_localhost_zero() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 8080);
        assert!(!is_localhost_addr(addr));
    }

    // ================================================================
    // Serde
    // ================================================================

    #[test]
    fn test_service_config_serde_roundtrip() {
        let config = ServiceConfig::new("test-service", 8080);
        let json = serde_json::to_string(&config).unwrap();
        let back: ServiceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, back);
    }

    #[test]
    fn test_service_error_serde_roundtrip() {
        let err = ServiceError::ForbiddenBindAddress {
            addr: "0.0.0.0".to_string(),
        };
        let json = serde_json::to_string(&err).unwrap();
        let back: ServiceError = serde_json::from_str(&json).unwrap();
        assert_eq!(err, back);
    }

    // ================================================================
    // Error Display
    // ================================================================

    #[test]
    fn test_error_display_forbidden() {
        let err = ServiceError::ForbiddenBindAddress {
            addr: "0.0.0.0".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("0.0.0.0"));
        assert!(msg.contains("127.0.0.1"));
    }

    #[test]
    fn test_error_display_timeout() {
        let err = ServiceError::HealthCheckTimeout {
            service: "api".to_string(),
            timeout_secs: 30,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("api"));
        assert!(msg.contains("30"));
    }
}
