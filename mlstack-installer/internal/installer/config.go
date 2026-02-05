// internal/installer/config.go
package installer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Config holds installer configuration with enhanced features
type Config struct {
	// Basic configuration
	ScriptsDir          string
	LogDir              string
	WorkDir             string
	BatchMode           bool
	EnvFile             string
	InstallPath         string
	PerformanceSettings map[string]string
	SecuritySettings    SecuritySettings

	// User preferences and settings
	UserPreferences     UserPreferences      `json:"user_preferences"`
	InstallationHistory []InstallationRecord `json:"installation_history"`
	ComponentSettings   ComponentSettings    `json:"component_settings"`
	EnvironmentSettings EnvironmentSettings  `json:"environment_settings"`
	NetworkSettings     NetworkSettings      `json:"network_settings"`

	// State tracking
	CurrentState   InstallationState `json:"current_state"`
	RecoveryPoints []RecoveryPoint   `json:"recovery_points"`
	SessionInfo    SessionInfo       `json:"session_info"`
	ErrorHistory   []ErrorRecord     `json:"error_history"`

	// Security and validation
	SecurityConfig  SecurityConfig   `json:"security_config"`
	ValidationRules []ValidationRule `json:"validation_rules"`

	// Performance settings
	PerformanceConfig PerformanceConfig `json:"performance_config"`
	CacheConfig       CacheConfig       `json:"cache_config"`

	// Locking mechanism
	mu            sync.RWMutex
	configFile    string
	configVersion string
	lastModified  time.Time
	isModified    bool
}

// UserPreferences holds user-specific preferences
type UserPreferences struct {
	InstallationPath     string               `json:"installation_path"`
	AutoConfirm          bool                 `json:"auto_confirm"`
	EnableAutoUpdates    bool                 `json:"enable_auto_updates"`
	Theme                string               `json:"theme"`
	Language             string               `json:"language"`
	Timezone             string               `json:"timezone"`
	PreferredComponents  []string             `json:"preferred_components"`
	PerformanceProfile   string               `json:"performance_profile"`
	BackupLocation       string               `json:"backup_location"`
	LogLevel             string               `json:"log_level"`
	NotificationSettings NotificationSettings `json:"notification_settings"`
}

// ComponentSettings holds component-specific settings
type ComponentSettings struct {
	PyTorch       ComponentConfig            `json:"pytorch"`
	FlashAttn     ComponentConfig            `json:"flash_attention"`
	Megatron      ComponentConfig            `json:"megatron"`
	ONNXRuntime   ComponentConfig            `json:"onnx_runtime"`
	Triton        ComponentConfig            `json:"triton"`
	vLLM          ComponentConfig            `json:"vllm"`
	ROCm          ComponentConfig            `json:"rocm"`
	MPI           ComponentConfig            `json:"mpi"`
	WandB         ComponentConfig            `json:"wandb"`
	CustomConfigs map[string]ComponentConfig `json:"custom_configs"`
}

// ComponentConfig holds configuration for individual components
type ComponentConfig struct {
	Enabled              bool              `json:"enabled"`
	Version              string            `json:"version"`
	InstallPath          string            `json:"install_path"`
	CustomFlags          []string          `json:"custom_flags"`
	EnvironmentVars      map[string]string `json:"environment_vars"`
	PostInstallScript    string            `json:"post_install_script"`
	PreInstallScript     string            `json:"pre_install_script"`
	Dependencies         []string          `json:"dependencies"`
	OptionalDependencies []string          `json:"optional_dependencies"`
	PerformanceSettings  map[string]string `json:"performance_settings"`
	RetryCount           int               `json:"retry_count"`
	RetryDelay           time.Duration     `json:"retry_delay"`
	Timeout              time.Duration     `json:"timeout"`
}

// EnvironmentSettings holds environment configuration
type EnvironmentSettings struct {
	GlobalEnvironmentVars map[string]string `json:"global_environment_vars"`
	PathExtensions        []string          `json:"path_extensions"`
	Aliases               map[string]string `json:"aliases"`
	ProfileScripts        []string          `json:"profile_scripts"`
	SystemServices        []SystemService   `json:"system_services"`
}

// SystemService represents a system service configuration
type SystemService struct {
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Command     string            `json:"command"`
	AutoStart   bool              `json:"auto_start"`
	DependsOn   []string          `json:"depends_on"`
	Environment map[string]string `json:"environment"`
}

// NetworkSettings holds network configuration
type NetworkSettings struct {
	Proxy          ProxyConfig    `json:"proxy"`
	DownloadConfig DownloadConfig `json:"download_config"`
	MirrorConfig   MirrorConfig   `json:"mirror_config"`
	FirewallRules  []FirewallRule `json:"firewall_rules"`
	TimeoutSeconds int            `json:"timeout_seconds"`
}

// ProxyConfig holds proxy configuration
type ProxyConfig struct {
	HTTPProxy     string `json:"http_proxy"`
	HTTPSProxy    string `json:"https_proxy"`
	NoProxy       string `json:"no_proxy"`
	ProxyUser     string `json:"proxy_user"`
	ProxyPassword string `json:"proxy_password"`
	AutoDetect    bool   `json:"auto_detect"`
}

// DownloadConfig holds download configuration
type DownloadConfig struct {
	MaxConcurrentDownloads int           `json:"max_concurrent_downloads"`
	Timeout                time.Duration `json:"timeout"`
	RetryAttempts          int           `json:"retry_attempts"`
	DownloadPath           string        `json:"download_path"`
	VerifyChecksums        bool          `json:"verify_checksums"`
	CacheDownloads         bool          `json:"cache_downloads"`
}

// MirrorConfig holds mirror configuration
type MirrorConfig struct {
	PreferredMirror    string        `json:"preferred_mirror"`
	AlternativeMirrors []string      `json:"alternative_mirrors"`
	MirrorTimeout      time.Duration `json:"mirror_timeout"`
	FallbackToDefault  bool          `json:"fallback_to_default"`
	MirrorRegions      []string      `json:"mirror_regions"`
}

// FirewallRule represents a firewall configuration rule
type FirewallRule struct {
	Name          string `json:"name"`
	Description   string `json:"description"`
	Port          int    `json:"port"`
	Protocol      string `json:"protocol"`
	Action        string `json:"action"`
	SourceIP      string `json:"source_ip"`
	DestinationIP string `json:"destination_ip"`
	Enabled       bool   `json:"enabled"`
}

// InstallationState tracks installation state
type InstallationState struct {
	CurrentStage     string         `json:"current_stage"`
	CurrentComponent string         `json:"current_component"`
	InstallationID   string         `json:"installation_id"`
	StartTime        time.Time      `json:"start_time"`
	LastUpdateTime   time.Time      `json:"last_update_time"`
	Progress         float64        `json:"progress"`
	Status           string         `json:"status"`
	Error            string         `json:"error"`
	CompletedSteps   []string       `json:"completed_steps"`
	PendingSteps     []string       `json:"pending_steps"`
	FailedSteps      []string       `json:"failed_steps"`
	ResumePoint      *ResumePoint   `json:"resume_point,omitempty"`
	RollbackPoint    *RollbackPoint `json:"rollback_point,omitempty"`
}

// ResumePoint stores information for resuming installations
type ResumePoint struct {
	StepName     string      `json:"step_name"`
	ComponentID  string      `json:"component_id"`
	Progress     float64     `json:"progress"`
	State        interface{} `json:"state"`
	Timestamp    time.Time   `json:"timestamp"`
	CheckpointID string      `json:"checkpoint_id"`
}

// SecuritySettings holds security configuration
type SecuritySettings struct {
	VerifySSL        bool     `json:"verify_ssl"`
	DisableRoot      bool     `json:"disable_root"`
	SudoPassword     string   `json:"sudo_password"`
	AllowedUsers     []string `json:"allowed_users"`
	RequireAuth      bool     `json:"require_auth"`
	Encryption       string   `json:"encryption"`
	ScriptValidation bool     `json:"script_validation"`
	RequireSudo      bool     `json:"require_sudo"`
	BackupSystem     bool     `json:"backup_system"`
}

// RollbackPoint stores information for rollback
type RollbackPoint struct {
	StepName      string    `json:"step_name"`
	ComponentID   string    `json:"component_id"`
	BackupPath    string    `json:"backup_path"`
	Timestamp     time.Time `json:"timestamp"`
	CheckpointID  string    `json:"checkpoint_id"`
	RestoreScript string    `json:"restore_script"`
}

// RecoveryPoint represents a recovery checkpoint
type RecoveryPoint struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Timestamp   time.Time `json:"timestamp"`
	Stage       string    `json:"stage"`
	Components  []string  `json:"components"`
	BackupPath  string    `json:"backup_path"`
	Description string    `json:"description"`
	Valid       bool      `json:"valid"`
	DataSize    int64     `json:"data_size"`
}

// SessionInfo tracks session information
type SessionInfo struct {
	SessionID    string          `json:"session_id"`
	UserID       string          `json:"user_id"`
	StartTime    time.Time       `json:"start_time"`
	LastActivity time.Time       `json:"last_activity"`
	IPAddr       string          `json:"ip_addr"`
	UserAgent    string          `json:"user_agent"`
	Permissions  []string        `json:"permissions"`
	ActivityLog  []ActivityEntry `json:"activity_log"`
}

// ActivityEntry represents a user activity entry
type ActivityEntry struct {
	Timestamp time.Time     `json:"timestamp"`
	Action    string        `json:"action"`
	Target    string        `json:"target"`
	Details   string        `json:"details"`
	Result    string        `json:"result"`
	Duration  time.Duration `json:"duration"`
}

// ErrorRecord represents an error entry
type ErrorRecord struct {
	ID         string    `json:"id"`
	Timestamp  time.Time `json:"timestamp"`
	Component  string    `json:"component"`
	ErrorType  string    `json:"error_type"`
	Message    string    `json:"message"`
	StackTrace string    `json:"stack_trace"`
	UserAction string    `json:"user_action"`
	Resolved   bool      `json:"resolved"`
	Resolution string    `json:"resolution"`
}

// SecurityConfig holds security-related configuration
type SecurityConfig struct {
	EnableEncryption bool               `json:"enable_encryption"`
	EncryptionKey    string             `json:"encryption_key"`
	EnableAudit      bool               `json:"enable_audit"`
	AuditLogLevel    string             `json:"audit_log_level"`
	Permissions      PermissionConfig   `json:"permissions"`
	Verification     VerificationConfig `json:"verification"`
}

// PermissionConfig holds permission configuration
type PermissionConfig struct {
	RequireRoot          bool          `json:"require_root"`
	AllowUserInstall     bool          `json:"allow_user_install"`
	RequireConfirmation  bool          `json:"require_confirmation"`
	MaxRetryAttempts     int           `json:"max_retry_attempts"`
	SessionTimeout       time.Duration `json:"session_timeout"`
	EnableSignatureCheck bool          `json:"enable_signature_check"`
	AllowedUsers         []string      `json:"allowed_users"`
	DeniedUsers          []string      `json:"denied_users"`
}

// VerificationConfig holds verification configuration
type VerificationConfig struct {
	EnableChecksumVerify  bool          `json:"enable_checksum_verify"`
	EnableSignatureVerify bool          `json:"enable_signature_verify"`
	EnableManifestVerify  bool          `json:"enable_manifest_verify"`
	TrustedSources        []string      `json:"trusted_sources"`
	VerificationTimeout   time.Duration `json:"verification_timeout"`
}

// ValidationRule represents a validation rule for configuration
type ValidationRule struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Field       string `json:"field"`
	Rule        string `json:"rule"`
	ErrorMsg    string `json:"error_msg"`
	Severity    string `json:"severity"`
}

// PerformanceConfig holds performance-related configuration
type PerformanceConfig struct {
	MaxConcurrentTasks int           `json:"max_concurrent_tasks"`
	TaskTimeout        time.Duration `json:"task_timeout"`
	MemoryLimit        int64         `json:"memory_limit"`
	DiskSpaceLimit     int64         `json:"disk_space_limit"`
	EnableParallelism  bool          `json:"enable_parallelism"`
	OptimizeForCPU     bool          `json:"optimize_for_cpu"`
	OptimizeForMemory  bool          `json:"optimize_for_memory"`
	EnableCaching      bool          `json:"enable_caching"`
	CacheSize          int64         `json:"cache_size"`
	PrefetchComponents bool          `json:"prefetch_components"`
}

// CacheConfig holds caching configuration
type CacheConfig struct {
	Enabled           bool          `json:"enabled"`
	CachePath         string        `json:"cache_path"`
	MaxCacheSize      int64         `json:"max_cache_size"`
	TTL               time.Duration `json:"ttl"`
	Compression       bool          `json:"compression"`
	EnablePersistence bool          `json:"enable_persistence"`
	CleanupInterval   time.Duration `json:"cleanup_interval"`
}

// NotificationSettings holds notification preferences
type NotificationSettings struct {
	EnableEmail        bool     `json:"enable_email"`
	EnableDesktop      bool     `json:"enable_desktop"`
	EnableSystem       bool     `json:"enable_system"`
	EmailRecipient     string   `json:"email_recipient"`
	EmailProvider      string   `json:"email_provider"`
	EventSubscriptions []string `json:"event_subscriptions"`
	CriticalOnly       bool     `json:"critical_only"`
}

// NewConfig creates a new enhanced configuration instance
func NewConfig() (*Config, error) {
	// Get working directory
	workDir, err := os.Getwd()
	if err != nil {
		return nil, err
	}

	// Default scripts directory (relative to workdir)
	scriptsDir := filepath.Join(workDir, "scripts")

	// Check if scripts directory exists in parent directory (for development)
	parentScriptsDir := filepath.Join(filepath.Dir(workDir), "scripts")
	if _, err := os.Stat(parentScriptsDir); err == nil {
		scriptsDir = parentScriptsDir
	}

	// Create directories
	logDir := filepath.Join(getUserHomeDir(), ".mlstack", "logs")
	configDir := filepath.Join(getUserHomeDir(), ".mlstack", "config")
	cacheDir := filepath.Join(getUserHomeDir(), ".mlstack", "cache")
	backupDir := filepath.Join(getUserHomeDir(), ".mlstack", "backups")

	dirs := []string{logDir, configDir, cacheDir, backupDir}
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return nil, fmt.Errorf("failed to create directory %s: %v", dir, err)
		}
	}

	// Check for batch mode
	batchMode := os.Getenv("MLSTACK_BATCH_MODE") == "1"

	// Create default configuration
	config := &Config{
		ScriptsDir:    scriptsDir,
		LogDir:        logDir,
		WorkDir:       workDir,
		BatchMode:     batchMode,
		configFile:    filepath.Join(configDir, "config.json"),
		configVersion: "1.0.0",
		isModified:    true,
	}

	// Initialize default settings
	config.initializeDefaults()

	// Try to load existing configuration
	if err := config.Load(); err != nil {
		// If config doesn't exist, save the defaults
		if os.IsNotExist(err) {
			return config, config.Save()
		}
		return nil, err
	}

	return config, nil
}

// initializeDefaults sets up default values for configuration
func (c *Config) initializeDefaults() {
	// Set default user preferences
	c.UserPreferences = UserPreferences{
		InstallationPath:   "/opt/rocm",
		AutoConfirm:        false,
		EnableAutoUpdates:  true,
		Theme:              "dark",
		Language:           "en",
		Timezone:           "UTC",
		PerformanceProfile: "balanced",
		BackupLocation:     filepath.Join(getUserHomeDir(), ".mlstack", "backups"),
		LogLevel:           "info",
		NotificationSettings: NotificationSettings{
			EnableDesktop: true,
			EnableSystem:  true,
			CriticalOnly:  false,
		},
	}

	// Set default component settings
	c.ComponentSettings = ComponentSettings{
		PyTorch: ComponentConfig{
			Enabled:         true,
			Version:         "2.6.0+rocm6.4.43482",
			InstallPath:     "/opt/pytorch",
			CustomFlags:     []string{"-DCMAKE_BUILD_TYPE=Release"},
			EnvironmentVars: map[string]string{"PYTORCH_ROCM_ARCH": "GFX1100"},
			RetryCount:      3,
			RetryDelay:      30 * time.Second,
			Timeout:         10 * time.Minute,
		},
		FlashAttn: ComponentConfig{
			Enabled:     true,
			Version:     "2.5.6",
			InstallPath: "/opt/flash-attention",
			RetryCount:  3,
			RetryDelay:  30 * time.Second,
			Timeout:     15 * time.Minute,
		},
		Megatron: ComponentConfig{
			Enabled:     true,
			Version:     "Megatron-LM",
			InstallPath: "/opt/megatron-lm",
			RetryCount:  3,
			RetryDelay:  30 * time.Second,
			Timeout:     20 * time.Minute,
		},
		ONNXRuntime: ComponentConfig{
			Enabled:     true,
			Version:     "1.22.0",
			InstallPath: "/opt/onnxruntime",
			RetryCount:  3,
			RetryDelay:  30 * time.Second,
			Timeout:     10 * time.Minute,
		},
		Triton: ComponentConfig{
			Enabled:     true,
			Version:     "3.2.0",
			InstallPath: "/opt/triton",
			RetryCount:  3,
			RetryDelay:  30 * time.Second,
			Timeout:     15 * time.Minute,
		},
		vLLM: ComponentConfig{
			Enabled:     true,
			Version:     "0.8.5",
			InstallPath: "/opt/vllm",
			RetryCount:  3,
			RetryDelay:  30 * time.Second,
			Timeout:     10 * time.Minute,
		},
		ROCm: ComponentConfig{
			Enabled:     true,
			Version:     "6.4.43482",
			InstallPath: "/opt/rocm",
			RetryCount:  3,
			RetryDelay:  60 * time.Second,
			Timeout:     30 * time.Minute,
		},
		MPI: ComponentConfig{
			Enabled:     true,
			Version:     "Open MPI 5.0.7",
			InstallPath: "/opt/mpi",
			RetryCount:  3,
			RetryDelay:  30 * time.Second,
			Timeout:     10 * time.Minute,
		},
		WandB: ComponentConfig{
			Enabled:     true,
			Version:     "0.19.9",
			InstallPath: "/opt/wandb",
			RetryCount:  3,
			RetryDelay:  30 * time.Second,
			Timeout:     5 * time.Minute,
		},
		CustomConfigs: make(map[string]ComponentConfig),
	}

	// Set default environment settings
	c.EnvironmentSettings = EnvironmentSettings{
		GlobalEnvironmentVars: map[string]string{
			"ROCM_PATH":                "/opt/rocm",
			"HSA_OVERRIDE_GFX_VERSION": "11.0.0",
			"PYTORCH_ROCM_ARCH":        "GFX1100",
		},
		PathExtensions: []string{
			"/opt/rocm/bin",
			"/opt/rocm/hip/bin",
			"/opt/rocm/profiler/bin",
			"/opt/rocm/opencl/bin",
			"/usr/local/cuda/bin",
		},
		ProfileScripts: []string{
			"source /opt/rocm/.env",
		},
	}

	// Set default network settings
	c.NetworkSettings = NetworkSettings{
		DownloadConfig: DownloadConfig{
			MaxConcurrentDownloads: 5,
			Timeout:                30 * time.Minute,
			RetryAttempts:          3,
			DownloadPath:           filepath.Join(getUserHomeDir(), ".mlstack", "downloads"),
			VerifyChecksums:        true,
			CacheDownloads:         true,
		},
		MirrorConfig: MirrorConfig{
			MirrorTimeout:     30 * time.Second,
			FallbackToDefault: true,
			MirrorRegions:     []string{"us", "eu", "asia"},
		},
	}

	// Set default security settings
	c.SecurityConfig = SecurityConfig{
		EnableEncryption: false,
		EnableAudit:      true,
		AuditLogLevel:    "info",
		Permissions: PermissionConfig{
			RequireRoot:         true,
			AllowUserInstall:    false,
			RequireConfirmation: true,
			MaxRetryAttempts:    5,
			SessionTimeout:      30 * time.Minute,
		},
		Verification: VerificationConfig{
			EnableChecksumVerify:  true,
			EnableSignatureVerify: false,
			EnableManifestVerify:  false,
			VerificationTimeout:   5 * time.Minute,
		},
	}

	// Set default performance settings
	c.PerformanceConfig = PerformanceConfig{
		MaxConcurrentTasks: 4,
		TaskTimeout:        1 * time.Hour,
		MemoryLimit:        8 * 1024 * 1024 * 1024,   // 8GB
		DiskSpaceLimit:     100 * 1024 * 1024 * 1024, // 100GB
		EnableParallelism:  true,
		EnableCaching:      true,
		CacheSize:          1 * 1024 * 1024 * 1024, // 1GB
		PrefetchComponents: true,
	}

	// Set default cache settings
	c.CacheConfig = CacheConfig{
		Enabled:           true,
		CachePath:         filepath.Join(getUserHomeDir(), ".mlstack", "cache"),
		MaxCacheSize:      10 * 1024 * 1024 * 1024, // 10GB
		TTL:               24 * time.Hour,
		Compression:       true,
		EnablePersistence: true,
		CleanupInterval:   1 * time.Hour,
	}

	// Initialize session info
	c.SessionInfo = SessionInfo{
		SessionID:    generateSessionID(),
		UserID:       getCurrentUserID(),
		StartTime:    time.Now(),
		LastActivity: time.Now(),
		IPAddr:       getCurrentIP(),
		ActivityLog:  []ActivityEntry{},
	}

	// Initialize current state
	c.CurrentState = InstallationState{
		CurrentStage:   "initialized",
		Status:         "ready",
		StartTime:      time.Now(),
		LastUpdateTime: time.Now(),
	}
}

// Load loads configuration from file
func (c *Config) Load() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if config file exists
	if _, err := os.Stat(c.configFile); os.IsNotExist(err) {
		return fmt.Errorf("configuration file not found: %s", c.configFile)
	}

	// Read configuration file
	data, err := os.ReadFile(c.configFile)
	if err != nil {
		return fmt.Errorf("failed to read configuration file: %v", err)
	}

	// Parse JSON configuration
	if err := json.Unmarshal(data, c); err != nil {
		return fmt.Errorf("failed to parse configuration: %v", err)
	}

	// Get file modification time
	if stat, err := os.Stat(c.configFile); err == nil {
		c.lastModified = stat.ModTime()
	}

	// Validate loaded configuration
	if err := c.validate(); err != nil {
		return fmt.Errorf("configuration validation failed: %v", err)
	}

	c.isModified = false
	return nil
}

// Save saves configuration to file
func (c *Config) Save() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Validate configuration before saving
	if err := c.validate(); err != nil {
		return fmt.Errorf("configuration validation failed: %v", err)
	}

	// Create backup of existing configuration
	if _, err := os.Stat(c.configFile); err == nil {
		backupFile := c.configFile + ".backup." + time.Now().Format("20060102_150405")
		if err := c.createBackup(backupFile); err != nil {
			fmt.Printf("Warning: Failed to create configuration backup: %v\n", err)
		}
	}

	// Convert to JSON with formatting
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal configuration: %v", err)
	}

	// Write configuration file
	if err := os.WriteFile(c.configFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write configuration file: %v", err)
	}

	// Update modification time
	if stat, err := os.Stat(c.configFile); err == nil {
		c.lastModified = stat.ModTime()
	}

	c.isModified = false
	return nil
}

// validate validates the configuration
func (c *Config) validate() error {
	// Define validation rules
	rules := []ValidationRule{
		{
			Name:        "InstallationPath",
			Description: "Installation path must be valid",
			Field:       "UserPreferences.InstallationPath",
			Rule:        "required",
			ErrorMsg:    "Installation path is required",
			Severity:    "critical",
		},
		{
			Name:        "LogLevel",
			Description: "Log level must be valid",
			Field:       "UserPreferences.LogLevel",
			Rule:        "in:debug,info,warn,error",
			ErrorMsg:    "Invalid log level",
			Severity:    "warning",
		},
	}

	// Apply validation rules
	for _, rule := range rules {
		if err := c.validateRule(rule); err != nil {
			if rule.Severity == "critical" {
				return err
			}
			fmt.Printf("Warning: %s\n", err)
		}
	}

	return nil
}

// validateRule validates a specific rule
func (c *Config) validateRule(rule ValidationRule) error {
	// Implement rule validation based on field and rule
	switch rule.Field {
	case "UserPreferences.InstallationPath":
		if rule.Rule == "required" {
			if c.UserPreferences.InstallationPath == "" {
				return fmt.Errorf(rule.ErrorMsg)
			}
			// Check if path exists or can be created
			if err := os.MkdirAll(c.UserPreferences.InstallationPath, 0755); err != nil {
				return fmt.Errorf("cannot create installation path: %v", err)
			}
		}
	case "UserPreferences.LogLevel":
		if rule.Rule == "in:debug,info,warn,error" {
			validLevels := map[string]bool{
				"debug": true,
				"info":  true,
				"warn":  true,
				"error": true,
			}
			if !validLevels[c.UserPreferences.LogLevel] {
				return fmt.Errorf(rule.ErrorMsg)
			}
		}
	}

	return nil
}

// createBackup creates a backup of the configuration
func (c *Config) createBackup(backupFile string) error {
	data, err := os.ReadFile(c.configFile)
	if err != nil {
		return err
	}

	return os.WriteFile(backupFile, data, 0644)
}

// GetSetting gets a setting by path
func (c *Config) GetSetting(path string) (interface{}, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// Implement setting retrieval by path
	// This is a simplified implementation - in production you'd use reflection or a proper path parser
	switch path {
	case "user.preferences.installation_path":
		return c.UserPreferences.InstallationPath, nil
	case "user.preferences.theme":
		return c.UserPreferences.Theme, nil
	case "component.settings.pytorch.enabled":
		return c.ComponentSettings.PyTorch.Enabled, nil
	case "performance.config.max_concurrent_tasks":
		return c.PerformanceConfig.MaxConcurrentTasks, nil
	default:
		return nil, fmt.Errorf("setting not found: %s", path)
	}
}

// SetSetting sets a setting by path
func (c *Config) SetSetting(path string, value interface{}) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.isModified = true

	// Implement setting setting by path
	// This is a simplified implementation - in production you'd use reflection or a proper path parser
	switch path {
	case "user.preferences.installation_path":
		if str, ok := value.(string); ok {
			c.UserPreferences.InstallationPath = str
		} else {
			return fmt.Errorf("invalid value type for installation path")
		}
	case "user.preferences.theme":
		if str, ok := value.(string); ok {
			c.UserPreferences.Theme = str
		} else {
			return fmt.Errorf("invalid value type for theme")
		}
	case "component.settings.pytorch.enabled":
		if b, ok := value.(bool); ok {
			c.ComponentSettings.PyTorch.Enabled = b
		} else {
			return fmt.Errorf("invalid value type for pytorch enabled")
		}
	case "performance.config.max_concurrent_tasks":
		if i, ok := value.(int); ok {
			c.PerformanceConfig.MaxConcurrentTasks = i
		} else {
			return fmt.Errorf("invalid value type for max concurrent tasks")
		}
	default:
		return fmt.Errorf("setting not found: %s", path)
	}

	return nil
}

// AddToHistory adds an installation record to history
func (c *Config) AddToHistory(record InstallationRecord) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.InstallationHistory = append(c.InstallationHistory, record)
	c.isModified = true

	return nil
}

// GetHistory gets installation history
func (c *Config) GetHistory() []InstallationRecord {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.InstallationHistory
}

// AddErrorRecord adds an error record
func (c *Config) AddErrorRecord(err ErrorRecord) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.ErrorHistory = append(c.ErrorHistory, err)
	c.isModified = true

	return nil
}

// GetErrors gets error history
func (c *Config) GetErrors() []ErrorRecord {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.ErrorHistory
}

// CreateRecoveryPoint creates a recovery point
func (c *Config) CreateRecoveryPoint(name string, stage string, components []string, description string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	recoveryPoint := RecoveryPoint{
		ID:          generateRecoveryID(),
		Name:        name,
		Timestamp:   time.Now(),
		Stage:       stage,
		Components:  components,
		Description: description,
		Valid:       false,
	}

	// Create backup for recovery
	backupPath := filepath.Join(c.UserPreferences.BackupLocation, recoveryPoint.ID)
	if err := os.MkdirAll(backupPath, 0755); err != nil {
		return fmt.Errorf("failed to create backup directory: %v", err)
	}

	recoveryPoint.BackupPath = backupPath
	recoveryPoint.Valid = true

	c.RecoveryPoints = append(c.RecoveryPoints, recoveryPoint)
	c.isModified = true

	return nil
}

// GetRecoveryPoints gets all recovery points
func (c *Config) GetRecoveryPoints() []RecoveryPoint {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.RecoveryPoints
}

// IsModified returns whether the configuration has been modified
func (c *Config) IsModified() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.isModified
}

// SetModified sets the modification flag
func (c *Config) SetModified(modified bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.isModified = modified
}

// GetSessionInfo gets current session information
func (c *Config) GetSessionInfo() SessionInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.SessionInfo
}

// UpdateActivity updates user activity
func (c *Config) UpdateActivity(action, target, details, result string, duration time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	activity := ActivityEntry{
		Timestamp: time.Now(),
		Action:    action,
		Target:    target,
		Details:   details,
		Result:    result,
		Duration:  duration,
	}

	c.SessionInfo.ActivityLog = append(c.SessionInfo.ActivityLog, activity)
	c.SessionInfo.LastActivity = time.Now()
	c.isModified = true

	return nil
}

// ExportConfiguration exports configuration to a file
func (c *Config) ExportConfiguration(filePath string) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to export configuration: %v", err)
	}

	return os.WriteFile(filePath, data, 0644)
}

// ImportConfiguration imports configuration from a file
func (c *Config) ImportConfiguration(filePath string) error {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read import file: %v", err)
	}

	var importedConfig Config
	if err := json.Unmarshal(data, &importedConfig); err != nil {
		return fmt.Errorf("failed to parse import file: %v", err)
	}

	// Validate imported configuration
	if err := importedConfig.validate(); err != nil {
		return fmt.Errorf("imported configuration validation failed: %v", err)
	}

	// Merge configurations
	c.mu.Lock()
	defer c.mu.Unlock()

	// Merge user preferences
	if importedConfig.UserPreferences.InstallationPath != "" {
		c.UserPreferences.InstallationPath = importedConfig.UserPreferences.InstallationPath
	}

	// Add to history
	importedConfig.InstallationHistory = append(c.InstallationHistory, importedConfig.InstallationHistory...)
	c.InstallationHistory = importedConfig.InstallationHistory

	// Add recovery points
	c.RecoveryPoints = append(c.RecoveryPoints, importedConfig.RecoveryPoints...)

	c.isModified = true

	return nil
}

// Utility functions

func getUserHomeDir() string {
	if home := os.Getenv("HOME"); home != "" {
		return home
	}
	return "/root"
}

func generateSessionID() string {
	return fmt.Sprintf("session_%d", time.Now().UnixNano())
}

func generateRecoveryID() string {
	return fmt.Sprintf("recovery_%d", time.Now().UnixNano())
}

func getCurrentUserID() string {
	if uid := os.Getenv("USER"); uid != "" {
		return uid
	}
	return "unknown"
}

func getCurrentIP() string {
	// This is a simplified implementation
	// In production, you'd get the actual IP address
	return "127.0.0.1"
}

// InstallationRecord represents an installation record in history
type InstallationRecord struct {
	ID           string        `json:"id"`
	Timestamp    time.Time     `json:"timestamp"`
	Version      string        `json:"version"`
	Components   []string      `json:"components"`
	Status       string        `json:"status"`
	Duration     time.Duration `json:"duration"`
	SuccessCount int           `json:"success_count"`
	ErrorCount   int           `json:"error_count"`
	LogFile      string        `json:"log_file"`
}
