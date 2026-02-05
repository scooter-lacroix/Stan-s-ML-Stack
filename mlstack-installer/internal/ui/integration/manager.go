// internal/ui/integration/manager.go
package integration

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/scooter-lacroix/mlstack-installer/internal/installer"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// Manager defines the interface for integration manager operations
type Manager interface {
	Initialize() error
	DetectHardware() (installer.GPUInfo, installer.SystemInfo, error)
	RunScript(scriptPath string, args []string) (string, error)
	VerifyInstallation() (*installer.PreFlightCheckResults, error)
	SaveConfig(config *installer.Config) error
	LoadConfig() (*installer.Config, error)
	GetScriptList() ([]string, error)
	GetComponentList() ([]types.Component, error)
	GetConfigPath() string
	GetScriptDir() string
	GetBackupDir() string
	GetStatus() map[string]interface{}
	SetConfigPath(path string)
	SetScriptDir(dir string)
	SetBackupDir(dir string)
	SetTimeout(timeout time.Duration)
	SetRetryCount(count int)
	GetLastError() error
	ClearLastError()
	GetOperationCount() int
	IsInitialized() bool
	Shutdown() error
	GetHardwareDetectionCommand() (func() tea.Msg, error)
	GetScriptExecutionCommand(scriptPath string, args []string) (func() tea.Msg, error)
	GetVerificationCommand() (func() tea.Msg, error)
	GetConfigLoadCommand() (func() tea.Msg, error)
	GetConfigSaveCommand(config *installer.Config) (func() tea.Msg, error)
	CreateBatchOperation(commands []func() tea.Msg) *types.BatchOperation
	ExecuteBatchOperation(batch *types.BatchOperation) (func() tea.Msg, error)
}

// IntegrationManager implements integration with existing installer functionality
type IntegrationManager struct {
	hardwareDetector    *HardwareDetector
	scriptExecutor      *ScriptExecutor
	configManager       *ConfigManager
	verificationService *VerificationService

	// Configuration
	scriptDir  string
	configPath string
	backupDir  string
	timeout    time.Duration
	retryCount int

	// Concurrency
	mu         sync.RWMutex
	cancelFunc context.CancelFunc
	ctx        context.Context

	// State
	isInitialized  bool
	lastError      error
	operationCount int
}

// HardwareDetector handles hardware detection
type HardwareDetector struct {
	manager *IntegrationManager
}

// ScriptExecutor handles script execution
type ScriptExecutor struct {
	manager *IntegrationManager
}

// ConfigManager handles configuration management
type ConfigManager struct {
	manager *IntegrationManager
}

// VerificationService handles verification
type VerificationService struct {
	manager *IntegrationManager
}

// NewHardwareDetector creates a new hardware detector wrapper
func NewHardwareDetector() *HardwareDetector {
	return &HardwareDetector{}
}

// NewPreFlightChecker creates a new pre-flight checker wrapper
func NewPreFlightChecker() *VerificationService {
	return &VerificationService{}
}

// NewManager creates a new integration manager
func NewManager() *IntegrationManager {
	ctx, cancel := context.WithCancel(context.Background())

	return &IntegrationManager{
		scriptDir:     "./scripts",
		configPath:    "./config.yaml",
		backupDir:     "./backups",
		timeout:       30 * time.Second,
		retryCount:    3,
		ctx:           ctx,
		cancelFunc:    cancel,
		isInitialized: false,
	}
}

// Initialize sets up the integration manager
func (im *IntegrationManager) Initialize() error {
	im.mu.Lock()
	defer im.mu.Unlock()

	if im.isInitialized {
		return fmt.Errorf("integration manager already initialized")
	}

	// Create sub-services
	im.hardwareDetector = &HardwareDetector{manager: im}
	im.scriptExecutor = &ScriptExecutor{manager: im}
	im.configManager = &ConfigManager{manager: im}
	im.verificationService = &VerificationService{manager: im}

	// Validate directories
	if err := im.validateDirectories(); err != nil {
		return fmt.Errorf("directory validation failed: %v", err)
	}

	im.isInitialized = true
	return nil
}

// validateDirectories validates required directories
func (im *IntegrationManager) validateDirectories() error {
	directories := []string{
		im.scriptDir,
		im.backupDir,
		filepath.Dir(im.configPath),
	}

	for _, dir := range directories {
		if _, err := os.Stat(dir); os.IsNotExist(err) {
			return fmt.Errorf("directory does not exist: %s", dir)
		}
	}

	return nil
}

// DetectHardware detects system hardware and returns GPU and system info
func (im *IntegrationManager) DetectHardware() (installer.GPUInfo, installer.SystemInfo, error) {
	if !im.isInitialized {
		return installer.GPUInfo{}, installer.SystemInfo{}, fmt.Errorf("integration manager not initialized")
	}

	im.mu.RLock()
	defer im.mu.RUnlock()

	im.operationCount++
	defer im.recordOperation("hardware_detection")

	return im.hardwareDetector.Detect()
}

// Detect is the actual hardware detection implementation
func (hd *HardwareDetector) Detect() (installer.GPUInfo, installer.SystemInfo, error) {
	// Detect hardware directly using existing functions
	gpuInfo, err := installer.DetectGPU()
	if err != nil {
		return installer.GPUInfo{}, installer.SystemInfo{}, fmt.Errorf("GPU detection failed: %v", err)
	}

	// Detect system info directly
	systemInfo, err := installer.DetectSystem()
	if err != nil {
		return installer.GPUInfo{}, installer.SystemInfo{}, fmt.Errorf("system detection failed: %v", err)
	}

	return gpuInfo, systemInfo, nil
}

// RunScript executes a script with the given arguments
func (im *IntegrationManager) RunScript(scriptPath string, args []string) (string, error) {
	if !im.isInitialized {
		return "", fmt.Errorf("integration manager not initialized")
	}

	im.mu.RLock()
	defer im.mu.RUnlock()

	im.operationCount++
	defer im.recordOperation("script_execution")

	return im.scriptExecutor.Execute(scriptPath, args)
}

// Execute is the actual script execution implementation
func (se *ScriptExecutor) Execute(scriptPath string, args []string) (string, error) {
	// Validate script path
	fullPath := filepath.Join(se.manager.scriptDir, scriptPath)
	if err := se.manager.validateScriptPath(fullPath); err != nil {
		return "", fmt.Errorf("script validation failed: %v", err)
	}

	// Build command
	cmd := exec.CommandContext(se.manager.ctx, fullPath, args...)

	// Capture output
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("script execution failed: %v\nOutput: %s", err, string(output))
	}

	return string(output), nil
}

// validateScriptPath validates script execution path
func (im *IntegrationManager) validateScriptPath(scriptPath string) error {
	// Check if path exists
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		return fmt.Errorf("script file does not exist: %s", scriptPath)
	}

	// Check if it's executable
	info, err := os.Stat(scriptPath)
	if err != nil {
		return fmt.Errorf("cannot access script file: %v", err)
	}

	if info.Mode().Perm()&0111 == 0 {
		return fmt.Errorf("script is not executable: %s", scriptPath)
	}

	return nil
}

// VerifyInstallation verifies the current installation
func (im *IntegrationManager) VerifyInstallation() (*installer.PreFlightCheckResults, error) {
	if !im.isInitialized {
		return nil, fmt.Errorf("integration manager not initialized")
	}

	im.mu.RLock()
	defer im.mu.RUnlock()

	im.operationCount++
	defer im.recordOperation("verification")

	return im.verificationService.Verify()
}

// Verify is the actual verification implementation
func (vs *VerificationService) Verify() (*installer.PreFlightCheckResults, error) {
	// Get system info for pre-flight checks
	sysInfo, err := installer.DetectSystem()
	if err != nil {
		return nil, fmt.Errorf("failed to detect system info: %v", err)
	}

	// Get GPU info for pre-flight checks
	gpuInfo, err := installer.DetectGPU()
	if err != nil {
		return nil, fmt.Errorf("failed to detect GPU info: %v", err)
	}

	// Run pre-flight checks directly using existing function
	results, err := installer.RunPreFlightChecks(sysInfo, gpuInfo)
	if err != nil {
		return nil, fmt.Errorf("verification failed: %v", err)
	}

	return results, nil
}

// SaveConfig saves configuration to file
func (im *IntegrationManager) SaveConfig(config *installer.Config) error {
	if !im.isInitialized {
		return fmt.Errorf("integration manager not initialized")
	}

	im.mu.Lock()
	defer im.mu.Unlock()

	im.operationCount++
	defer im.recordOperation("config_save")

	return im.configManager.Save(config)
}

// Save is the actual config save implementation
func (cm *ConfigManager) Save(config *installer.Config) error {
	if config == nil {
		return fmt.Errorf("config cannot be nil")
	}

	// Set config file path (using public method if available)
	if config.ScriptsDir == "" {
		config.ScriptsDir = cm.manager.scriptDir
	}

	// Create backup before saving
	if err := cm.createBackup(); err != nil {
		return fmt.Errorf("failed to create backup: %v", err)
	}

	// Save config
	if err := config.Save(); err != nil {
		return fmt.Errorf("failed to save config: %v", err)
	}

	return nil
}

// LoadConfig loads configuration from file
func (im *IntegrationManager) LoadConfig() (*installer.Config, error) {
	if !im.isInitialized {
		return nil, fmt.Errorf("integration manager not initialized")
	}

	im.mu.RLock()
	defer im.mu.RUnlock()

	im.operationCount++
	defer im.recordOperation("config_load")

	return im.configManager.Load()
}

// Load is the actual config load implementation
func (cm *ConfigManager) Load() (*installer.Config, error) {
	// Create default config if not exists
	if _, err := os.Stat(cm.manager.configPath); os.IsNotExist(err) {
		return cm.createDefaultConfig(), nil
	}

	// Create config instance
	config := cm.createDefaultConfig()
	if config == nil {
		return nil, fmt.Errorf("failed to create default config")
	}

	// Load existing config
	if err := config.Load(); err != nil {
		return nil, fmt.Errorf("failed to load config: %v", err)
	}

	return config, nil
}

// createBackup creates a backup of the current config
func (cm *ConfigManager) createBackup() error {
	if _, err := os.Stat(cm.manager.configPath); os.IsNotExist(err) {
		return nil // No config to backup
	}

	timestamp := time.Now().Format("20060102_150405")
	backupPath := filepath.Join(cm.manager.backupDir, "config_backup_"+timestamp+".yaml")

	// Copy file
	data, err := os.ReadFile(cm.manager.configPath)
	if err != nil {
		return err
	}

	return os.WriteFile(backupPath, data, 0644)
}

// createDefaultConfig creates a default configuration
func (cm *ConfigManager) createDefaultConfig() *installer.Config {
	config, err := installer.NewConfig()
	if err != nil {
		return nil
	}

	// Set default values
	config.UserPreferences.AutoConfirm = false
	config.UserPreferences.EnableAutoUpdates = true
	config.UserPreferences.LogLevel = "info"
	config.UserPreferences.PreferredComponents = []string{
		"rocm",
		"pytorch",
		"triton",
		"mpi4py",
		"deepspeed",
	}

	config.EnvironmentSettings.GlobalEnvironmentVars = map[string]string{
		"ROCM_PATH":                "/opt/rocm",
		"HSA_OVERRIDE_GFX_VERSION": "11.0.0",
		"PYTORCH_ROCM_ARCH":        "GFX1100",
	}

	// Note: InstallationPath and TempDir are handled by the installer package itself

	return config
}

// GetScriptList returns a list of available scripts
func (im *IntegrationManager) GetScriptList() ([]string, error) {
	if !im.isInitialized {
		return nil, fmt.Errorf("integration manager not initialized")
	}

	im.mu.RLock()
	defer im.mu.RUnlock()

	files, err := os.ReadDir(im.scriptDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read script directory: %v", err)
	}

	var scripts []string
	for _, file := range files {
		if !file.IsDir() && strings.HasSuffix(file.Name(), ".sh") {
			scripts = append(scripts, file.Name())
		}
	}

	return scripts, nil
}

// GetComponentList returns a list of available components
func (im *IntegrationManager) GetComponentList() ([]types.Component, error) {
	if !im.isInitialized {
		return nil, fmt.Errorf("integration manager not initialized")
	}

	// Return empty list for now - actual component management would be implemented differently
	return []types.Component{}, nil
}

// GetConfigPath returns the current config path
func (im *IntegrationManager) GetConfigPath() string {
	return im.configPath
}

// GetScriptDir returns the script directory
func (im *IntegrationManager) GetScriptDir() string {
	return im.scriptDir
}

// GetBackupDir returns the backup directory
func (im *IntegrationManager) GetBackupDir() string {
	return im.backupDir
}

// GetStatus returns the current status of the integration manager
func (im *IntegrationManager) GetStatus() map[string]interface{} {
	im.mu.RLock()
	defer im.mu.RUnlock()

	return map[string]interface{}{
		"initialized":     im.isInitialized,
		"operation_count": im.operationCount,
		"last_error":      im.lastError,
		"config_path":     im.configPath,
		"script_dir":      im.scriptDir,
		"backup_dir":      im.backupDir,
		"timeout":         im.timeout,
		"retry_count":     im.retryCount,
	}
}

// recordOperation records an operation and updates error state
func (im *IntegrationManager) recordOperation(operation string) {
	// In a real implementation, this would record metrics and track performance
}

// SetConfigPath sets the configuration path
func (im *IntegrationManager) SetConfigPath(path string) {
	im.mu.Lock()
	defer im.mu.Unlock()

	im.configPath = path
}

// SetScriptDir sets the script directory
func (im *IntegrationManager) SetScriptDir(dir string) {
	im.mu.Lock()
	defer im.mu.Unlock()

	im.scriptDir = dir
}

// SetBackupDir sets the backup directory
func (im *IntegrationManager) SetBackupDir(dir string) {
	im.mu.Lock()
	defer im.mu.Unlock()

	im.backupDir = dir
}

// SetTimeout sets the operation timeout
func (im *IntegrationManager) SetTimeout(timeout time.Duration) {
	im.mu.Lock()
	defer im.mu.Unlock()

	im.timeout = timeout
}

// SetRetryCount sets the retry count for failed operations
func (im *IntegrationManager) SetRetryCount(count int) {
	im.mu.Lock()
	defer im.mu.Unlock()

	im.retryCount = count
}

// GetLastError returns the last error encountered
func (im *IntegrationManager) GetLastError() error {
	im.mu.RLock()
	defer im.mu.RUnlock()

	return im.lastError
}

// ClearLastError clears the last error
func (im *IntegrationManager) ClearLastError() {
	im.mu.Lock()
	defer im.mu.Unlock()

	im.lastError = nil
}

// GetOperationCount returns the number of operations performed
func (im *IntegrationManager) GetOperationCount() int {
	im.mu.RLock()
	defer im.mu.RUnlock()

	return im.operationCount
}

// IsInitialized returns whether the integration manager is initialized
func (im *IntegrationManager) IsInitialized() bool {
	im.mu.RLock()
	defer im.mu.RUnlock()

	return im.isInitialized
}

// Shutdown shuts down the integration manager
func (im *IntegrationManager) Shutdown() error {
	im.mu.Lock()
	defer im.mu.Unlock()

	if im.cancelFunc != nil {
		im.cancelFunc()
	}

	im.isInitialized = false
	return nil
}

// GetHardwareDetectionCommand returns a command for hardware detection
func (im *IntegrationManager) GetHardwareDetectionCommand() (func() tea.Msg, error) {
	if !im.isInitialized {
		return nil, fmt.Errorf("integration manager not initialized")
	}

	return func() tea.Msg {
		gpuInfo, systemInfo, err := im.DetectHardware()
		if err != nil {
			return types.HardwareDetectionCompleteMsg{
				Error: err,
			}
		}
		return types.HardwareDetectionCompleteMsg{
			GPUInfo:    gpuInfo,
			SystemInfo: systemInfo,
		}
	}, nil
}

// GetScriptExecutionCommand returns a command for script execution
func (im *IntegrationManager) GetScriptExecutionCommand(scriptPath string, args []string) (func() tea.Msg, error) {
	if !im.isInitialized {
		return nil, fmt.Errorf("integration manager not initialized")
	}

	return func() tea.Msg {
		startTime := time.Now()
		output, err := im.RunScript(scriptPath, args)
		duration := time.Since(startTime)

		return types.ScriptExecutionCompleteMsg{
			ScriptPath: scriptPath,
			Output:     output,
			Error:      err,
			Duration:   duration,
		}
	}, nil
}

// GetVerificationCommand returns a command for verification
func (im *IntegrationManager) GetVerificationCommand() (func() tea.Msg, error) {
	if !im.isInitialized {
		return nil, fmt.Errorf("integration manager not initialized")
	}

	return func() tea.Msg {
		results, err := im.VerifyInstallation()
		if err != nil {
			return types.VerificationCompleteMsg{
				Error: err,
			}
		}
		return types.VerificationCompleteMsg{
			Results: results,
		}
	}, nil
}

// GetConfigLoadCommand returns a command for config loading
func (im *IntegrationManager) GetConfigLoadCommand() (func() tea.Msg, error) {
	if !im.isInitialized {
		return nil, fmt.Errorf("integration manager not initialized")
	}

	return func() tea.Msg {
		config, err := im.LoadConfig()
		if err != nil {
			return types.ConfigLoadMsg{
				Error: err,
			}
		}
		return types.ConfigLoadMsg{
			Config: config,
		}
	}, nil
}

// GetConfigSaveCommand returns a command for config saving
func (im *IntegrationManager) GetConfigSaveCommand(config *installer.Config) (func() tea.Msg, error) {
	if !im.isInitialized {
		return nil, fmt.Errorf("integration manager not initialized")
	}

	return func() tea.Msg {
		err := im.SaveConfig(config)
		return types.ConfigSaveMsg{
			Success: err == nil,
			Message: fmt.Sprintf("Config saved: %v", err),
		}
	}, nil
}

// CreateBatchOperation creates a batch operation with multiple commands
func (im *IntegrationManager) CreateBatchOperation(commands []func() tea.Msg) *types.BatchOperation {
	batchID := fmt.Sprintf("batch_%d", time.Now().UnixNano())
	batch := &types.BatchOperation{
		ID:       batchID,
		Commands: make([]types.Command, len(commands)),
		Done:     false,
		Progress: 0.0,
	}

	for i, cmd := range commands {
		// Convert func() tea.Msg to func() tea.Cmd
		cmdFunc := cmd
		batch.Commands[i] = types.AsyncCommand{
			ID: batchID,
			Executor: func() tea.Cmd {
				return tea.Cmd(func() tea.Msg {
					return cmdFunc()
				})
			},
			CancelFn: func() error { return nil },
			Timeout:  im.timeout,
			Context:  im.ctx,
			Done:     false,
			Progress: 0.0,
		}
	}

	return batch
}

// ExecuteBatchOperation executes a batch operation
func (im *IntegrationManager) ExecuteBatchOperation(batch *types.BatchOperation) (func() tea.Msg, error) {
	if !im.isInitialized {
		return nil, fmt.Errorf("integration manager not initialized")
	}

	return func() tea.Msg {
		results := make([]tea.Msg, len(batch.Commands))

		for i, cmd := range batch.Commands {
			results[i] = cmd.Execute()

			// Update progress
			progress := float64(i+1) / float64(len(batch.Commands))
			batch.Progress = progress
		}

		batch.Done = true

		return types.BatchCompleteMsg{
			BatchID: batch.ID,
			Success: true,
		}
	}, nil
}
