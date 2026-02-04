// internal/ui/types/types.go
package types

import (
	"context"
	"time"
	"fmt"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/scooter-lacroix/mlstack-installer/internal/installer"
)

// Stage represents the main application stage
type Stage int

const (
	StageWelcome Stage = iota
	StageHardwareDetect
	StagePreFlightCheck
	StageComponentSelect
	StageConfiguration
	StageConfirm
	StageInstalling
	StageComplete
	StageRecovery
)

func (s Stage) String() string {
	switch s {
	case StageWelcome:
		return "welcome"
	case StageHardwareDetect:
		return "hardware"
	case StagePreFlightCheck:
		return "preflight"
	case StageComponentSelect:
		return "components"
	case StageConfiguration:
		return "configuration"
	case StageConfirm:
		return "confirm"
	case StageInstalling:
		return "installing"
	case StageComplete:
		return "complete"
	case StageRecovery:
		return "recovery"
	default:
		return "unknown"
	}
}

// Component represents an installable ML stack component
type Component struct {
	ID          string
	Name        string
	Description string
	Script      string
	Category    string
	Required    bool
	Selected    bool
	Installed   bool
	Progress    float64
	Size        int64
	Estimate    string
}

// State represents the complete application state
type State struct {
	// Core UI state
	Stage      Stage
	Width      int
	Height     int
	Ready      bool
	Quitting   bool
	LastUpdate time.Time

	// Hardware and system information
	GPUInfo          installer.GPUInfo
	SystemInfo       installer.SystemInfo
	PreflightResults *installer.PreFlightCheckResults
	HardwareDetected bool
	PreFlightPassed  bool

	// Component management
	Components         []Component
	SelectedCategories map[string]bool
	CurrentComponent   int

	// Progress and status
	Progress   progress.Model
	Spinner    spinner.Model
	InstallLog []string
	ErrorLog   []string

	// Configuration
	Config         *installer.Config
	ConfigModified bool
	ConfigLoaded   bool

	// Recovery and error handling
	RecoveryMode       bool
	ErrorCount         int
	RecoveryOptions    []installer.RecoveryOption
	AvailableSnapshots []installer.SnapshotInfo
	CurrentRecovery    string
	RecoveryProgress   float64

	// UI specific state
	ShowDetails       bool
	SelectedCheck     int
	AutoFixMode       bool
	FixProgress       float64
	SystemScore       int
	TotalTimeEstimate time.Duration
	CurrentPhase      string
}

// Message is the base interface for all UI messages
type Message interface {
	IsMessage()
}

// Navigation messages
type NavigateToStageMsg struct {
	Stage Stage
}

func (NavigateToStageMsg) IsMessage() {}

type NavigateBackMsg struct{}

func (NavigateBackMsg) IsMessage() {}

type NavigateForwardMsg struct{}

func (NavigateForwardMsg) IsMessage() {}

// Hardware detection messages
type HardwareDetectedMsg struct {
	GPUInfo          installer.GPUInfo
	SystemInfo       installer.SystemInfo
	PreflightResults *installer.PreFlightCheckResults
	Error            error
}

func (HardwareDetectedMsg) IsMessage() {}

type HardwareProgressMsg struct {
	Step     string
	Progress float64
	Total    int
}

func (HardwareProgressMsg) IsMessage() {}

// Installation messages
type ComponentToggleMsg struct {
	ComponentID string
}

func (ComponentToggleMsg) IsMessage() {}

type InstallationStartMsg struct {
	Components []Component
}

func (InstallationStartMsg) IsMessage() {}

type InstallationProgressMsg struct {
	ComponentID string
	Progress    float64
	Message     string
}

func (InstallationProgressMsg) IsMessage() {}

type InstallationCompleteMsg struct {
	ComponentID string
	Success     bool
	Error       error
}

func (InstallationCompleteMsg) IsMessage() {}

// Configuration messages
type ConfigLoadMsg struct {
	Config *installer.Config
	Error  error
}

func (ConfigLoadMsg) IsMessage() {}

type ConfigSaveMsg struct {
	Success bool
	Message string
}

func (ConfigSaveMsg) IsMessage() {}

// Recovery messages
type RecoveryTriggerMsg struct {
	ErrorType    string
	ErrorMessage string
}

func (RecoveryTriggerMsg) IsMessage() {}

type RecoveryOptionSelectMsg struct {
	OptionIndex int
}

func (RecoveryOptionSelectMsg) IsMessage() {}

// UI state messages
type WindowSizeMsg struct {
	Width  int
	Height int
}

func (WindowSizeMsg) IsMessage() {}

type TickMsg struct {
	Time time.Time
}

func (TickMsg) IsMessage() {}

// Component interface for all UI components
type UIComponent interface {
	Init() tea.Cmd
	Update(tea.Msg) (tea.Model, tea.Cmd)
	View() string
}

// LayoutComponent interface for layout-aware components
type LayoutComponent interface {
	UIComponent
	SetBounds(x, y, width, height int)
	GetBounds() (x, y, width, height int)
}

// StateManager interface for state management
type StateManager interface {
	GetState() State
	UpdateState(State) StateManager
	TransitionTo(Stage) StateManager
	// Window and layout management
	SetWindowSize(width, height int) StateManager
	// Validation and error handling
	IsValidTransition(from, to Stage) bool
	LogError(error)
	// Component management
	GetSelectedComponents() []Component
	ToggleComponent(componentID string) StateManager
	StartInstallation() StateManager
	UpdateComponentProgress(componentID string, progress float64, message string) StateManager
	CompleteComponent(componentID string, success bool, err error) StateManager
	// Hardware and configuration
	SetHardwareInfo(gpu GPUInfo, system SystemInfo) StateManager
	SetPreflightResults(results *installer.PreFlightCheckResults) StateManager
	SetConfig(config *installer.Config) StateManager
	TriggerRecovery(errorType, errorMessage string) StateManager
	// Additional utilities
	GetRequiredComponents() []Component
	GetCurrentComponent() *Component
	GetLogs() []string
	GetErrorLog() []string
}

// Logger interface for debugging and logging
type Logger interface {
	Log(message string)
	LogError(err error)
	LogProgress(message string)
	GetLogs() []string
}

// ErrorHandler interface for error handling
type ErrorHandler interface {
	HandleError(err error) tea.Cmd
	TriggerRecovery(errorType, errorMessage string) tea.Cmd
	CanRecover() bool
	GetRecoveryOptions() []installer.RecoveryOption
}

// Security interfaces
type SecurityValidator interface {
	ValidateInput(input string, context string) error
	ValidateScriptPath(path string) error
	ValidatePrivileges(operation string) error
	SanitizeInput(input string) string
	PreventCommandInjection(command string) string
	PerformSecurityCheck(input string, context string, operation string) *SecurityCheckResult
	GetSecurityReport() map[string]interface{}
}

// SecurityCheckResult represents the result of a security check
type SecurityCheckResult struct {
	Valid  bool
	Error  error
	Checks map[string]bool
	Log    []string
}

// Performance monitoring interfaces
type PerformanceMonitor interface {
	StartOperation(name string)
	FinishOperation(name string, success bool)
	RecordMetric(metricName string, value float64)
	GetMetrics() map[string]float64
}

// PerformanceAlert represents a performance alert
type PerformanceAlert struct {
	Type       string
	Message    string
	MetricName string
	Value      float64
	Threshold  float64
	Timestamp  time.Time
	Severity   string // "warning" or "error"
}

// GPUInfo represents GPU hardware information
type GPUInfo struct {
	Vendor        string  `json:"vendor"`
	Model         string  `json:"model"`
	Driver        string  `json:"driver"`
	ComputeUnits  int     `json:"compute_units"`
	MemoryGB      float64 `json:"memory_gb"`
	Architecture  string  `json:"architecture"`
	GFXVersion    string  `json:"gfx_version"`
	Temperature   float64 `json:"temperature"`
	GPUCount      int     `json:"gpu_count"`
	Status        string  `json:"status"`
	Optimizations string  `json:"optimizations"`
	PowerUsage    float64 `json:"power_usage"`
}

// SystemInfo represents system hardware information
type SystemInfo struct {
	OS            string        `json:"os"`
	Distribution  string        `json:"distribution"`
	KernelVersion string        `json:"kernel_version"`
	Architecture  string        `json:"architecture"`
	CPU           CPUInfo       `json:"cpu"`
	Memory        MemoryInfo    `json:"memory"`
	Storage       []StorageInfo `json:"storage"`
	Timestamp     time.Time     `json:"timestamp"`
}

// CPUInfo represents CPU information
type CPUInfo struct {
	Model      string   `json:"model"`
	Cores      int      `json:"cores"`
	Threads    int      `json:"threads"`
	ClockSpeed float64  `json:"clock_speed"`
	CacheSize  uint64   `json:"cache_size"`
	Flags      []string `json:"flags"`
}

// MemoryInfo represents memory information
type MemoryInfo struct {
	TotalGB     float64 `json:"total_gb"`
	AvailableGB float64 `json:"available_gb"`
	UsedGB      float64 `json:"used_gb"`
	SwapTotalGB float64 `json:"swap_total_gb"`
	SwapUsedGB  float64 `json:"swap_used_gb"`
}

// StorageInfo represents storage information
type StorageInfo struct {
	Path        string  `json:"path"`
	Type        string  `json:"type"`
	SizeGB      float64 `json:"size_gb"`
	UsedGB      float64 `json:"used_gb"`
	AvailableGB float64 `json:"available_gb"`
}

// PreFlightCheckResults represents pre-flight check results
type PreFlightCheckResults struct {
	CanContinue     bool     `json:"can_continue"`
	PassedCount     int      `json:"passed_count"`
	FailedCount     int      `json:"failed_count"`
	Score           int      `json:"score"`
	Recommendations []string `json:"recommendations"`
	Errors          []string `json:"errors"`
	Warnings        []string `json:"warnings"`
}

// Integration interfaces
type IntegrationManager interface {
	DetectHardware() (installer.GPUInfo, installer.SystemInfo, error)
	RunScript(scriptPath string, args []string) (string, error)
	VerifyInstallation() (*installer.PreFlightCheckResults, error)
	SaveConfig(config *installer.Config) error
	LoadConfig() (*installer.Config, error)
}

// Batch operations
type BatchOperation struct {
	ID       string
	Commands []Command
	Done     bool
	Progress float64
}

// Command interface with enhanced features
type Command interface {
	Execute() tea.Msg
	Cancel() error
	GetID() string
	GetTimeout() time.Duration
}

// Async command with context
type AsyncCommand struct {
	ID       string
	Executor func() tea.Cmd
	CancelFn func() error
	Timeout  time.Duration
	Context  context.Context
	Done     bool
	Progress float64
}

// Execute implements the Command interface
func (a AsyncCommand) Execute() tea.Msg {
	return a.Executor()
}

func (a AsyncCommand) Cancel() error {
	if a.CancelFn != nil {
		return a.CancelFn()
	}
	return nil
}

func (a AsyncCommand) GetID() string {
	return a.ID
}

func (a AsyncCommand) GetTimeout() time.Duration {
	return a.Timeout
}

// Enhanced message types for comprehensive functionality

// Security-related messages
type SecurityValidationMsg struct {
	Input      string
	Validation error
	Context    string
}

func (SecurityValidationMsg) IsMessage() {}

type SecurityCheckMsg struct {
	Operation string
	Path      string
	Success   bool
	Error     error
}

func (SecurityCheckMsg) IsMessage() {}

// Performance monitoring messages
type MetricRecordMsg struct {
	Name      string
	Value     float64
	Timestamp time.Time
}

func (MetricRecordMsg) IsMessage() {}

type PerformanceWarningMsg struct {
	Operation string
	Duration  time.Duration
	Threshold time.Duration
}

func (p PerformanceWarningMsg) String() string {
	return fmt.Sprintf("Operation '%s' took %s, exceeding threshold of %s", p.Operation, p.Duration, p.Threshold)
}

func (PerformanceWarningMsg) IsMessage() {}

// Integration messages
type HardwareDetectionCompleteMsg struct {
	GPUInfo    installer.GPUInfo
	SystemInfo installer.SystemInfo
	Error      error
}

func (HardwareDetectionCompleteMsg) IsMessage() {}

type ScriptExecutionCompleteMsg struct {
	ScriptPath string
	Output     string
	Error      error
	Duration   time.Duration
}

func (ScriptExecutionCompleteMsg) IsMessage() {}

type VerificationCompleteMsg struct {
	Results *installer.PreFlightCheckResults
	Error   error
}

func (VerificationCompleteMsg) IsMessage() {}

// Configuration messages
type ConfigUpdateMsg struct {
	Config *installer.Config
	Field  string
	Value  interface{}
}

func (ConfigUpdateMsg) IsMessage() {}

type ConfigValidationErrorMsg struct {
	Field   string
	Message string
}

func (ConfigValidationErrorMsg) IsMessage() {}

// Batch operation messages
type BatchStartMsg struct {
	BatchID string
}

func (BatchStartMsg) IsMessage() {}

type BatchProgressMsg struct {
	BatchID  string
	Index    int
	Total    int
	Progress float64
}

func (BatchProgressMsg) IsMessage() {}

type BatchCompleteMsg struct {
	BatchID string
	Success bool
	Error   error
}

func (BatchCompleteMsg) IsMessage() {}

// Context cancellation
type CancelOperationMsg struct {
	OperationID string
	Reason      string
}

func (CancelOperationMsg) IsMessage() {}

// Debug and diagnostic messages
type DebugMsg struct {
	Level   string
	Message string
	Context map[string]interface{}
}

func (DebugMsg) IsMessage() {}

type DiagnosticMsg struct {
	Type     string
	Message  string
	Details  interface{}
	Severity string
}

func (DiagnosticMsg) IsMessage() {}

// Error recovery enhanced messages
type RecoveryAttemptMsg struct {
	ErrorType   string
	Error       error
	Strategy    string
	Attempt     int
	MaxAttempts int
}

func (RecoveryAttemptMsg) IsMessage() {}

type RecoverySuccessMsg struct {
	ErrorType string
	Message   string
	Steps     []string
}

func (RecoverySuccessMsg) IsMessage() {}

type RecoveryFailureMsg struct {
	ErrorType string
	Error     error
	Options   []installer.RecoveryOption
}

func (RecoveryFailureMsg) IsMessage() {}

// UI state enhancement messages
type FocusChangeMsg struct {
	Element string
	Active  bool
}

func (FocusChangeMsg) IsMessage() {}

type VisibilityChangeMsg struct {
	Element string
	Visible bool
}

func (VisibilityChangeMsg) IsMessage() {}

type LayoutUpdateMsg struct {
	Width    int
	Height   int
	Elements []string
}

func (LayoutUpdateMsg) IsMessage() {}

// Enhanced component selection messages
type CategoryFilterMsg struct {
	Category string
	Enabled  bool
}

func (CategoryFilterMsg) IsMessage() {}

type SearchQueryMsg struct {
	Query string
}

func (SearchQueryMsg) IsMessage() {}

type SortComponentsMsg struct {
	By    string
	Order string // "asc" or "desc"
}

func (SortComponentsMsg) IsMessage() {}

// Installation progress enhanced messages
type InstallationStartComponentMsg struct {
	ComponentID string
	StartTime   time.Time
}

func (InstallationStartComponentMsg) IsMessage() {}

type InstallationPhaseChangeMsg struct {
	ComponentID string
	Phase       string
	Percent     float64
}

func (InstallationPhaseChangeMsg) IsMessage() {}

type InstallationRollbackMsg struct {
	ComponentID string
	Reason      string
	Steps       []string
}

func (InstallationRollbackMsg) IsMessage() {}

// System status messages
type SystemStatusUpdateMsg struct {
	CPUUsage    float64
	MemoryUsage float64
	DiskUsage   float64
	GPUUsage    []float64
}

func (SystemStatusUpdateMsg) IsMessage() {}

type ResourceWarningMsg struct {
	Resource  string
	Usage     float64
	Threshold float64
	Message   string
}

func (ResourceWarningMsg) IsMessage() {}

// Theme and styling messages
type ThemeChangeMsg struct {
	Theme string
}

func (ThemeChangeMsg) IsMessage() {}

type StyleUpdateMsg struct {
	Element string
	Style   map[string]string
}

func (StyleUpdateMsg) IsMessage() {}

// Export and import messages
type ExportConfigMsg struct {
	Path   string
	Format string
}

func (ExportConfigMsg) IsMessage() {}

type ImportConfigMsg struct {
	Path   string
	Format string
	Config *installer.Config
	Error  error
}

func (ImportConfigMsg) IsMessage() {}

// Help and documentation messages
type ShowHelpMsg struct {
	Topic string
}

func (ShowHelpMsg) IsMessage() {}

type DocumentationRequestMsg struct {
	Topic string
	Page  string
}

func (DocumentationRequestMsg) IsMessage() {}

// User interaction messages
type UserInputMsg struct {
	Input string
	Field string
}

func (UserInputMsg) IsMessage() {}

type ConfirmationRequestMsg struct {
	Message string
	Yes     bool
}

func (ConfirmationRequestMsg) IsMessage() {}

// InitCompleteMsg for initialization completion
type InitCompleteMsg struct{}

func (InitCompleteMsg) IsMessage() {}
