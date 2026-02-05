// Package terminal provides robust terminal management solutions for Bubble Tea UI
package terminal

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"sync"
	"syscall"
	"time"

	"github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/mattn/go-isatty"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/diagnostics"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
	"golang.org/x/term"
)

// TerminalManager provides robust terminal management for Bubble Tea applications
type TerminalManager struct {
	ctx                  context.Context
	cancel               context.CancelFunc
	config              TerminalConfig
	state               TerminalState
	diagnostics         *diagnostics.TerminalDiagnosticContext
	safetyMode          bool
	recoveryMode       bool
	gracefulShutdown    bool
	emergencyMode       bool
	terminalLost        bool
	terminalLostTime    time.Time
	recoveryAttempts    int
	maxRecoveryAttempts  int
	recoveryDelay       time.Duration
	mu                  sync.RWMutex
	initialized         bool
	startTime           time.Time
	lastActivity        time.Time
	activityTimeout    time.Duration
	handlers           *TerminalEventHandlers
	metrics            *TerminalMetrics
	profiling          *TerminalProfiling
	strategies         map[string]TerminalStrategy
	activeStrategy     string
	fallbackStrategies []string
}

// TerminalConfig defines terminal configuration
type TerminalConfig struct {
	EnableAltScreen     bool
	EnableMouse        bool
	EnableBracketedPaste bool
	EnableMouseCellMotion bool
	InputBuffer        int
	OutputBuffer       int
	MouseCellMotion    bool
	SafetyMode         bool
	RecoveryMode       bool
	ActivityTimeout    time.Duration
	MaxRecoveryAttempts int
	RecoveryDelay      time.Duration
	DiagnosticMode     bool
	ProfilingMode     bool
	ProfileDir        string
	LogLevel          string
	GracefulShutdown   bool
	TerminalType      string
	ForceTTY         bool
	FallbackStrategies []string
}

// TerminalState represents terminal state
type TerminalState struct {
	Initialized     bool
	Ready          bool
	Active         bool
	Safe           bool
	RecoveryMode   bool
	EmergencyMode  bool
	HasTTY         bool
	TTYPath        string
	Size           TerminalSize
	InputMode      TerminalInputMode
	OutputMode     TerminalOutputMode
	SignalHandlers map[string]bool
	ErrorCount     int
	LastError      error
	LastErrorTime  time.Time
	StartTime      time.Time
	Uptime         time.Duration
	LastActivity   time.Time
	ActivityCount  int
}

// TerminalSize represents terminal dimensions
type TerminalSize struct {
	Width  int
	Height int
	Cols   int
	Rows   int
}

// TerminalInputMode represents terminal input mode
type TerminalInputMode struct {
	Canonical    bool
	Echo         bool
	Interactive  bool
	Raw         bool
	Cooked      bool
}

// TerminalOutputMode represents terminal output mode
type TerminalOutputMode struct {
	Buffered   bool
	LineBuffered bool
	BlockBuffered bool
}

// TerminalEventHandlers handles terminal events
type TerminalEventHandlers struct {
	OnTerminalLost     func()
	OnTerminalReady    func()
	OnTerminalError    func(error)
	OnActivity         func()
	OnIdle            func(time.Duration)
	OnRecoverySuccess  func()
	OnRecoveryFailure  func(error)
	OnGracefulShutdown func()
	OnEmergencyMode    func()
}

// TerminalMetrics tracks terminal metrics
type TerminalMetrics struct {
	Uptime           time.Duration
	ActivityCount    int
	ErrorCount       int
	RecoveryCount    int
	RecoverySuccess  int
	RecoveryFailure  int
	ProfileCount     int
	DiagnosticCount  int
	TerminalLost     int
	LastCheck        time.Time
	Availability     float64
	HealthScore      float64
	Performance      float64
}

// TerminalProfiling manages terminal profiling
type TerminalProfiling struct {
	Active        bool
	ProfileDir    string
	CPUProfile    *os.File
	MemProfile    *os.File
	BlockProfile  *os.File
	GCProfile     *os.File
	TraceProfile  *os.File
	StartTime     time.Time
	ProfileCount  int
	ProfileInterval time.Duration
}

// TerminalStrategy defines terminal strategy interface
type TerminalStrategy interface {
	Name() string
	Initialize(config TerminalConfig) error
	Start(program *tea.Program) error
	Stop() error
	Recover() error
	IsSafe() bool
	CanHandleTerminalState(TerminalState) bool
	GetPriority() int
}

// NewTerminalManager creates a new terminal manager
func NewTerminalManager() *TerminalManager {
	ctx, cancel := context.WithCancel(context.Background())

	manager := &TerminalManager{
		ctx:                 ctx,
		cancel:              cancel,
		config:              getDefaultTerminalConfig(),
		state:               TerminalState{},
		safetyMode:          false,
		recoveryMode:       false,
		gracefulShutdown:    false,
		emergencyMode:      false,
		terminalLost:       false,
		maxRecoveryAttempts: 3,
		recoveryDelay:     2 * time.Second,
		mu:                sync.RWMutex{},
		initialized:        false,
		startTime:         time.Now(),
		lastActivity:      time.Now(),
		activityTimeout:   30 * time.Minute,
		handlers:          &TerminalEventHandlers{},
		metrics:           &TerminalMetrics{},
		profiling:         &TerminalProfiling{},
		strategies:        make(map[string]TerminalStrategy),
		activeStrategy:    "default",
		fallbackStrategies: []string{"fallback", "emergency"},
	}

	// Initialize strategies
	manager.initializeStrategies()

	return manager
}

// getDefaultTerminalConfig returns default terminal configuration
func getDefaultTerminalConfig() TerminalConfig {
	return TerminalConfig{
		EnableAltScreen:         true,
		EnableMouse:            false,
		EnableBracketedPaste:    true,
		EnableMouseCellMotion:  true,
		InputBuffer:            100,
		OutputBuffer:           100,
		MouseCellMotion:        true,
		SafetyMode:             true,
		RecoveryMode:           true,
		ActivityTimeout:        30 * time.Minute,
		MaxRecoveryAttempts:    3,
		RecoveryDelay:         2 * time.Second,
		DiagnosticMode:        true,
		ProfilingMode:         false,
		ProfileDir:            ".terminal_profiles",
		LogLevel:              "info",
		GracefulShutdown:      true,
		ForceTTY:             true,
		FallbackStrategies:   []string{"fallback", "emergency"},
	}
}

// initializeStrategies initializes terminal strategies
func (tm *TerminalManager) initializeStrategies() {
	// Register default strategy
	tm.registerStrategy(&DefaultStrategy{manager: tm})

	// Register fallback strategy
	tm.registerStrategy(&FallbackStrategy{manager: tm})

	// Register emergency strategy
	tm.registerStrategy(&EmergencyStrategy{manager: tm})

	// Register safe strategy
	tm.registerStrategy(&SafeStrategy{manager: tm})

	// Register profiling strategy
	tm.registerStrategy(&ProfilingStrategy{manager: tm})
}

// registerStrategy registers a terminal strategy
func (tm *TerminalManager) registerStrategy(strategy TerminalStrategy) {
	tm.strategies[strategy.Name()] = strategy
}

// Initialize initializes the terminal manager
func (tm *TerminalManager) Initialize() error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if tm.initialized {
		return errors.New("terminal manager already initialized")
	}

	// Check terminal environment
	if err := tm.checkTerminalEnvironment(); err != nil {
		return fmt.Errorf("terminal environment check failed: %w", err)
	}

	// Set up signal handling
	if err := tm.setupSignalHandling(); err != nil {
		return fmt.Errorf("signal handling setup failed: %w", err)
	}

	// Set up safety mechanisms
	if tm.config.SafetyMode {
		tm.enableSafetyMode()
	}

	// Initialize diagnostics
	if tm.config.DiagnosticMode {
		tm.initializeDiagnostics()
	}

	// Initialize profiling
	if tm.config.ProfilingMode {
		tm.initializeProfiling()
	}

	// Set initial state
	tm.state = TerminalState{
		Initialized:     true,
		Ready:          false,
		Active:         false,
		Safe:           tm.config.SafetyMode,
		RecoveryMode:   tm.config.RecoveryMode,
		HasTTY:         tm.checkTTYAccess(),
		TTYPath:        "/dev/tty",
		Size:          tm.getTerminalSize(),
		InputMode:     tm.getInputMode(),
		OutputMode:    tm.getOutputMode(),
		SignalHandlers: make(map[string]bool),
		ErrorCount:    0,
		StartTime:     time.Now(),
		LastActivity:  time.Now(),
		ActivityCount: 0,
	}

	// Set initial strategy
	tm.selectBestStrategy()

	tm.initialized = true
	tm.startTime = time.Now()

	// Call initialization event handler
	if tm.handlers.OnTerminalReady != nil {
		tm.handlers.OnTerminalReady()
	}

	return nil
}

// Start starts the terminal manager and creates Bubble Tea program
func (tm *TerminalManager) Start(model tea.Model) (*tea.Program, error) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if !tm.initialized {
		return nil, errors.New("terminal manager not initialized")
	}

	// Get the best strategy
	strategy := tm.getBestStrategy()
	if strategy == nil {
		return nil, errors.New("no suitable terminal strategy found")
	}

	// Create program options
	programOptions := tm.createProgramOptions(model)

	// Create the program
	program := tea.NewProgram(model, programOptions...)

	// Start the strategy
	if err := strategy.Start(program); err != nil {
		return nil, fmt.Errorf("strategy start failed: %w", err)
	}

	// Update state
	tm.state.Active = true
	tm.state.Ready = true
	tm.startTime = time.Now()

	return program, nil
}

// Stop stops the terminal manager gracefully
func (tm *TerminalManager) Stop() error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if !tm.state.Active {
		return nil
	}

	// Graceful shutdown
	if tm.config.GracefulShutdown {
		tm.gracefulShutdown = true
		if tm.handlers.OnGracefulShutdown != nil {
			tm.handlers.OnGracefulShutdown()
		}
	}

	// Stop the active strategy
	if strategy := tm.getBestStrategy(); strategy != nil {
		if err := strategy.Stop(); err != nil {
			return fmt.Errorf("strategy stop failed: %w", err)
		}
	}

	// Cancel context
	tm.cancel()

	// Stop profiling
	if tm.profiling.Active {
		tm.stopProfiling()
	}

	// Update state
	tm.state.Active = false
	tm.state.Ready = false

	return nil
}

// Recover attempts to recover from terminal issues
func (tm *TerminalManager) Recover() error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	// Check if we're in recovery mode
	if tm.recoveryMode {
		return nil
	}

	// Check recovery attempts
	if tm.recoveryAttempts >= tm.config.MaxRecoveryAttempts {
		return fmt.Errorf("max recovery attempts (%d) exceeded", tm.config.MaxRecoveryAttempts)
	}

	tm.recoveryMode = true
	tm.recoveryAttempts++

	// Call recovery event handler
	if tm.handlers.OnTerminalLost != nil {
		tm.handlers.OnTerminalLost()
	}

	// Get fallback strategy
	strategy := tm.getFallbackStrategy()
	if strategy == nil {
		return errors.New("no fallback strategy available")
	}

	// Attempt recovery
	if err := strategy.Recover(); err != nil {
		tm.recoveryMode = false
		if tm.handlers.OnRecoveryFailure != nil {
			tm.handlers.OnRecoveryFailure(err)
		}
		return fmt.Errorf("recovery failed: %w", err)
	}

	// Recovery successful
	tm.recoveryMode = false
	tm.state.RecoveryMode = false
	tm.state.Safe = true

	// Update metrics
	tm.metrics.RecoveryCount++
	tm.metrics.RecoverySuccess++

	// Call recovery success event handler
	if tm.handlers.OnRecoverySuccess != nil {
		tm.handlers.OnRecoverySuccess()
	}

	return nil
}

// EmergencyMode enables emergency mode for critical failures
func (tm *TerminalManager) EmergencyMode() error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if tm.emergencyMode {
		return nil
	}

	tm.emergencyMode = true
	tm.config.SafetyMode = true
	tm.config.RecoveryMode = false

	// Call emergency mode event handler
	if tm.handlers.OnEmergencyMode != nil {
		tm.handlers.OnEmergencyMode()
	}

	// Update metrics
	tm.metrics.ErrorCount++

	return nil
}

// CreateProgramOptions creates program options based on terminal state
func (tm *TerminalManager) createProgramOptions(model tea.Model) []tea.ProgramOption {
	options := make([]tea.ProgramOption, 0)

	// Determine output stream
	outputStream := tm.getOutputStream()

	// Add output option
	options = append(options, tea.WithOutput(outputStream))

	// Add input handling
	if tm.canUseInput() {
		options = append(options, tea.WithInput(os.Stdin))
	}

	// Add alt screen if enabled and safe
	if tm.shouldUseAltScreen() {
		options = append(options, tea.WithAltScreen())
	}

	// Add mouse support if enabled and safe
	if tm.shouldUseMouse() {
		options = append(options, tea.WithMouseCellMotion())
	}

	// Add signal handling if enabled
	if tm.config.GracefulShutdown {
		options = append(options, tm.gracefulShutdownOption())
	}

	// Add safety options
	if tm.config.SafetyMode {
		options = append(options, tm.safetyOptions()...)
	}

	return options
}

// SafetyMode enables safety mode
func (tm *TerminalManager) enableSafetyMode() {
	tm.safetyMode = true
	tm.config.SafetyMode = true

	// Add safety options
	tm.config.EnableAltScreen = false
	tm.config.EnableMouse = false
	tm.config.InputBuffer = 50
	tm.config.OutputBuffer = 50
}

// checkTerminalEnvironment checks terminal environment
func (tm *TerminalManager) checkTerminalEnvironment() error {
	// Check if terminal is available
	if tm.config.ForceTTY && !tm.checkTTYAccess() {
		return errors.New("terminal not available or inaccessible")
	}

	// Check environment variables
	if err := tm.checkEnvironment(); err != nil {
		return fmt.Errorf("environment check failed: %w", err)
	}

	// Check permissions
	if err := tm.checkPermissions(); err != nil {
		return fmt.Errorf("permissions check failed: %w", err)
	}

	return nil
}

// setupSignalHandling sets up signal handling
func (tm *TerminalManager) setupSignalHandling() error {
	// Create signal channel
	sigChan := make(chan os.Signal, 1)

	// Register signals
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM, syscall.SIGPIPE)

	// Start signal handler
	go func() {
		for {
			select {
			case <-tm.ctx.Done():
				return
			case sig := <-sigChan:
				tm.handleSignal(sig)
			}
		}
	}()

	// Update signal handlers
	tm.state.SignalHandlers["SIGINT"] = true
	tm.state.SignalHandlers["SIGTERM"] = true
	tm.state.SignalHandlers["SIGPIPE"] = true

	return nil
}

// handleSignal handles signals
func (tm *TerminalManager) handleSignal(sig os.Signal) {
	switch sig {
	case syscall.SIGINT:
		tm.handleSIGINT()
	case syscall.SIGTERM:
		tm.handleSIGTERM()
	case syscall.SIGPIPE:
		tm.handleSIGPIPE()
	}
}

// handleSIGINT handles SIGINT
func (tm *TerminalManager) handleSIGINT() {
	if tm.config.GracefulShutdown {
		tm.gracefulShutdown = true
		if tm.handlers.OnGracefulShutdown != nil {
			tm.handlers.OnGracefulShutdown()
		}
	}
	tm.stop()
}

// handleSIGTERM handles SIGTERM
func (tm *TerminalManager) handleSIGTERM() {
	if tm.config.GracefulShutdown {
		tm.gracefulShutdown = true
		if tm.handlers.OnGracefulShutdown != nil {
			tm.handlers.OnGracefulShutdown()
		}
	}
	tm.stop()
}

// handleSIGPIPE handles SIGPIPE
func (tm *TerminalManager) handleSIGPIPE() {
	// SIGPIPE typically indicates broken pipe
	tm.handleError(fmt.Errorf("SIGPIPE received - broken pipe detected"))
}

// stop stops the terminal manager
func (tm *TerminalManager) stop() {
	tm.cancel()
	tm.Stop()
}

// handleError handles errors
func (tm *TerminalManager) handleError(err error) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	tm.state.ErrorCount++
	tm.state.LastError = err
	tm.state.LastErrorTime = time.Now()

	// Update metrics
	tm.metrics.ErrorCount++

	// Call error event handler
	if tm.handlers.OnTerminalError != nil {
		tm.handlers.OnTerminalError(err)
	}

	// Attempt recovery if in recovery mode
	if tm.config.RecoveryMode && tm.recoveryMode {
		go func() {
			time.Sleep(tm.config.RecoveryDelay)
			if recoveryErr := tm.Recover(); recoveryErr != nil {
				tm.handleError(recoveryErr)
			}
		}()
	}
}

// checkTTYAccess checks TTY access
func (tm *TerminalManager) checkTTYAccess() bool {
	// Check if stdin is a terminal
	if !isatty.IsTerminal(os.Stdin.Fd()) {
		return false
	}

	// Check if we can open /dev/tty
	if _, err := os.Stat("/dev/tty"); err != nil {
		return false
	}

	// Try to open /dev/tty for reading/writing
	file, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		return false
	}
	file.Close()

	return true
}

// getTerminalSize gets terminal size
func (tm *TerminalManager) getTerminalSize() TerminalSize {
	size := TerminalSize{
		Width:  80,
		Height: 24,
		Cols:   80,
		Rows:   24,
	}

	// Try to get actual terminal size
	if term.IsTerminal(os.Stdin.Fd()) {
		if width, height, err := term.GetSize(int(os.Stdin.Fd())); err == nil {
			size.Width = width
			size.Height = height
			size.Cols = width
			size.Rows = height
		}
	}

	return size
}

// getInputMode gets terminal input mode
func (tm *TerminalManager) getInputMode() TerminalInputMode {
	return TerminalInputMode{
		Canonical:   true,
		Echo:        false,
		Interactive: isatty.IsTerminal(os.Stdin.Fd()),
		Raw:         false,
		Cooked:      true,
	}
}

// getOutputMode gets terminal output mode
func (tm *TerminalManager) getOutputMode() TerminalOutputMode {
	return TerminalOutputMode{
		Buffered:   false,
		LineBuffered: false,
		BlockBuffered: false,
	}
}

// getOutputStream determines the appropriate output stream
func (tm *TerminalManager) getOutputStream() io.Writer {
	// Try stderr first for better TTY compatibility
	if isatty.IsTerminal(os.Stderr.Fd()) {
		return os.Stderr
	}

	// Fall back to stdout
	return os.Stdout
}

// canUseInput determines if we can use input
func (tm *TerminalManager) canUseInput() bool {
	return isatty.IsTerminal(os.Stdin.Fd()) && tm.state.InputMode.Interactive
}

// shouldUseAltScreen determines if we should use alt screen
func (tm *TerminalManager) shouldUseAltScreen() bool {
	return tm.config.EnableAltScreen && tm.state.Safe && !tm.emergencyMode
}

// shouldUseMouse determines if we should use mouse support
func (tm *TerminalManager) shouldUseMouse() bool {
	return tm.config.EnableMouse && tm.state.Safe && !tm.emergencyMode
}

// gracefulShutdownOption creates graceful shutdown option
func (tm *TerminalManager) gracefulShutdownOption() tea.ProgramOption {
	return tea.WithExitSubscriber(func() tea.QuitMsg {
		if tm.gracefulShutdown {
			return tea.QuitMsg{}
		}
		return nil
	})
}

// safetyOptions creates safety options
func (tm *TerminalManager) safetyOptions() []tea.ProgramOption {
	options := make([]tea.ProgramOption, 0)

	// Disable blocking operations
	if tm.safetyMode {
		options = append(options, tea.WithInput(os.Stdin))
	}

	return options
}

// selectBestStrategy selects the best available strategy
func (tm *TerminalManager) selectBestStrategy() {
	bestStrategy := "default"
	bestPriority := 0

	for name, strategy := range tm.strategies {
		if strategy.CanHandleTerminalState(tm.state) && strategy.GetPriority() > bestPriority {
			bestStrategy = name
			bestPriority = strategy.GetPriority()
		}
	}

	tm.activeStrategy = bestStrategy
}

// getBestStrategy gets the best available strategy
func (tm *TerminalManager) getBestStrategy() TerminalStrategy {
	if strategy, exists := tm.strategies[tm.activeStrategy]; exists {
		return strategy
	}

	// Fall back to default
	if strategy, exists := tm.strategies["default"]; exists {
		return strategy
	}

	return nil
}

// getFallbackStrategy gets the fallback strategy
func (tm *TerminalManager) getFallbackStrategy() TerminalStrategy {
	for _, name := range tm.fallbackStrategies {
		if strategy, exists := tm.strategies[name]; exists {
			return strategy
		}
	}
	return nil
}

// initializeDiagnostics initializes diagnostics
func (tm *TerminalManager) initializeDiagnostics() {
	tm.diagnostics = diagnostics.NewDiagnosticContext()
}

// initializeProfiling initializes profiling
func (tm *TerminalManager) initializeProfiling() {
	tm.profiling.ProfileDir = tm.config.ProfileDir
	tm.profiling.ProfileInterval = 30 * time.Second
	tm.profiling.Active = true
}

// stopProfiling stops profiling
func (tm *TerminalManager) stopProfiling() {
	if tm.profiling.CPUProfile != nil {
		tm.profiling.CPUProfile.Close()
	}
	if tm.profiling.MemProfile != nil {
		tm.profiling.MemProfile.Close()
	}
	if tm.profiling.BlockProfile != nil {
		tm.profiling.BlockProfile.Close()
	}
	if tm.profiling.GCProfile != nil {
		tm.profiling.GCProfile.Close()
	}
	if tm.profiling.TraceProfile != nil {
		tm.profiling.TraceProfile.Close()
	}
	tm.profiling.Active = false
}

// checkEnvironment checks environment
func (tm *TerminalManager) checkEnvironment() error {
	// Check terminal environment variables
	requiredVars := []string{"TERM", "HOME", "SHELL"}

	for _, varName := range requiredVars {
		if os.Getenv(varName) == "" {
			return fmt.Errorf("required environment variable %s not set", varName)
		}
	}

	return nil
}

// checkPermissions checks permissions
func (tm *TerminalManager) checkPermissions() error {
	// Check if we can access terminal devices
	if tm.state.HasTTY {
		if _, err := os.Stat("/dev/tty"); err != nil {
			return fmt.Errorf("cannot access terminal device: %w", err)
		}
	}

	return nil
}

// Monitor monitors terminal activity
func (tm *TerminalManager) Monitor() {
	ticker := time.NewTicker(tm.config.ActivityTimeout / 2)
	defer ticker.Stop()

	for {
		select {
		case <-tm.ctx.Done():
			return
		case <-ticker.C:
			tm.checkActivity()
		}
	}
}

// checkActivity checks terminal activity
func (tm *TerminalManager) checkActivity() {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	now := time.Now()
	idleTime := now.Sub(tm.state.LastActivity)

	if idleTime > tm.config.ActivityTimeout {
		// Terminal is idle
		if tm.config.RecoveryMode {
			go func() {
				time.Sleep(tm.config.RecoveryDelay)
				if err := tm.Recover(); err != nil {
					tm.handleError(err)
				}
			}()
		}

		// Call idle event handler
		if tm.handlers.OnIdle != nil {
			tm.handlers.OnIdle(idleTime)
		}
	}

	// Update metrics
	tm.metrics.LastCheck = now
	tm.metrics.Availability = tm.calculateAvailability()
	tm.metrics.HealthScore = tm.calculateHealthScore()
	tm.metrics.Performance = tm.calculatePerformance()
}

// calculateAvailability calculates availability
func (tm *TerminalManager) calculateAvailability() float64 {
	if tm.metrics.ErrorCount == 0 {
		return 1.0
	}

	return 1.0 - float64(tm.metrics.ErrorCount)/float64(tm.metrics.ActivityCount+1)
}

// calculateHealthScore calculates health score
func (tm *TerminalManager) calculateHealthScore() float64 {
	if tm.state.Safe && !tm.state.RecoveryMode && !tm.emergencyMode {
		return 1.0
	}

	if tm.state.RecoveryMode {
		return 0.7
	}

	if tm.emergencyMode {
		return 0.3
	}

	return 0.8
}

// calculatePerformance calculates performance score
func (tm *TerminalManager) calculatePerformance() float64 {
	if tm.metrics.ActivityCount == 0 {
		return 1.0
	}

	return 1.0 - float64(tm.metrics.ErrorCount)/float64(tm.metrics.ActivityCount)
}

// RegisterEventHandlers registers event handlers
func (tm *TerminalManager) RegisterEventHandlers(handlers TerminalEventHandlers) {
	tm.handlers = &handlers
}

// GetState gets current terminal state
func (tm *TerminalManager) GetState() TerminalState {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	return tm.state
}

// GetMetrics gets terminal metrics
func (tm *TerminalManager) GetMetrics() TerminalMetrics {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	return *tm.metrics
}

// GetDiagnostics gets diagnostic results
func (tm *TerminalManager) GetDiagnostics() *diagnostics.TerminalDiagnosticResult {
	if tm.diagnostics == nil {
		return nil
	}

	return tm.diagnostics.GenerateDiagnosticReport()
}

// IsInitialized checks if terminal manager is initialized
func (tm *TerminalManager) IsInitialized() bool {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	return tm.initialized
}

// IsSafe checks if terminal manager is in safe mode
func (tm *TerminalManager) IsSafe() bool {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	return tm.state.Safe
}

// IsEmergencyMode checks if emergency mode is enabled
func (tm *TerminalManager) IsEmergencyMode() bool {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	return tm.emergencyMode
}

// GetActiveStrategy gets active strategy name
func (tm *TerminalManager) GetActiveStrategy() string {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	return tm.activeStrategy
}

// GenerateReport generates a comprehensive terminal management report
func (tm *TerminalManager) GenerateReport() string {
	var report strings.Builder

	// Header
	report.WriteString("🔍 TERMINAL MANAGEMENT REPORT\n")
	report.WriteString("=" * 60 + "\n\n")

	// Current state
	report.WriteString("📊 CURRENT STATE\n")
	report.WriteString("-" * 40 + "\n")
	state := tm.GetState()
	report.WriteString(fmt.Sprintf("   Initialized: %v\n", state.Initialized))
	report.WriteString(fmt.Sprintf("   Active: %v\n", state.Active))
	report.WriteString(fmt.Sprintf("   Ready: %v\n", state.Ready))
	report.WriteString(fmt.Sprintf("   Safe: %v\n", state.Safe))
	report.WriteString(fmt.Sprintf("   Recovery Mode: %v\n", state.RecoveryMode))
	report.WriteString(fmt.Sprintf("   Emergency Mode: %v\n", tm.emergencyMode))
	report.WriteString(fmt.Sprintf("   Has TTY: %v\n", state.HasTTY))
	report.WriteString(fmt.Sprintf("   Terminal Size: %dx%d\n", state.Size.Width, state.Size.Height))
	report.WriteString("\n")

	// Configuration
	report.WriteString("⚙️  CONFIGURATION\n")
	report.WriteString("-" * 40 + "\n")
	config := tm.config
	report.WriteString(fmt.Sprintf("   Safety Mode: %v\n", config.SafetyMode))
	report.WriteString(fmt.Sprintf("   Recovery Mode: %v\n", config.RecoveryMode))
	report.WriteString(fmt.Sprintf("   Alt Screen: %v\n", config.EnableAltScreen))
	report.WriteString(fmt.Sprintf("   Mouse Support: %v\n", config.EnableMouse))
	report.WriteString(fmt.Sprintf("   Activity Timeout: %v\n", config.ActivityTimeout))
	report.WriteString(fmt.Sprintf("   Max Recovery Attempts: %d\n", config.MaxRecoveryAttempts))
	report.WriteString(fmt.Sprintf("   Graceful Shutdown: %v\n", config.GracefulShutdown))
	report.WriteString("\n")

	// Metrics
	report.WriteString("📈 METRICS\n")
	report.WriteString("-" * 40 + "\n")
	metrics := tm.GetMetrics()
	report.WriteString(fmt.Sprintf("   Uptime: %v\n", metrics.Uptime))
	report.WriteString(fmt.Sprintf("   Activity Count: %d\n", metrics.ActivityCount))
	report.WriteString(fmt.Sprintf("   Error Count: %d\n", metrics.ErrorCount))
	report.WriteString(fmt.Sprintf("   Recovery Count: %d\n", metrics.RecoveryCount))
	report.WriteString(fmt.Sprintf("   Recovery Success: %d\n", metrics.RecoverySuccess))
	report.WriteString(fmt.Sprintf("   Recovery Failure: %d\n", metrics.RecoveryFailure))
	report.WriteString(fmt.Sprintf("   Availability: %.2f%%\n", metrics.Availability*100))
	report.WriteString(fmt.Sprintf("   Health Score: %.2f\n", metrics.HealthScore))
	report.WriteString(fmt.Sprintf("   Performance: %.2f\n", metrics.Performance))
	report.WriteString("\n")

	// Active strategy
	report.WriteString("🎯 ACTIVE STRATEGY\n")
	report.WriteString("-" * 40 + "\n")
	report.WriteString(fmt.Sprintf("   Name: %s\n", tm.GetActiveStrategy()))
	report.WriteString(fmt.Sprintf("   Safety Mode: %v\n", tm.IsSafe()))
	report.WriteString(fmt.Sprintf("   Emergency Mode: %v\n", tm.IsEmergencyMode()))
	report.WriteString("\n")

	// Diagnostics
	if diagnostics := tm.GetDiagnostics(); diagnostics != nil {
		report.WriteString("🔍 DIAGNOSTICS\n")
		report.WriteString("-" * 40 + "\n")
		report.WriteString(diagnostics)
		report.WriteString("\n")
	}

	return report.String()
}

// TerminalStrategy implementations

// DefaultStrategy is the default terminal strategy
type DefaultStrategy struct {
	manager *TerminalManager
}

func (s *DefaultStrategy) Name() string { return "default" }

func (s *DefaultStrategy) Initialize(config TerminalConfig) error {
	return nil
}

func (s *DefaultStrategy) Start(program *tea.Program) error {
	return nil
}

func (s *DefaultStrategy) Stop() error {
	return nil
}

func (s *DefaultStrategy) Recover() error {
	return nil
}

func (s *DefaultStrategy) IsSafe() bool {
	return true
}

func (s *DefaultStrategy) CanHandleTerminalState(state TerminalState) bool {
	return state.Initialized && !state.RecoveryMode && !state.EmergencyMode
}

func (s *DefaultStrategy) GetPriority() int {
	return 1
}

// FallbackStrategy is a fallback terminal strategy
type FallbackStrategy struct {
	manager *TerminalManager
}

func (s *FallbackStrategy) Name() string { return "fallback" }

func (s *FallbackStrategy) Initialize(config TerminalConfig) error {
	// Disable advanced features in fallback mode
	config.EnableAltScreen = false
	config.EnableMouse = false
	config.InputBuffer = 50
	config.OutputBuffer = 50
	return nil
}

func (s *FallbackStrategy) Start(program *tea.Program) error {
	return nil
}

func (s *FallbackStrategy) Stop() error {
	return nil
}

func (s *FallbackStrategy) Recover() error {
	return nil
}

func (s *FallbackStrategy) IsSafe() bool {
	return true
}

func (s *FallbackStrategy) CanHandleTerminalState(state TerminalState) bool {
	return state.Initialized && !state.EmergencyMode
}

func (s *FallbackStrategy) GetPriority() int {
	return 2
}

// EmergencyStrategy is an emergency terminal strategy
type EmergencyStrategy struct {
	manager *TerminalManager
}

func (s *EmergencyStrategy) Name() string { return "emergency" }

func (s *EmergencyStrategy) Initialize(config TerminalConfig) error {
	// Disable all advanced features in emergency mode
	config.EnableAltScreen = false
	config.EnableMouse = false
	config.SafetyMode = true
	config.RecoveryMode = false
	return nil
}

func (s *EmergencyStrategy) Start(program *tea.Program) error {
	return nil
}

func (s *EmergencyStrategy) Stop() error {
	return nil
}

func (s *EmergencyStrategy) Recover() error {
	return nil
}

func (s *EmergencyStrategy) IsSafe() bool {
	return true
}

func (s *EmergencyStrategy) CanHandleTerminalState(state TerminalState) bool {
	return state.Initialized
}

func (s *EmergencyStrategy) GetPriority() int {
	return 3
}

// SafeStrategy is a safe terminal strategy
type SafeStrategy struct {
	manager *TerminalManager
}

func (s *SafeStrategy) Name() string { return "safe" }

func (s *SafeStrategy) Initialize(config TerminalConfig) error {
	// Enable safety features
	config.SafetyMode = true
	config.EnableAltScreen = false
	config.EnableMouse = false
	config.InputBuffer = 100
	config.OutputBuffer = 100
	return nil
}

func (s *SafeStrategy) Start(program *tea.Program) error {
	return nil
}

func (s *SafeStrategy) Stop() error {
	return nil
}

func (s *SafeStrategy) Recover() error {
	return nil
}

func (s *SafeStrategy) IsSafe() bool {
	return true
}

func (s *SafeStrategy) CanHandleTerminalState(state TerminalState) bool {
	return state.Safe && !state.EmergencyMode
}

func (s *SafeStrategy) GetPriority() int {
	return 4
}

// ProfilingStrategy is a profiling terminal strategy
type ProfilingStrategy struct {
	manager *TerminalManager
}

func (s *ProfilingStrategy) Name() string { return "profiling" }

func (s *ProfilingStrategy) Initialize(config TerminalConfig) error {
	// Enable profiling features
	config.ProfilingMode = true
	return nil
}

func (s *ProfilingStrategy) Start(program *tea.Program) error {
	return nil
}

func (s *ProfilingStrategy) Stop() error {
	return nil
}

func (s *ProfilingStrategy) Recover() error {
	return nil
}

func (s *ProfilingStrategy) IsSafe() bool {
	return true
}

func (s *ProfilingStrategy) CanHandleTerminalState(state TerminalState) bool {
	return state.Initialized && state.Safe
}

func (s *ProfilingStrategy) GetPriority() int {
	return 5
}

// CreateSafeBubbleTeaProgram creates a safe Bubble Tea program using terminal manager
func CreateSafeBubbleTeaProgram(model tea.Model, config TerminalConfig) (*tea.Program, error) {
	// Create terminal manager
	manager := NewTerminalManager()
	manager.config = config

	// Register event handlers
	manager.RegisterEventHandlers(TerminalEventHandlers{
		OnTerminalReady: func() {
			fmt.Println("Terminal ready")
		},
		OnTerminalError: func(err error) {
			fmt.Printf("Terminal error: %v\n", err)
		},
		OnRecoverySuccess: func() {
			fmt.Println("Terminal recovery successful")
		},
		OnRecoveryFailure: func(err error) {
			fmt.Printf("Terminal recovery failed: %v\n", err)
		},
	})

	// Initialize manager
	if err := manager.Initialize(); err != nil {
		return nil, fmt.Errorf("terminal manager initialization failed: %w", err)
	}

	// Start program
	program, err := manager.Start(model)
	if err != nil {
		return nil, fmt.Errorf("program start failed: %w", err)
	}

	// Start monitoring
	go manager.Monitor()

	return program, nil
}