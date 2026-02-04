// Package patterns provides alternative UI patterns and scaffolding options for Bubble Tea
package patterns

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/charmbracelet/bubbletea"
)

// UIPattern represents different UI architectural patterns
type UIPattern string

const (
	// ProgressiveProgressivePattern starts simple and gradually adds complexity
	ProgressivePattern UIPattern = "progressive"

	// GracefulDegradationPattern maintains basic functionality when advanced features fail
	GracefulDegradationPattern UIPattern = "graceful_degradation"

	// FailFastPattern catches and handles errors immediately
	FailFastPattern UIPattern = "fail_fast"

	// MinimalViableUIPattern provides core functionality only
	MinimalViableUIPattern UIPattern = "minimal_viable"

	// RecoveryOrientedPattern focuses on automatic recovery from failures
	RecoveryOrientedPattern UIPattern = "recovery_oriented"

	// ModularPattern separates concerns into independent components
	ModularPattern UIPattern = "modular"

	// HybridPattern combines multiple patterns for robustness
	HybridPattern UIPattern = "hybrid"

	// DebugPattern prioritizes debugging and diagnostics
	DebugPattern UIPattern = "debug"
)

// UIComponents represents the core components needed for any UI
type UIComponents struct {
	Terminal    TerminalStrategy
	State       StateManager
	ErrorHandling ErrorHandler
	Logging     Logger
	Signals     SignalHandler
	Performance PerformanceMonitor
}

// TerminalStrategy defines how terminal interactions are handled
type TerminalStrategy interface {
	Initialize() error
	Cleanup() error
	IsResponsive() bool
	BackupStrategy() TerminalStrategy
}

// StateManager handles application state transitions
type StateManager interface {
	GetCurrentState() string
	Transition(newState string) error
	IsValidTransition(from, to string) bool
	GetStateHistory() []string
}

// ErrorHandler provides comprehensive error handling capabilities
type ErrorHandler interface {
	HandleError(error) error
	Recover() interface{}
	RegisterPanicHandler()
	LogError(error, string)
}

// Logger provides structured logging capabilities
type Logger interface {
	Log(message string, level string, context map[string]interface{})
	Error(message string, err error)
	Debug(message string, context map[string]interface{})
	Info(message string, context map[string]interface{})
}

// SignalHandler manages system signals
type SignalHandler interface {
	RegisterSignalHandlers() error
	HandleSignal(sig os.Signal) error
	GracefulShutdown(ctx context.Context) error
}

// PerformanceMonitor tracks performance metrics
type PerformanceMonitor interface {
	TrackMetric(name string, value float64)
	GetMetrics() map[string]float64
	AlertOnThreshold(name string, threshold float64) bool
}

// ProgressiveUI implements the progressive enhancement pattern
type ProgressiveUI struct {
	components    UIComponents
	currentPhase  int
	maxPhases     int
	baseModel     bubbletea.Model
	advancedModel bubbletea.Model
}

// NewProgressiveUI creates a new progressive UI pattern
func NewProgressiveUI(base, advanced bubbletea.Model) *ProgressiveUI {
	return &ProgressiveUI{
		components:    createDefaultComponents(),
		currentPhase:  0,
		maxPhases:     3,
		baseModel:     base,
		advancedModel: advanced,
	}
}

// Init initializes the progressive UI
func (p *ProgressiveUI) Init() tea.Cmd {
	p.components.Logging.Log("Starting progressive UI initialization", "INFO", map[string]interface{}{
		"pattern":        ProgressivePattern,
		"current_phase":  p.currentPhase,
		"max_phases":     p.maxPhases,
	})

	// Phase 1: Basic initialization
	if p.currentPhase == 0 {
		return p.initializePhase1()
	}

	// Phase 2: Enhanced functionality
	if p.currentPhase == 1 {
		return p.initializePhase2()
	}

	// Phase 3: Full features
	if p.currentPhase == 2 {
		return p.initializePhase3()
	}

	return nil
}

// Update handles messages for progressive UI
func (p *ProgressiveUI) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case PhaseCompleteMsg:
		p.currentPhase++
		if p.currentPhase < p.maxPhases {
			return p, p.Init()
		}

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			return p, p.gracefulShutdown()
		case tea.KeySpace:
			// Manually trigger next phase
			if p.currentPhase < p.maxPhases-1 {
				p.currentPhase++
				return p, p.Init()
			}
		}
	}

	// Route to appropriate model based on current phase
	var model bubbletea.Model = p.baseModel
	if p.currentPhase >= 1 {
		model = p.advancedModel
	}

	return model.Update(msg)
}

// View renders the UI based on current phase
func (p *ProgressiveUI) View() string {
	if p.currentPhase == 0 {
		return p.baseModel.View()
	}
	return p.advancedModel.View()
}

// initializePhase1 handles basic UI initialization
func (p *ProgressiveUI) initializePhase1() tea.Cmd {
	p.components.Logging.Log("Initializing phase 1: Basic UI", "INFO", map[string]interface{}{
		"phase": p.currentPhase,
	})
	return func() tea.Msg {
		return PhaseCompleteMsg{Phase: 1}
	}
}

// initializePhase2 handles enhanced initialization
func (p *ProgressiveUI) initializePhase2() tea.Cmd {
	p.components.Logging.Log("Initializing phase 2: Enhanced UI", "INFO", map[string]interface{}{
		"phase": p.currentPhase,
	})
	return func() tea.Msg {
		return PhaseCompleteMsg{Phase: 2}
	}
}

// initializePhase3 handles full feature initialization
func (p *ProgressiveUI) initializePhase3() tea.Cmd {
	p.components.Logging.Log("Initializing phase 3: Full features", "INFO", map[string]interface{}{
		"phase": p.currentPhase,
	})
	return func() tea.Msg {
		return PhaseCompleteMsg{Phase: 3}
	}
}

// gracefulShutdown handles graceful shutdown
func (p *ProgressiveUI) gracefulShutdown() tea.Cmd {
	return func() tea.Msg {
		p.components.Logging.Log("Starting graceful shutdown", "INFO", map[string]interface{}{
			"phase": p.currentPhase,
		})
		// Cleanup logic here
		return tea.Quit()
	}
}

// GracefulDegradationUI implements graceful degradation pattern
type GracefulDegradationUI struct {
	components         UIComponents
	primaryModel       bubbletea.Model
	fallbackModels     []bubbletea.Model
	currentModelIndex  int
	healthCheckInterval time.Duration
}

// NewGracefulDegradationUI creates a new graceful degradation UI
func NewGracefulDegradationUI(primary bubbletea.Model, fallbacks []bubbletea.Model) *GracefulDegradationUI {
	return &GracefulDegradationUI{
		components:         createDefaultComponents(),
		primaryModel:       primary,
		fallbackModels:     fallbacks,
		currentModelIndex:  0,
		healthCheckInterval: 5 * time.Second,
	}
}

// Init initializes graceful degradation UI
func (g *GracefulDegradationUI) Init() tea.Cmd {
	g.components.Logging.Log("Starting graceful degradation UI", "INFO", map[string]interface{}{
		"pattern":          GracefulDegradationPattern,
		"current_model":    g.currentModelIndex,
		"total_models":     len(g.fallbackModels) + 1,
	})

	return g.startHealthMonitoring()
}

// Update handles messages with graceful degradation
func (g *GracefulDegradationUI) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	// Check if current model is responsive
	if !g.components.Terminal.IsResponsive() {
		g.degradeToNextModel()
	}

	// Handle fallback logic
	if g.currentModelIndex > 0 {
		return g.handleDegradedUpdate(msg)
	}

	return g.primaryModel.Update(msg)
}

// View renders the current UI
func (g *GracefulDegradationUI) View() string {
	var currentModel bubbletea.Model
	if g.currentModelIndex == 0 {
		currentModel = g.primaryModel
	} else {
		currentModel = g.fallbackModels[g.currentModelIndex-1]
	}
	return currentModel.View()
}

// degradeToNextModel degrades to the next fallback model
func (g *GracefulDegradationUI) degradeToNextModel() {
	if g.currentModelIndex < len(g.fallbackModels) {
		g.currentModelIndex++
		g.components.Logging.Log("Degraded to fallback model", "WARNING", map[string]interface{}{
			"from_model": g.currentModelIndex - 1,
			"to_model":   g.currentModelIndex,
		})
	}
}

// handleDegradedUpdate handles updates in degraded mode
func (g *GracefulDegradationUI) handleDegradedUpdate(msg tea.Msg) (bubbletea.Model, tea.Cmd) {
	g.components.ErrorHandling.LogError(fmt.Errorf("operating in degraded mode"), "degraded_mode")

	if g.currentModelIndex <= len(g.fallbackModels) {
		return g.fallbackModels[g.currentModelIndex-1].Update(msg)
	}

	return g, nil
}

// startHealthMonitoring starts health monitoring
func (g *GracefulDegradationUI) startHealthMonitoring() tea.Cmd {
	return tea.Tick(g.healthCheckInterval, func(t time.Time) tea.Msg {
		return HealthCheckMsg{}
	})
}

// FailFastUI implements the fail-fast pattern
type FailFastUI struct {
	components    UIComponents
	preFlightChecks []PreFlightCheck
	model         bubbletea.Model
	errored       bool
}

// PreFlightCheck defines a pre-flight check function
type PreFlightCheck func() error

// NewFailFastUI creates a new fail-fast UI
func NewFailFastUI(model bubbletea.Model, checks []PreFlightCheck) *FailFastUI {
	return &FailFastUI{
		components:     createDefaultComponents(),
		preFlightChecks: checks,
		model:          model,
		errored:        false,
	}
}

// Init performs pre-flight checks before starting
func (f *FailFastUI) Init() tea.Cmd {
	f.components.Logging.Log("Starting fail-fast UI", "INFO", map[string]interface{}{
		"pattern": FailFastPattern,
		"checks":  len(f.preFlightChecks),
	})

	// Run all pre-flight checks
	for i, check := range f.preFlightChecks {
		err := check()
		if err != nil {
			f.errored = true
			f.components.LogError(err, fmt.Sprintf("pre-flight check %d failed", i))
			return f.displayError(err)
		}
	}

	f.components.Logging.Log("All pre-flight checks passed", "INFO", nil)
	return f.model.Init()
}

// Update handles messages with error checking
func (f *FailFastUI) Update(msg tea.Msg) (bubbletea.Model, tea.Cmd) {
	if f.errored {
		return f, nil
	}

	switch msg := msg.(type) {
	case tea.KeyMsg:
		if msg.Type == tea.KeyCtrlC {
			return f, f.gracefulShutdown()
		}
	}

	return f.model.Update(msg)
}

// View renders the UI or error message
func (f *FailFastUI) View() string {
	if f.errored {
		return f.components.ErrorHandling.Recover().(string)
	}
	return f.model.View()
}

// displayError displays an error message
func (f *FailFastUI) displayError(err error) tea.Cmd {
	return func() tea.Msg {
		f.components.LogError(err, "fail_fast_error")
		return ErrorDisplayMsg{Error: err}
	}
}

// gracefulShutdown handles graceful shutdown
func (f *FailFastUI) gracefulShutdown() tea.Cmd {
	return func() tea.Msg {
		f.components.Logging.Log("Starting graceful shutdown", "INFO", nil)
		return tea.Quit()
	}
}

// RecoveryOrientedUI implements recovery-oriented pattern
type RecoveryOrientedUI struct {
	components        UIComponents
	primaryModel      bubbletea.Model
	recoveryActions   map[string]func() bubbletea.Model
	errorHistory      []error
	maxErrorHistory   int
	autoRecoveryCount int
}

// NewRecoveryOrientedUI creates a new recovery-oriented UI
func NewRecoveryOrientedUI(primary bubbletea.Model, recoveryActions map[string]func() bubbletea.Model) *RecoveryOrientedUI {
	return &RecoveryOrientedUI{
		components:        createDefaultComponents(),
		primaryModel:      primary,
		recoveryActions:   recoveryActions,
		errorHistory:      make([]error, 0),
		maxErrorHistory:   10,
		autoRecoveryCount: 0,
	}
}

// Init initializes recovery-oriented UI
func (r *RecoveryOrientedUI) Init() tea.Cmd {
	r.components.Logging.Log("Starting recovery-oriented UI", "INFO", map[string]interface{}{
		"pattern":             RecoveryOrientedPattern,
		"recovery_actions":    len(r.recoveryActions),
		"auto_recovery_count": r.autoRecoveryCount,
	})

	return r.startErrorMonitoring()
}

// Update handles messages with automatic recovery
func (r *RecoveryOrientedUI) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		if msg.Type == tea.KeyCtrlC {
			return r, r.gracefulShutdown()
		}
	case ErrorDetectedMsg:
		r.handleError(msg.Error)
		return r, nil
	case RecoveryCompleteMsg:
		r.autoRecoveryCount++
		r.components.Logging.Log("Auto-recovery completed", "INFO", map[string]interface{}{
			"recovery_count": r.autoRecoveryCount,
		})
	}

	return r.primaryModel.Update(msg)
}

// View renders the current UI
func (r *RecoveryOrientedUI) View() string {
	if len(r.errorHistory) > 0 {
		return r.displayRecoveryMode()
	}
	return r.primaryModel.View()
}

// handleError handles detected errors with automatic recovery
func (r *RecoveryOrientedUI) handleError(err error) {
	r.errorHistory = append(r.errorHistory, err)
	if len(r.errorHistory) > r.maxErrorHistory {
		r.errorHistory = r.errorHistory[1:]
	}

	r.components.LogError(err, "recovery_error")
	r.components.Logging.Log("Error detected, attempting recovery", "WARNING", map[string]interface{}{
		"error_count":    len(r.errorHistory),
		"last_error":    err.Error(),
	})

	// Try to recover
	if recoveryAction, exists := r.recoveryActions["general"]; exists {
		recoveredModel := recoveryAction()
		r.primaryModel = recoveredModel
	}
}

// displayRecoveryMode shows recovery mode UI
func (r *RecoveryOrientedUI) displayRecoveryMode() string {
	return fmt.Sprintf("Recovery Mode - Recent Errors: %d\n%s\nPress Space to continue, Ctrl+C to quit",
		len(r.errorHistory), r.primaryModel.View())
}

// startErrorMonitoring starts error monitoring
func (r *RecoveryOrientedUI) startErrorMonitoring() tea.Cmd {
	return tea.Tick(2*time.Second, func(t time.Time) tea.Msg {
		return ErrorMonitoringMsg{}
	})
}

// gracefulShutdown handles graceful shutdown
func (r *RecoveryOrientedUI) gracefulShutdown() tea.Cmd {
	return func() tea.Msg {
		r.components.Logging.Log("Starting graceful shutdown", "INFO", map[string]interface{}{
			"recovery_count": r.autoRecoveryCount,
		})
		return tea.Quit()
	}
}

// ModularUI implements modular pattern
type ModularUI struct {
	components    UIComponents
	modules       map[string]bubbletea.Model
	activeModule  string
	moduleTransitions map[string][]string
}

// NewModularUI creates a new modular UI
func NewModularUI(modules map[string]bubbletea.Model, transitions map[string][]string) *ModularUI {
	return &ModularUI{
		components:       createDefaultComponents(),
		modules:          modules,
		activeModule:     "main",
		moduleTransitions: transitions,
	}
}

// Init initializes modular UI
func (m *ModularUI) Init() tea.Cmd {
	m.components.Logging.Log("Starting modular UI", "INFO", map[string]interface{}{
		"pattern":       ModularPattern,
		"modules":       len(m.modules),
		"active_module": m.activeModule,
	})

	if _, exists := m.modules[m.activeModule]; !exists {
		m.components.LogError(fmt.Errorf("module %s not found", m.activeModule), "module_not_found")
		m.activeModule = "main"
	}

	return m.modules[m.activeModule].Init()
}

// Update handles messages with module routing
func (m *ModularUI) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, m.gracefulShutdown()
		case tea.KeyTab:
			// Switch to next module
			return m.switchModule()
		}
	case ModuleSwitchMsg:
		if m.isValidTransition(m.activeModule, msg.TargetModule) {
			m.activeModule = msg.TargetModule
			return m, m.modules[m.activeModule].Init()
		}
	}

	return m.modules[m.activeModule].Update(msg)
}

// View renders the active module
func (m *ModularUI) View() string {
	return m.modules[m.activeModule].View()
}

// switchModule switches to the next module
func (m *ModularUI) switchModule() (tea.Model, tea.Cmd) {
	transitionList := m.moduleTransitions[m.activeModule]
	if len(transitionList) > 0 {
		nextModule := transitionList[0]
		return m, func() tea.Msg {
			return ModuleSwitchMsg{TargetModule: nextModule}
		}
	}
	return m, nil
}

// isValidTransition checks if module transition is valid
func (m *ModularUI) isValidTransition(from, to string) bool {
	transitions, exists := m.moduleTransitions[from]
	if !exists {
		return false
	}

	for _, validTarget := range transitions {
		if validTarget == to {
			return true
		}
	}
	return false
}

// gracefulShutdown handles graceful shutdown
func (m *ModularUI) gracefulShutdown() tea.Cmd {
	return func() tea.Msg {
		m.components.Logging.Log("Starting graceful shutdown", "INFO", map[string]interface{}{
			"active_module": m.activeModule,
		})
		return tea.Quit()
	}
}

// createDefaultComponents creates default UI components
func createDefaultComponents() UIComponents {
	return UIComponents{
		Terminal:    &BasicTerminalStrategy{},
		State:       &BasicStateManager{},
		ErrorHandling: &BasicErrorHandler{},
		Logging:     &BasicLogger{},
		Signals:     &BasicSignalHandler{},
		Performance: &BasicPerformanceMonitor{},
	}
}

// Basic implementations for interfaces
type BasicTerminalStrategy struct{}

func (b *BasicTerminalStrategy) Initialize() error { return nil }
func (b *BasicTerminalStrategy) Cleanup() error { return nil }
func (b *BasicTerminalStrategy) IsResponsive() bool { return true }
func (b *BasicTerminalStrategy) BackupStrategy() TerminalStrategy { return &BasicTerminalStrategy{} }

type BasicStateManager struct{}

func (b *BasicStateManager) GetCurrentState() string { return "initial" }
func (b *BasicStateManager) Transition(newState string) error { return nil }
func (b *BasicStateManager) IsValidTransition(from, to string) bool { return true }
func (b *BasicStateManager) GetStateHistory() []string { return []string{"initial"} }

type BasicErrorHandler struct{}

func (b *BasicErrorHandler) HandleError(err error) error { return err }
func (b *BasicErrorHandler) Recover() interface{} { return "Recovered from error" }
func (b *BasicErrorHandler) RegisterPanicHandler() {}
func (b *BasicErrorHandler) LogError(err error, context string) {}

type BasicLogger struct{}

func (b *BasicLogger) Log(message string, level string, context map[string]interface{}) {}
func (b *BasicLogger) Error(message string, err error) {}
func (b *BasicLogger) Debug(message string, context map[string]interface{}) {}
func (b *BasicLogger) Info(message string, context map[string]interface{}) {}

type BasicSignalHandler struct{}

func (b *BasicSignalHandler) RegisterSignalHandlers() error { return nil }
func (b *BasicSignalHandler) HandleSignal(sig os.Signal) error { return nil }
func (b *BasicSignalHandler) GracefulShutdown(ctx context.Context) error { return nil }

type BasicPerformanceMonitor struct{}

func (b *BasicPerformanceMonitor) TrackMetric(name string, value float64) {}
func (b *BasicPerformanceMonitor) GetMetrics() map[string]float64 { return make(map[string]float64) }
func (b *BasicPerformanceMonitor) AlertOnThreshold(name string, threshold float64) bool { return false }

// Message types
type PhaseCompleteMsg struct{ Phase int }
type HealthCheckMsg struct{}
type ErrorDisplayMsg struct{ Error error }
type ErrorDetectedMsg struct{ Error error }
type RecoveryCompleteMsg struct{}
type ErrorMonitoringMsg struct{}
type ModuleSwitchMsg struct{ TargetModule string }