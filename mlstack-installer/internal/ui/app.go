// internal/ui/app.go
package ui

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/scooter-lacroix/mlstack-installer/internal/installer"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/components"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/integration"
	intg "github.com/scooter-lacroix/mlstack-installer/internal/ui/integration"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/performance"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/security"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/state"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// Pure MVU Architecture Implementation

// Model represents the immutable state in MVU architecture
type Model struct {
	// Core state (immutable)
	State        types.State
	StateManager types.StateManager
	Security     types.SecurityValidator
	PerfMonitor  types.PerformanceMonitor
	Integration  types.IntegrationManager

	// Components
	Welcome       types.LayoutComponent
	Hardware      types.LayoutComponent
	ComponentList types.LayoutComponent
	Configuration types.LayoutComponent
	Progress      types.LayoutComponent
	Recovery      types.LayoutComponent

	// Layout and viewport
	Viewport viewport.Model
	Help     help.Model

	// Key bindings
	HelpKey   key.Binding
	QuitKey   key.Binding
	UpKey     key.Binding
	DownKey   key.Binding
	LeftKey   key.Binding
	RightKey  key.Binding
	EnterKey  key.Binding
	EscapeKey key.Binding

	// Context for cancellation
	Ctx    context.Context
	Cancel context.CancelFunc

	// Performance tracking
	LastUpdate time.Time
	FrameTime  time.Duration
	FrameCount int64

	// Batch operations
	ActiveBatches map[string]*types.BatchOperation

	// Security context
	SecurityContext map[string]bool

	// UI state
	FocusedElement string
	LastError      error
	WarningCount   int
}

// KeyMap defines the default key bindings
type KeyMap struct {
	HelpKey   key.Binding
	QuitKey   key.Binding
	UpKey     key.Binding
	DownKey   key.Binding
	LeftKey   key.Binding
	RightKey  key.Binding
	EnterKey  key.Binding
	EscapeKey key.Binding
}

// DefaultKeyBindings returns the default key bindings
func DefaultKeyBindings() KeyMap {
	return KeyMap{
		HelpKey: key.NewBinding(
			key.WithKeys("h", "?"),
			key.WithHelp("h/?", "show help"),
		),
		QuitKey: key.NewBinding(
			key.WithKeys("q", "ctrl+c"),
			key.WithHelp("q", "quit"),
		),
		UpKey: key.NewBinding(
			key.WithKeys("k", "up"),
			key.WithHelp("↑/k", "move up"),
		),
		DownKey: key.NewBinding(
			key.WithKeys("j", "down"),
			key.WithHelp("↓/j", "move down"),
		),
		LeftKey: key.NewBinding(
			key.WithKeys("h", "left"),
			key.WithHelp("←/h", "move left"),
		),
		RightKey: key.NewBinding(
			key.WithKeys("l", "right"),
			key.WithHelp("→/l", "move right"),
		),
		EnterKey: key.NewBinding(
			key.WithKeys("enter", " "),
			key.WithHelp("enter", "select"),
		),
		EscapeKey: key.NewBinding(
			key.WithKeys("esc"),
			key.WithHelp("esc", "go back"),
		),
	}
}

// NewModel creates a new immutable model with default values
func NewModel() *Model {
	// Create context with cancellation
	ctx, cancel := context.WithCancel(context.Background())

	// Initialize core components
	stateManager := state.NewManager()
	securityValidator := security.NewValidator()
	perfMonitor := performance.NewMonitor()
	integrationManager := integration.NewManager()

	// Assign interfaces (concrete types should implement interfaces)
	var stateManagerInterface types.StateManager = stateManager
	var securityValidatorInterface types.SecurityValidator = securityValidator
	var perfMonitorInterface types.PerformanceMonitor = perfMonitor
	var integrationManagerInterface types.IntegrationManager = integrationManager

	// Create model with all required interfaces
	model := &Model{
		State:           stateManager.GetState(),
		StateManager:    stateManagerInterface,
		Security:        securityValidatorInterface,
		PerfMonitor:     perfMonitorInterface,
		Integration:     integrationManagerInterface,
		Help:            help.New(),
		Viewport:        viewport.New(80, 24),
		Ctx:             ctx,
		Cancel:          cancel,
		ActiveBatches:   make(map[string]*types.BatchOperation),
		SecurityContext: make(map[string]bool),
		LastUpdate:      time.Now(),

		// Initialize key bindings
		HelpKey: key.NewBinding(
			key.WithKeys("h", "?"),
			key.WithHelp("h/?", "help"),
		),
		QuitKey: key.NewBinding(
			key.WithKeys("q", "ctrl+c"),
			key.WithHelp("q", "quit"),
		),
		UpKey: key.NewBinding(
			key.WithKeys("up", "k"),
			key.WithHelp("↑/k", "up"),
		),
		DownKey: key.NewBinding(
			key.WithKeys("down", "j"),
			key.WithHelp("↓/j", "down"),
		),
		LeftKey: key.NewBinding(
			key.WithKeys("left", "h"),
			key.WithHelp("←/h", "left"),
		),
		RightKey: key.NewBinding(
			key.WithKeys("right", "l"),
			key.WithHelp("→/l", "right"),
		),
		EnterKey: key.NewBinding(
			key.WithKeys("enter", " "),
			key.WithHelp("enter/⏎", "select"),
		),
		EscapeKey: key.NewBinding(
			key.WithKeys("esc"),
			key.WithHelp("esc", "back"),
		),
	}

	// Initialize components
	model.initializeComponents()

	// Initialize security context
	model.initializeSecurityContext()

	return model
}

// initializeComponents creates all UI components
func (m *Model) initializeComponents() {
	// Cast integration manager back to concrete type for component constructors
	concreteIntegration, ok := m.Integration.(intg.Manager)
	if !ok {
		// Handle the error appropriately - for now, we'll create a new manager
		concreteIntegration = intg.NewManager()
	}

	// Create welcome component with AMD branding
	m.Welcome = components.NewWelcomeComponent(m.State.Width, m.State.Height, concreteIntegration)

	// Create hardware detection component
	m.Hardware = components.NewHardwareDetectComponent(m.State.Width, m.State.Height, concreteIntegration)

	// Create component selection component
	m.ComponentList = components.NewComponentSelectComponent(
		m.State.Components,
		m.State.SelectedCategories,
		m.State.Width,
		m.State.Height,
	)

	// Create configuration component
	m.Configuration = components.NewConfigurationComponent(m.State.Width, m.State.Height, m.State.Config)

	// Create installation progress component
	m.Progress = components.NewInstallationProgressComponent(m.State.Width, m.State.Height)

	// Create recovery component
	m.Recovery = components.NewRecoveryComponent(m.State.Width, m.State.Height)

	// Set initial focus
	m.FocusedElement = "welcome"
}

// initializeSecurityContext sets up security validation context
func (m *Model) initializeSecurityContext() {
	m.SecurityContext["script_validation"] = true
	m.SecurityContext["input_sanitization"] = true
	m.SecurityContext["privilege_check"] = true
	m.SecurityContext["path_protection"] = true
	m.SecurityContext["command_injection_prevention"] = true
}

// Clone creates a deep copy of the model (for immutable updates)
func (m *Model) Clone() *Model {
	if m == nil {
		return nil
	}

	// Create deep copy of state
	newState := m.State

	// Create deep copy of selected categories
	newSelectedCategories := make(map[string]bool)
	for k, v := range m.State.SelectedCategories {
		newSelectedCategories[k] = v
	}
	newState.SelectedCategories = newSelectedCategories

	// Create deep copy of components
	newComponents := make([]types.Component, len(m.State.Components))
	copy(newComponents, m.State.Components)
	newState.Components = newComponents

	// Create deep copy of logs
	newInstallLog := make([]string, len(m.State.InstallLog))
	copy(newInstallLog, m.State.InstallLog)
	newState.InstallLog = newInstallLog

	newErrorLog := make([]string, len(m.State.ErrorLog))
	copy(newErrorLog, m.State.ErrorLog)
	newState.ErrorLog = newErrorLog

	// Create deep copy of active batches
	newActiveBatches := make(map[string]*types.BatchOperation)
	for k, v := range m.ActiveBatches {
		if v != nil {
			newBatch := *v
			newActiveBatches[k] = &newBatch
		}
	}

	// Create deep copy of security context
	newSecurityContext := make(map[string]bool)
	for k, v := range m.SecurityContext {
		newSecurityContext[k] = v
	}

	// Create new model with copied data
	newModel := *m
	newModel.State = newState
	newModel.ActiveBatches = newActiveBatches
	newModel.SecurityContext = newSecurityContext
	newModel.LastUpdate = time.Now()

	return &newModel
}

// Update is the pure function that handles all state transitions
func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	// Performance tracking
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		m.FrameTime = duration
		m.FrameCount++
		m.PerfMonitor.RecordMetric("frame_time_ms", float64(duration.Milliseconds()))
	}()

	// Handle different message types
	switch msg := msg.(type) {
	case tea.KeyMsg:
		newModel, newCmd := m.handleKey(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case tea.WindowSizeMsg:
		newModel, newCmd := m.handleWindowSize(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case spinner.TickMsg:
		newModel, newCmd := m.handleSpinnerTick(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	// Handle progress updates through custom messages instead
	// case progress.FrameMsg is handled in component-specific updates

	// Navigation messages
	case types.NavigateToStageMsg:
		newModel, newCmd := m.handleNavigateToStage(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.NavigateBackMsg:
		newModel, newCmd := m.handleNavigateBack(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.NavigateForwardMsg:
		newModel, newCmd := m.handleNavigateForward(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	// Hardware detection messages
	case types.HardwareDetectedMsg:
		newModel, newCmd := m.handleHardwareDetected(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.HardwareProgressMsg:
		newModel, newCmd := m.handleHardwareProgress(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	// Installation messages
	case types.ComponentToggleMsg:
		newModel, newCmd := m.handleComponentToggle(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.InstallationStartMsg:
		newModel, newCmd := m.handleInstallationStart(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.InstallationProgressMsg:
		newModel, newCmd := m.handleInstallationProgress(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.InstallationCompleteMsg:
		newModel, newCmd := m.handleInstallationComplete(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	// Configuration messages
	case types.ConfigLoadMsg:
		newModel, newCmd := m.handleConfigLoad(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.ConfigSaveMsg:
		newModel, newCmd := m.handleConfigSave(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	// Recovery messages
	case types.RecoveryTriggerMsg:
		newModel, newCmd := m.handleRecoveryTrigger(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.RecoveryOptionSelectMsg:
		newModel, newCmd := m.handleRecoveryOptionSelect(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	// Security messages
	case types.SecurityValidationMsg:
		newModel, newCmd := m.handleSecurityValidation(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.SecurityCheckMsg:
		newModel, newCmd := m.handleSecurityCheck(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	// Performance messages
	case types.MetricRecordMsg:
		newModel, newCmd := m.handleMetricRecord(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.PerformanceWarningMsg:
		newModel, newCmd := m.handlePerformanceWarning(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	// Integration messages
	case types.HardwareDetectionCompleteMsg:
		newModel, newCmd := m.handleHardwareDetectionComplete(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.ScriptExecutionCompleteMsg:
		newModel, newCmd := m.handleScriptExecutionComplete(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.VerificationCompleteMsg:
		newModel, newCmd := m.handleVerificationComplete(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	// Batch operation messages
	case types.BatchStartMsg:
		newModel, newCmd := m.handleBatchStart(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.BatchProgressMsg:
		newModel, newCmd := m.handleBatchProgress(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.BatchCompleteMsg:
		newModel, newCmd := m.handleBatchComplete(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.CancelOperationMsg:
		newModel, newCmd := m.handleCancelOperation(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	// Error recovery messages
	case types.RecoveryAttemptMsg:
		newModel, newCmd := m.handleRecoveryAttempt(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.RecoverySuccessMsg:
		newModel, newCmd := m.handleRecoverySuccess(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.RecoveryFailureMsg:
		newModel, newCmd := m.handleRecoveryFailure(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	// System status messages
	case types.SystemStatusUpdateMsg:
		newModel, newCmd := m.handleSystemStatusUpdate(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case types.ResourceWarningMsg:
		newModel, newCmd := m.handleResourceWarning(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	// Other messages
	case types.TickMsg:
		newModel, newCmd := m.handleTick(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	case tea.QuitMsg:
		newModel, newCmd := m.handleQuit(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}

	default:
		// Let current component handle the message
		newModel, newCmd := m.handleComponentMessage(msg)
		if model, ok := newModel.(*Model); ok {
			m = model
			cmd = newCmd
		}
	}

	// Update model reference
	m.State = m.StateManager.GetState()

	return m, cmd
}

// View renders the current UI state
func (m *Model) View() string {
	// Get current stage and render appropriate view
	switch m.State.Stage {
	case types.StageWelcome:
		return m.Welcome.View()

	case types.StageHardwareDetect:
		return m.Hardware.View()

	case types.StagePreFlightCheck:
		return m.PreFlightView()

	case types.StageComponentSelect:
		return m.ComponentList.View()

	case types.StageConfiguration:
		return m.Configuration.View()

	case types.StageConfirm:
		return m.ConfirmView()

	case types.StageInstalling:
		return m.Progress.View()

	case types.StageComplete:
		return m.CompleteView()

	case types.StageRecovery:
		return m.Recovery.View()

	default:
		return "Unknown stage: " + m.State.Stage.String()
	}
}

// Init initializes the application
func (m *Model) Init() tea.Cmd {
	// Start performance monitoring
	m.PerfMonitor.StartOperation("application_startup")

	// Initialize components with commands
	var cmds []tea.Cmd

	// Initialize welcome component
	if m.Welcome != nil {
		cmds = append(cmds, m.Welcome.Init())
	}

	// Initialize hardware detection component (bypassed for now - was hanging)
	// TODO: Fix hardware component Init() method to not block
	// if m.Hardware != nil {
	// 	cmds = append(cmds, m.Hardware.Init())
	// }

	// Start periodic updates
	cmds = append(cmds, m.tickCommand())

	// Return batch command
	return tea.Batch(cmds...)
}

// tickCommand returns a tick command for periodic updates
func (m *Model) tickCommand() tea.Cmd {
	return tea.Tick(time.Second, func(t time.Time) tea.Msg {
		return types.TickMsg{Time: t}
	})
}

// handleKey processes key input
func (m *Model) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch {
	case key.Matches(msg, m.HelpKey):
		m.FocusedElement = "help"
		return m, nil

	case key.Matches(msg, m.QuitKey):
		if m.QuitKey.Keys()[0] == "ctrl+c" {
			return m, tea.Quit
		}
		m.FocusedElement = "quit"
		return m, m.confirmQuit()

	case key.Matches(msg, m.UpKey):
		m.FocusedElement = "up"
		return m, m.handleUpNavigation()

	case key.Matches(msg, m.DownKey):
		m.FocusedElement = "down"
		return m, m.handleDownNavigation()

	case key.Matches(msg, m.LeftKey):
		m.FocusedElement = "left"
		return m, m.handleLeftNavigation()

	case key.Matches(msg, m.RightKey):
		m.FocusedElement = "right"
		return m, m.handleRightNavigation()

	case key.Matches(msg, m.EnterKey):
		m.FocusedElement = "enter"
		return m, m.handleEnter()

	case key.Matches(msg, m.EscapeKey):
		m.FocusedElement = "escape"
		return m, m.handleEscape()

	default:
		return m, nil
	}
}

// handleWindowSize handles window size changes
func (m *Model) handleWindowSize(msg tea.WindowSizeMsg) (tea.Model, tea.Cmd) {
	// Update viewport size
	m.Viewport.Width = msg.Width
	m.Viewport.Height = msg.Height

	// Update state manager
	m.StateManager = m.StateManager.SetWindowSize(msg.Width, msg.Height)

	// Update all components with new dimensions
	if m.Welcome != nil {
		m.Welcome.SetBounds(0, 0, msg.Width, msg.Height)
	}
	if m.Hardware != nil {
		m.Hardware.SetBounds(0, 0, msg.Width, msg.Height)
	}
	if m.ComponentList != nil {
		m.ComponentList.SetBounds(0, 0, msg.Width, msg.Height)
	}
	if m.Configuration != nil {
		m.Configuration.SetBounds(0, 0, msg.Width, msg.Height)
	}
	if m.Progress != nil {
		m.Progress.SetBounds(0, 0, msg.Width, msg.Height)
	}
	if m.Recovery != nil {
		m.Recovery.SetBounds(0, 0, msg.Width, msg.Height)
	}

	// Send layout update message
	return m, m.sendLayoutUpdate(msg.Width, msg.Height)
}

// handleSpinnerTick handles spinner tick messages
func (m *Model) handleSpinnerTick(msg spinner.TickMsg) (tea.Model, tea.Cmd) {
	// Update component spinners
	if m.Hardware != nil {
		_, cmd := m.Hardware.Update(msg)
		return m, cmd
	}
	if m.Progress != nil {
		_, cmd := m.Progress.Update(msg)
		return m, cmd
	}
	return m, nil
}

// handleProgressFrame handles progress bar frame messages
func (m *Model) handleProgressFrame(msg tea.Msg) (tea.Model, tea.Cmd) {
	// Update progress bar - this will be called when progress needs updating
	// The actual progress updates come through InstallationProgressMsg
	return m, nil
}

// handleNavigateToStage handles navigation to a specific stage
func (m *Model) handleNavigateToStage(msg types.NavigateToStageMsg) (tea.Model, tea.Cmd) {
	// Validate transition
	if m.StateManager.IsValidTransition(m.State.Stage, msg.Stage) {
		m.StateManager = m.StateManager.TransitionTo(msg.Stage)
		m.FocusedElement = "navigation"
		return m, m.sendStageTransition(msg.Stage)
	}

	// Log invalid transition attempt
	m.StateManager.LogError(nil)
	m.StateManager.LogError(nil)
	return m, nil
}

// handleNavigateBack handles backward navigation
func (m *Model) handleNavigateBack(msg types.NavigateBackMsg) (tea.Model, tea.Cmd) {
	// Implement backward navigation logic
	switch m.State.Stage {
	case types.StageConfiguration:
		return m.handleNavigateToStage(types.NavigateToStageMsg{Stage: types.StageComponentSelect})
	case types.StageConfirm:
		return m.handleNavigateToStage(types.NavigateToStageMsg{Stage: types.StageComponentSelect})
	case types.StageInstalling:
		return m.handleNavigateToStage(types.NavigateToStageMsg{Stage: types.StageConfirm})
	case types.StageRecovery:
		return m.handleNavigateToStage(types.NavigateToStageMsg{Stage: types.StageWelcome})
	default:
		return m, nil
	}
}

// handleNavigateForward handles forward navigation
func (m *Model) handleNavigateForward(msg types.NavigateForwardMsg) (tea.Model, tea.Cmd) {
	// Implement forward navigation logic
	switch m.State.Stage {
	case types.StageWelcome:
		return m.handleNavigateToStage(types.NavigateToStageMsg{Stage: types.StageHardwareDetect})
	case types.StageHardwareDetect:
		return m.handleNavigateToStage(types.NavigateToStageMsg{Stage: types.StageComponentSelect})
	case types.StageComponentSelect:
		return m.handleNavigateToStage(types.NavigateToStageMsg{Stage: types.StageConfiguration})
	case types.StageConfiguration:
		return m.handleNavigateToStage(types.NavigateToStageMsg{Stage: types.StageConfirm})
	case types.StageConfirm:
		return m.handleNavigateToStage(types.NavigateToStageMsg{Stage: types.StageInstalling})
	default:
		return m, nil
	}
}

// confirmQuit shows confirmation dialog for quitting
func (m *Model) confirmQuit() tea.Cmd {
	return tea.Sequence(
		tea.Printf("Are you sure you want to quit? (y/n): "),
	)
}

// Helper methods for navigation
func (m *Model) handleUpNavigation() tea.Cmd {
	// Delegate to current component
	if m.ComponentList != nil {
		_, cmd := m.ComponentList.Update(tea.KeyMsg{Type: tea.KeyUp})
		return cmd
	}
	return nil
}

func (m *Model) handleDownNavigation() tea.Cmd {
	// Delegate to current component
	if m.ComponentList != nil {
		_, cmd := m.ComponentList.Update(tea.KeyMsg{Type: tea.KeyDown})
		return cmd
	}
	return nil
}

func (m *Model) handleLeftNavigation() tea.Cmd {
	// Delegate to current component
	if m.ComponentList != nil {
		_, cmd := m.ComponentList.Update(tea.KeyMsg{Type: tea.KeyLeft})
		return cmd
	}
	return nil
}

func (m *Model) handleRightNavigation() tea.Cmd {
	// Delegate to current component
	if m.ComponentList != nil {
		_, cmd := m.ComponentList.Update(tea.KeyMsg{Type: tea.KeyRight})
		return cmd
	}
	return nil
}

func (m *Model) handleEnter() tea.Cmd {
	// Delegate to current component
	switch m.State.Stage {
	case types.StageComponentSelect:
		if m.ComponentList != nil {
			_, cmd := m.ComponentList.Update(tea.KeyMsg{Type: tea.KeyEnter})
			return cmd
		}
	case types.StageConfirm:
		// Start installation
		// Directly call StateManager instead of going through the handler
		m.StateManager = m.StateManager.StartInstallation()
		return m.installationCommand()
	}
	return nil
}

func (m *Model) handleEscape() tea.Cmd {
	// Implement escape logic
	switch m.State.Stage {
	case types.StageComponentSelect, types.StageConfiguration, types.StageConfirm:
		_, cmd := m.handleNavigateBack(types.NavigateBackMsg{})
		return cmd
	case types.StageInstalling:
		// Handle cancel installation
		_, cmd := m.handleCancelOperation(types.CancelOperationMsg{OperationID: "installation", Reason: "User cancelled"})
		return cmd
	default:
		return tea.Quit
	}
}

// View methods for different stages
func (m *Model) PreFlightView() string {
	return "Pre-flight checks view"
}

func (m *Model) ConfirmView() string {
	return "Confirmation view"
}

func (m *Model) CompleteView() string {
	var b strings.Builder
	b.WriteString("\n")
	b.WriteString(titleStyle.Render(" Installation Summary "))
	b.WriteString("\n\n")

	installed := []types.Component{}
	failed := []types.Component{}
	skipped := []types.Component{}
	envConfigured := []types.Component{}
	verification := []types.Component{}

	for _, comp := range m.State.Components {
		if !comp.Selected {
			continue
		}

		switch comp.Category {
		case "environment":
			envConfigured = append(envConfigured, comp)
		case "verification":
			verification = append(verification, comp)
		}

		if comp.Installed {
			installed = append(installed, comp)
		} else if comp.Progress > 0 {
			failed = append(failed, comp)
		} else {
			skipped = append(skipped, comp)
		}
	}

	b.WriteString(headerStyle.Render("Installed Components") + "\n")
	if len(installed) == 0 {
		b.WriteString(dimStyle.Render("  None\n"))
	} else {
		for _, comp := range installed {
			b.WriteString(successStyle.Render("  ✓ " + comp.Name))
			b.WriteString("\n")
		}
	}

	b.WriteString("\n")
	b.WriteString(headerStyle.Render("Failed Components") + "\n")
	if len(failed) == 0 {
		b.WriteString(dimStyle.Render("  None\n"))
	} else {
		for _, comp := range failed {
			b.WriteString(errorStyle.Render("  ✗ " + comp.Name))
			b.WriteString("\n")
		}
	}

	b.WriteString("\n")
	b.WriteString(headerStyle.Render("Skipped / Not Installed") + "\n")
	if len(skipped) == 0 {
		b.WriteString(dimStyle.Render("  None\n"))
	} else {
		for _, comp := range skipped {
			b.WriteString(warningStyle.Render("  • " + comp.Name))
			b.WriteString("\n")
		}
	}

	b.WriteString("\n")
	b.WriteString(headerStyle.Render("Environment & Verification") + "\n")
	if len(envConfigured) == 0 && len(verification) == 0 {
		b.WriteString(dimStyle.Render("  No environment or verification tasks selected.\n"))
	} else {
		for _, comp := range envConfigured {
			status := "configured"
			style := infoStyle
			if comp.Installed {
				status = "complete"
				style = successStyle
			}
			b.WriteString(style.Render(fmt.Sprintf("  %s (%s)", comp.Name, status)))
			b.WriteString("\n")
		}
		for _, comp := range verification {
			status := "pending"
			style := warningStyle
			if comp.Installed {
				status = "completed"
				style = successStyle
			}
			b.WriteString(style.Render(fmt.Sprintf("  %s (%s)", comp.Name, status)))
			b.WriteString("\n")
		}
	}

	benchmarkCount := 0
	testCount := 0
	for _, entry := range m.State.InstallLog {
		lower := strings.ToLower(entry)
		if strings.Contains(lower, "benchmark") {
			benchmarkCount++
		}
		if strings.Contains(lower, "test") || strings.Contains(lower, "verify") {
			testCount++
		}
	}

	b.WriteString("\n")
	b.WriteString(headerStyle.Render("Benchmarks & Tests") + "\n")
	if benchmarkCount == 0 && testCount == 0 {
		b.WriteString(dimStyle.Render("  No benchmark/test results logged.\n"))
	} else {
		b.WriteString(infoStyle.Render(fmt.Sprintf("  Benchmark log entries: %d", benchmarkCount)))
		b.WriteString("\n")
		b.WriteString(infoStyle.Render(fmt.Sprintf("  Test/verification log entries: %d", testCount)))
		b.WriteString("\n")
	}

	b.WriteString("\n")
	b.WriteString(helpStyle.Render("Press q to quit • Press esc to go back"))

	return b.String()
}

// Message sending helper methods
func (m *Model) sendStageTransition(stage types.Stage) tea.Cmd {
	return func() tea.Msg {
		return types.DebugMsg{
			Level:   "info",
			Message: "Stage transition",
			Context: map[string]interface{}{
				"from": m.State.Stage.String(),
				"to":   stage.String(),
			},
		}
	}
}

func (m *Model) sendLayoutUpdate(width, height int) tea.Cmd {
	return func() tea.Msg {
		return types.LayoutUpdateMsg{
			Width:    width,
			Height:   height,
			Elements: []string{"viewport", "components"},
		}
	}
}

// Handle component-specific messages
func (m *Model) handleComponentMessage(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch m.State.Stage {
	case types.StageWelcome:
		if m.Welcome != nil {
			return m.Welcome.Update(msg)
		}
	case types.StageHardwareDetect:
		if m.Hardware != nil {
			return m.Hardware.Update(msg)
		}
	case types.StageComponentSelect:
		if m.ComponentList != nil {
			return m.ComponentList.Update(msg)
		}
	case types.StageConfiguration:
		if m.Configuration != nil {
			return m.Configuration.Update(msg)
		}
	case types.StageInstalling:
		if m.Progress != nil {
			return m.Progress.Update(msg)
		}
	case types.StageRecovery:
		if m.Recovery != nil {
			return m.Recovery.Update(msg)
		}
	}
	return m, nil
}

// Tick handler for periodic updates
func (m *Model) handleTick(msg types.TickMsg) (tea.Model, tea.Cmd) {
	// Update performance metrics
	m.PerfMonitor.RecordMetric("tick_count", float64(m.FrameCount))

	// Check for resource warnings
	cmd := m.checkResourceWarnings()

	// Update system status periodically
	if time.Since(m.LastUpdate) > 5*time.Second {
		cmd = tea.Batch(cmd, m.updateSystemStatus())
	}

	m.LastUpdate = msg.Time
	return m, cmd
}

// handleQuit handles quit message
func (m *Model) handleQuit(msg tea.QuitMsg) (tea.Model, tea.Cmd) {
	// Cancel all running operations
	m.Cancel()

	// Clean up resources
	m.cleanupResources()

	return m, tea.Quit
}

// cleanupResources cleans up application resources
func (m *Model) cleanupResources() {
	// Cancel all active batches
	for id, batch := range m.ActiveBatches {
		if batch != nil && !batch.Done {
			// Mark batch as cancelled
			batch.Done = true
			m.ActiveBatches[id] = batch
		}
	}

	// Finish performance monitoring
	m.PerfMonitor.FinishOperation("application_startup", true)
}

// checkResourceWarnings checks for resource usage warnings
func (m *Model) checkResourceWarnings() tea.Cmd {
	// This would check actual system resources in a real implementation
	return func() tea.Msg {
		return types.SystemStatusUpdateMsg{
			CPUUsage:    25.0,
			MemoryUsage: 40.0,
			DiskUsage:   60.0,
			GPUUsage:    []float64{45.0, 30.0},
		}
	}
}

// updateSystemStatus updates system status
func (m *Model) updateSystemStatus() tea.Cmd {
	return func() tea.Msg {
		return types.SystemStatusUpdateMsg{
			CPUUsage:    25.0,
			MemoryUsage: 40.0,
			DiskUsage:   60.0,
			GPUUsage:    []float64{45.0, 30.0},
		}
	}
}

// Placeholder implementations for other handlers (would be implemented in detail)
func (m *Model) handleHardwareDetected(msg types.HardwareDetectedMsg) (tea.Model, tea.Cmd) {
	// Update state with hardware detection results
	// Convert installer types to types for interface compatibility
	m.StateManager.SetHardwareInfo(types.GPUInfo{
		Vendor:        msg.GPUInfo.Vendor,
		Model:         msg.GPUInfo.Model,
		Driver:        msg.GPUInfo.Driver,
		ComputeUnits:  msg.GPUInfo.ComputeUnits,
		MemoryGB:      msg.GPUInfo.MemoryGB,
		Architecture:  msg.GPUInfo.Architecture,
		GFXVersion:    msg.GPUInfo.GFXVersion,
		Temperature:   msg.GPUInfo.Temperature,
		GPUCount:      msg.GPUInfo.GPUCount,
		Status:        msg.GPUInfo.Status,
		Optimizations: msg.GPUInfo.Optimizations,
		PowerUsage:    msg.GPUInfo.PowerUsage,
	}, types.SystemInfo{
		OS:            msg.SystemInfo.OS,
		Distribution:  msg.SystemInfo.Distribution,
		KernelVersion: msg.SystemInfo.KernelVersion,
		Architecture:  msg.SystemInfo.Architecture,
		CPU:           convertCPUInfo(msg.SystemInfo.CPU),
		Memory:        convertMemoryInfo(msg.SystemInfo.Memory),
		Storage:       convertStorageInfo(msg.SystemInfo.Storage),
		Timestamp:     msg.SystemInfo.Timestamp,
	})
	if msg.PreflightResults != nil {
		m.StateManager.SetPreflightResults(msg.PreflightResults)
	}
	return m, nil
}

func (m *Model) handleHardwareProgress(msg types.HardwareProgressMsg) (tea.Model, tea.Cmd) {
	// Update hardware detection progress
	return m, nil
}

func (m *Model) handleComponentToggle(msg types.ComponentToggleMsg) (tea.Model, tea.Cmd) {
	// Toggle component selection
	m.StateManager.ToggleComponent(msg.ComponentID)
	return m, nil
}

func (m *Model) handleInstallationStart(msg types.InstallationStartMsg) (tea.Model, tea.Cmd) {
	// Start installation process
	m.StateManager.StartInstallation()
	return m, m.installationCommand()
}

func (m *Model) installationCommand() tea.Cmd {
	return func() tea.Msg {
		// Simulate installation progress
		for i := 0; i < 100; i += 10 {
			time.Sleep(100 * time.Millisecond)
			return types.InstallationProgressMsg{
				ComponentID: "rocm",
				Progress:    float64(i) / 100.0,
				Message:     "Installing ROCm platform...",
			}
		}
		return types.InstallationCompleteMsg{
			ComponentID: "rocm",
			Success:     true,
		}
	}
}

func (m *Model) handleInstallationProgress(msg types.InstallationProgressMsg) (tea.Model, tea.Cmd) {
	// Update installation progress
	m.StateManager.UpdateComponentProgress(msg.ComponentID, msg.Progress, msg.Message)
	return m, nil
}

func (m *Model) handleInstallationComplete(msg types.InstallationCompleteMsg) (tea.Model, tea.Cmd) {
	// Mark component as complete
	m.StateManager.CompleteComponent(msg.ComponentID, msg.Success, msg.Error)

	// Check if all components are done
	if m.State.Stage == types.StageComplete {
		return m, m.sendStageTransition(types.StageComplete)
	}

	return m, nil
}

func (m *Model) handleConfigLoad(msg types.ConfigLoadMsg) (tea.Model, tea.Cmd) {
	// Handle config loading
	if msg.Error != nil {
		m.StateManager.LogError(msg.Error)
		return m, nil
	}
	m.StateManager.SetConfig(msg.Config)
	return m, nil
}

func (m *Model) handleConfigSave(msg types.ConfigSaveMsg) (tea.Model, tea.Cmd) {
	// Handle config saving
	return m, nil
}

func (m *Model) handleRecoveryTrigger(msg types.RecoveryTriggerMsg) (tea.Model, tea.Cmd) {
	// Trigger recovery mode
	m.StateManager.TriggerRecovery(msg.ErrorType, msg.ErrorMessage)
	return m, nil
}

func (m *Model) handleRecoveryOptionSelect(msg types.RecoveryOptionSelectMsg) (tea.Model, tea.Cmd) {
	// Handle recovery option selection
	return m, nil
}

func (m *Model) handleSecurityValidation(msg types.SecurityValidationMsg) (tea.Model, tea.Cmd) {
	// Handle security validation
	return m, nil
}

func (m *Model) handleSecurityCheck(msg types.SecurityCheckMsg) (tea.Model, tea.Cmd) {
	// Handle security check
	return m, nil
}

func (m *Model) handleMetricRecord(msg types.MetricRecordMsg) (tea.Model, tea.Cmd) {
	// Record performance metric
	m.PerfMonitor.RecordMetric(msg.Name, msg.Value)
	return m, nil
}

func (m *Model) handlePerformanceWarning(msg types.PerformanceWarningMsg) (tea.Model, tea.Cmd) {
	// Handle performance warning
	return m, nil
}

func (m *Model) handleHardwareDetectionComplete(msg types.HardwareDetectionCompleteMsg) (tea.Model, tea.Cmd) {
	// Handle hardware detection completion
	return m, nil
}

func (m *Model) handleScriptExecutionComplete(msg types.ScriptExecutionCompleteMsg) (tea.Model, tea.Cmd) {
	// Handle script execution completion
	return m, nil
}

func (m *Model) handleVerificationComplete(msg types.VerificationCompleteMsg) (tea.Model, tea.Cmd) {
	// Handle verification completion
	return m, nil
}

func (m *Model) handleBatchStart(msg types.BatchStartMsg) (tea.Model, tea.Cmd) {
	// Handle batch operation start
	return m, nil
}

func (m *Model) handleBatchProgress(msg types.BatchProgressMsg) (tea.Model, tea.Cmd) {
	// Handle batch operation progress
	return m, nil
}

func (m *Model) handleBatchComplete(msg types.BatchCompleteMsg) (tea.Model, tea.Cmd) {
	// Handle batch operation completion
	return m, nil
}

func (m *Model) handleCancelOperation(msg types.CancelOperationMsg) (tea.Model, tea.Cmd) {
	// Handle operation cancellation
	return m, nil
}

// Type conversion helper functions
func convertCPUInfo(cpu installer.CPUInfo) types.CPUInfo {
	return types.CPUInfo{
		Model:      cpu.Model,
		Cores:      cpu.Cores,
		Threads:    cpu.Threads,
		ClockSpeed: cpu.ClockSpeed,
		CacheSize:  cpu.CacheSize,
		Flags:      cpu.Flags,
	}
}

func convertMemoryInfo(memory installer.MemoryInfo) types.MemoryInfo {
	return types.MemoryInfo{
		TotalGB:     memory.TotalGB,
		AvailableGB: memory.AvailableGB,
		UsedGB:      memory.UsedGB,
		SwapTotalGB: memory.SwapTotalGB,
		SwapUsedGB:  memory.SwapUsedGB,
	}
}

func convertStorageInfo(storage []installer.StorageInfo) []types.StorageInfo {
	result := make([]types.StorageInfo, len(storage))
	for i, s := range storage {
		result[i] = types.StorageInfo{
			Path:        s.Path,
			Type:        s.Type,
			SizeGB:      s.SizeGB,
			UsedGB:      s.UsedGB,
			AvailableGB: s.AvailableGB,
		}
	}
	return result
}

func (m *Model) handleRecoveryAttempt(msg types.RecoveryAttemptMsg) (tea.Model, tea.Cmd) {
	// Handle recovery attempt
	return m, nil
}

func (m *Model) handleRecoverySuccess(msg types.RecoverySuccessMsg) (tea.Model, tea.Cmd) {
	// Handle recovery success
	return m, nil
}

func (m *Model) handleRecoveryFailure(msg types.RecoveryFailureMsg) (tea.Model, tea.Cmd) {
	// Handle recovery failure
	return m, nil
}

func (m *Model) handleSystemStatusUpdate(msg types.SystemStatusUpdateMsg) (tea.Model, tea.Cmd) {
	// Handle system status update
	return m, nil
}

func (m *Model) handleResourceWarning(msg types.ResourceWarningMsg) (tea.Model, tea.Cmd) {
	// Handle resource warning
	return m, nil
}

// Getters for external access
func (m *Model) GetStateManager() types.StateManager {
	return m.StateManager
}

func (m *Model) GetSecurityValidator() types.SecurityValidator {
	return m.Security
}

func (m *Model) GetPerformanceMonitor() types.PerformanceMonitor {
	return m.PerfMonitor
}

func (m *Model) GetIntegrationManager() types.IntegrationManager {
	return m.Integration
}

func (m *Model) GetActiveBatches() map[string]*types.BatchOperation {
	return m.ActiveBatches
}

func (m *Model) GetSecurityContext() map[string]bool {
	return m.SecurityContext
}
