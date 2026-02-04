// internal/ui/app.go
package ui

import (
	"fmt"
	"time"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/scooter-lacroix/mlstack-installer/internal/installer"
)

// Messages for async operations
type (
	hardwareDetectedMsg struct {
		gpuInfo          installer.GPUInfo
		systemInfo       installer.SystemInfo
		preflightResults *installer.PreFlightCheckResults
		progressMsg      struct {
			componentID string
			progress    float64
			message     string
		}
	}

	systemInfoMsg struct {
		systemInfo installer.SystemInfo
	}

	preflightCheckMsg struct {
		results *installer.PreFlightCheckResults
		error   error
	}

	configLoadMsg struct {
		config *installer.Config
		error  error
	}

	errorMsg struct {
		message   string
		component string
		errorType string
	}

	// Enhanced tick messages for UI updates
	tickMsg time.Time

	// Phase 4 enhanced messages
	hardwareProgressMsg struct {
		step       string
		progress   float64
		totalSteps int
	}

	preflightProgressMsg struct {
		checkName string
		status    string
		progress  float64
	}

	configSaveMsg struct {
		success bool
		message string
	}

	// Phase 5 recovery messages
	checkpointCreatedMsg struct {
		id   string
		name string
	}

	snapshotCreatedMsg struct {
		id   string
		name string
	}

	rollbackMsg struct {
		targetID string
		success  bool
		error    error
	}

	recoveryCompleteMsg struct {
		success bool
		message string
	}
)

// Stage constants for enhanced workflow
const (
	StageWelcome = iota
	StageHardwareDetect
	StagePreFlightCheck
	StageSystemInfo
	StageComponentSelect
	StageConfiguration
	StageConfirm
	StageInstalling
	StageRecovery
	StageComplete
)

// Model represents the enhanced UI state
type Model struct {
	width            int
	height           int
	ready            bool
	quitting         bool
	stage            int
	currentComponent int

	// Enhanced configuration
	config         *installer.Config
	configModified bool
	configLoaded   bool

	// Enhanced hardware detection
	gpuInfo                 installer.GPUInfo
	systemInfo              installer.SystemInfo
	preflightResults        *installer.PreFlightCheckResults
	detectionProgress       installer.DetectionProgress
	hardwareRecommendations []string

	// Component selection with enhanced state
	components         []Component
	selectedCategories map[string]bool
	hardwareDetected   bool
	preFlightPassed    bool

	// Progress tracking
	progress   progress.Model
	spinner    spinner.Model
	installLog []string
	errorLog   []string

	// Enhanced logging and monitoring
	systemScore       int
	totalTimeEstimate time.Duration
	currentPhase      string

	// UI state management
	showDetails   bool
	selectedCheck int
	autoFixMode   bool
	fixProgress   float64

	// Error recovery state
	recoveryMode       bool
	recoveryManager    *installer.RecoveryManager
	rollbackPoint      string
	errorCount         int
	recoveryOptions    []installer.RecoveryOption
	availableSnapshots []installer.SnapshotInfo
	currentRecovery    string
	recoveryProgress   float64
}

// NewModel creates a new enhanced model with all features
func NewModel() Model {
	return Model{
		ready:            false,
		quitting:         false,
		stage:            StageWelcome,
		currentComponent: 0,
		config:           nil,
		configModified:   false,
		configLoaded:     false,
		hardwareDetected: false,
		preFlightPassed:  false,
		selectedCategories: map[string]bool{
			"foundation":   true,
			"core":         true,
			"extension":    false,
			"environment":  false,
			"verification": false,
		},
		installLog:              []string{},
		errorLog:                []string{},
		systemScore:             0,
		recoveryMode:            false,
		recoveryManager:         nil,
		rollbackPoint:           "",
		errorCount:              0,
		recoveryOptions:         []installer.RecoveryOption{},
		availableSnapshots:      []installer.SnapshotInfo{},
		currentRecovery:         "",
		recoveryProgress:        0.0,
		totalTimeEstimate:       0,
		currentPhase:            "initialization",
		showDetails:             false,
		selectedCheck:           0,
		autoFixMode:             false,
		fixProgress:             0,
		progress:                progress.New(),
		spinner:                 spinner.New(spinner.WithSpinner(spinner.Line)),
		hardwareRecommendations: []string{},
	}
}

// Init initializes the model with enhanced initialization sequence
func (m Model) Init() tea.Cmd {
	return tea.Batch(
		m.spinner.Tick,
		loadConfiguration,
		detectHardwareEnhanced,
	)
}

// Update handles messages and updates state with comprehensive error handling and new features
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.ready = true

		// Update progress bar width if needed
		m.progress.Width = m.width - 20
		if m.progress.Width < 20 {
			m.progress.Width = 20
		}

		return m, nil

	case tea.KeyMsg:
		return m.handleEnhancedKeyPress(msg)

	case hardwareDetectedMsg:
		m.gpuInfo = msg.gpuInfo
		m.systemInfo = msg.systemInfo
		m.preflightResults = msg.preflightResults
		m.hardwareDetected = true

		// Calculate system score
		m.systemScore = msg.systemInfo.TotalScore

		// Update total time estimate based on hardware
		m.calculateTimeEstimate()

		// Log comprehensive hardware detection
		detectionLog := fmt.Sprintf("[HARDWARE] Detected: %s | System: %s | Score: %d",
			msg.gpuInfo.Model, msg.systemInfo.Distribution, m.systemScore)
		m.installLog = append(m.installLog, detectionLog)

		// Log pre-flight results
		if msg.preflightResults != nil {
			preflightLog := fmt.Sprintf("[PREFLIGHT] Status: %s | Passed: %d | Failed: %d | Warnings: %d",
				msg.preflightResults.OverallStatus,
				msg.preflightResults.PassedCount,
				msg.preflightResults.FailedCount,
				msg.preflightResults.WarningCount)
			m.installLog = append(m.installLog, preflightLog)

			// Check if installation can proceed
			if msg.preflightResults.CanContinue {
				m.stage = StageComponentSelect
			} else {
				m.stage = StagePreFlightCheck
				m.recoveryMode = true
			}
		} else {
			m.stage = StageComponentSelect
		}

		return m, nil

	case systemInfoMsg:
		m.systemInfo = msg.systemInfo
		m.systemScore = msg.systemInfo.TotalScore

		// Log system information
		sysInfoLog := fmt.Sprintf("[SYSTEM] OS: %s | CPU: %s | Memory: %.1fGB | Score: %d",
			msg.systemInfo.Distribution,
			msg.systemInfo.CPU.Model,
			msg.systemInfo.Memory.TotalGB,
			m.systemScore)
		m.installLog = append(m.installLog, sysInfoLog)

		return m, nil

	case preflightCheckMsg:
		if msg.error != nil {
			m.errorLog = append(m.errorLog, fmt.Sprintf("[ERROR] Pre-flight checks failed: %v", msg.error))
			m.stage = StagePreFlightCheck
			m.recoveryMode = true
			return m, nil
		}

		m.preflightResults = msg.results
		m.preFlightPassed = msg.results.CanContinue

		// Log pre-flight completion
		completionLog := fmt.Sprintf("[PREFLIGHT] Completed with %d passed, %d failed, %d warnings",
			msg.results.PassedCount, msg.results.FailedCount, msg.results.WarningCount)
		m.installLog = append(m.installLog, completionLog)

		// Proceed to next stage based on results
		if msg.results.CanContinue {
			m.stage = StageComponentSelect
		} else {
			m.stage = StagePreFlightCheck
			m.recoveryMode = true
		}

		return m, nil

	case configLoadMsg:
		if msg.error != nil {
			m.errorLog = append(m.errorLog, fmt.Sprintf("[ERROR] Failed to load configuration: %v", msg.error))
			// Continue with default configuration
			return m, nil
		}

		m.config = msg.config
		m.configModified = false

		// Log configuration loading
		configLog := fmt.Sprintf("[CONFIG] Loaded configuration for session %s", msg.config.SessionInfo.SessionID)
		m.installLog = append(m.installLog, configLog)

		return m, nil

	case installer.CompletedMsg:
		return m.handleEnhancedInstallationComplete(msg)

	case installer.ProgressMsg:
		return m.handleEnhancedProgressUpdate(msg)

	case hardwareProgressMsg:
		// Update detection progress
		m.detectionProgress.CurrentStep = msg.step
		m.detectionProgress.TotalSteps = msg.totalSteps
		m.currentPhase = fmt.Sprintf("Hardware Detection (%.0f%%)", msg.progress*100)

		return m, nil

	case preflightProgressMsg:
		m.currentPhase = fmt.Sprintf("Pre-flight Check: %s (%s)", msg.checkName, msg.status)

		// Update progress for the specific check
		for i := range m.preflightResults.Checks {
			if m.preflightResults.Checks[i].Name == msg.checkName {
				// Update based on status
				switch msg.status {
				case "passed":
					m.preflightResults.Checks[i].Status = "passed"
				case "failed":
					m.preflightResults.Checks[i].Status = "failed"
				case "warning":
					m.preflightResults.Checks[i].Status = "warning"
				}
				break
			}
		}

		return m, nil

	case configSaveMsg:
		m.configModified = !msg.success
		if msg.success {
			m.installLog = append(m.installLog, fmt.Sprintf("[CONFIG] %s", msg.message))
		} else {
			m.errorLog = append(m.errorLog, fmt.Sprintf("[ERROR] %s", msg.message))
		}
		return m, nil

	// Phase 5 recovery message handlers
	case checkpointCreatedMsg:
		m.installLog = append(m.installLog, fmt.Sprintf("[RECOVERY] Checkpoint created: %s (%s)", msg.name, msg.id))
		m.rollbackPoint = msg.id
		return m, nil

	case snapshotCreatedMsg:
		m.installLog = append(m.installLog, fmt.Sprintf("[RECOVERY] Snapshot created: %s (%s)", msg.name, msg.id))
		m.rollbackPoint = msg.id
		return m, nil

	case rollbackMsg:
		if msg.success {
			m.installLog = append(m.installLog, fmt.Sprintf("[RECOVERY] Rollback to %s completed successfully", msg.targetID))
			m.stage = StageWelcome
			m.recoveryMode = false
		} else {
			m.errorLog = append(m.errorLog, fmt.Sprintf("[ERROR] Rollback to %s failed: %v", msg.targetID, msg.error))
			m.stage = StageRecovery
		}
		return m, nil

	case recoveryCompleteMsg:
		if msg.success {
			m.installLog = append(m.installLog, fmt.Sprintf("[RECOVERY] %s", msg.message))
			m.recoveryMode = false
			m.stage = StageComponentSelect
		} else {
			m.errorLog = append(m.errorLog, fmt.Sprintf("[ERROR] Recovery failed: %s", msg.message))
			m.stage = StageRecovery
		}
		return m, nil

	case tickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		_ = cmd // Use cmd to avoid unused variable error
		return m, nil
	}

	return m, nil
}

// calculateTimeEstimate calculates installation time based on system capabilities
func (m Model) calculateTimeEstimate() {
	baseTime := 60 * time.Minute // Base estimate: 60 minutes

	// Adjust based on system score
	scoreMultiplier := float64(m.systemScore) / 100.0
	adjustedTime := time.Duration(float64(baseTime) / scoreMultiplier)

	// Adjust based on GPU capabilities
	if m.gpuInfo.GPUCount > 0 {
		if m.gpuInfo.MemoryGB >= 24 {
			adjustedTime = time.Duration(float64(adjustedTime) * 0.8) // 20% faster with high VRAM
		} else if m.gpuInfo.MemoryGB < 16 {
			adjustedTime = time.Duration(float64(adjustedTime) * 1.2) // 20% slower with low VRAM
		}

		if m.gpuInfo.ComputeUnits >= 96 {
			adjustedTime = time.Duration(float64(adjustedTime) * 0.9) // 10% faster with high CUs
		}
	}

	// Adjust based on CPU
	if m.systemInfo.CPU.Cores >= 8 {
		adjustedTime = time.Duration(float64(adjustedTime) * 0.8) // 20% faster with more cores
	} else if m.systemInfo.CPU.Cores < 4 {
		adjustedTime = time.Duration(float64(adjustedTime) * 1.3) // 30% slower with fewer cores
	}

	m.totalTimeEstimate = adjustedTime
}

// Enhanced key press handler with new features
func (m Model) handleEnhancedKeyPress(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if !m.ready {
		return m, nil
	}

	switch msg.String() {
	case "ctrl+c", "q":
		if m.stage != StageInstalling {
			m.quitting = true
			return m, tea.Quit
		}
		return m, nil

	case "enter":
		return m.handleEnhancedEnter()

	case " ", "space":
		if m.stage == StageComponentSelect {
			return m.toggleComponent()
		} else if m.stage == StagePreFlightCheck {
			return m.toggleCheckSelection()
		} else if m.stage == StageConfiguration {
			return m.toggleConfiguration()
		}

	case "up", "k":
		if m.stage == StageComponentSelect {
			return m.navigateSelection(-1)
		} else if m.stage == StagePreFlightCheck {
			return m.navigateCheckSelection(-1)
		} else if m.stage == StageConfiguration {
			return m.navigateConfiguration(-1)
		}

	case "down", "j":
		if m.stage == StageComponentSelect {
			return m.navigateSelection(1)
		} else if m.stage == StagePreFlightCheck {
			return m.navigateCheckSelection(1)
		} else if m.stage == StageConfiguration {
			return m.navigateConfiguration(1)
		}

	case "tab":
		if m.stage == StageComponentSelect {
			return m.switchCategory()
		} else if m.stage == StageConfiguration {
			return m.switchConfigurationTab()
		}

	case "h", "left":
		if m.stage == StageComponentSelect {
			return m.navigateCategory(-1)
		} else if m.stage == StageConfiguration {
			return m.navigateConfigurationTab(-1)
		}

	case "l", "right":
		if m.stage == StageComponentSelect {
			return m.navigateCategory(1)
		} else if m.stage == StageConfiguration {
			return m.navigateConfigurationTab(1)
		}

	case "f":
		// Apply auto-fixes
		if m.stage == StagePreFlightCheck && m.preflightResults != nil {
			return m.applyAutoFixes()
		} else if m.stage == StageRecovery {
			return m.executeRecoveryOption()
		}

	case "r":
		// Restart/recover
		if m.stage == StageComplete || m.stage == StageConfirm || m.recoveryMode {
			m.restartInstallation()
			return m, nil
		} else if m.stage == StageRecovery {
			return m.restartRecovery()
		}

	case "s":
		// Save configuration
		if m.config != nil && m.configModified {
			m.saveConfiguration()
			return m, nil
		} else if m.stage == StageRecovery {
			return m.createRecoverySnapshot()
		}

	case "d":
		// Toggle details
		m.showDetails = !m.showDetails
		return m, nil

	case "c":
		// Show configuration
		if m.stage == StageWelcome || m.stage == StageComplete {
			m.stage = StageConfiguration
			return m, nil
		} else if m.stage == StageConfirm || m.stage == StageRecovery {
			// Show configuration view - this would typically switch to a configuration view
			m.showDetails = true // For now, just show details
			return m, nil
		}

	case "b":
		// Backup/rollback
		if m.stage == StageRecovery {
			return m.createRecoveryCheckpoint()
		}

	case "o":
		// Restore from snapshot
		if m.stage == StageRecovery {
			return m.restoreFromSnapshot()
		}

		// case "h":
		// // Show help
		// return m.showHelp()
	}

	return m, nil
}

// Enhanced enter handler
func (m Model) handleEnhancedEnter() (tea.Model, tea.Cmd) {
	switch m.stage {
	case StageWelcome:
		m.stage = StageHardwareDetect

		// Log start of installation
		startLog := fmt.Sprintf("[START] ML Stack installation started at %s",
			time.Now().Format("2006-01-02 15:04:05"))
		m.installLog = append(m.installLog, startLog)

		return m, detectHardwareEnhanced

	case StageHardwareDetect:
		m.stage = StagePreFlightCheck
		return m, runPreFlightChecks

	case StagePreFlightCheck:
		if m.preflightResults != nil && m.preflightResults.CanContinue {
			m.stage = StageComponentSelect
			return m, nil
		}
		return m, nil

	case StageSystemInfo:
		m.stage = StageComponentSelect
		return m, nil

	case StageComponentSelect:
		m.stage = StageConfirm

		// Log component selection
		selected := m.GetSelectedComponents()
		selectionLog := fmt.Sprintf("[SELECTION] Selected %d components for installation", len(selected))
		m.installLog = append(m.installLog, selectionLog)

		return m, nil

	case StageConfiguration:
		m.stage = StageWelcome
		return m, nil

	case StageConfirm:
		m.stage = StageInstalling
		m.currentComponent = 0

		selected := m.GetSelectedComponents()
		if len(selected) > 0 {
			// Start first installation
			executor := installer.NewScriptExecutor()

			// Log start of first component
			startLog := fmt.Sprintf("[INSTALL] Starting installation of %s", selected[0].Name)
			m.installLog = append(m.installLog, startLog)

			return m, executor.Execute(selected[0].ID, selected[0].Script)
		}
		return m, nil

	case StageComplete:
		m.quitting = true
		return m, tea.Quit
	}

	return m, nil
}

// Enhanced installation complete handler
func (m Model) handleEnhancedInstallationComplete(msg installer.CompletedMsg) (tea.Model, tea.Cmd) {
	componentName := m.getComponentNameByID(msg.ComponentID)

	if !msg.Success {
		errorMsg := fmt.Sprintf("❌ Failed to install %s: %v", componentName, msg.Error)
		m.errorLog = append(m.errorLog, errorMsg)
		m.errorCount++

		// Log detailed error
		logEntry := fmt.Sprintf("[FAILED] %s - Error: %v | Time: %s",
			componentName, msg.Error, time.Now().Format("2006-01-02 15:04:05"))
		m.installLog = append(m.installLog, logEntry)

		// Mark component as failed
		for i := range m.components {
			if m.components[i].ID == msg.ComponentID {
				m.components[i].Installed = false
				m.components[i].Progress = 0.0
				break
			}
		}

		// Check if we should enter recovery mode
		if m.errorCount > 3 {
			m.recoveryMode = true
			m.stage = StagePreFlightCheck
			return m, m.showRecoveryOptions()
		}
	} else {
		successMsg := fmt.Sprintf("✅ Successfully installed %s", componentName)
		m.installLog = append(m.installLog, successMsg)

		// Log successful completion
		logEntry := fmt.Sprintf("[SUCCESS] %s - Completed at %s",
			componentName, time.Now().Format("2006-01-02 15:04:05"))
		m.installLog = append(m.installLog, logEntry)

		// Mark component as installed
		for i := range m.components {
			if m.components[i].ID == msg.ComponentID {
				m.components[i].Installed = true
				m.components[i].Progress = 1.0
				break
			}
		}
	}

	// Install next component
	m.currentComponent++
	if m.currentComponent < len(m.GetSelectedComponents()) {
		nextComp := m.GetSelectedComponents()[m.currentComponent]
		executor := installer.NewScriptExecutor()

		// Log start of next component
		startLog := fmt.Sprintf("[START] Installing %s (%s) - Estimated: %s",
			nextComp.Name, nextComp.ID, nextComp.Estimate)
		m.installLog = append(m.installLog, startLog)

		return m, executor.Execute(nextComp.ID, nextComp.Script)
	}

	// All components done
	m.stage = StageComplete
	completeLog := fmt.Sprintf("[COMPLETE] Installation completed at %s with %d errors",
		time.Now().Format("2006-01-02 15:04:05"), len(m.errorLog))
	m.installLog = append(m.installLog, completeLog)

	// Save installation record
	if m.config != nil {
		record := installer.InstallationRecord{
			ID:           fmt.Sprintf("install_%d", time.Now().Unix()),
			Timestamp:    time.Now(),
			Version:      "0.1.5",
			Components:   m.getSelectedComponentIDs(),
			Status:       "completed",
			Duration:     time.Since(m.config.CurrentState.StartTime),
			SuccessCount: len(m.GetSelectedComponents()) - len(m.errorLog),
			ErrorCount:   len(m.errorLog),
			LogFile:      "",
		}
		m.config.AddToHistory(record)
		m.configModified = true
		m.config.Save()
	}

	return m, nil
}

// Enhanced progress update handler
func (m Model) handleEnhancedProgressUpdate(msg installer.ProgressMsg) (tea.Model, tea.Cmd) {
	// Update progress for current component
	for i := range m.components {
		if m.components[i].ID == msg.ComponentID {
			m.components[i].Progress = msg.Progress

			// Log progress updates
			if msg.Progress > 0 && msg.Progress <= 1.0 {
				progressLog := fmt.Sprintf("[PROGRESS] %s: %.1f%% - %s",
					m.components[i].Name, msg.Progress*100, msg.Message)
				m.installLog = append(m.installLog, progressLog)
			}

			break
		}
	}

	// Update progress bar
	cmd := m.progress.SetPercent(msg.Progress)

	// Create progress message for UI
	progressMsg := struct {
		componentID string
		progress    float64
		message     string
	}{
		componentID: msg.ComponentID,
		progress:    msg.Progress,
		message:     msg.Message,
	}

	return m, tea.Sequence(cmd, func() tea.Msg { return progressMsg })
}

// Toggle component selection with enhanced features
func (m Model) toggleComponent() (tea.Model, tea.Cmd) {
	// Find currently selected component and toggle its selection
	for i := range m.components {
		if m.components[i].Selected {
			// Toggle selection
			m.components[i].Selected = !m.components[i].Selected

			// Log selection change
			action := "selected"
			if !m.components[i].Selected {
				action = "deselected"
			}
			logEntry := fmt.Sprintf("[UI] %s %s (%s)", action, m.components[i].Name, m.components[i].ID)
			m.installLog = append(m.installLog, logEntry)

			// Update configuration if available
			if m.config != nil {
				compSettings := m.GetComponentSettings(m.components[i].ID)
				if compSettings != nil {
					compSettings.Enabled = m.components[i].Selected
					m.configModified = true
				}
			}

			break
		}
	}
	return m, nil
}

// Navigate selection with enhanced features
func (m Model) navigateSelection(direction int) (tea.Model, tea.Cmd) {
	// Find currently selected component and move to adjacent one
	currentIndex := -1

	for i, comp := range m.components {
		if comp.Selected {
			currentIndex = i
			break
		}
	}

	if currentIndex >= 0 {
		// Calculate new index
		newIndex := currentIndex + direction
		if newIndex < 0 {
			newIndex = len(m.components) - 1
		} else if newIndex >= len(m.components) {
			newIndex = 0
		}

		// Deselect current and select new
		m.components[currentIndex].Selected = false
		m.components[newIndex].Selected = true

		// Log navigation
		navLog := fmt.Sprintf("[UI] Navigated to %s (%s)", m.components[newIndex].Name, m.components[newIndex].ID)
		m.installLog = append(m.installLog, navLog)
	}

	return m, nil
}

// Switch category with enhanced features
func (m Model) switchCategory() (tea.Model, tea.Cmd) {
	// Cycle through categories: foundation -> core -> extension -> environment -> verification
	categories := []string{"foundation", "core", "extension", "environment", "verification"}

	for i, cat := range categories {
		if m.selectedCategories[cat] {
			// Find next category
			nextIndex := (i + 1) % len(categories)
			nextCat := categories[nextIndex]

			m.selectedCategories[cat] = false
			m.selectedCategories[nextCat] = true

			// Log category switch
			switchLog := fmt.Sprintf("[UI] Switched to %s components category", nextCat)
			m.installLog = append(m.installLog, switchLog)

			// Update configuration if available
			if m.config != nil {
				m.configModified = true
			}

			break
		}
	}

	return m, nil
}

// Navigate category with enhanced features
func (m Model) navigateCategory(direction int) (tea.Model, tea.Cmd) {
	categories := []string{"foundation", "core", "extension", "environment", "verification"}

	// Find current category
	var currentCategory string
	for cat, selected := range m.selectedCategories {
		if selected {
			currentCategory = cat
			break
		}
	}

	// Find category index
	currentIndex := -1
	for i, cat := range categories {
		if cat == currentCategory {
			currentIndex = i
			break
		}
	}

	// Navigate to adjacent category
	if currentIndex >= 0 {
		newIndex := currentIndex + direction
		if newIndex < 0 {
			newIndex = len(categories) - 1
		} else if newIndex >= len(categories) {
			newIndex = 0
		}

		newCategory := categories[newIndex]

		// Update selected categories
		m.selectedCategories[currentCategory] = false
		m.selectedCategories[newCategory] = true

		// Log navigation
		navLog := fmt.Sprintf("[UI] Navigated to %s components category", newCategory)
		m.installLog = append(m.installLog, navLog)
	}

	return m, nil
}

// Restart installation with enhanced recovery
func (m Model) restartInstallation() {
	m.stage = StageComponentSelect
	m.currentComponent = 0
	m.errorLog = []string{}
	m.installLog = []string{}
	m.errorCount = 0
	m.recoveryMode = false

	// Reset component progress but keep selections
	for i := range m.components {
		m.components[i].Installed = false
		m.components[i].Progress = 0.0
	}

	// Log restart
	restartLog := fmt.Sprintf("[RESTART] Installation restarted at %s",
		time.Now().Format("2006-01-02 15:04:05"))
	m.installLog = append(m.installLog, restartLog)

	// Reset configuration state
	if m.config != nil {
		m.config.CurrentState = installer.InstallationState{
			CurrentStage:   "restarted",
			Status:         "ready",
			StartTime:      time.Now(),
			LastUpdateTime: time.Now(),
		}
		m.configModified = true
	}
}

// Get component name by ID
func (m Model) getComponentNameByID(componentID string) string {
	for _, comp := range m.components {
		if comp.ID == componentID {
			return comp.Name
		}
	}
	return componentID
}

// Get selected components with enhanced filtering
func (m Model) GetSelectedComponents() []Component {
	var selected []Component
	for _, comp := range m.components {
		if comp.Selected {
			selected = append(selected, comp)
		}
	}
	return selected
}

// Get required components
func (m Model) GetRequiredComponents() []Component {
	var required []Component
	for _, comp := range m.components {
		if comp.Required {
			required = append(required, comp)
		}
	}
	return required
}

// Get total size of selected components
func (m Model) GetTotalSize() int64 {
	var total int64
	for _, comp := range m.GetSelectedComponents() {
		total += comp.Size
	}
	return total
}

// Get total estimated time
func (m Model) GetTotalTime() string {
	if m.totalTimeEstimate > 0 {
		return m.totalTimeEstimate.Truncate(time.Minute).String()
	}
	return "Unknown"
}

// Get selected component IDs
func (m Model) getSelectedComponentIDs() []string {
	var ids []string
	for _, comp := range m.GetSelectedComponents() {
		ids = append(ids, comp.ID)
	}
	return ids
}

// Get component settings from configuration
func (m Model) GetComponentSettings(componentID string) *installer.ComponentConfig {
	if m.config == nil {
		return nil
	}

	switch componentID {
	case "pytorch":
		return &m.config.ComponentSettings.PyTorch
	case "flash_attention":
		return &m.config.ComponentSettings.FlashAttn
	case "megatron":
		return &m.config.ComponentSettings.Megatron
	case "onnx_runtime":
		return &m.config.ComponentSettings.ONNXRuntime
	case "triton":
		return &m.config.ComponentSettings.Triton
	case "rocm":
		return &m.config.ComponentSettings.ROCm
	case "mpi":
		return &m.config.ComponentSettings.MPI
	case "wandb":
		return &m.config.ComponentSettings.WandB
	default:
		if settings, exists := m.config.ComponentSettings.CustomConfigs[componentID]; exists {
			return &settings
		}
		return nil
	}
}

// Enhanced hardware detection
func detectHardwareEnhanced() tea.Msg {
	// Use enhanced hardware detection
	gpuInfo, err := installer.DetectGPU()
	if err != nil {
		return hardwareDetectedMsg{
			gpuInfo: installer.GPUInfo{
				Model:    "Unknown",
				Status:   "error",
				GPUCount: 0,
			},
			systemInfo:       installer.SystemInfo{},
			preflightResults: nil,
		}
	}

	// Get system information
	systemInfo, err := installer.DetectSystem()
	if err != nil {
		return hardwareDetectedMsg{
			gpuInfo:          gpuInfo,
			systemInfo:       systemInfo,
			preflightResults: nil,
		}
	}

	// Run pre-flight checks
	preflightResults, err := installer.RunPreFlightChecks(systemInfo, gpuInfo)
	if err != nil {
		return hardwareDetectedMsg{
			gpuInfo:          gpuInfo,
			systemInfo:       systemInfo,
			preflightResults: preflightResults,
		}
	}

	return hardwareDetectedMsg{
		gpuInfo:          gpuInfo,
		systemInfo:       systemInfo,
		preflightResults: preflightResults,
	}
}

// Load configuration
func loadConfiguration() tea.Msg {
	config, err := installer.NewConfig()
	if err != nil {
		return configLoadMsg{
			config: nil,
			error:  err,
		}
	}
	return configLoadMsg{
		config: config,
		error:  nil,
	}
}

// Run pre-flight checks
func runPreFlightChecks() tea.Msg {
	// This would be called after hardware detection
	return nil
}

// Save configuration
func (m Model) saveConfiguration() tea.Msg {
	if m.config == nil {
		return configSaveMsg{
			success: false,
			message: "No configuration loaded",
		}
	}

	if err := m.config.Save(); err != nil {
		return configSaveMsg{
			success: false,
			message: fmt.Sprintf("Failed to save configuration: %v", err),
		}
	}

	return configSaveMsg{
		success: true,
		message: "Configuration saved successfully",
	}
}

// Show recovery options
func (m Model) showRecoveryOptions() tea.Cmd {
	// Implement recovery options
	return nil
}

// Apply auto-fixes
func (m Model) applyAutoFixes() (tea.Model, tea.Cmd) {
	if m.preflightResults == nil {
		return m, nil
	}

	m.autoFixMode = true
	m.fixProgress = 0.0

	// Apply fixes sequentially
	return m, m.applyNextFix(0)
}

// Apply next fix
func (m Model) applyNextFix(index int) tea.Cmd {
	if index >= len(m.preflightResults.AutoFixes) {
		m.autoFixMode = false
		// Re-run checks
		return runPreFlightChecks
	}

	fix := m.preflightResults.AutoFixes[index]
	m.fixProgress = float64(index) / float64(len(m.preflightResults.AutoFixes))

	// Apply fix (this would be implemented in the installer)
	fmt.Printf("Applying fix: %s\n", fix.Name)

	// Return command to apply next fix after delay
	return tea.Tick(2*time.Second, func(t time.Time) tea.Msg {
		return m.applyNextFix(index + 1)
	})
}

// Toggle check selection in pre-flight screen
func (m Model) toggleCheckSelection() (tea.Model, tea.Cmd) {
	// Implement check selection toggle
	return m, nil
}

// Navigate check selection
func (m Model) navigateCheckSelection(direction int) (tea.Model, tea.Cmd) {
	// Implement check selection navigation
	return m, nil
}

// Toggle configuration
func (m Model) toggleConfiguration() (tea.Model, tea.Cmd) {
	// Implement configuration toggle
	return m, nil
}

// Navigate configuration
func (m Model) navigateConfiguration(direction int) (tea.Model, tea.Cmd) {
	// Implement configuration navigation
	return m, nil
}

// Switch configuration tab
func (m Model) switchConfigurationTab() (tea.Model, tea.Cmd) {
	// Implement tab switching
	return m, nil
}

// Navigate configuration tab
func (m Model) navigateConfigurationTab(direction int) (tea.Model, tea.Cmd) {
	// Implement tab navigation
	return m, nil
}

// Show help
func (m Model) showHelp() (tea.Model, tea.Cmd) {
	// Implement help display
	return m, nil
}

// Recovery helper functions

// initializeRecoveryManager initializes the recovery manager
func (m Model) initializeRecoveryManager() (tea.Model, tea.Cmd) {
	if m.recoveryManager == nil && m.config != nil {
		recoveryManager, err := installer.NewRecoveryManager(m.config)
		if err != nil {
			m.errorLog = append(m.errorLog, fmt.Sprintf("[ERROR] Failed to initialize recovery manager: %v", err))
			return m, nil
		}
		m.recoveryManager = recoveryManager

		// Load available snapshots
		snapshots := recoveryManager.GetAvailableSnapshots()
		m.availableSnapshots = snapshots

		// Initialize recovery options
		m.recoveryOptions = m.getRecoveryOptions()

		// Log initialization
		m.installLog = append(m.installLog, "[RECOVERY] Recovery manager initialized")
	}

	return m, nil
}

// getRecoveryOptions generates recovery options based on current state
func (m Model) getRecoveryOptions() []installer.RecoveryOption {
	stageName := "unknown"
	switch m.stage {
	case StageHardwareDetect:
		stageName = "hardware"
	case StagePreFlightCheck:
		stageName = "preflight"
	case StageComponentSelect:
		stageName = "component"
	case StageInstalling:
		stageName = "installation"
	}

	lastComponent := ""
	if m.currentComponent > 0 && m.currentComponent < len(m.GetSelectedComponents()) {
		lastComponent = m.GetSelectedComponents()[m.currentComponent-1].Name
	}

	return m.recoveryManager.GetRecoveryOptions(stageName, lastComponent, m.errorLog)
}

// createRecoveryCheckpoint creates a recovery checkpoint
func (m Model) createRecoveryCheckpoint() (tea.Model, tea.Cmd) {
	if m.recoveryManager == nil {
		return m, nil
	}

	name := "manual_checkpoint"
	if m.currentComponent > 0 && m.currentComponent < len(m.GetSelectedComponents()) {
		name = m.GetSelectedComponents()[m.currentComponent-1].Name
	}

	err := m.recoveryManager.CreateRecoveryCheckpoint(name, "")
	if err != nil {
		m.errorLog = append(m.errorLog, fmt.Sprintf("[ERROR] Failed to create checkpoint: %v", err))
	} else {
		// Send checkpoint created message
		cmd := func() tea.Msg {
			return checkpointCreatedMsg{
				id:   fmt.Sprintf("ckpt_%d", time.Now().Unix()),
				name: name,
			}
		}
		return m, cmd
	}

	return m, nil
}

// createRecoverySnapshot creates a recovery snapshot
func (m Model) createRecoverySnapshot() (tea.Model, tea.Cmd) {
	if m.recoveryManager == nil {
		return m, nil
	}

	name := "manual_snapshot"
	selected := m.GetSelectedComponents()
	componentNames := make([]string, len(selected))
	for i, comp := range selected {
		componentNames[i] = comp.Name
	}

	err := m.recoveryManager.CreateRecoverySnapshot(name, componentNames)
	if err != nil {
		m.errorLog = append(m.errorLog, fmt.Sprintf("[ERROR] Failed to create snapshot: %v", err))
	} else {
		// Send snapshot created message
		cmd := func() tea.Msg {
			return snapshotCreatedMsg{
				id:   fmt.Sprintf("snap_%d", time.Now().Unix()),
				name: name,
			}
		}
		return m, cmd
	}

	return m, nil
}

// executeRecoveryOption executes the selected recovery option
func (m Model) executeRecoveryOption() (tea.Model, tea.Cmd) {
	if m.recoveryManager == nil || len(m.recoveryOptions) == 0 {
		return m, nil
	}

	// Execute the first available recovery option
	selectedOption := m.recoveryOptions[0]

	go func() {
		err := m.recoveryManager.ExecuteRecoveryOption(selectedOption)

		// Send completion message
		cmd := func() tea.Msg {
			return recoveryCompleteMsg{
				success: err == nil,
				message: fmt.Sprintf("Recovery %s: %s",
					func() string {
						if err != nil {
							return "failed"
						}
						return "completed"
					}(), selectedOption.Name),
			}
		}

		// Send the command
		_ = cmd()
	}()

	return m, nil
}

// restoreFromSnapshot restores from a selected snapshot
func (m Model) restoreFromSnapshot() (tea.Model, tea.Cmd) {
	if m.recoveryManager == nil || len(m.availableSnapshots) == 0 {
		return m, nil
	}

	// Restore from the most recent snapshot
	targetSnapshot := m.availableSnapshots[0]

	go func() {
		err := m.recoveryManager.RestoreFromSnapshot(targetSnapshot.ID)

		// Send rollback message
		cmd := func() tea.Msg {
			return rollbackMsg{
				targetID: targetSnapshot.ID,
				success:  err == nil,
				error:    err,
			}
		}

		// Send the command
		_ = cmd()
	}()

	return m, nil
}

// restartRecovery restarts the recovery process
func (m Model) restartRecovery() (tea.Model, tea.Cmd) {
	m.stage = StageWelcome
	m.recoveryMode = false
	m.recoveryOptions = []installer.RecoveryOption{}
	m.availableSnapshots = []installer.SnapshotInfo{}

	// Log restart
	m.installLog = append(m.installLog, "[RECOVERY] Recovery process restarted")

	return m, nil
}

// triggerRecoveryMode triggers recovery mode when errors occur
func (m Model) triggerRecoveryMode(errorType string, errorMessage string) (tea.Model, tea.Cmd) {
	m.recoveryMode = true
	m.stage = StageRecovery

	// Add error to error log
	m.errorLog = append(m.errorLog, fmt.Sprintf("[ERROR] %s: %s", errorType, errorMessage))

	// Generate recovery options
	m.recoveryOptions = m.getRecoveryOptions()

	// Log recovery activation
	m.installLog = append(m.installLog, fmt.Sprintf("[RECOVERY] Recovery mode activated: %s", errorType))

	// Initialize recovery manager
	return m.initializeRecoveryManager()
}

// updateRecoveryProgress updates recovery progress for UI display
func (m Model) updateRecoveryProgress(progress float64, message string) (tea.Model, tea.Cmd) {
	m.recoveryProgress = progress

	// Log progress update
	progressLog := fmt.Sprintf("[RECOVERY] %.1f%%: %s", progress*100, message)
	m.installLog = append(m.installLog, progressLog)

	return m, nil
}

// getAvailableSnapshotsForUI returns formatted snapshot information for UI
func (m Model) getAvailableSnapshotsForUI() []string {
	snapshots := []string{}
	for _, snapshot := range m.availableSnapshots {
		snapshotInfo := fmt.Sprintf("📸 %s (%s)",
			snapshot.ID,
			snapshot.Timestamp.Format("2006-01-02 15:04:05"))
		snapshots = append(snapshots, snapshotInfo)
	}
	return snapshots
}

// getRecoveryOptionsForUI returns formatted recovery options for UI
func (m Model) getRecoveryOptionsForUI() []string {
	options := []string{}
	for i, option := range m.recoveryOptions {
		optionInfo := fmt.Sprintf("[%d] %s (%s)", i, option.Name, option.RiskLevel)
		options = append(options, optionInfo)
	}
	return options
}
