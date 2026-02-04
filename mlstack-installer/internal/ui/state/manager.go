// internal/ui/state/manager.go
package state

import (
	"fmt"
	"time"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/scooter-lacroix/mlstack-installer/internal/installer"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// Manager implements centralized state management
type Manager struct {
	state               types.State
	lastProgressPercent map[string]int
	lastProgressMessage map[string]string
}

// NewManager creates a new state manager with initial state
func NewManager() *Manager {
	// Initialize progress bar
	prog := progress.New(
		progress.WithDefaultGradient(),
		progress.WithWidth(60),
	)

	// Initialize spinner
	s := spinner.New()
	s.Spinner = spinner.Line

	return &Manager{
		state: types.State{
			Stage:      types.StageWelcome,
			Width:      80,
			Height:     24,
			Ready:      false,
			Quitting:   false,
			LastUpdate: time.Now(),

			// Component selection defaults
			SelectedCategories: map[string]bool{
				"foundation":   true,
				"core":         true,
				"extension":    false,
				"environment":  false,
				"verification": false,
			},

			// UI elements
			Progress: prog,
			Spinner:  s,

			// Arrays and slices
			InstallLog:         []string{},
			ErrorLog:           []string{},
			Components:         AllComponents(),
			RecoveryOptions:    []installer.RecoveryOption{},
			AvailableSnapshots: []installer.SnapshotInfo{},

			// Default values
			ShowDetails:       false,
			SelectedCheck:     0,
			AutoFixMode:       false,
			FixProgress:       0.0,
			SystemScore:       0,
			TotalTimeEstimate: 0,
			CurrentPhase:      "initialization",
			CurrentComponent:  -1,
			RecoveryProgress:  0.0,
			ErrorCount:        0,
			ConfigModified:    false,
			ConfigLoaded:      false,
			HardwareDetected:  false,
			PreFlightPassed:   false,
			RecoveryMode:      false,
		},
		lastProgressPercent: make(map[string]int),
		lastProgressMessage: make(map[string]string),
	}
}

// GetState returns the current state
func (m *Manager) GetState() types.State {
	return m.state
}

// UpdateState returns a new manager with updated state
func (m *Manager) UpdateState(newState types.State) types.StateManager {
	newManager := &Manager{
		state:               newState,
		lastProgressPercent: m.lastProgressPercent,
		lastProgressMessage: m.lastProgressMessage,
	}
	if newManager.lastProgressPercent == nil {
		newManager.lastProgressPercent = make(map[string]int)
	}
	if newManager.lastProgressMessage == nil {
		newManager.lastProgressMessage = make(map[string]string)
	}
	return newManager
}

// TransitionTo performs a safe stage transition
func (m *Manager) TransitionTo(stage types.Stage) types.StateManager {
	// Validate transition
	if !m.isValidTransition(m.state.Stage, stage) {
		// Log invalid transition attempt
		m.log("Invalid transition attempted: " + m.state.Stage.String() + " -> " + stage.String())
		return m
	}

	// Create new state with transition
	newState := m.state
	newState.Stage = stage
	newState.LastUpdate = time.Now()

	// Log transition
	m.log("Transitioning to stage: " + stage.String())

	return m.UpdateState(newState)
}

// SetWindowSize updates window dimensions
func (m *Manager) SetWindowSize(width, height int) types.StateManager {
	newState := m.state
	newState.Width = width
	newState.Height = height
	newState.Ready = true

	// Update progress bar width
	if width > 20 {
		newState.Progress.Width = width - 20
	}

	return m.UpdateState(newState)
}

// SetHardwareInfo updates hardware detection results (implements interface)
func (m *Manager) SetHardwareInfo(gpu types.GPUInfo, system types.SystemInfo) types.StateManager {
	// Convert types to installer types for internal processing
	installerGPU := installer.GPUInfo{
		Vendor:        gpu.Vendor,
		Model:         gpu.Model,
		Driver:        gpu.Driver,
		ComputeUnits:  gpu.ComputeUnits,
		MemoryGB:      gpu.MemoryGB,
		Architecture:  gpu.Architecture,
		GFXVersion:    gpu.GFXVersion,
		Temperature:   gpu.Temperature,
		GPUCount:      gpu.GPUCount,
		Status:        gpu.Status,
		Optimizations: gpu.Optimizations,
		PowerUsage:    gpu.PowerUsage,
	}

	installerSystem := installer.SystemInfo{
		OS:            system.OS,
		Distribution:  system.Distribution,
		KernelVersion: system.KernelVersion,
		Architecture:  system.Architecture,
		CPU:           convertToInstallerCPU(system.CPU),
		Memory:        convertToInstallerMemory(system.Memory),
		Storage:       convertToInstallerStorage(system.Storage),
		Timestamp:     system.Timestamp,
	}

	newState := m.state
	newState.GPUInfo = installerGPU
	newState.SystemInfo = installerSystem
	newState.HardwareDetected = true

	// Calculate system score
	m.calculateSystemScore(&newState)

	// Log hardware detection
	m.log("Hardware detected: " + gpu.Model)

	return m.UpdateState(newState)
}

// SetHardwareInfoWithSystem updates hardware detection results with installer types (internal)
func (m *Manager) SetHardwareInfoWithSystem(gpu installer.GPUInfo, system installer.SystemInfo) types.StateManager {
	newState := m.state
	newState.GPUInfo = gpu
	newState.SystemInfo = system
	newState.HardwareDetected = true

	// Calculate system score
	m.calculateSystemScore(&newState)

	// Log hardware detection
	m.log("Hardware detected: " + gpu.Model)

	return m.UpdateState(newState)
}

// SetPreflightResults updates pre-flight check results (implements interface)
func (m *Manager) SetPreflightResults(results *installer.PreFlightCheckResults) types.StateManager {
	newState := m.state
	newState.PreflightResults = results
	newState.PreFlightPassed = results != nil && results.CanContinue

	if results != nil {
		m.log(fmt.Sprintf("Pre-flight checks completed: %d passed, %d failed", results.PassedCount, results.FailedCount))
	}

	return m.UpdateState(newState)
}

// ToggleComponent toggles component selection
func (m *Manager) ToggleComponent(componentID string) types.StateManager {
	newState := m.state

	// Find and toggle component
	for i := range newState.Components {
		if newState.Components[i].ID == componentID {
			newState.Components[i].Selected = !newState.Components[i].Selected

			action := "selected"
			if !newState.Components[i].Selected {
				action = "deselected"
			}

			m.log(action + " component: " + newState.Components[i].Name)
			break
		}
	}

	return m.UpdateState(newState)
}

// StartInstallation begins the installation process
func (m *Manager) StartInstallation() types.StateManager {
	newState := m.state
	newState.Stage = types.StageInstalling
	newState.CurrentComponent = 0
	newState.ErrorLog = []string{} // Clear previous errors

	// Get selected components
	selected := m.GetSelectedComponents()
	if len(selected) > 0 {
		m.log(fmt.Sprintf("Starting installation of %d components", len(selected)))
	}

	return m.UpdateState(newState)
}

// UpdateComponentProgress updates installation progress for a component
func (m *Manager) UpdateComponentProgress(componentID string, progress float64, message string) types.StateManager {
	newState := m.state

	// Update component progress
	for i := range newState.Components {
		if newState.Components[i].ID == componentID {
			newState.Components[i].Progress = progress
			break
		}
	}

	// Update overall progress bar
	newState.Progress.SetPercent(progress)

	// Log progress (throttled to avoid UI log spam)
	if progress > 0 && progress <= 1.0 {
		progressPercent := int(progress * 100)
		lastPercent := m.lastProgressPercent[componentID]
		lastMessage := m.lastProgressMessage[componentID]

		shouldLog := message != lastMessage
		if progressPercent != lastPercent && (progressPercent%5 == 0 || progressPercent == 100) {
			shouldLog = true
		}

		if shouldLog {
			m.log(fmt.Sprintf("Progress: %s (%d%%)", message, progressPercent))
			m.lastProgressPercent[componentID] = progressPercent
			m.lastProgressMessage[componentID] = message
		} else if progressPercent > lastPercent {
			m.lastProgressPercent[componentID] = progressPercent
		}
	}

	return m.UpdateState(newState)
}

// CompleteComponent marks a component installation as complete
func (m *Manager) CompleteComponent(componentID string, success bool, err error) types.StateManager {
	newState := m.state

	// Update component status
	for i := range newState.Components {
		if newState.Components[i].ID == componentID {
			newState.Components[i].Installed = success
			if success {
				newState.Components[i].Progress = 1.0
				m.log("Completed: " + newState.Components[i].Name)
			} else {
				newState.ErrorCount++
				m.log("Failed: " + newState.Components[i].Name + " - " + err.Error())
			}
			break
		}
	}

	// Move to next component or complete
	newState.CurrentComponent++
	selected := m.GetSelectedComponents()

	if newState.CurrentComponent >= len(selected) {
		newState.Stage = types.StageComplete
		m.log(fmt.Sprintf("Installation completed with %d errors", newState.ErrorCount))
	}

	return m.UpdateState(newState)
}

// TriggerRecovery enters recovery mode
func (m *Manager) TriggerRecovery(errorType, errorMessage string) types.StateManager {
	newState := m.state
	newState.RecoveryMode = true
	newState.Stage = types.StageRecovery
	newState.ErrorLog = append(newState.ErrorLog, errorType+": "+errorMessage)

	m.log("Entering recovery mode: " + errorType)

	return m.UpdateState(newState)
}

// SetConfig updates configuration
func (m *Manager) SetConfig(config *installer.Config) types.StateManager {
	newState := m.state
	newState.Config = config
	newState.ConfigLoaded = true

	if config != nil {
		m.log("Configuration loaded successfully")
	}

	return m.UpdateState(newState)
}

// GetSelectedComponents returns all selected components (implements interface)
func (m *Manager) GetSelectedComponents() []types.Component {
	var selected []types.Component
	for _, comp := range m.state.Components {
		if comp.Selected {
			selected = append(selected, comp)
		}
	}
	return selected
}

// GetRequiredComponents returns all required components
func (m *Manager) GetRequiredComponents() []types.Component {
	var required []types.Component
	for _, comp := range m.state.Components {
		if comp.Required {
			required = append(required, comp)
		}
	}
	return required
}

// GetCurrentComponent returns the currently installing component
func (m *Manager) GetCurrentComponent() *types.Component {
	selected := m.GetSelectedComponents()
	if m.state.CurrentComponent >= 0 && m.state.CurrentComponent < len(selected) {
		return &selected[m.state.CurrentComponent]
	}
	return nil
}

// IsValidTransition checks if a stage transition is valid (implements interface)
func (m *Manager) IsValidTransition(from, to types.Stage) bool {
	return m.isValidTransition(from, to)
}

// isValidTransition checks if a stage transition is valid
func (m *Manager) isValidTransition(from, to types.Stage) bool {
	// Define valid transitions
	validTransitions := map[types.Stage][]types.Stage{
		types.StageWelcome:         {types.StageHardwareDetect, types.StageConfiguration, types.StageComponentSelect},
		types.StageHardwareDetect:  {types.StagePreFlightCheck, types.StageComponentSelect},
		types.StagePreFlightCheck:  {types.StageComponentSelect, types.StageRecovery},
		types.StageComponentSelect: {types.StageConfiguration, types.StageConfirm},
		types.StageConfiguration:   {types.StageWelcome, types.StageComponentSelect},
		types.StageConfirm:         {types.StageInstalling, types.StageComponentSelect, types.StageConfiguration},
		types.StageInstalling:      {types.StageComplete, types.StageRecovery},
		types.StageComplete:        {}, // Terminal state
		types.StageRecovery:        {types.StageWelcome, types.StageComponentSelect, types.StageHardwareDetect},
	}

	allowed, exists := validTransitions[from]
	if !exists {
		return false
	}

	for _, valid := range allowed {
		if valid == to {
			return true
		}
	}

	return false
}

// calculateSystemScore calculates a system compatibility score
func (m *Manager) calculateSystemScore(state *types.State) {
	score := 0

	// GPU detection
	if state.GPUInfo.Model != "" && state.GPUInfo.Model != "Unknown" {
		score += 40

		// GPU memory bonus
		if state.GPUInfo.MemoryGB >= 24 {
			score += 20
		} else if state.GPUInfo.MemoryGB >= 16 {
			score += 15
		} else if state.GPUInfo.MemoryGB >= 8 {
			score += 10
		}
	}

	// System memory
	if state.SystemInfo.Memory.TotalGB >= 32 {
		score += 20
	} else if state.SystemInfo.Memory.TotalGB >= 16 {
		score += 15
	} else if state.SystemInfo.Memory.TotalGB >= 8 {
		score += 10
	}

	// CPU cores
	if state.SystemInfo.CPU.Cores >= 12 {
		score += 15
	} else if state.SystemInfo.CPU.Cores >= 8 {
		score += 10
	} else if state.SystemInfo.CPU.Cores >= 4 {
		score += 5
	}

	// Disk space (if available)
	if len(state.SystemInfo.Storage) > 0 {
		totalAvailable := 0.0
		for _, storage := range state.SystemInfo.Storage {
			totalAvailable += storage.AvailableGB
		}
		if totalAvailable >= 100 {
			score += 5
		}
	}

	state.SystemScore = score
}

// Log adds a message to the installation log (public)
func (m *Manager) Log(message string) {
	m.log(message)
}

// log adds a message to the installation log (private)
func (m *Manager) log(message string) {
	timestamp := time.Now().Format("15:04:05")
	logEntry := "[" + timestamp + "] " + message
	m.state.InstallLog = append(m.state.InstallLog, logEntry)

	// Keep log size manageable
	if len(m.state.InstallLog) > 1000 {
		m.state.InstallLog = m.state.InstallLog[500:]
	}
}

// LogError adds an error to the error log (implements interface)
func (m *Manager) LogError(err error) {
	m.logError(err) // Delegate to private method
}

// logError adds an error to the error log (private method)
func (m *Manager) logError(err error) {
	timestamp := time.Now().Format("15:04:05")
	logEntry := "[" + timestamp + "] ERROR: " + err.Error()
	m.state.ErrorLog = append(m.state.ErrorLog, logEntry)
	m.Log("ERROR: " + err.Error())
}

// GetLogs returns the installation log
func (m *Manager) GetLogs() []string {
	return m.state.InstallLog
}

// GetErrorLog returns the error log
func (m *Manager) GetErrorLog() []string {
	return m.state.ErrorLog
}

// Type conversion helper functions for interface compatibility
func convertToInstallerCPU(cpu types.CPUInfo) installer.CPUInfo {
	return installer.CPUInfo{
		Model:      cpu.Model,
		Cores:      cpu.Cores,
		Threads:    cpu.Threads,
		ClockSpeed: cpu.ClockSpeed,
		CacheSize:  cpu.CacheSize,
		Flags:      cpu.Flags,
	}
}

func convertToInstallerMemory(memory types.MemoryInfo) installer.MemoryInfo {
	return installer.MemoryInfo{
		TotalGB:     memory.TotalGB,
		AvailableGB: memory.AvailableGB,
		UsedGB:      memory.UsedGB,
		SwapTotalGB: memory.SwapTotalGB,
		SwapUsedGB:  memory.SwapUsedGB,
	}
}

func convertToInstallerStorage(storage []types.StorageInfo) []installer.StorageInfo {
	result := make([]installer.StorageInfo, len(storage))
	for i, s := range storage {
		result[i] = installer.StorageInfo{
			Path:        s.Path,
			Type:        s.Type,
			SizeGB:      s.SizeGB,
			UsedGB:      s.UsedGB,
			AvailableGB: s.AvailableGB,
		}
	}
	return result
}
