// internal/ui/monitoring/monitoring.go
package monitoring

import (
	"fmt"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// MonitoringSystem provides the complete monitoring dashboard system
type MonitoringSystem struct {
	dashboard      *Dashboard
	integration    *DashboardIntegration
	optimizer      *OptimizedRenderer
	accessibility  *AccessibilityManager
	resources      *ResourceManager
	service        *SystemMonitorService
	initialized    bool
	width, height  int
	theme          *AMDTheme
	lastUpdate     time.Time
	updateInterval time.Duration
}

// MonitoringConfig holds configuration for the monitoring system
type MonitoringConfig struct {
	Enabled         bool          `json:"enabled"`
	UpdateInterval  time.Duration `json:"update_interval"`
	Theme           string        `json:"theme"`
	Optimization    string        `json:"optimization_level"`
	Accessibility   bool          `json:"accessibility_enabled"`
	HighContrast    bool          `json:"high_contrast"`
	LargeText       bool          `json:"large_text"`
	ReducedMotion   bool          `json:"reduced_motion"`
	ScreenReader    bool          `json:"screen_reader"`
	ColorBlindMode  string        `json:"color_blind_mode"`
	PerformanceMode string        `json:"performance_mode"`
}

// NewMonitoringSystem creates a new monitoring system
func NewMonitoringSystem(config MonitoringConfig) *MonitoringSystem {
	system := &MonitoringSystem{
		initialized:    false,
		updateInterval: config.UpdateInterval,
		theme:          NewAMDTheme(),
	}

	// Apply configuration
	system.applyConfig(config)

	return system
}

// applyConfig applies configuration to the monitoring system
func (m *MonitoringSystem) applyConfig(config MonitoringConfig) {
	// Set theme based on configuration
	switch config.Theme {
	case "amd":
		m.theme.SetTheme(ThemeAMD)
	case "dark":
		m.theme.SetTheme(ThemeDark)
	case "light":
		m.theme.SetTheme(ThemeLight)
	case "high_contrast":
		m.theme.SetTheme(ThemeHighContrast)
	default:
		m.theme.SetTheme(ThemeAMD)
	}

	// Create components
	m.integration = NewDashboardIntegration(m.theme)
	m.optimizer = NewOptimizedRenderer(m.theme)
	m.accessibility = NewAccessibilityManager(m.theme)
	m.resources = NewResourceManager()
	m.service = NewSystemMonitorService(config.UpdateInterval)

	// Apply accessibility settings
	m.accessibility.SetHighContrast(config.HighContrast)
	m.accessibility.SetLargeText(config.LargeText)
	m.accessibility.SetReducedMotion(config.ReducedMotion)
	m.accessibility.SetScreenReaderMode(config.ScreenReader)

	// Apply color blind mode
	switch config.ColorBlindMode {
	case "protanopia":
		m.accessibility.SetColorBlindMode(ColorBlindProtanopia)
	case "deuteranopia":
		m.accessibility.SetColorBlindMode(ColorBlindDeuteranopia)
	case "tritanopia":
		m.accessibility.SetColorBlindMode(ColorBlindTritanopia)
	case "achromatopsia":
		m.accessibility.SetColorBlindMode(ColorBlindAchromatopsia)
	}

	// Set optimization level
	m.optimizer.SetOptimizationLevel(config.Optimization)
}

// Initialize initializes the monitoring system
func (m *MonitoringSystem) Initialize(width, height int) tea.Cmd {
	m.width = width
	m.height = height

	// Initialize integration
	m.integration.Initialize(width, height)
	m.dashboard = m.integration.GetDashboard()

	// Set bounds for dashboard
	m.dashboard.SetBounds(0, 0, width, height)

	// Start monitoring service if enabled
	var cmd tea.Cmd
	if m.service != nil {
		cmd = m.service.Start()
	}

	m.initialized = true
	m.lastUpdate = time.Now()

	return cmd
}

// Update handles updates to the monitoring system
func (m *MonitoringSystem) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	if !m.initialized {
		return m, nil
	}

	var cmds []tea.Cmd

	// Handle system messages
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.dashboard.SetBounds(0, 0, msg.Width, msg.Height)

	case tea.KeyMsg:
		// Handle global key bindings for monitoring
		cmd := m.handleKeyMsg(msg)
		if cmd != nil {
			cmds = append(cmds, cmd)
		}

	// Handle monitoring-specific messages
	case MonitoringTickMsg:
		cmd = m.handleMonitoringTick(msg)
		cmds = append(cmds, cmd)

	case StartMonitoringMsg:
		if m.service != nil && !m.service.IsRunning() {
			cmd = m.service.Start()
			cmds = append(cmds, cmd)
		}

	case StopMonitoringMsg:
		if m.service != nil && m.service.IsRunning() {
			m.service.Stop()
		}

	case RefreshMonitoringMsg:
		cmd = m.handleRefresh(msg)
		cmds = append(cmds, cmd)

	case ToggleThemeMsg:
		m.theme.CycleTheme()
		m.dashboard.theme = m.theme
		m.optimizer.theme = m.theme
		m.accessibility.theme = m.theme

	case NavigateToPanelMsg:
		if m.dashboard != nil {
			m.dashboard.activePanel = msg.PanelName
		}

	case TogglePanelCollapseMsg:
		if m.dashboard != nil {
			m.dashboard.toggleCollapsePanel()
		}

	case RunDiagnosticsMsg:
		cmd = m.handleRunDiagnostics(msg)
		cmds = append(cmds, cmd)

	case DiagnosticCompleteMsg:
		m.handleDiagnosticComplete(msg)

	case DiagnosticExportCompleteMsg:
		m.handleDiagnosticExportComplete(msg)

	case PerformanceWarningMsg:
		m.handlePerformanceWarning(msg)

	case ResourceWarningMsg:
		m.handleResourceWarning(msg)
	}

	// Update dashboard
	if m.dashboard != nil {
		model, cmd := m.dashboard.Update(msg)
		if dashboard, ok := model.(*Dashboard); ok {
			m.dashboard = dashboard
		}
		if cmd != nil {
			cmds = append(cmds, cmd)
		}
	}

	// Update performance metrics
	if m.optimizer != nil {
		m.optimizer.CheckMemoryUsage()
	}

	// Update last update time
	m.lastUpdate = time.Now()

	return m, tea.Batch(cmds...)
}

// View renders the monitoring system
func (m *MonitoringSystem) View() string {
	if !m.initialized || m.dashboard == nil {
		return "Monitoring system not initialized"
	}

	// Use optimized renderer
	return m.optimizer.Render("main_dashboard", func() string {
		return m.dashboard.View()
	}, m.width, m.height)
}

// handleKeyMsg handles key messages for monitoring system
func (m *MonitoringSystem) handleKeyMsg(msg tea.KeyMsg) tea.Cmd {
	// Implement global key bindings
	switch msg.String() {
	case "m":
		// Toggle monitoring
		if m.service.IsRunning() {
			return func() tea.Msg {
				return StopMonitoringMsg{StopTime: time.Now()}
			}
		} else {
			return func() tea.Msg {
				return StartMonitoringMsg{StartTime: time.Now()}
			}
		}
	case "r":
		// Refresh monitoring
		return func() tea.Msg {
			return RefreshMonitoringMsg{Timestamp: time.Now()}
		}
	case "t":
		// Toggle theme
		return func() tea.Msg {
			return ToggleThemeMsg{Timestamp: time.Now()}
		}
	case "d":
		// Run diagnostics
		return func() tea.Msg {
			return RunDiagnosticsMsg{Timestamp: time.Now()}
		}
	case "e":
		// Export diagnostics
		return func() tea.Msg {
			return ExportDiagnosticMsg{Timestamp: time.Now()}
		}
	case "h":
		// Toggle high contrast
		m.accessibility.SetHighContrast(!m.accessibility.highContrast)
		return nil
	case "l":
		// Toggle large text
		m.accessibility.SetLargeText(!m.accessibility.largeText)
		return nil
	}

	return nil
}

// handleMonitoringTick handles monitoring tick messages
func (m *MonitoringSystem) handleMonitoringTick(msg MonitoringTickMsg) tea.Cmd {
	// Collect performance metrics
	metrics := CollectPerformanceMetrics()

	// Convert to system status message
	statusMsg := ConvertMetricsToSystemStatus(metrics)

	// Check resource limits
	overLimits := m.resources.IsOverLimit()
	for resource, over := range overLimits {
		if over {
			warningMsg := ResourceWarningMsg{
				Resource:  resource,
				Usage:     getResourceUsage(metrics, resource),
				Threshold: getResourceLimit(m.resources.limits, resource),
				Message:   fmt.Sprintf("%s resource usage exceeds limit", resource),
			}
			return func() tea.Msg { return warningMsg }
		}
	}

	// Schedule next tick
	return tea.Tick(m.updateInterval, func(t time.Time) tea.Msg {
		return MonitoringTickMsg{Time: t}
	})
}

// handleRefresh handles refresh monitoring messages
func (m *MonitoringSystem) handleRefresh(msg RefreshMonitoringMsg) tea.Cmd {
	// Force refresh of all data
	m.optimizer.InvalidateCache()

	// Collect fresh metrics
	metrics := CollectPerformanceMetrics()
	statusMsg := ConvertMetricsToSystemStatus(metrics)

	return func() tea.Msg { return statusMsg }
}

// handleRunDiagnostics handles run diagnostics messages
func (m *MonitoringSystem) handleRunDiagnostics(msg RunDiagnosticsMsg) tea.Cmd {
	// Create diagnostic engine
	engine := NewDiagnosticEngine(m.theme)

	// Run diagnostics
	return engine.RunDiagnostics()
}

// handleDiagnosticComplete handles diagnostic completion
func (m *MonitoringSystem) handleDiagnosticComplete(msg DiagnosticCompleteMsg) {
	// Add alerts to dashboard if available
	if m.dashboard != nil && m.dashboard.alertsPanel != nil {
		for _, result := range msg.Results {
			switch result.Status {
			case "fail":
				m.dashboard.alertsPanel.AddAlert("error", result.Message, result.Category)
			case "warning":
				m.dashboard.alertsPanel.AddAlert("warning", result.Message, result.Category)
			case "info":
				m.dashboard.alertsPanel.AddAlert("info", result.Message, result.Category)
			}
		}
	}
}

// handleDiagnosticExportComplete handles diagnostic export completion
func (m *MonitoringSystem) handleDiagnosticExportComplete(msg DiagnosticExportCompleteMsg) {
	// Add notification about export completion
	if m.dashboard != nil && m.dashboard.alertsPanel != nil {
		if msg.Success {
			m.dashboard.alertsPanel.AddAlert("info",
				fmt.Sprintf("Diagnostic report exported to %s", msg.FilePath),
				"system")
		} else {
			m.dashboard.alertsPanel.AddAlert("error", "Failed to export diagnostic report", "system")
		}
	}
}

// handlePerformanceWarning handles performance warnings
func (m *MonitoringSystem) handlePerformanceWarning(msg types.PerformanceWarningMsg) {
	// Add performance warning to alerts
	if m.dashboard != nil && m.dashboard.alertsPanel != nil {
		warningMsg := fmt.Sprintf("Performance warning: %s", msg.String())
		m.dashboard.alertsPanel.AddAlert("warning", warningMsg, "performance")
	}
}

// handleResourceWarning handles resource warnings
func (m *MonitoringSystem) handleResourceWarning(msg types.ResourceWarningMsg) {
	// Add resource warning to alerts
	if m.dashboard != nil && m.dashboard.alertsPanel != nil {
		warningMsg := fmt.Sprintf("Resource warning: %s at %.1f%%", msg.Resource, msg.Usage)
		m.dashboard.alertsPanel.AddAlert("warning", warningMsg, "system")
	}
}

// GetPerformanceMetrics returns current performance metrics
func (m *MonitoringSystem) GetPerformanceMetrics() PerformanceMetrics {
	if m.optimizer != nil {
		return m.optimizer.GetMetrics()
	}
	return PerformanceMetrics{}
}

// GetSystemStatus returns current system status
func (m *MonitoringSystem) GetSystemStatus() map[string]interface{} {
	if !m.initialized {
		return map[string]interface{}{"status": "not_initialized"}
	}

	status := map[string]interface{}{
		"initialized":       m.initialized,
		"monitoring_active": m.service.IsRunning(),
		"last_update":       m.lastUpdate,
		"dimensions": map[string]int{
			"width":  m.width,
			"height": m.height,
		},
		"theme": m.theme.CurrentTheme,
	}

	// Add performance metrics
	perfMetrics := m.GetPerformanceMetrics()
	status["performance"] = map[string]interface{}{
		"render_time":    perfMetrics.RenderTime,
		"frame_rate":     perfMetrics.FrameRate,
		"memory_usage":   perfMetrics.MemoryUsage,
		"cache_hit_rate": perfMetrics.CacheHitRate,
	}

	// Add resource usage
	if m.resources != nil {
		usage := m.resources.GetUsage()
		status["resources"] = map[string]interface{}{
			"cpu_usage":    usage.CPU,
			"memory_usage": usage.Memory,
			"gpu_usage":    usage.GPU,
			"disk_io":      usage.DiskIO,
			"network_io":   usage.NetIO,
		}
	}

	// Add accessibility settings
	if m.accessibility != nil {
		status["accessibility"] = map[string]interface{}{
			"high_contrast":    m.accessibility.highContrast,
			"large_text":       m.accessibility.largeText,
			"reduced_motion":   m.accessibility.reducedMotion,
			"screen_reader":    m.accessibility.screenReaderMode,
			"color_blind_mode": m.accessibility.colorBlindMode,
		}
	}

	return status
}

// ExportDiagnosticReport exports a comprehensive diagnostic report
func (m *MonitoringSystem) ExportDiagnosticReport() (string, error) {
	if !m.initialized {
		return "", fmt.Errorf("monitoring system not initialized")
	}

	report := map[string]interface{}{
		"system_status":       m.GetSystemStatus(),
		"performance_metrics": m.GetPerformanceMetrics(),
		"diagnostics":         m.collectDiagnosticData(),
		"timestamp":           time.Now(),
	}

	// Convert to JSON (simplified)
	return fmt.Sprintf("Diagnostic Report:\n%+v", report), nil
}

// collectDiagnosticData collects diagnostic data from all components
func (m *MonitoringSystem) collectDiagnosticData() map[string]interface{} {
	data := make(map[string]interface{})

	// Collect system performance metrics
	data["performance"] = CollectPerformanceMetrics()

	// Collect dashboard state
	if m.dashboard != nil {
		data["dashboard"] = map[string]interface{}{
			"active_panel":    m.dashboard.activePanel,
			"panel_collapsed": m.dashboard.panelCollapsed,
		}
	}

	// Collect resource usage
	if m.resources != nil {
		data["resources"] = m.resources.GetUsage()
		data["resource_limits"] = m.resources.limits
	}

	return data
}

// Cleanup cleans up monitoring system resources
func (m *MonitoringSystem) Cleanup() {
	if m.service != nil {
		m.service.Stop()
	}

	if m.optimizer != nil {
		m.optimizer.Cleanup()
	}

	if m.resources != nil {
		m.resources.Cleanup()
	}

	m.initialized = false
}

// GetMonitoringCommands returns available monitoring commands
func (m *MonitoringSystem) GetMonitoringCommands() map[string]string {
	return map[string]string{
		"m":   "Toggle monitoring on/off",
		"r":   "Refresh monitoring data",
		"t":   "Toggle theme",
		"d":   "Run diagnostics",
		"e":   "Export diagnostic report",
		"h":   "Toggle high contrast mode",
		"l":   "Toggle large text mode",
		"tab": "Navigate between panels",
		"c":   "Toggle panel collapse",
		"esc": "Go back",
		"q":   "Quit monitoring",
		"?":   "Show help",
	}
}

// SetConfig updates monitoring system configuration
func (m *MonitoringSystem) SetConfig(config MonitoringConfig) {
	m.applyConfig(config)
	m.updateInterval = config.UpdateInterval

	// Update service interval
	if m.service != nil {
		m.service.Stop()
		m.service = NewSystemMonitorService(config.UpdateInterval)
		if m.initialized {
			m.service.Start()
		}
	}
}

// GetConfig returns current monitoring system configuration
func (m *MonitoringSystem) GetConfig() MonitoringConfig {
	return MonitoringConfig{
		Enabled:         m.initialized,
		UpdateInterval:  m.updateInterval,
		Theme:           m.theme.CurrentTheme,
		Optimization:    "medium", // This would be tracked
		Accessibility:   true,
		HighContrast:    m.accessibility.highContrast,
		LargeText:       m.accessibility.largeText,
		ReducedMotion:   m.accessibility.reducedMotion,
		ScreenReader:    m.accessibility.screenReaderMode,
		ColorBlindMode:  "none", // This would be tracked
		PerformanceMode: "balanced",
	}
}

// Utility functions
func getResourceUsage(metrics PerformanceMetrics, resource string) float64 {
	switch resource {
	case "cpu":
		return metrics.CPU.Usage
	case "memory":
		return metrics.Memory.Used
	case "gpu":
		if len(metrics.GPU) > 0 {
			return metrics.GPU[0].Usage
		}
		return 0
	case "disk":
		return float64(metrics.Storage.Devices[0].Used) / float64(metrics.Storage.Devices[0].Size) * 100
	case "network":
		return 0 // Network usage would be calculated differently
	default:
		return 0
	}
}

func getResourceLimit(limits ResourceLimits, resource string) float64 {
	switch resource {
	case "cpu":
		return limits.MaxCPU
	case "memory":
		return float64(limits.MaxMemory)
	case "gpu":
		return limits.MaxGPU
	case "disk":
		return float64(limits.MaxDiskIO)
	case "network":
		return float64(limits.MaxNetIO)
	default:
		return 0
	}
}
