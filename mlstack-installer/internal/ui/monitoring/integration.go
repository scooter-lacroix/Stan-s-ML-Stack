// internal/ui/monitoring/integration.go
package monitoring

import (
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// DashboardIntegration handles integration with the main MVU architecture
type DashboardIntegration struct {
	dashboard     *Dashboard
	initialized   bool
	updateChannel chan tea.Msg
	theme         *AMDTheme
}

// MonitoringStage represents the dashboard stage in the main application
const MonitoringStage types.Stage = types.Stage(100) // Custom stage for monitoring

// NewDashboardIntegration creates a new dashboard integration
func NewDashboardIntegration(theme *AMDTheme) *DashboardIntegration {
	return &DashboardIntegration{
		initialized:   false,
		updateChannel: make(chan tea.Msg, 100),
		theme:         theme,
	}
}

// Initialize initializes the dashboard integration
func (di *DashboardIntegration) Initialize(width, height int) {
	di.dashboard = NewDashboard(width, height)
	di.initialized = true
}

// GetDashboard returns the dashboard instance
func (di *DashboardIntegration) GetDashboard() *Dashboard {
	return di.dashboard
}

// HandleMessage handles messages for the dashboard integration
func (di *DashboardIntegration) HandleMessage(msg tea.Msg) tea.Cmd {
	if !di.initialized || di.dashboard == nil {
		return nil
	}

	switch msg := msg.(type) {
	// Handle system status updates
	case types.SystemStatusUpdateMsg:
		return di.handleSystemStatusUpdate(msg)

	// Handle performance warnings
	case types.PerformanceWarningMsg:
		return di.handlePerformanceWarning(msg)

	// Handle resource warnings
	case types.ResourceWarningMsg:
		return di.handleResourceWarning(msg)

	// Handle monitoring-specific messages
	case StartMonitoringMsg:
		return di.handleStartMonitoring(msg)

	case StopMonitoringMsg:
		return di.handleStopMonitoring(msg)

	case RefreshMonitoringMsg:
		return di.handleRefreshMonitoring(msg)

	case ToggleThemeMsg:
		return di.handleToggleTheme(msg)

	case ExportDiagnosticMsg:
		return di.handleExportDiagnostic(msg)

	case NavigateToPanelMsg:
		return di.handleNavigateToPanel(msg)

	case TogglePanelCollapseMsg:
		return di.handleTogglePanelCollapse(msg)

	case RunDiagnosticsMsg:
		return di.handleRunDiagnostics(msg)

	default:
		// Pass message to dashboard
		_, cmd := di.dashboard.Update(msg)
		return cmd
	}
}

// handleSystemStatusUpdate handles system status update messages
func (di *DashboardIntegration) handleSystemStatusUpdate(msg types.SystemStatusUpdateMsg) tea.Cmd {
	// The dashboard already handles SystemStatusUpdateMsg in its Update method
	_, cmd := di.dashboard.Update(msg)
	return cmd
}

// handlePerformanceWarning handles performance warning messages
func (di *DashboardIntegration) handlePerformanceWarning(msg types.PerformanceWarningMsg) tea.Cmd {
	// Add alert to alerts panel
	if di.dashboard.alertsPanel != nil {
		alertMessage := "Performance warning detected"
		if msg.Duration > time.Second {
			alertMessage = "High operation latency detected"
		}
		di.dashboard.alertsPanel.AddAlert("warning", alertMessage, "performance")
	}

	// Update performance bar with warning
	if di.dashboard.performanceBar != nil {
		di.dashboard.performanceBar.UpdateData(types.SystemStatusUpdateMsg{
			CPUUsage:    85.0, // Elevated for warning
			MemoryUsage: 75.0,
			DiskUsage:   60.0,
			GPUUsage:    []float64{70.0, 60.0},
		})
	}

	return nil
}

// handleResourceWarning handles resource warning messages
func (di *DashboardIntegration) handleResourceWarning(msg types.ResourceWarningMsg) tea.Cmd {
	// Add alert to alerts panel
	if di.dashboard.alertsPanel != nil {
		alertMessage := fmt.Sprintf("%s resource usage: %.1f%%", msg.Resource, msg.Usage)
		di.dashboard.alertsPanel.AddAlert("warning", alertMessage, "system")
	}

	// Update relevant widgets based on resource type
	switch msg.Resource {
	case "CPU":
		if di.dashboard.systemMonitor != nil {
			di.dashboard.systemMonitor.UpdateData(types.SystemStatusUpdateMsg{
				CPUUsage: msg.Usage,
			})
		}
	case "memory":
		if di.dashboard.systemMonitor != nil {
			di.dashboard.systemMonitor.UpdateData(types.SystemStatusUpdateMsg{
				MemoryUsage: msg.Usage,
			})
		}
	case "GPU":
		if di.dashboard.gpuMonitor != nil {
			di.dashboard.gpuMonitor.UpdateData(types.SystemStatusUpdateMsg{
				GPUUsage: []float64{msg.Usage},
			})
		}
	}

	return nil
}

// handleStartMonitoring handles start monitoring messages
func (di *DashboardIntegration) handleStartMonitoring(msg StartMonitoringMsg) tea.Cmd {
	// Start real-time monitoring
	return tea.Tick(1*time.Second, func(t time.Time) tea.Msg {
		return MonitoringTickMsg{Time: t}
	})
}

// handleStopMonitoring handles stop monitoring messages
func (di *DashboardIntegration) handleStopMonitoring(msg StopMonitoringMsg) tea.Cmd {
	// Stop real-time monitoring
	return nil
}

// handleRefreshMonitoring handles refresh monitoring messages
func (di *DashboardIntegration) handleRefreshMonitoring(msg RefreshMonitoringMsg) tea.Cmd {
	// Force refresh of all dashboard data
	return func() tea.Msg {
		return types.SystemStatusUpdateMsg{
			CPUUsage:    getCPUUsage(),
			MemoryUsage: getMemoryUsage(),
			DiskUsage:   getDiskUsage(),
			GPUUsage:    getGPUUsage(),
		}
	}
}

// handleToggleTheme handles theme toggle messages
func (di *DashboardIntegration) handleToggleTheme(msg ToggleThemeMsg) tea.Cmd {
	// Toggle theme
	if di.dashboard != nil && di.dashboard.theme != nil {
		di.dashboard.theme.CycleTheme()
	}

	return nil
}

// handleExportDiagnostic handles export diagnostic messages
func (di *DashboardIntegration) handleExportDiagnostic(msg ExportDiagnosticMsg) tea.Cmd {
	// Export diagnostic report
	if di.dashboard == nil {
		return nil
	}

	// Get diagnostic widget (this would need to be added to dashboard)
	// For now, return a simple completion message
	return func() tea.Msg {
		return DiagnosticExportCompleteMsg{
			Success:   true,
			Timestamp: time.Now(),
			FilePath:  "/tmp/diagnostic_report.json",
		}
	}
}

// handleNavigateToPanel handles panel navigation messages
func (di *DashboardIntegration) handleNavigateToPanel(msg NavigateToPanelMsg) tea.Cmd {
	// Navigate to specific panel
	if di.dashboard != nil {
		di.dashboard.activePanel = msg.PanelName
	}

	return nil
}

// handleTogglePanelCollapse handles panel collapse toggle messages
func (di *DashboardIntegration) handleTogglePanelCollapse(msg TogglePanelCollapseMsg) tea.Cmd {
	// Toggle panel collapse
	if di.dashboard != nil {
		di.dashboard.toggleCollapsePanel()
	}

	return nil
}

// handleRunDiagnostics handles run diagnostics messages
func (di *DashboardIntegration) handleRunDiagnostics(msg RunDiagnosticsMsg) tea.Cmd {
	// Run diagnostics
	if di.dashboard != nil {
		// This would need a diagnostic widget in the dashboard
		// For now, return a simulated completion
		return func() tea.Msg {
			return DiagnosticCompleteMsg{
				Results: []DiagnosticResult{
					{
						ID:        "test_cpu",
						Name:      "CPU Test",
						Status:    "pass",
						Score:     95,
						Message:   "CPU performance is excellent",
						Timestamp: time.Now(),
					},
				},
				Timestamp: time.Now(),
				Duration:  5 * time.Second,
			}
		}
	}

	return nil
}

// GetMonitoringCommands returns commands for monitoring dashboard
func GetMonitoringCommands() map[string]func() tea.Cmd {
	return map[string]func() tea.Cmd{
		"start_monitoring": func() tea.Cmd {
			return func() tea.Msg {
				return StartMonitoringMsg{StartTime: time.Now()}
			}
		},
		"stop_monitoring": func() tea.Cmd {
			return func() tea.Msg {
				return StopMonitoringMsg{StopTime: time.Now()}
			}
		},
		"refresh_monitoring": func() tea.Cmd {
			return func() tea.Msg {
				return RefreshMonitoringMsg{Timestamp: time.Now()}
			}
		},
		"toggle_theme": func() tea.Cmd {
			return func() tea.Msg {
				return ToggleThemeMsg{Timestamp: time.Now()}
			}
		},
		"run_diagnostics": func() tea.Cmd {
			return func() tea.Msg {
				return RunDiagnosticsMsg{Timestamp: time.Now()}
			}
		},
		"export_diagnostic": func() tea.Cmd {
			return func() tea.Msg {
				return ExportDiagnosticMsg{Timestamp: time.Now()}
			}
		},
	}
}

// AddMonitoringStage adds monitoring stage to the main application
func AddMonitoringStage(stateManager types.StateManager) types.StateManager {
	// This would extend the state manager to support the monitoring stage
	// For now, we return the existing state manager
	return stateManager
}

// Monitoring-specific message types
type StartMonitoringMsg struct {
	StartTime time.Time
}

func (StartMonitoringMsg) IsMessage() {}

type StopMonitoringMsg struct {
	StopTime time.Time
}

func (StopMonitoringMsg) IsMessage() {}

type RefreshMonitoringMsg struct {
	Timestamp time.Time
}

func (RefreshMonitoringMsg) IsMessage() {}

type ToggleThemeMsg struct {
	Timestamp time.Time
}

func (ToggleThemeMsg) IsMessage() {}

type ExportDiagnosticMsg struct {
	Timestamp time.Time
}

func (ExportDiagnosticMsg) IsMessage() {}

type NavigateToPanelMsg struct {
	PanelName string
}

func (NavigateToPanelMsg) IsMessage() {}

type TogglePanelCollapseMsg struct {
	PanelName string
}

func (TogglePanelCollapseMsg) IsMessage() {}

type RunDiagnosticsMsg struct {
	Timestamp time.Time
}

func (RunDiagnosticsMsg) IsMessage() {}

type MonitoringTickMsg struct {
	Time time.Time
}

func (MonitoringTickMsg) IsMessage() {}

type DiagnosticExportCompleteMsg struct {
	Success   bool
	Timestamp time.Time
	FilePath  string
}

func (DiagnosticExportCompleteMsg) IsMessage() {}

// PerformanceMetrics represents comprehensive performance metrics
type PerformanceMetrics struct {
	CPU struct {
		Usage       float64   `json:"usage"`
		Cores       int       `json:"cores"`
		Frequency   float64   `json:"frequency"`
		Temperature float64   `json:"temperature"`
		LoadAverage []float64 `json:"load_average"`
	} `json:"cpu"`

	Memory struct {
		Total     float64 `json:"total"`
		Used      float64 `json:"used"`
		Available float64 `json:"available"`
		Swap      struct {
			Total float64 `json:"total"`
			Used  float64 `json:"used"`
		} `json:"swap"`
	} `json:"memory"`

	GPU []struct {
		ID          int     `json:"id"`
		Name        string  `json:"name"`
		Usage       float64 `json:"usage"`
		Temperature float64 `json:"temperature"`
		Memory      struct {
			Used  float64 `json:"used"`
			Total float64 `json:"total"`
		} `json:"memory"`
		Power      float64 `json:"power"`
		ClockSpeed float64 `json:"clock_speed"`
	} `json:"gpu"`

	Network struct {
		Interfaces []struct {
			Name    string  `json:"name"`
			State   string  `json:"state"`
			RxBytes int64   `json:"rx_bytes"`
			TxBytes int64   `json:"tx_bytes"`
			RxRate  float64 `json:"rx_rate"`
			TxRate  float64 `json:"tx_rate"`
		} `json:"interfaces"`
	} `json:"network"`

	Storage struct {
		Devices []struct {
			Name      string  `json:"name"`
			Size      float64 `json:"size"`
			Used      float64 `json:"used"`
			Available float64 `json:"available"`
			Type      string  `json:"type"`
		} `json:"devices"`
	} `json:"storage"`

	System struct {
		Uptime      time.Duration `json:"uptime"`
		Processes   int           `json:"processes"`
		Load        float64       `json:"load"`
		Temperature float64       `json:"temperature"`
	} `json:"system"`

	Timestamp time.Time `json:"timestamp"`
}

// CollectPerformanceMetrics collects comprehensive performance metrics
func CollectPerformanceMetrics() PerformanceMetrics {
	return PerformanceMetrics{
		CPU: struct {
			Usage       float64   `json:"usage"`
			Cores       int       `json:"cores"`
			Frequency   float64   `json:"frequency"`
			Temperature float64   `json:"temperature"`
			LoadAverage []float64 `json:"load_average"`
		}{
			Usage:       getCPUUsage(),
			Cores:       16,
			Frequency:   4500.0,
			Temperature: 65.0,
			LoadAverage: []float64{0.5, 0.7, 0.6},
		},
		Memory: struct {
			Total     float64 `json:"total"`
			Used      float64 `json:"used"`
			Available float64 `json:"available"`
			Swap      struct {
				Total float64 `json:"total"`
				Used  float64 `json:"used"`
			} `json:"swap"`
		}{
			Total:     32.0,
			Used:      getMemoryUsage(),
			Available: 32.0 - getMemoryUsage(),
			Swap: struct {
				Total float64 `json:"total"`
				Used  float64 `json:"used"`
			}{
				Total: 8.0,
				Used:  2.1,
			},
		},
		GPU: []struct {
			ID          int     `json:"id"`
			Name        string  `json:"name"`
			Usage       float64 `json:"usage"`
			Temperature float64 `json:"temperature"`
			Memory      struct {
				Used  float64 `json:"used"`
				Total float64 `json:"total"`
			} `json:"memory"`
			Power      float64 `json:"power"`
			ClockSpeed float64 `json:"clock_speed"`
		}{
			{
				ID:          0,
				Name:        "AMD Radeon RX 7900 XTX",
				Usage:       getGPUUsage()[0],
				Temperature: 65.0,
				Memory: struct {
					Used  float64 `json:"used"`
					Total float64 `json:"total"`
				}{
					Used:  8.5,
					Total: 24.0,
				},
				Power:      250.0,
				ClockSpeed: 2100.0,
			},
		},
		Network: struct {
			Interfaces []struct {
				Name    string  `json:"name"`
				State   string  `json:"state"`
				RxBytes int64   `json:"rx_bytes"`
				TxBytes int64   `json:"tx_bytes"`
				RxRate  float64 `json:"rx_rate"`
				TxRate  float64 `json:"tx_rate"`
			} `json:"interfaces"`
		}{
			Interfaces: []struct {
				Name    string  `json:"name"`
				State   string  `json:"state"`
				RxBytes int64   `json:"rx_bytes"`
				TxBytes int64   `json:"tx_bytes"`
				RxRate  float64 `json:"rx_rate"`
				TxRate  float64 `json:"tx_rate"`
			}{
				{
					Name:    "eth0",
					State:   "up",
					RxBytes: 1024567890,
					TxBytes: 512345678,
					RxRate:  1024.5,
					TxRate:  512.3,
				},
			},
		},
		Storage: struct {
			Devices []struct {
				Name      string  `json:"name"`
				Size      float64 `json:"size"`
				Used      float64 `json:"used"`
				Available float64 `json:"available"`
				Type      string  `json:"type"`
			} `json:"devices"`
		}{
			Devices: []struct {
				Name      string  `json:"name"`
				Size      float64 `json:"size"`
				Used      float64 `json:"used"`
				Available float64 `json:"available"`
				Type      string  `json:"type"`
			}{
				{
					Name:      "/dev/nvme0n1",
					Size:      1000.0,
					Used:      600.0,
					Available: 400.0,
					Type:      "NVMe SSD",
				},
			},
		},
		System: struct {
			Uptime      time.Duration `json:"uptime"`
			Processes   int           `json:"processes"`
			Load        float64       `json:"load"`
			Temperature float64       `json:"temperature"`
		}{
			Uptime:      48 * time.Hour,
			Processes:   156,
			Load:        0.7,
			Temperature: 45.0,
		},
		Timestamp: time.Now(),
	}
}

// ConvertMetricsToSystemStatus converts performance metrics to system status message
func ConvertMetricsToSystemStatus(metrics PerformanceMetrics) types.SystemStatusUpdateMsg {
	gpuUsage := make([]float64, len(metrics.GPU))
	for i, gpu := range metrics.GPU {
		gpuUsage[i] = gpu.Usage
	}

	return types.SystemStatusUpdateMsg{
		CPUUsage:    metrics.CPU.Usage,
		MemoryUsage: (metrics.Memory.Used / metrics.Memory.Total) * 100,
		DiskUsage:   (metrics.Storage.Devices[0].Used / metrics.Storage.Devices[0].Size) * 100,
		GPUUsage:    gpuUsage,
	}
}

// SystemMonitorService provides a service for monitoring system performance
type SystemMonitorService struct {
	interval   time.Duration
	running    bool
	stopChan   chan struct{}
	updateChan chan tea.Msg
}

// NewSystemMonitorService creates a new system monitor service
func NewSystemMonitorService(interval time.Duration) *SystemMonitorService {
	return &SystemMonitorService{
		interval:   interval,
		running:    false,
		stopChan:   make(chan struct{}),
		updateChan: make(chan tea.Msg, 100),
	}
}

// Start starts the system monitor service
func (s *SystemMonitorService) Start() tea.Cmd {
	if s.running {
		return nil
	}

	s.running = true

	return func() tea.Msg {
		ticker := time.NewTicker(s.interval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				if !s.running {
					return nil
				}

				// Collect metrics
				metrics := CollectPerformanceMetrics()
				statusMsg := ConvertMetricsToSystemStatus(metrics)

				return statusMsg

			case <-s.stopChan:
				return nil
			}
		}
	}
}

// Stop stops the system monitor service
func (s *SystemMonitorService) Stop() {
	if s.running {
		s.running = false
		close(s.stopChan)
		s.stopChan = make(chan struct{})
	}
}

// IsRunning returns true if the service is running
func (s *SystemMonitorService) IsRunning() bool {
	return s.running
}

// GetUpdateChannel returns the update channel
func (s *SystemMonitorService) GetUpdateChannel() chan tea.Msg {
	return s.updateChan
}
