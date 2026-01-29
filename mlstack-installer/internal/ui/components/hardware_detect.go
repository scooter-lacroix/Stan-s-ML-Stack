// internal/ui/components/hardware_detect.go
package components

import (
	"fmt"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/lipgloss"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/integration"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// HardwareDetectComponent implements hardware detection with AMD branding
type HardwareDetectComponent struct {
	// Dimensions
	Width  int
	Height int
	X, Y   int

	// State
	Ready        bool
	Focused      bool
	Detecting    bool
	Complete     bool
	GPUInfo      types.GPUInfo
	SystemInfo   types.SystemInfo
	Error        error
	Progress     float64
	CurrentStep  string
	StartTime    time.Time

	// Integration
	integration integration.Manager

	// UI elements
	Spinner spinner.Model

	// AMD styling
	TitleStyle       lipgloss.Style
	SubtitleStyle    lipgloss.Style
	ContentStyle     lipgloss.Style
	SuccessStyle     lipgloss.Style
	ErrorStyle       lipgloss.Style
	ProgressStyle    lipgloss.Style
	BorderStyle      lipgloss.Style
	GPUInfoStyle     lipgloss.Style
	SystemInfoStyle  lipgloss.Style
}

// NewHardwareDetectComponent creates a new hardware detection component
func NewHardwareDetectComponent(width, height int, integration integration.Manager) *HardwareDetectComponent {
	// Initialize spinner
	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color(AMDOrange))

	// Calculate dimensions
	borderWidth := 4
	contentWidth := width - borderWidth*2

	// Initialize AMD-themed styles
	titleStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDRed)).
		Bold(true).
		Width(contentWidth).
		Align(lipgloss.Center).
		MarginBottom(1)

	subtitleStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDOrange)).
		Italic(true).
		Width(contentWidth).
		Align(lipgloss.Center).
		MarginBottom(2)

	contentStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDGrayDark)).
		Width(contentWidth).
		MarginBottom(1)

	successStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDSuccess)).
		Bold(true).
		Width(contentWidth).
		MarginBottom(1)

	errorStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDError)).
		Bold(true).
		Width(contentWidth).
		MarginBottom(1)

	progressStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDInfo)).
		Width(contentWidth).
		MarginBottom(1)

	borderStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(AMDRed)).
		Padding(1)

	gpuInfoStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDGrayDark)).
		Background(lipgloss.Color("#EBF8FF")).
		Padding(1).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(AMDGray)).
		Width(contentWidth).
		MarginBottom(1)

	systemInfoStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDGrayDark)).
		Width(contentWidth).
		MarginBottom(1)

	return &HardwareDetectComponent{
		Width:        width,
		Height:       height,
		Ready:        true,
		Detecting:    false,
		Complete:     false,
		StartTime:    time.Now(),
		integration:  integration,
		Spinner:      s,
		TitleStyle:   titleStyle,
		SubtitleStyle: subtitleStyle,
		ContentStyle: contentStyle,
		SuccessStyle: successStyle,
		ErrorStyle:   errorStyle,
		ProgressStyle: progressStyle,
		BorderStyle:  borderStyle,
		GPUInfoStyle: gpuInfoStyle,
		SystemInfoStyle: systemInfoStyle,
	}
}

// Init initializes the hardware detection component
func (c *HardwareDetectComponent) Init() tea.Cmd {
	// Return a command that starts hardware detection
	return func() tea.Msg {
		// Simulate detection start
		return types.HardwareProgressMsg{
			Step:     "Initializing detection...",
			Progress: 0.0,
			Total:    100,
		}
	}
}

// Update handles messages for the hardware detection component
func (c *HardwareDetectComponent) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "enter", " ":
			if c.Complete {
				return c, nil
			}
		case "q", "ctrl+c":
			return c, tea.Quit
		}

	case spinner.TickMsg:
		var cmd tea.Cmd
		c.Spinner, cmd = c.Spinner.Update(msg)
		cmds = append(cmds, cmd)

	case types.HardwareProgressMsg:
		c.Detecting = true
		c.CurrentStep = msg.Step
		c.Progress = msg.Progress

	case types.HardwareDetectedMsg:
		c.Detecting = false
		c.Complete = true
		// Convert installer types to ui types
		c.GPUInfo = types.GPUInfo{
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
		}
		c.SystemInfo = types.SystemInfo{
			OS:            msg.SystemInfo.OS,
			Distribution:  msg.SystemInfo.Distribution,
			KernelVersion: msg.SystemInfo.KernelVersion,
			Architecture:  msg.SystemInfo.Architecture,
			CPU: types.CPUInfo{
				Model:      msg.SystemInfo.CPU.Model,
				Cores:      msg.SystemInfo.CPU.Cores,
				Threads:    msg.SystemInfo.CPU.Threads,
				ClockSpeed: msg.SystemInfo.CPU.ClockSpeed,
				CacheSize:  msg.SystemInfo.CPU.CacheSize,
				Flags:      msg.SystemInfo.CPU.Flags,
			},
			Memory: types.MemoryInfo{
				TotalGB:     msg.SystemInfo.Memory.TotalGB,
				AvailableGB: msg.SystemInfo.Memory.AvailableGB,
				UsedGB:      msg.SystemInfo.Memory.UsedGB,
				SwapTotalGB: msg.SystemInfo.Memory.SwapTotalGB,
				SwapUsedGB:  msg.SystemInfo.Memory.SwapUsedGB,
			},
			Storage: func() []types.StorageInfo {
				result := make([]types.StorageInfo, len(msg.SystemInfo.Storage))
				for i, s := range msg.SystemInfo.Storage {
					result[i] = types.StorageInfo{
						Path:        s.Path,
						Type:        s.Type,
						SizeGB:      s.SizeGB,
						UsedGB:      s.UsedGB,
						AvailableGB: s.AvailableGB,
					}
				}
				return result
			}(),
			Timestamp: msg.SystemInfo.Timestamp,
		}
		c.Error = msg.Error
	}

	var cmd tea.Cmd
	if len(cmds) > 0 {
		cmd = tea.Batch(cmds...)
	}
	return c, cmd
}

// View renders the hardware detection screen
func (c *HardwareDetectComponent) View() string {
	content := ""

	// Title section
	content += c.TitleStyle.Render("Hardware Detection") + "\n\n"

	// Subtitle
	content += c.SubtitleStyle.Render("Detecting AMD GPU and system configuration") + "\n\n"

	if c.Detecting {
		// Show detection progress
		content += c.ProgressStyle.Render(fmt.Sprintf("%s %s", c.Spinner.View(), c.CurrentStep)) + "\n\n"
		content += c.ContentStyle.Render("Scanning for AMD GPUs...") + "\n"
	} else if c.Complete {
		if c.Error != nil {
			// Show error
			content += c.ErrorStyle.Render(fmt.Sprintf("Detection failed: %v", c.Error)) + "\n\n"
		} else {
			// Show success
			content += c.SuccessStyle.Render("Hardware detection complete!") + "\n\n"

			// GPU Info
			if c.GPUInfo.Model != "" {
				gpuText := fmt.Sprintf(
					"GPU: %s %s\nDriver: %s\nCompute Units: %d\nMemory: %.1f GB",
					c.GPUInfo.Vendor,
					c.GPUInfo.Model,
					c.GPUInfo.Driver,
					c.GPUInfo.ComputeUnits,
					c.GPUInfo.MemoryGB,
				)
				content += c.GPUInfoStyle.Render(gpuText) + "\n"
			}

			// System Info
			sysText := fmt.Sprintf(
				"OS: %s %s\nKernel: %s\nArchitecture: %s",
				c.SystemInfo.OS,
				c.SystemInfo.Distribution,
				c.SystemInfo.KernelVersion,
				c.SystemInfo.Architecture,
			)
			content += c.SystemInfoStyle.Render(sysText) + "\n"
		}

		// Help text
		content += c.ContentStyle.Render("\nPress ENTER to continue") + "\n"
	} else {
		// Initial state
		content += c.ContentStyle.Render("Preparing hardware detection...") + "\n"
	}

	// Apply border
	return c.BorderStyle.Width(c.Width).Render(content)
}

// SetBounds sets the component bounds
func (c *HardwareDetectComponent) SetBounds(x, y, width, height int) {
	c.X = x
	c.Y = y
	c.Width = width
	c.Height = height
}

// GetBounds returns the component bounds
func (c *HardwareDetectComponent) GetBounds() (x, y, width, height int) {
	return c.X, c.Y, c.Width, c.Height
}
