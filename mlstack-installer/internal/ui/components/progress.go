// internal/ui/components/progress.go
package components

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// InstallationProgressComponent implements installation progress display with AMD theming
type InstallationProgressComponent struct {
	// Dimensions
	Width  int
	Height int
	X, Y   int

	// State
	Ready         bool
	Installing    bool
	Completed     bool
	Failed        bool
	CurrentStep   int
	TotalSteps    int
	Progress      float64
	CurrentPhase  string
	ErrorMessage  string
	StartTime     time.Time
	EstimatedTime time.Duration

	// UI elements
	ProgressBars    map[string]progress.Model
	Spinner         spinner.Model
	LogMessages     []string
	MaxLogMessages  int
	lastLogProgress float64
	lastLogMessage  string

	// AMD styling
	TitleStyle    lipgloss.Style
	PhaseStyle    lipgloss.Style
	ProgressStyle lipgloss.Style
	LogStyle      lipgloss.Style
	SuccessStyle  lipgloss.Style
	ErrorStyle    lipgloss.Style
	InfoStyle     lipgloss.Style
	ButtonStyle   lipgloss.Style

	// Key bindings
	KeyBindings map[string]key.Binding
}

// InstallationStep represents an installation step
type InstallationStep struct {
	ID          string
	Name        string
	Description string
	Progress    float64
	Completed   bool
	Failed      bool
	ErrorMsg    string
	StartTime   time.Time
	EndTime     time.Time
}

// NewInstallationProgressComponent creates a new installation progress component
func NewInstallationProgressComponent(width, height int) *InstallationProgressComponent {
	// Initialize styles
	titleStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDRed)).
		Bold(true).
		Width(width).
		Align(lipgloss.Center)

	phaseStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDOrange)).
		Bold(true).
		Width(width).
		Align(lipgloss.Center)

	progressStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDWhite)).
		Background(lipgloss.Color(AMDBlack)).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(AMDGray)).
		Padding(1).
		Width(width - 4)

	logStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDGray)).
		Background(lipgloss.Color(AMDBlack)).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(AMDGray)).
		Padding(1).
		Width(width - 4).
		Height(15)

	successStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDSuccess)).
		Bold(true).
		Width(width).
		Align(lipgloss.Center)

	errorStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDError)).
		Bold(true).
		Width(width).
		Align(lipgloss.Center)

	infoStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDGray)).
		Width(width).
		Align(lipgloss.Left)

	buttonStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDWhite)).
		Background(lipgloss.Color(AMDRed)).
		Bold(true).
		Padding(0, 4).
		Margin(1, 1)

	// Initialize progress bars
	progressBars := make(map[string]progress.Model)
	overallProgress := progress.New(progress.WithDefaultScaledGradient())
	overallProgress.Width = width - 20
	progressBars["overall"] = overallProgress

	// Initialize spinner
	spinner := spinner.New(spinner.WithSpinner(spinner.Dot))
	spinner.Style = lipgloss.NewStyle().Foreground(lipgloss.Color(AMDOrange))

	// Create component
	component := &InstallationProgressComponent{
		Width:           width,
		Height:          height,
		X:               0,
		Y:               0,
		Ready:           false,
		Installing:      false,
		Completed:       false,
		Failed:          false,
		CurrentStep:     0,
		TotalSteps:      1,
		Progress:        0.0,
		CurrentPhase:    "Initializing",
		StartTime:       time.Now(),
		EstimatedTime:   0,
		ProgressBars:    progressBars,
		Spinner:         spinner,
		LogMessages:     make([]string, 0),
		MaxLogMessages:  100,
		lastLogProgress: -1,
		lastLogMessage:  "",
		TitleStyle:      titleStyle,
		PhaseStyle:      phaseStyle,
		ProgressStyle:   progressStyle,
		LogStyle:        logStyle,
		SuccessStyle:    successStyle,
		ErrorStyle:      errorStyle,
		InfoStyle:       infoStyle,
		ButtonStyle:     buttonStyle,
	}

	// Initialize key bindings
	component.initializeKeyBindings()

	// Default installation steps
	component.initializeSteps()

	return component
}

// initializeSteps sets up default installation steps
func (c *InstallationProgressComponent) initializeSteps() {
	// Add default progress bars
	steps := []string{
		"System Check",
		"ROCm Installation",
		"PyTorch Setup",
		"Flash Attention",
		"Configuration",
		"Verification",
	}

	for _, step := range steps {
		stepProgress := progress.New(progress.WithDefaultScaledGradient())
		stepProgress.Width = c.Width - 40
		c.ProgressBars[step] = stepProgress
	}

	c.TotalSteps = len(steps)
}

// initializeKeyBindings sets up keyboard shortcuts
func (c *InstallationProgressComponent) initializeKeyBindings() {
	c.KeyBindings = map[string]key.Binding{
		"cancel": key.NewBinding(
			key.WithKeys("c", "ctrl+c"),
			key.WithHelp("C/Ctrl+C", "Cancel Installation"),
		),
		"retry": key.NewBinding(
			key.WithKeys("r"),
			key.WithHelp("R", "Retry Failed Step"),
		),
		"logs": key.NewBinding(
			key.WithKeys("l"),
			key.WithHelp("L", "Toggle Detailed Logs"),
		),
		"continue": key.NewBinding(
			key.WithKeys("enter", " "),
			key.WithHelp("Enter", "Continue"),
		),
	}
}

// Init initializes the installation progress component
func (c *InstallationProgressComponent) Init() tea.Cmd {
	c.Ready = true
	return c.Spinner.Tick
}

// Update handles messages for the installation progress component
func (c *InstallationProgressComponent) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch {
		case key.Matches(msg, c.KeyBindings["cancel"]):
			if c.Installing && !c.Completed {
				c.Failed = true
				c.ErrorMessage = "Installation cancelled by user"
				c.Installing = false
				return c, func() tea.Msg {
					return types.CancelOperationMsg{
						OperationID: "installation",
						Reason:      "User cancelled",
					}
				}
			}

		case key.Matches(msg, c.KeyBindings["retry"]):
			if c.Failed {
				c.Failed = false
				c.ErrorMessage = ""
				c.Installing = true
				c.addLogMessage("Retrying installation...")
			}

		case key.Matches(msg, c.KeyBindings["continue"]):
			if c.Completed || c.Failed {
				return c, func() tea.Msg {
					if c.Completed {
						return types.NavigateToStageMsg{Stage: types.StageComplete}
					}
					return types.NavigateBackMsg{}
				}
			}

		case key.Matches(msg, c.KeyBindings["logs"]):
			// Toggle detailed logs (would show/hide log panel)
		}

	case spinner.TickMsg:
		c.Spinner, cmd = c.Spinner.Update(msg)

	case types.InstallationProgressMsg:
		c.updateProgress(msg)

	case types.InstallationCompleteMsg:
		c.handleInstallationComplete(msg)

	case types.InstallationStartMsg:
		c.handleInstallationStart(msg)

	default:
		// Let progress bars handle messages
		for name, progBar := range c.ProgressBars {
			updatedProgBar, progCmd := progBar.Update(msg)
			if updatedProgress, ok := updatedProgBar.(progress.Model); ok {
				c.ProgressBars[name] = updatedProgress
			}
			if progCmd != nil {
				cmd = progCmd
			}
		}
	}

	return c, cmd
}

// View renders the installation progress component
func (c *InstallationProgressComponent) View() string {
	if !c.Ready {
		return "Initializing progress display..."
	}

	var builder strings.Builder

	// Title
	if c.Completed {
		builder.WriteString(c.SuccessStyle.Render("✓ Installation Complete!") + "\n\n")
	} else if c.Failed {
		builder.WriteString(c.ErrorStyle.Render("✗ Installation Failed") + "\n\n")
	} else {
		builder.WriteString(c.TitleStyle.Render("Installing Stan's ML Stack") + "\n\n")
	}

	// Current phase
	if c.Installing {
		phaseText := fmt.Sprintf("%s %s", c.Spinner.View(), c.CurrentPhase)
		builder.WriteString(c.PhaseStyle.Render(phaseText) + "\n\n")
	}

	// Overall progress
	builder.WriteString(c.renderOverallProgress() + "\n\n")

	// Individual component progress
	builder.WriteString(c.renderComponentProgress() + "\n\n")

	// Error message if failed
	if c.Failed && c.ErrorMessage != "" {
		builder.WriteString(c.ErrorStyle.Render("Error: "+c.ErrorMessage) + "\n\n")
	}

	// Log messages
	if len(c.LogMessages) > 0 {
		builder.WriteString(c.renderLogs() + "\n\n")
	}

	// Help text
	builder.WriteString(c.renderHelp())

	return builder.String()
}

// SetBounds updates the component dimensions
func (c *InstallationProgressComponent) SetBounds(x, y, width, height int) {
	c.X = x
	c.Y = y
	c.Width = width
	c.Height = height

	// Update styles
	c.TitleStyle = c.TitleStyle.Width(width)
	c.PhaseStyle = c.PhaseStyle.Width(width)
	c.ProgressStyle = c.ProgressStyle.Width(width - 4)
	c.SuccessStyle = c.SuccessStyle.Width(width)
	c.ErrorStyle = c.ErrorStyle.Width(width)
	c.InfoStyle = c.InfoStyle.Width(width)

	// Update progress bar widths
	for name, progBar := range c.ProgressBars {
		if name == "overall" {
			progBar.Width = width - 20
		} else {
			progBar.Width = width - 40
		}
		c.ProgressBars[name] = progBar
	}
}

// GetBounds returns the current component bounds
func (c *InstallationProgressComponent) GetBounds() (x, y, width, height int) {
	return c.X, c.Y, c.Width, c.Height
}

// updateProgress updates the installation progress
func (c *InstallationProgressComponent) updateProgress(msg types.InstallationProgressMsg) {
	c.Progress = msg.Progress
	c.CurrentPhase = msg.Message

	// Update overall progress bar
	if progBar, exists := c.ProgressBars["overall"]; exists {
		progBar.SetPercent(msg.Progress)
		c.ProgressBars["overall"] = progBar
	}

	// Update component-specific progress
	if progBar, exists := c.ProgressBars[msg.ComponentID]; exists {
		progBar.SetPercent(msg.Progress)
		c.ProgressBars[msg.ComponentID] = progBar
	}

	progressDelta := msg.Progress - c.lastLogProgress
	shouldLog := msg.Message != c.lastLogMessage || progressDelta >= 0.05 || msg.Progress >= 0.999
	if shouldLog {
		c.addLogMessage(fmt.Sprintf("Progress: %s - %.1f%%", msg.Message, msg.Progress*100))
		c.lastLogProgress = msg.Progress
		c.lastLogMessage = msg.Message
	}
}

// handleInstallationComplete handles installation completion
func (c *InstallationProgressComponent) handleInstallationComplete(msg types.InstallationCompleteMsg) {
	c.Installing = false
	if msg.Success {
		c.Completed = true
		c.Progress = 1.0
		c.CurrentPhase = "Installation completed successfully!"
		c.addLogMessage("✓ Installation completed successfully!")

		// Set all progress bars to 100%
		for name, progBar := range c.ProgressBars {
			progBar.SetPercent(1.0)
			c.ProgressBars[name] = progBar
		}
	} else {
		c.Failed = true
		c.ErrorMessage = msg.Error.Error()
		c.CurrentPhase = "Installation failed"
		c.addLogMessage(fmt.Sprintf("✗ Installation failed: %v", msg.Error))
	}
}

// handleInstallationStart handles installation start
func (c *InstallationProgressComponent) handleInstallationStart(msg types.InstallationStartMsg) {
	c.Installing = true
	c.Completed = false
	c.Failed = false
	c.Progress = 0.0
	c.CurrentPhase = "Starting installation..."
	c.StartTime = time.Now()
	c.LogMessages = make([]string, 0)
	c.lastLogProgress = -1
	c.lastLogMessage = ""

	c.addLogMessage("Starting installation...")
	c.addLogMessage(fmt.Sprintf("Components to install: %d", len(msg.Components)))
}

// addLogMessage adds a message to the log
func (c *InstallationProgressComponent) addLogMessage(message string) {
	timestamp := time.Now().Format("15:04:05")
	logEntry := fmt.Sprintf("[%s] %s", timestamp, message)

	c.LogMessages = append(c.LogMessages, logEntry)

	// Limit log messages
	if len(c.LogMessages) > c.MaxLogMessages {
		c.LogMessages = c.LogMessages[1:]
	}
}

// renderOverallProgress renders the overall progress bar
func (c *InstallationProgressComponent) renderOverallProgress() string {
	progBar, exists := c.ProgressBars["overall"]
	if !exists {
		return ""
	}

	elapsed := time.Since(c.StartTime)
	progressPercent := int(c.Progress * 100)

	progressText := fmt.Sprintf("Overall Progress: %d%%", progressPercent)
	if elapsed > 0 && c.Progress > 0 {
		estimated := time.Duration(float64(elapsed) / c.Progress)
		remaining := estimated - elapsed
		if remaining > 0 {
			progressText += fmt.Sprintf(" (ETA: %v)", remaining.Round(time.Second))
		}
	}

	return c.InfoStyle.Render(progressText) + "\n" + progBar.View()
}

// renderComponentProgress renders individual component progress bars
func (c *InstallationProgressComponent) renderComponentProgress() string {
	var builder strings.Builder

	componentOrder := []string{
		"System Check",
		"ROCm Installation",
		"PyTorch Setup",
		"Flash Attention",
		"Configuration",
		"Verification",
	}

	for _, componentName := range componentOrder {
		if progBar, exists := c.ProgressBars[componentName]; exists {
			// Check if component is in a good state (progress > 0 or completed)
			if progBar.Percent() > 0 || c.Completed {
				builder.WriteString(c.InfoStyle.Render(componentName) + "\n")
				builder.WriteString(progBar.View() + "\n\n")
			}
		}
	}

	return builder.String()
}

// renderLogs renders the log messages
func (c *InstallationProgressComponent) renderLogs() string {
	if len(c.LogMessages) == 0 {
		return ""
	}

	var builder strings.Builder
	builder.WriteString(c.InfoStyle.Render("Installation Log:") + "\n")

	// Show last N log messages
	maxLines := 10
	start := 0
	if len(c.LogMessages) > maxLines {
		start = len(c.LogMessages) - maxLines
	}

	for i := start; i < len(c.LogMessages); i++ {
		builder.WriteString(c.LogMessages[i] + "\n")
	}

	return c.LogStyle.Render(builder.String())
}

// renderHelp renders the help text
func (c *InstallationProgressComponent) renderHelp() string {
	var helpItems []string

	if c.Installing {
		helpItems = append(helpItems, fmt.Sprintf("%s: Cancel", c.KeyBindings["cancel"].Help().Key))
	}

	if c.Failed {
		helpItems = append(helpItems, fmt.Sprintf("%s: Retry", c.KeyBindings["retry"].Help().Key))
	}

	if c.Completed || c.Failed {
		helpItems = append(helpItems, fmt.Sprintf("%s: Continue", c.KeyBindings["continue"].Help().Key))
	}

	helpItems = append(helpItems, fmt.Sprintf("%s: Toggle Logs", c.KeyBindings["logs"].Help().Key))

	return c.InfoStyle.Render(strings.Join(helpItems, " • "))
}

// IsCompleted returns whether installation is completed
func (c *InstallationProgressComponent) IsCompleted() bool {
	return c.Completed
}

// IsFailed returns whether installation failed
func (c *InstallationProgressComponent) IsFailed() bool {
	return c.Failed
}

// IsInstalling returns whether installation is in progress
func (c *InstallationProgressComponent) IsInstalling() bool {
	return c.Installing
}

// GetProgress returns current progress percentage
func (c *InstallationProgressComponent) GetProgress() float64 {
	return c.Progress
}

// GetElapsedTime returns elapsed installation time
func (c *InstallationProgressComponent) GetElapsedTime() time.Duration {
	return time.Since(c.StartTime)
}

// GetState returns the component state
func (c *InstallationProgressComponent) GetState() map[string]interface{} {
	return map[string]interface{}{
		"ready":         c.Ready,
		"installing":    c.Installing,
		"completed":     c.Completed,
		"failed":        c.Failed,
		"progress":      c.Progress,
		"current_phase": c.CurrentPhase,
		"error_message": c.ErrorMessage,
		"elapsed_time":  c.GetElapsedTime(),
		"log_count":     len(c.LogMessages),
		"width":         c.Width,
		"height":        c.Height,
		"start_time":    c.StartTime,
	}
}
