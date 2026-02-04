// internal/ui/components/recovery.go
package components

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/scooter-lacroix/mlstack-installer/internal/installer"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// RecoveryComponent implements error recovery and troubleshooting with AMD theming
type RecoveryComponent struct {
	// Dimensions
	Width  int
	Height int
	X, Y   int

	// State
	Ready            bool
	Loading          bool
	Focused          bool
	ErrorType        string
	ErrorMessage     string
	RecoveryOptions  []installer.RecoveryOption
	SelectedIndex    int
	Recovering       bool
	RecoveryProgress float64

	// UI elements
	List        list.Model
	Spinner     spinner.Model
	Details     map[string]string
	ShowDetails bool

	// AMD styling
	TitleStyle    lipgloss.Style
	SubtitleStyle lipgloss.Style
	ErrorStyle    lipgloss.Style
	InfoStyle     lipgloss.Style
	SuccessStyle  lipgloss.Style
	WarningStyle  lipgloss.Style
	ButtonStyle   lipgloss.Style
	DetailsStyle  lipgloss.Style

	// Key bindings
	KeyBindings map[string]key.Binding

	// Performance tracking
	startTime time.Time
}

// RecoveryItem implements list.Item for recovery options
type RecoveryItem struct {
	Option installer.RecoveryOption
}

func (i RecoveryItem) Title() string {
	riskLevel := ""
	switch i.Option.RiskLevel {
	case "low":
		riskLevel = "🟢"
	case "medium":
		riskLevel = "🟡"
	case "high":
		riskLevel = "🔴"
	default:
		riskLevel = "⚪"
	}

	return fmt.Sprintf("%s %s", riskLevel, i.Option.Name)
}

func (i RecoveryItem) Description() string {
	return i.Option.Description
}

func (i RecoveryItem) FilterValue() string {
	return fmt.Sprintf("%s %s %s", i.Option.Name, i.Option.Description, i.Option.Category)
}

// NewRecoveryComponent creates a new recovery component
func NewRecoveryComponent(width, height int) *RecoveryComponent {
	// Initialize styles
	titleStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDRed)).
		Bold(true).
		Width(width).
		Align(lipgloss.Center)

	subtitleStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDOrange)).
		Italic(true).
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

	successStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDSuccess)).
		Bold(true).
		Width(width).
		Align(lipgloss.Center)

	warningStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDWarning)).
		Bold(true).
		Width(width).
		Align(lipgloss.Center)

	buttonStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDWhite)).
		Background(lipgloss.Color(AMDRed)).
		Bold(true).
		Padding(0, 4).
		Margin(1, 1)

	detailsStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDWhite)).
		Background(lipgloss.Color(AMDBlack)).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(AMDGray)).
		Padding(1).
		Width(width - 4)

	// Initialize spinner
	spinner := spinner.New(spinner.WithSpinner(spinner.Dot))
	spinner.Style = lipgloss.NewStyle().Foreground(lipgloss.Color(AMDOrange))

	// Create delegate for list
	delegate := list.NewDefaultDelegate()
	delegate.Styles.SelectedTitle = lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDWhite)).
		Background(lipgloss.Color(AMDRed)).
		Bold(true)
	delegate.Styles.SelectedDesc = delegate.Styles.SelectedTitle

	// Initialize list
	listModel := list.New([]list.Item{}, delegate, width-10, height-20)
	listModel.Title = "Recovery Options"
	listModel.SetShowStatusBar(true)
	listModel.SetShowHelp(false)

	// Create component
	component := &RecoveryComponent{
		Width:            width,
		Height:           height,
		X:                0,
		Y:                0,
		Ready:            false,
		Loading:          true,
		Focused:          true,
		SelectedIndex:    0,
		Recovering:       false,
		RecoveryProgress: 0.0,
		List:             listModel,
		Spinner:          spinner,
		Details:          make(map[string]string),
		ShowDetails:      false,
		TitleStyle:       titleStyle,
		SubtitleStyle:    subtitleStyle,
		ErrorStyle:       errorStyle,
		InfoStyle:        infoStyle,
		SuccessStyle:     successStyle,
		WarningStyle:     warningStyle,
		ButtonStyle:      buttonStyle,
		DetailsStyle:     detailsStyle,
		startTime:        time.Now(),
	}

	// Initialize key bindings
	component.initializeKeyBindings()

	// Load default recovery options
	component.loadDefaultRecoveryOptions()

	component.Ready = true
	component.Loading = false

	return component
}

// initializeKeyBindings sets up keyboard shortcuts
func (c *RecoveryComponent) initializeKeyBindings() {
	c.KeyBindings = map[string]key.Binding{
		"up": key.NewBinding(
			key.WithKeys("up", "k"),
			key.WithHelp("↑/K", "Move Up"),
		),
		"down": key.NewBinding(
			key.WithKeys("down", "j"),
			key.WithHelp("↓/J", "Move Down"),
		),
		"select": key.NewBinding(
			key.WithKeys("enter", " "),
			key.WithHelp("Enter", "Select Recovery Option"),
		),
		"details": key.NewBinding(
			key.WithKeys("d", "i"),
			key.WithHelp("D/I", "Show Details"),
		),
		"retry": key.NewBinding(
			key.WithKeys("r"),
			key.WithHelp("R", "Retry Last Operation"),
		),
		"diagnostic": key.NewBinding(
			key.WithKeys("ctrl+d"),
			key.WithHelp("Ctrl+D", "Run Diagnostic"),
		),
		"back": key.NewBinding(
			key.WithKeys("esc", "q"),
			key.WithHelp("Esc/Q", "Back to Installer"),
		),
	}
}

// loadDefaultRecoveryOptions loads default recovery options
func (c *RecoveryComponent) loadDefaultRecoveryOptions() {
	c.RecoveryOptions = []installer.RecoveryOption{
		{
			ID:          "check_dependencies",
			Name:        "Check System Dependencies",
			Description: "Verify all required dependencies are installed",
			Category:    "system",
			RiskLevel:   "low",
			Command:     "dependency_check",
		},
		{
			ID:          "repair_permissions",
			Name:        "Repair File Permissions",
			Description: "Fix permission issues with installation directories",
			Category:    "permissions",
			RiskLevel:   "low",
			Command:     "permission_repair",
		},
		{
			ID:          "cleanup_partial",
			Name:        "Clean Up Partial Installation",
			Description: "Remove partially installed components and clean up",
			Category:    "cleanup",
			RiskLevel:   "medium",
			Command:     "partial_cleanup",
		},
		{
			ID:          "reinstall_rocm",
			Name:        "Reinstall ROCm",
			Description: "Remove and reinstall ROCm platform",
			Category:    "reinstall",
			RiskLevel:   "high",
			Command:     "rocm_reinstall",
		},
		{
			ID:          "reset_environment",
			Name:        "Reset Environment",
			Description: "Reset environment variables and configuration",
			Category:    "environment",
			RiskLevel:   "medium",
			Command:     "environment_reset",
		},
		{
			ID:          "system_diagnostic",
			Name:        "Run Full System Diagnostic",
			Description: "Comprehensive system and hardware diagnostic",
			Category:    "diagnostic",
			RiskLevel:   "low",
			Command:     "full_diagnostic",
		},
		{
			ID:          "backup_restore",
			Name:        "Restore from Backup",
			Description: "Restore ML stack from previous backup",
			Category:    "backup",
			RiskLevel:   "medium",
			Command:     "backup_restore",
		},
	}

	// Convert to list items
	items := make([]list.Item, len(c.RecoveryOptions))
	for i, option := range c.RecoveryOptions {
		items[i] = RecoveryItem{Option: option}
	}
	c.List.SetItems(items)

	// Load details for each option
	c.loadOptionDetails()
}

// loadOptionDetails loads detailed information for each recovery option
func (c *RecoveryComponent) loadOptionDetails() {
	c.Details = map[string]string{
		"check_dependencies": `This option will:
• Check for required system packages
• Verify GPU drivers are properly installed
• Validate environment variables
• Check disk space and memory availability
• Verify network connectivity for downloads

Estimated time: 1-2 minutes`,
		"repair_permissions": `This option will:
• Check ownership of installation directories
• Fix permission issues with scripts
• Set appropriate execute permissions
• Verify user access to required paths

Estimated time: 30 seconds`,
		"cleanup_partial": `This option will:
• Identify partially installed components
• Remove incomplete installations
• Clean up temporary files
• Reset installation state

Warning: This will remove any partial progress
Estimated time: 2-5 minutes`,
		"reinstall_rocm": `This option will:
• Completely remove existing ROCm installation
• Download and reinstall ROCm platform
• Reconfigure GPU drivers
• Reset GPU environment variables

Warning: This will remove current ROCm installation
Estimated time: 15-30 minutes`,
		"reset_environment": `This option will:
• Reset environment variables to defaults
• Clear shell configuration changes
• Remove temporary environment modifications
• Reinitialize ML stack environment

Estimated time: 1 minute`,
		"system_diagnostic": `This option will:
• Check hardware compatibility
• Test GPU functionality
• Verify software versions
• Check system resources
• Generate diagnostic report

Estimated time: 3-5 minutes`,
		"backup_restore": `This option will:
• Search for available backups
• Allow selection of backup to restore
• Restore configuration and components
• Verify restored installation

Note: Requires existing backup to be available
Estimated time: 5-10 minutes`,
	}
}

// Init initializes the recovery component
func (c *RecoveryComponent) Init() tea.Cmd {
	return c.Spinner.Tick
}

// Update handles messages for the recovery component
func (c *RecoveryComponent) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch {
		case key.Matches(msg, c.KeyBindings["up"], c.KeyBindings["down"]):
			c.List, cmd = c.List.Update(msg)

		case key.Matches(msg, c.KeyBindings["select"]):
			if !c.Recovering {
				c.executeRecoveryOption()
			}

		case key.Matches(msg, c.KeyBindings["details"]):
			c.ShowDetails = !c.ShowDetails

		case key.Matches(msg, c.KeyBindings["retry"]):
			if !c.Recovering {
				c.retryLastOperation()
			}

		case key.Matches(msg, c.KeyBindings["diagnostic"]):
			if !c.Recovering {
				c.runDiagnostic()
			}

		case key.Matches(msg, c.KeyBindings["back"]):
			if !c.Recovering {
				return c, func() tea.Msg {
					return types.NavigateBackMsg{}
				}
			}
		}

	case spinner.TickMsg:
		if c.Loading || c.Recovering {
			c.Spinner, cmd = c.Spinner.Update(msg)
		}

	case types.RecoveryOptionSelectMsg:
		c.handleRecoveryOptionSelection(msg)

	case types.RecoveryAttemptMsg:
		c.handleRecoveryAttempt(msg)

	case types.RecoverySuccessMsg:
		c.handleRecoverySuccess(msg)

	case types.RecoveryFailureMsg:
		c.handleRecoveryFailure(msg)

	default:
		// Let list handle other messages
		c.List, cmd = c.List.Update(msg)
	}

	return c, cmd
}

// View renders the recovery component
func (c *RecoveryComponent) View() string {
	if !c.Ready {
		return "Initializing recovery options..."
	}

	var builder strings.Builder

	// Title
	builder.WriteString(c.TitleStyle.Render("System Recovery") + "\n\n")

	// Error information
	if c.ErrorMessage != "" {
		builder.WriteString(c.ErrorStyle.Render("Last Error: "+c.ErrorMessage) + "\n\n")
	}

	// Current action
	if c.Recovering {
		actionText := fmt.Sprintf("%s Recovering...", c.Spinner.View())
		builder.WriteString(c.WarningStyle.Render(actionText) + "\n\n")
	}

	// Recovery options list
	builder.WriteString(c.List.View())

	// Option details
	if c.ShowDetails && c.List.SelectedItem() != nil {
		builder.WriteString("\n\n" + c.renderOptionDetails())
	}

	// Help text
	builder.WriteString("\n\n" + c.renderHelp())

	return builder.String()
}

// SetBounds updates the component dimensions
func (c *RecoveryComponent) SetBounds(x, y, width, height int) {
	c.X = x
	c.Y = y
	c.Width = width
	c.Height = height

	// Update styles
	c.TitleStyle = c.TitleStyle.Width(width)
	c.SubtitleStyle = c.SubtitleStyle.Width(width)
	c.ErrorStyle = c.ErrorStyle.Width(width)
	c.InfoStyle = c.InfoStyle.Width(width)
	c.SuccessStyle = c.SuccessStyle.Width(width)
	c.WarningStyle = c.WarningStyle.Width(width)
	c.DetailsStyle = c.DetailsStyle.Width(width - 4)

	// Update list dimensions
	c.List.SetSize(width-10, height-20)
}

// GetBounds returns the current component bounds
func (c *RecoveryComponent) GetBounds() (x, y, width, height int) {
	return c.X, c.Y, c.Width, c.Height
}

// executeRecoveryOption executes the selected recovery option
func (c *RecoveryComponent) executeRecoveryOption() {
	if selectedItem, ok := c.List.SelectedItem().(RecoveryItem); ok {
		c.Recovering = true
		c.RecoveryProgress = 0.0

		// Log the recovery option being executed
		fmt.Printf("Executing recovery option: %s\n", selectedItem.Option.Name)

		// Send recovery attempt message
		// In a real implementation, this would trigger the actual recovery
		go func() {
			// Simulate recovery process
			time.Sleep(2 * time.Second)
			// Send success message
			// This would be done through proper tea.Cmd in real implementation
		}()
	}
}

// retryLastOperation retries the last failed operation
func (c *RecoveryComponent) retryLastOperation() {
	c.ErrorMessage = ""
	c.Recovering = true
	// Implementation would retry the last operation
}

// runDiagnostic runs a full system diagnostic
func (c *RecoveryComponent) runDiagnostic() {
	c.Loading = true
	// Implementation would run diagnostic and show results
}

// handleRecoveryOptionSelection handles recovery option selection
func (c *RecoveryComponent) handleRecoveryOptionSelection(msg types.RecoveryOptionSelectMsg) {
	c.SelectedIndex = msg.OptionIndex
}

// handleRecoveryAttempt handles recovery attempt notification
func (c *RecoveryComponent) handleRecoveryAttempt(msg types.RecoveryAttemptMsg) {
	c.Recovering = true
	c.ErrorMessage = msg.Error.Error()
}

// handleRecoverySuccess handles recovery success notification
func (c *RecoveryComponent) handleRecoverySuccess(msg types.RecoverySuccessMsg) {
	c.Recovering = false
	c.RecoveryProgress = 1.0
	c.ErrorMessage = ""
}

// handleRecoveryFailure handles recovery failure notification
func (c *RecoveryComponent) handleRecoveryFailure(msg types.RecoveryFailureMsg) {
	c.Recovering = false
	c.RecoveryProgress = 0.0
	c.ErrorMessage = msg.Error.Error()
}

// renderOptionDetails renders details for the selected option
func (c *RecoveryComponent) renderOptionDetails() string {
	if selectedItem, ok := c.List.SelectedItem().(RecoveryItem); ok {
		if details, exists := c.Details[selectedItem.Option.ID]; exists {
			return c.DetailsStyle.Render(details)
		}
	}
	return ""
}

// renderHelp renders the help text
func (c *RecoveryComponent) renderHelp() string {
	var helpItems []string

	if !c.Recovering {
		helpItems = append(helpItems, []string{
			fmt.Sprintf("%s: Select Option", c.KeyBindings["select"].Help().Key),
			fmt.Sprintf("%s: Show Details", c.KeyBindings["details"].Help().Key),
			fmt.Sprintf("%s: Retry", c.KeyBindings["retry"].Help().Key),
			fmt.Sprintf("%s: Diagnostic", c.KeyBindings["diagnostic"].Help().Key),
			fmt.Sprintf("%s: Back", c.KeyBindings["back"].Help().Key),
		}...)
	}

	return c.InfoStyle.Render(strings.Join(helpItems, " • "))
}

// SetError sets the current error information
func (c *RecoveryComponent) SetError(errorType, errorMessage string) {
	c.ErrorType = errorType
	c.ErrorMessage = errorMessage
}

// GetSelectedOption returns the currently selected recovery option
func (c *RecoveryComponent) GetSelectedOption() *installer.RecoveryOption {
	if selectedItem, ok := c.List.SelectedItem().(RecoveryItem); ok {
		return &selectedItem.Option
	}
	return nil
}

// IsRecovering returns whether a recovery operation is in progress
func (c *RecoveryComponent) IsRecovering() bool {
	return c.Recovering
}

// GetState returns the component state
func (c *RecoveryComponent) GetState() map[string]interface{} {
	return map[string]interface{}{
		"ready":             c.Ready,
		"loading":           c.Loading,
		"recovering":        c.Recovering,
		"error_type":        c.ErrorType,
		"error_message":     c.ErrorMessage,
		"selected_index":    c.SelectedIndex,
		"recovery_progress": c.RecoveryProgress,
		"option_count":      len(c.RecoveryOptions),
		"show_details":      c.ShowDetails,
		"width":             c.Width,
		"height":            c.Height,
		"start_time":        c.startTime,
	}
}
