// internal/ui/components/welcome.go
package components

import (
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/integration"
)

// WelcomeComponent implements the welcome screen with AMD branding
type WelcomeComponent struct {
	// Dimensions
	Width  int
	Height int
	X, Y   int

	// State
	Ready    bool
	Focused  bool
	StartTime time.Time

	// Integration
	integration integration.Manager

	// AMD styling
	TitleStyle    lipgloss.Style
	SubtitleStyle lipgloss.Style
	ContentStyle  lipgloss.Style
	HelpStyle     lipgloss.Style
	BorderStyle   lipgloss.Style
}

// NewWelcomeComponent creates a new welcome component with AMD branding
func NewWelcomeComponent(width, height int, integration integration.Manager) *WelcomeComponent {
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
		Align(lipgloss.Center).
		MarginBottom(1)

	helpStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDGray)).
		Width(contentWidth).
		Align(lipgloss.Center).
		MarginTop(2)

	borderStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(AMDRed)).
		Padding(1)

	return &WelcomeComponent{
		Width:      width,
		Height:     height,
		Ready:      true,
		StartTime:  time.Now(),
		integration: integration,
		TitleStyle:  titleStyle,
		SubtitleStyle: subtitleStyle,
		ContentStyle: contentStyle,
		HelpStyle:   helpStyle,
		BorderStyle: borderStyle,
	}
}

// Init initializes the welcome component
func (c *WelcomeComponent) Init() tea.Cmd {
	return nil
}

// Update handles messages for the welcome component
func (c *WelcomeComponent) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "enter", " ":
			// Navigate to next stage
			return c, nil
		case "q", "ctrl+c":
			return c, tea.Quit
		}
	}
	return c, nil
}

// View renders the welcome screen
func (c *WelcomeComponent) View() string {
	// Build the welcome screen with AMD branding
	content := ""

	// Title section
	content += c.TitleStyle.Render("Stan's ML Stack Installer") + "\n\n"

	// Subtitle
	content += c.SubtitleStyle.Render("Powered by AMD ROCm - GPU Computing Platform") + "\n\n"

	// Version info
	content += c.ContentStyle.Render("Version 0.1.5") + "\n\n"

	// Welcome message
	content += c.ContentStyle.Render("Welcome to the ML Stack Installer!") + "\n\n"
	content += c.ContentStyle.Render("This installer will guide you through installing") + "\n"
	content += c.ContentStyle.Render("AMD-optimized machine learning components.") + "\n\n"

	// Features list
	features := []string{
		"ROCm Platform for AMD GPU computing",
		"PyTorch with HIP support",
		"Flash Attention for efficient training",
		"vLLM for fast LLM inference",
		"Complete ML development environment",
	}

	for _, feature := range features {
		content += c.ContentStyle.Render("  • " + feature) + "\n"
	}

	content += "\n"

	// Help text
	content += c.HelpStyle.Render("Press ENTER to continue • Press Q to quit") + "\n"

	// Apply border
	return c.BorderStyle.Width(c.Width).Render(content)
}

// SetBounds sets the component bounds
func (c *WelcomeComponent) SetBounds(x, y, width, height int) {
	c.X = x
	c.Y = y
	c.Width = width
	c.Height = height
}

// GetBounds returns the component bounds
func (c *WelcomeComponent) GetBounds() (x, y, width, height int) {
	return c.X, c.Y, c.Width, c.Height
}
