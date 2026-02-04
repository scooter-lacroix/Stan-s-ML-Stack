// internal/ui/monitoring/themes.go
package monitoring

import (
	"github.com/charmbracelet/lipgloss"
)

// AMDTheme implements the professional AMD branding theme system
type AMDTheme struct {
	// Color palette
	Primary      lipgloss.Color
	Secondary    lipgloss.Color
	Accent       lipgloss.Color
	Success      lipgloss.Color
	Warning      lipgloss.Color
	Error        lipgloss.Color
	Info         lipgloss.Color
	Background   lipgloss.Color
	Foreground   lipgloss.Color
	Muted        lipgloss.Color

	// Gradient colors for advanced effects
	GradientStart lipgloss.Color
	GradientEnd   lipgloss.Color

	// Component styles
	Container    lipgloss.Style
	Panel        lipgloss.Style
	Header       lipgloss.Style
	Title        lipgloss.Style
	SubTitle     lipgloss.Style
	Text         lipgloss.Style
	MutedText    lipgloss.Style
	Highlight    lipgloss.Style
	Border       lipgloss.Style
	ActiveBorder lipgloss.Style
	Button       lipgloss.Style
	ActiveButton lipgloss.Style
	ProgressBar  lipgloss.Style
	Help         lipgloss.Style
	Status       lipgloss.Style

	// Widget-specific styles
	Chart        lipgloss.Style
	Gauge        lipgloss.Style
	Metric       lipgloss.Style
	Alert        lipgloss.Style
	SuccessAlert lipgloss.Style
	WarningAlert lipgloss.Style
	ErrorAlert   lipgloss.Style

	// Accessibility
	HighContrast bool
	CurrentTheme string
}

// Theme constants
const (
	ThemeAMD          = "amd"
	ThemeDark         = "dark"
	ThemeLight        = "light"
	ThemeHighContrast = "high_contrast"
)

// NewAMDTheme creates a new AMD theme with professional branding
func NewAMDTheme() *AMDTheme {
	theme := &AMDTheme{
		CurrentTheme: ThemeAMD,
		HighContrast: false,
	}

	theme.setupColors()
	theme.setupStyles()

	return theme
}

// setupColors initializes the AMD color palette
func (t *AMDTheme) setupColors() {
	switch t.CurrentTheme {
	case ThemeAMD:
		// AMD Professional Brand Colors
		t.Primary = lipgloss.Color("#ED1C24")      // AMD Red
		t.Secondary = lipgloss.Color("#000000")    // Professional Black
		t.Accent = lipgloss.Color("#FF6B35")       // AMD Orange Accent
		t.Success = lipgloss.Color("#00C853")      // Success Green
		t.Warning = lipgloss.Color("#FFB300")      // Warning Amber
		t.Error = lipgloss.Color("#D32F2F")        // Error Red
		t.Info = lipgloss.Color("#1976D2")         // Info Blue
		t.Background = lipgloss.Color("#121212")   // Dark Background
		t.Foreground = lipgloss.Color("#FFFFFF")   // White Text
		t.Muted = lipgloss.Color("#B0BEC5")        // Muted Gray
		t.Border = lipgloss.Color("#37474F")       // Border Gray
		t.ActiveBorder = lipgloss.Color("#ED1C24") // AMD Red Active

		// Gradient colors for advanced effects
		t.GradientStart = lipgloss.Color("#ED1C24") // AMD Red
		t.GradientEnd = lipgloss.Color("#FF6B35")   // AMD Orange

	case ThemeDark:
		// Dark theme
		t.Primary = lipgloss.Color("#BB86FC")
		t.Secondary = lipgloss.Color("#03DAC6")
		t.Accent = lipgloss.Color("#CF6679")
		t.Success = lipgloss.Color("#4CAF50")
		t.Warning = lipgloss.Color("#FF9800")
		t.Error = lipgloss.Color("#F44336")
		t.Info = lipgloss.Color("#2196F3")
		t.Background = lipgloss.Color("#121212")
		t.Foreground = lipgloss.Color("#FFFFFF")
		t.Muted = lipgloss.Color("#9E9E9E")
		t.Border = lipgloss.Color("#424242")
		t.ActiveBorder = lipgloss.Color("#BB86FC")

		t.GradientStart = lipgloss.Color("#BB86FC")
		t.GradientEnd = lipgloss.Color("#03DAC6")

	case ThemeLight:
		// Light theme
		t.Primary = lipgloss.Color("#6200EE")
		t.Secondary = lipgloss.Color("#018786")
		t.Accent = lipgloss.Color("#3700B3")
		t.Success = lipgloss.Color("#388E3C")
		t.Warning = lipgloss.Color("#F57C00")
		t.Error = lipgloss.Color("#D32F2F")
		t.Info = lipgloss.Color("#1976D2")
		t.Background = lipgloss.Color("#FFFFFF")
		t.Foreground = lipgloss.Color("#000000")
		t.Muted = lipgloss.Color("#757575")
		t.Border = lipgloss.Color("#E0E0E0")
		t.ActiveBorder = lipgloss.Color("#6200EE")

		t.GradientStart = lipgloss.Color("#6200EE")
		t.GradientEnd = lipgloss.Color("#018786")

	case ThemeHighContrast:
		// High contrast theme for accessibility
		t.Primary = lipgloss.Color("#FFFF00")
		t.Secondary = lipgloss.Color("#00FFFF")
		t.Accent = lipgloss.Color("#FF00FF")
		t.Success = lipgloss.Color("#00FF00")
		t.Warning = lipgloss.Color("#FFB300")
		t.Error = lipgloss.Color("#FF0000")
		t.Info = lipgloss.Color("#0000FF")
		t.Background = lipgloss.Color("#000000")
		t.Foreground = lipgloss.Color("#FFFFFF")
		t.Muted = lipgloss.Color("#C0C0C0")
		t.Border = lipgloss.Color("#FFFFFF")
		t.ActiveBorder = lipgloss.Color("#FFFF00")

		t.GradientStart = lipgloss.Color("#FFFF00")
		t.GradientEnd = lipgloss.Color("#00FFFF")
	}
}

// setupStyles initializes all component styles
func (t *AMDTheme) setupStyles() {
	// Container styles
	t.Container = lipgloss.NewStyle().
		Background(t.Background).
		Foreground(t.Foreground).
		Padding(1, 2)

	t.Panel = lipgloss.NewStyle().
		Background(t.Background).
		Foreground(t.Foreground).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(t.Border).
		Padding(1)

	// Header styles
	t.Header = lipgloss.NewStyle().
		Bold(true).
		Foreground(t.Primary).
		MarginBottom(1)

	t.Title = lipgloss.NewStyle().
		Bold(true).
		Foreground(t.Primary).
		Underline(true)

	t.SubTitle = lipgloss.NewStyle().
		Bold(false).
		Foreground(t.Secondary).
		Italic(true)

	// Text styles
	t.Text = lipgloss.NewStyle().
		Foreground(t.Foreground)

	t.MutedText = lipgloss.NewStyle().
		Foreground(t.Muted).
		Italic(true)

	t.Highlight = lipgloss.NewStyle().
		Background(t.Accent).
		Foreground(t.Background).
		Bold(true)

	// Border styles
	t.Border = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(t.Border).
		Padding(1)

	t.ActiveBorder = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(t.ActiveBorder).
		Padding(1).
		BorderTop(true).
		BorderBottom(true).
		BorderLeft(true).
		BorderRight(true)

	// Button styles
	t.Button = lipgloss.NewStyle().
		Background(t.Primary).
		Foreground(t.Background).
		Bold(true).
		Padding(0, 2).
		Margin(1)

	t.ActiveButton = lipgloss.NewStyle().
		Background(t.Accent).
		Foreground(t.Background).
		Bold(true).
		Underline(true).
		Padding(0, 2).
		Margin(1)

	// Progress bar style
	t.ProgressBar = lipgloss.NewStyle().
		Background(t.Border).
		Foreground(t.Success).
		Bold(true)

	// Help style
	t.Help = lipgloss.NewStyle().
		Background(t.Background).
		Foreground(t.Muted).
		Border(lipgloss.NormalBorder()).
		BorderForeground(t.Border).
		Padding(1).
		MarginTop(1)

	// Status styles
	t.Status = lipgloss.NewStyle().
		Bold(true).
		Padding(0, 1)

	// Widget-specific styles
	t.Chart = lipgloss.NewStyle().
		Background(t.Background).
		Foreground(t.Foreground).
		Padding(1)

	t.Gauge = lipgloss.NewStyle().
		Background(t.Border).
		Foreground(t.Primary).
		Bold(true)

	t.Metric = lipgloss.NewStyle().
		Foreground(t.Secondary).
		Bold(true)

	// Alert styles
	t.Alert = lipgloss.NewStyle().
		Background(t.Border).
		Foreground(t.Foreground).
		Padding(1).
		MarginBottom(1)

	t.SuccessAlert = lipgloss.NewStyle().
		Background(t.Success).
		Foreground(t.Background).
		Bold(true).
		Padding(1).
		MarginBottom(1)

	t.WarningAlert = lipgloss.NewStyle().
		Background(t.Warning).
		Foreground(t.Background).
		Bold(true).
		Padding(1).
		MarginBottom(1)

	t.ErrorAlert = lipgloss.NewStyle().
		Background(t.Error).
		Foreground(t.Background).
		Bold(true).
		Padding(1).
		MarginBottom(1)
}

// CycleTheme cycles through available themes
func (t *AMDTheme) CycleTheme() {
	themes := []string{ThemeAMD, ThemeDark, ThemeLight, ThemeHighContrast}

	for i, theme := range themes {
		if theme == t.CurrentTheme {
			nextTheme := themes[(i+1)%len(themes)]
			t.SetTheme(nextTheme)
			return
		}
	}

	// Fallback to AMD theme
	t.SetTheme(ThemeAMD)
}

// SetTheme sets a specific theme
func (t *AMDTheme) SetTheme(themeName string) {
	t.CurrentTheme = themeName
	t.HighContrast = (themeName == ThemeHighContrast)
	t.setupColors()
	t.setupStyles()
}

// GetGradientText creates gradient text effect
func (t *AMDTheme) GetGradientText(text string) string {
	// Simple gradient implementation (can be enhanced with proper gradient calculations)
	return lipgloss.NewStyle().
		Foreground(t.GradientStart).
		Bold(true).
		Render(text)
}

// GetStatusStyle returns appropriate style for a status
func (t *AMDTheme) GetStatusStyle(status string) lipgloss.Style {
	switch status {
	case "success", "ok", "healthy":
		return t.SuccessAlert
	case "warning", "caution", "degraded":
		return t.WarningAlert
	case "error", "critical", "failed":
		return t.ErrorAlert
	case "info", "notice":
		return t.Alert
	default:
		return t.Alert
	}
}

// GetProgressBarStyle returns styled progress bar
func (t *AMDTheme) GetProgressBarStyle(percentage float64) lipgloss.Style {
	var color lipgloss.Color

	switch {
	case percentage >= 80:
		color = t.Error
	case percentage >= 60:
		color = t.Warning
	case percentage >= 40:
		color = t.Info
	default:
		color = t.Success
	}

	return lipgloss.NewStyle().
		Background(t.Border).
		Foreground(color).
		Bold(true)
}

// GetMetricStyle returns styled metric display
func (t *AMDTheme) GetMetricStyle(value float64, threshold float64) lipgloss.Style {
	if value > threshold {
		return lipgloss.NewStyle().
			Foreground(t.Warning).
			Bold(true)
	}

	return t.Metric
}

// GetAccessibilityOptions returns current accessibility settings
func (t *AMDTheme) GetAccessibilityOptions() map[string]bool {
	return map[string]bool{
		"high_contrast":  t.HighContrast,
		"large_text":     false, // Can be implemented
		"reduced_motion": false, // Can be implemented
		"screen_reader":  false, // Can be implemented
	}
}

// SetAccessibilityOption sets an accessibility option
func (t *AMDTheme) SetAccessibilityOption(option string, enabled bool) {
	switch option {
	case "high_contrast":
		if enabled && !t.HighContrast {
			t.SetTheme(ThemeHighContrast)
		} else if !enabled && t.HighContrast {
			t.SetTheme(ThemeAMD)
		}
	}
}

// ThemeConfig represents theme configuration
type ThemeConfig struct {
	Name          string            `json:"name"`
	DisplayName   string            `json:"display_name"`
	Colors        map[string]string `json:"colors"`
	Accessibility bool              `json:"accessibility"`
	AMD           bool              `json:"amd_branded"`
}

// GetAllThemes returns all available themes
func GetAllThemes() []ThemeConfig {
	return []ThemeConfig{
		{
			Name:        ThemeAMD,
			DisplayName: "AMD Professional",
			Colors: map[string]string{
				"primary":   "#ED1C24",
				"secondary": "#000000",
				"accent":    "#FF6B35",
			},
			Accessibility: false,
			AMD:           true,
		},
		{
			Name:        ThemeDark,
			DisplayName: "Dark Mode",
			Colors: map[string]string{
				"primary":   "#BB86FC",
				"secondary": "#03DAC6",
				"accent":    "#CF6679",
			},
			Accessibility: false,
			AMD:           false,
		},
		{
			Name:        ThemeLight,
			DisplayName: "Light Mode",
			Colors: map[string]string{
				"primary":   "#6200EE",
				"secondary": "#018786",
				"accent":    "#3700B3",
			},
			Accessibility: false,
			AMD:           false,
		},
		{
			Name:        ThemeHighContrast,
			DisplayName: "High Contrast",
			Colors: map[string]string{
				"primary":   "#FFFF00",
				"secondary": "#00FFFF",
				"accent":    "#FF00FF",
			},
			Accessibility: true,
			AMD:           false,
		},
	}
}
