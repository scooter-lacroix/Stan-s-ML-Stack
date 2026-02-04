// internal/ui/components/configuration.go
package components

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/scooter-lacroix/mlstack-installer/internal/installer"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// ConfigurationComponent implements configuration editing with AMD theming
type ConfigurationComponent struct {
	// Dimensions
	Width  int
	Height int
	X, Y   int

	// State
	Config         *installer.Config
	Ready          bool
	Loading        bool
	Focused        bool
	CurrentSection string
	Modified       bool

	// UI elements
	Inputs       map[string]textinput.Model
	Sections     []ConfigSection
	CurrentIndex int
	KeyBindings  map[string]key.Binding

	// AMD styling
	TitleStyle    lipgloss.Style
	SubtitleStyle lipgloss.Style
	SectionStyle  lipgloss.Style
	InputStyle    lipgloss.Style
	FocusedStyle  lipgloss.Style
	InfoStyle     lipgloss.Style
	ButtonStyle   lipgloss.Style
	ModifiedStyle lipgloss.Style

	// Performance tracking
	startTime time.Time
}

// ConfigSection represents a configuration section
type ConfigSection struct {
	Name        string
	Description string
	Fields      []ConfigField
}

// ConfigField represents a configuration field
type ConfigField struct {
	Key         string
	Label       string
	Type        string // "string", "number", "boolean", "path"
	Value       interface{}
	Description string
	Required    bool
}

// NewConfigurationComponent creates a new configuration component
func NewConfigurationComponent(width, height int, config *installer.Config) *ConfigurationComponent {
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

	sectionStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDWhite)).
		Background(lipgloss.Color(AMDGray)).
		Bold(true).
		Padding(0, 2).
		Margin(1, 0)

	inputStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDWhite)).
		Background(lipgloss.Color(AMDBlack)).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(AMDGray)).
		Padding(0, 1).
		Width(width - 20)

	focusedStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDWhite)).
		Background(lipgloss.Color(AMDRed)).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(AMDOrange)).
		Padding(0, 1)

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

	modifiedStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDOrange)).
		Bold(true)

	// Create default config if nil
	if config == nil {
		defaultConfig, err := installer.NewConfig()
		if err != nil {
			// Fallback to basic config
			config = &installer.Config{
				UserPreferences: installer.UserPreferences{
					InstallationPath:   "/opt/rocm",
					PerformanceProfile: "balanced",
					LogLevel:           "info",
					EnableAutoUpdates:  false,
				},
				PerformanceConfig: installer.PerformanceConfig{
					MaxConcurrentTasks: 4,
				},
				ComponentSettings: installer.ComponentSettings{
					ROCm: installer.ComponentConfig{
						Enabled: true,
					},
					PyTorch: installer.ComponentConfig{
						Enabled: false,
					},
				},
			}
		} else {
			config = defaultConfig
		}
	}

	// Create component
	component := &ConfigurationComponent{
		Width:         width,
		Height:        height,
		X:             0,
		Y:             0,
		Config:        config,
		Ready:         false,
		Loading:       true,
		Focused:       true,
		CurrentIndex:  0,
		Inputs:        make(map[string]textinput.Model),
		TitleStyle:    titleStyle,
		SubtitleStyle: subtitleStyle,
		SectionStyle:  sectionStyle,
		InputStyle:    inputStyle,
		FocusedStyle:  focusedStyle,
		InfoStyle:     infoStyle,
		ButtonStyle:   buttonStyle,
		ModifiedStyle: modifiedStyle,
		startTime:     time.Now(),
	}

	// Initialize configuration sections
	component.initializeSections()

	// Initialize key bindings
	component.initializeKeyBindings()

	// Create input fields
	component.createInputs()

	component.Ready = true
	component.Loading = false

	return component
}

// initializeSections sets up configuration sections
func (c *ConfigurationComponent) initializeSections() {
	c.Sections = []ConfigSection{
		{
			Name:        "Installation",
			Description: "Basic installation settings",
			Fields: []ConfigField{
				{
					Key:         "install_path",
					Label:       "Install Path",
					Type:        "path",
					Value:       c.Config.UserPreferences.InstallationPath,
					Description: "Directory where ML stack will be installed",
					Required:    true,
				},
				{
					Key:         "environment",
					Label:       "Environment",
					Type:        "string",
					Value:       c.Config.UserPreferences.PerformanceProfile,
					Description: "Installation environment (development/production)",
					Required:    true,
				},
			},
		},
		{
			Name:        "Performance",
			Description: "Performance and optimization settings",
			Fields: []ConfigField{
				{
					Key:         "enable_gpu",
					Label:       "Enable GPU",
					Type:        "boolean",
					Value:       c.isGPUEnabled(),
					Description: "Enable GPU acceleration with AMD ROCm",
					Required:    false,
				},
				{
					Key:         "parallel_jobs",
					Label:       "Parallel Jobs",
					Type:        "number",
					Value:       c.Config.PerformanceConfig.MaxConcurrentTasks,
					Description: "Number of parallel build jobs",
					Required:    false,
				},
			},
		},
		{
			Name:        "Logging",
			Description: "Logging and monitoring settings",
			Fields: []ConfigField{
				{
					Key:         "log_level",
					Label:       "Log Level",
					Type:        "string",
					Value:       c.Config.UserPreferences.LogLevel,
					Description: "Logging level (debug/info/warn/error)",
					Required:    false,
				},
			},
		},
		{
			Name:        "Features",
			Description: "Optional features and services",
			Fields: []ConfigField{
				{
					Key:         "auto_update",
					Label:       "Auto Update",
					Type:        "boolean",
					Value:       c.Config.UserPreferences.EnableAutoUpdates,
					Description: "Enable automatic updates",
					Required:    false,
				},
				{
					Key:         "telemetry",
					Label:       "Telemetry",
					Type:        "boolean",
					Value:       c.Config.ComponentSettings.PyTorch.Enabled,
					Description: "Send anonymous usage data",
					Required:    false,
				},
			},
		},
	}
}

// initializeKeyBindings sets up keyboard shortcuts
func (c *ConfigurationComponent) initializeKeyBindings() {
	c.KeyBindings = map[string]key.Binding{
		"up": key.NewBinding(
			key.WithKeys("up", "k"),
			key.WithHelp("↑/K", "Move Up"),
		),
		"down": key.NewBinding(
			key.WithKeys("down", "j"),
			key.WithHelp("↓/J", "Move Down"),
		),
		"edit": key.NewBinding(
			key.WithKeys("enter", " "),
			key.WithHelp("Enter", "Edit Field"),
		),
		"toggle": key.NewBinding(
			key.WithKeys("t"),
			key.WithHelp("T", "Toggle Boolean"),
		),
		"save": key.NewBinding(
			key.WithKeys("s", "ctrl+s"),
			key.WithHelp("S/Ctrl+S", "Save Config"),
		),
		"reset": key.NewBinding(
			key.WithKeys("r"),
			key.WithHelp("R", "Reset to Defaults"),
		),
		"back": key.NewBinding(
			key.WithKeys("esc", "q"),
			key.WithHelp("Esc/Q", "Back"),
		),
	}
}

// isGPUEnabled checks if GPU is enabled based on component settings
func (c *ConfigurationComponent) isGPUEnabled() bool {
	// Check if ROCm component is enabled
	return c.Config.ComponentSettings.ROCm.Enabled
}

// createInputs creates text input models for each field
func (c *ConfigurationComponent) createInputs() {
	for _, section := range c.Sections {
		for _, field := range section.Fields {
			if field.Type == "string" || field.Type == "number" || field.Type == "path" {
				input := textinput.New()
				input.Placeholder = field.Description

				switch field.Type {
				case "path":
					input.Width = c.Width - 30
				default:
					input.Width = 30
				}

				// Set current value
				if strValue, ok := field.Value.(string); ok {
					input.SetValue(strValue)
				}
				if numValue, ok := field.Value.(int); ok {
					input.SetValue(fmt.Sprintf("%d", numValue))
				}

				c.Inputs[field.Key] = input
			}
		}
	}
}

// Init initializes the configuration component
func (c *ConfigurationComponent) Init() tea.Cmd {
	return nil
}

// Update handles messages for the configuration component
func (c *ConfigurationComponent) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		// Check if any input is focused
		for fieldKey, input := range c.Inputs {
			if input.Focused() {
				updatedInput, inputCmd := input.Update(msg)
				c.Inputs[fieldKey] = updatedInput

				// Update config value on Enter/Escape
				enterBinding := key.NewBinding(key.WithKeys("enter"))
				if key.Matches(msg, enterBinding) {
					c.updateConfigValue(fieldKey, updatedInput.Value())
					updatedInput.Blur() // Blur command
					c.Modified = true
				}
				escapeBinding := key.NewBinding(key.WithKeys("escape"))
				if key.Matches(msg, escapeBinding) {
					input.Blur() // Blur command
				}

				return c, inputCmd
			}
		}

		// Normal navigation
		switch {
		case key.Matches(msg, c.KeyBindings["up"]):
			c.moveUp()

		case key.Matches(msg, c.KeyBindings["down"]):
			c.moveDown()

		case key.Matches(msg, c.KeyBindings["edit"]):
			c.editCurrentField()

		case key.Matches(msg, c.KeyBindings["toggle"]):
			c.toggleCurrentBoolean()

		case key.Matches(msg, c.KeyBindings["save"]):
			return c, c.saveConfig()

		case key.Matches(msg, c.KeyBindings["reset"]):
			c.resetToDefaults()

		case key.Matches(msg, c.KeyBindings["back"]):
			return c, func() tea.Msg {
				return types.NavigateBackMsg{}
			}
		}

	default:
		// Let inputs handle other messages
		for key, input := range c.Inputs {
			if input.Focused() {
				updatedInput, inputCmd := input.Update(msg)
				c.Inputs[key] = updatedInput
				return c, inputCmd
			}
		}
	}

	return c, cmd
}

// View renders the configuration component
func (c *ConfigurationComponent) View() string {
	if !c.Ready {
		return "Initializing configuration..."
	}

	var builder strings.Builder

	// Title with modified indicator
	title := "Configuration Settings"
	if c.Modified {
		title += " *"
	}
	builder.WriteString(c.TitleStyle.Render(title) + "\n\n")

	// Subtitle
	builder.WriteString(c.SubtitleStyle.Render("Customize your ML stack installation") + "\n\n")

	// Render sections
	c.renderSections(&builder)

	// Help text
	builder.WriteString("\n\n" + c.renderHelp())

	return builder.String()
}

// SetBounds updates the component dimensions
func (c *ConfigurationComponent) SetBounds(x, y, width, height int) {
	c.X = x
	c.Y = y
	c.Width = width
	c.Height = height

	// Update styles
	c.TitleStyle = c.TitleStyle.Width(width)
	c.SubtitleStyle = c.SubtitleStyle.Width(width)
	c.InfoStyle = c.InfoStyle.Width(width)

	// Recreate inputs with new dimensions
	c.createInputs()
}

// GetBounds returns the current component bounds
func (c *ConfigurationComponent) GetBounds() (x, y, width, height int) {
	return c.X, c.Y, c.Width, c.Height
}

// renderSections renders all configuration sections
func (c *ConfigurationComponent) renderSections(builder *strings.Builder) {
	globalIndex := 0
	for i, section := range c.Sections {
		// Section header
		sectionHeader := fmt.Sprintf("%d. %s", i+1, section.Name)
		builder.WriteString(c.SectionStyle.Render(sectionHeader) + "\n")
		builder.WriteString(c.InfoStyle.Render(section.Description) + "\n\n")

		// Section fields
		for j, field := range section.Fields {
			isSelected := globalIndex == c.CurrentIndex
			c.renderField(builder, field, i, j, isSelected)
			globalIndex++
		}

		builder.WriteString("\n")
	}
}

// renderField renders a single configuration field
func (c *ConfigurationComponent) renderField(builder *strings.Builder, field ConfigField, sectionIndex, fieldIndex int, isSelected bool) {
	var fieldLine strings.Builder

	// Field label
	label := field.Label
	if field.Required {
		label += " *"
	}

	if isSelected {
		fieldLine.WriteString(c.FocusedStyle.Render(fmt.Sprintf("  %s: ", label)))
	} else {
		fieldLine.WriteString(c.InputStyle.Render(fmt.Sprintf("  %s: ", label)))
	}

	// Field value based on type
	switch field.Type {
	case "boolean":
		boolValue, _ := field.Value.(bool)
		valueText := "No"
		if boolValue {
			valueText = "Yes"
		}
		if isSelected {
			fieldLine.WriteString(c.FocusedStyle.Render(valueText))
		} else {
			fieldLine.WriteString(c.InputStyle.Render(valueText))
		}

	case "string", "number", "path":
		if input, exists := c.Inputs[field.Key]; exists {
			if input.Focused() {
				fieldLine.WriteString(input.View())
			} else {
				value := fmt.Sprintf("%v", field.Value)
				if value == "" {
					value = "<not set>"
				}
				if isSelected {
					fieldLine.WriteString(c.FocusedStyle.Render(value))
				} else {
					fieldLine.WriteString(c.InputStyle.Render(value))
				}
			}
		}

	default:
		value := fmt.Sprintf("%v", field.Value)
		if isSelected {
			fieldLine.WriteString(c.FocusedStyle.Render(value))
		} else {
			fieldLine.WriteString(c.InputStyle.Render(value))
		}
	}

	builder.WriteString(fieldLine.String() + "\n")

	// Field description
	if field.Description != "" {
		if isSelected {
			builder.WriteString(c.ModifiedStyle.Render("    "+field.Description) + "\n")
		} else {
			builder.WriteString(c.InfoStyle.Render("    "+field.Description) + "\n")
		}
	}
}

// moveUp moves selection up
func (c *ConfigurationComponent) moveUp() {
	if c.CurrentIndex > 0 {
		c.CurrentIndex--
	}
}

// moveDown moves selection down
func (c *ConfigurationComponent) moveDown() {
	// Count total fields
	totalFields := 0
	for _, section := range c.Sections {
		totalFields += len(section.Fields)
	}

	if c.CurrentIndex < totalFields-1 {
		c.CurrentIndex++
	}
}

// editCurrentField focuses the current field for editing
func (c *ConfigurationComponent) editCurrentField() {
	field := c.getCurrentField()
	if field == nil {
		return
	}

	if field.Type == "string" || field.Type == "number" || field.Type == "path" {
		if input, exists := c.Inputs[field.Key]; exists {
			_ = input.Focus() // Focus command - we don't need the updated model here
		}
	}
}

// toggleCurrentBoolean toggles the current boolean field
func (c *ConfigurationComponent) toggleCurrentBoolean() {
	field := c.getCurrentField()
	if field == nil || field.Type != "boolean" {
		return
	}

	currentValue, _ := field.Value.(bool)
	field.Value = !currentValue
	c.updateConfigField(field)
	c.Modified = true
}

// getCurrentField returns the currently selected field
func (c *ConfigurationComponent) getCurrentField() *ConfigField {
	globalIndex := 0
	for _, section := range c.Sections {
		for i := range section.Fields {
			if globalIndex == c.CurrentIndex {
				return &section.Fields[i]
			}
			globalIndex++
		}
	}
	return nil
}

// updateConfigValue updates a configuration value from input
func (c *ConfigurationComponent) updateConfigValue(key string, value string) {
	// Find the field and update its value
	for _, section := range c.Sections {
		for i, field := range section.Fields {
			if field.Key == key {
				switch field.Type {
				case "string", "path":
					section.Fields[i].Value = value
				case "number":
					// Try to parse as number
					var numValue int
					fmt.Sscanf(value, "%d", &numValue)
					section.Fields[i].Value = numValue
				}
				c.updateConfigField(&section.Fields[i])
				return
			}
		}
	}
}

// updateConfigField updates the config object with field value
func (c *ConfigurationComponent) updateConfigField(field *ConfigField) {
	switch field.Key {
	case "install_path":
		if value, ok := field.Value.(string); ok {
			c.Config.UserPreferences.InstallationPath = value
		}
	case "environment":
		if value, ok := field.Value.(string); ok {
			c.Config.UserPreferences.PerformanceProfile = value
		}
	case "enable_gpu":
		if value, ok := field.Value.(bool); ok {
			c.Config.ComponentSettings.ROCm.Enabled = value
		}
	case "parallel_jobs":
		if value, ok := field.Value.(int); ok {
			c.Config.PerformanceConfig.MaxConcurrentTasks = value
		}
	case "log_level":
		if value, ok := field.Value.(string); ok {
			c.Config.UserPreferences.LogLevel = value
		}
	case "auto_update":
		if value, ok := field.Value.(bool); ok {
			c.Config.UserPreferences.EnableAutoUpdates = value
		}
	case "telemetry":
		if value, ok := field.Value.(bool); ok {
			// Use PyTorch enable status as telemetry proxy
			c.Config.ComponentSettings.PyTorch.Enabled = value
		}
	}
}

// saveConfig returns a command to save the configuration
func (c *ConfigurationComponent) saveConfig() tea.Cmd {
	return func() tea.Msg {
		// In a real implementation, this would save to disk
		c.Modified = false
		return types.ConfigSaveMsg{
			Success: true,
			Message: "Configuration saved successfully",
		}
	}
}

// resetToDefaults resets configuration to default values
func (c *ConfigurationComponent) resetToDefaults() {
	// Reset to default values based on actual installer.Config structure
	c.Config.UserPreferences.InstallationPath = "/opt/rocm"
	c.Config.UserPreferences.PerformanceProfile = "balanced"
	c.Config.ComponentSettings.ROCm.Enabled = true
	c.Config.PerformanceConfig.MaxConcurrentTasks = 4
	c.Config.UserPreferences.LogLevel = "info"
	c.Config.UserPreferences.EnableAutoUpdates = false
	c.Config.ComponentSettings.PyTorch.Enabled = false

	// Refresh sections and inputs
	c.initializeSections()
	c.createInputs()
	c.Modified = true
}

// renderHelp renders the help text
func (c *ConfigurationComponent) renderHelp() string {
	helpItems := []string{
		fmt.Sprintf("%s: Move Up/Down", c.KeyBindings["up"].Help().Key),
		fmt.Sprintf("%s: Edit Field", c.KeyBindings["edit"].Help().Key),
		fmt.Sprintf("%s: Toggle Boolean", c.KeyBindings["toggle"].Help().Key),
		fmt.Sprintf("%s: Save", c.KeyBindings["save"].Help().Key),
		fmt.Sprintf("%s: Reset", c.KeyBindings["reset"].Help().Key),
		fmt.Sprintf("%s: Back", c.KeyBindings["back"].Help().Key),
	}

	return c.InfoStyle.Render(strings.Join(helpItems, " • "))
}

// GetConfig returns the current configuration
func (c *ConfigurationComponent) GetConfig() *installer.Config {
	return c.Config
}

// IsModified returns whether the configuration has been modified
func (c *ConfigurationComponent) IsModified() bool {
	return c.Modified
}

// GetState returns the component state
func (c *ConfigurationComponent) GetState() map[string]interface{} {
	return map[string]interface{}{
		"ready":         c.Ready,
		"loading":       c.Loading,
		"modified":      c.Modified,
		"section_count": len(c.Sections),
		"current_index": c.CurrentIndex,
		"width":         c.Width,
		"height":        c.Height,
		"start_time":    c.startTime,
	}
}
