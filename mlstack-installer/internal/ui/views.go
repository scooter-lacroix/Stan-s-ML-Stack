// internal/ui/views.go
package ui

import (
	"github.com/charmbracelet/lipgloss"
)

// Enhanced styles with more visual appeal
var (
	// Main title styles
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#FAFAFA")).
			Background(lipgloss.Color("#7D56F4")).
			Padding(0, 2).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#FFFFFF"))

	// Header styles
	headerStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#7D56F4")).
			MarginTop(1).
			MarginBottom(1)

	subheaderStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#6C5CE7")).
			MarginTop(1)

	// Component styles
	componentStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#2D3748")).
			Background(lipgloss.Color("#F7FAFC")).
			Padding(0, 1).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#CBD5E0"))

	selectedComponentStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#FFFFFF")).
				Background(lipgloss.Color("#7D56F4")).
				Bold(true).
				Padding(0, 1).
				Border(lipgloss.RoundedBorder()).
				BorderForeground(lipgloss.Color("#FFFFFF"))

	checkboxStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#04B575"))

	selectedCheckboxStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#FFFFFF")).
				Bold(true)

	// Status styles
	successStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#04B575"))

	errorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FF6B6B"))

	warningStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FFA502"))

	// Progress and spinner styles
	progressStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#7D56F4")).
			Bold(true)

	// Help and info styles
	infoStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#626262")).
			MarginTop(1)

	dimStyle = lipgloss.NewStyle().
			Faint(true).
			Foreground(lipgloss.Color("#A0AEC0"))

	helpStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#4A5568")).
			MarginTop(1)

	// Enhanced hardware detection styles
	hardwareHeaderStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("#2B6CB0")).
				MarginTop(1).
				MarginBottom(1)

	gpuInfoStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#2D3748")).
			Background(lipgloss.Color("#EBF8FF")).
			Padding(0, 1).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#CBD5E0"))

	// Pre-flight check styles
	checkPassedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#04B575"))

	checkFailedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#FF6B6B"))

	checkWarningStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#FFA502"))

	checkListStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#2D3748")).
			Background(lipgloss.Color("#F7FAFC")).
			Padding(0, 1).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#CBD5E0"))

	// Configuration management styles
	configSectionStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("#553C9A")).
				MarginTop(2).
				MarginBottom(1)

	configItemStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#2D3748")).
			Padding(0, 1)

	configValueStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#4A5568")).
				Italic(true)

	// Error recovery styles
	recoveryHeaderStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("#C53030")).
				MarginTop(1).
				MarginBottom(1)

	recoveryActionStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#2B6CB0")).
				Underline(true)
)

// LegacyView renders the enhanced UI with comprehensive error handling
// NOTE: This is the old view system - DISABLED due to architecture mismatch
// Use the component-based View in app.go instead
/*
func (m Model) LegacyView() string {
	// Defensive check for model validity
	defer func() {
		if r := recover(); r != nil {
			// This catches any remaining panics and provides a safe fallback
			// In production, this would log the error
		}
	}()

	// Check if model is properly initialized
	if len(m.components) == 0 && m.stage > StageWelcome {
		// Model is in inconsistent state
		var b strings.Builder
		b.WriteString("\n")
		b.WriteString(titleStyle.Render(" System Error "))
		b.WriteString("\n\n")
		b.WriteString(errorStyle.Render("  ❌ Installer model is in an inconsistent state\n"))
		b.WriteString(infoStyle.Render("  Components not properly initialized\n\n"))
		b.WriteString(helpStyle.Render("  Press r to restart installer • Press q to quit"))
		return b.String()
	}

	if !m.ready {
		return m.loadingView()
	}

	if m.quitting {
		return m.completionView()
	}

	// Safe stage routing with error handling
	switch m.stage {
	case StageWelcome:
		return m.welcomeView()
	case StageHardwareDetect:
		return m.hardwareDetectView()
	case StagePreFlightCheck:
		return m.preflightCheckView()
	case StageComponentSelect:
		return m.componentSelectView()
	case StageConfiguration:
		return m.configurationView()
	case StageConfirm:
		return m.confirmView()
	case StageInstalling:
		return m.installingView()
	case StageRecovery:
		return m.recoveryView()
	case StageComplete:
		return m.completeView()
	default:
		// Unknown stage - provide safe fallback
		var b strings.Builder
		b.WriteString("\n")
		b.WriteString(titleStyle.Render(" Unknown State "))
		b.WriteString("\n\n")
		b.WriteString(warningStyle.Render(fmt.Sprintf("  ⚠️  Unknown installer stage: %d\n", m.stage)))
		b.WriteString(infoStyle.Render("  Returning to welcome screen...\n\n"))
		b.WriteString(helpStyle.Render("  Press Enter to continue"))
		return b.String()
	}
}
*/

/*
// Old view system - DISABLED due to architecture mismatch
// The following view functions use old model fields that no longer exist

// loadingView shows loading animation with defensive checks
func (m Model) loadingView() string {
	var b strings.Builder

	b.WriteString("\n")
	b.WriteString(titleStyle.Render(" Initializing ML Stack Installer "))
	b.WriteString("\n\n")

	// Safe spinner access
	spinnerView := "⠋" // Default fallback spinner
	if m.spinner.View != nil {
		spinnerView = m.spinner.View()
	}

	b.WriteString(fmt.Sprintf("  %s Initializing components...\n\n", spinnerView))
	b.WriteString(helpStyle.Render("  Please wait while the installer initializes..."))

	return b.String()
}

// completionView shows final completion screen
func (m Model) completionView() string {
	var b strings.Builder

	b.WriteString("\n")
	b.WriteString(titleStyle.Render(" Installation Complete! "))
	b.WriteString("\n\n")

	if len(m.errorLog) == 0 {
		b.WriteString(successStyle.Render("  ✅ All components installed successfully!\n\n"))
	} else {
		b.WriteString(warningStyle.Render("  ⚠️  Installation completed with warnings:\n\n"))
		for _, err := range m.errorLog {
			b.WriteString(errorStyle.Render(fmt.Sprintf("    • %s\n", err)))
		}
		b.WriteString("\n")
	}

	// Show next steps
	b.WriteString(headerStyle.Render("Next Steps:"))
	b.WriteString("\n")
	b.WriteString("  1. Source environment: source ~/.mlstack_env\n")
	b.WriteString("  2. Verify installation: ml-stack-verify\n")
	b.WriteString("  3. Test PyTorch: python -c 'import torch; print(torch.cuda.is_available())'\n")
	b.WriteString("  4. Check logs: ~/.mlstack/logs/\n\n")

	b.WriteString(helpStyle.Render("  Press Enter to exit installer"))

	return b.String()
}

// welcomeView shows enhanced welcome screen
func (m Model) welcomeView() string {
	var b strings.Builder

	b.WriteString("\n")
	b.WriteString(titleStyle.Render(" Stan's ML Stack Installer "))
	b.WriteString("\n\n")

	// Main description
	b.WriteString(infoStyle.Render("  A complete AMD GPU ML environment installer"))
	b.WriteString("\n")
	b.WriteString(infoStyle.Render("  Optimized for Radeon 7000 series GPUs"))
	b.WriteString("\n\n")

	// Features section
	b.WriteString(headerStyle.Render("Features:"))
	b.WriteString("\n")
	b.WriteString("  🔍 Automated GPU detection and compatibility checking")
	b.WriteString("\n")
	b.WriteString("  🛡️  Security-enhanced script execution with validation")
	b.WriteString("\n")
	b.WriteString("  📊 Real-time progress tracking and detailed logging")
	b.WriteString("\n")
	b.WriteString("  🎓 Complete ML stack with 20+ optimized components")
	b.WriteString("\n")
	b.WriteString("  ⚡ High-performance AMD GPU acceleration")
	b.WriteString("\n\n")

	// Component summary
	b.WriteString(subheaderStyle.Render("Components:"))
	b.WriteString("\n")
	b.WriteString(fmt.Sprintf("  • Foundation: %d core components (ROCm, PyTorch, etc.)\n", len(FoundationComponents)))
	b.WriteString(fmt.Sprintf("  • Core: %d ML stack components (Flash Attention, etc.)\n", len(CoreComponents)))
	b.WriteString(fmt.Sprintf("  • Extensions: %d advanced features (Megatron-LM, vLLM, etc.)\n", len(ExtensionComponents)))
	b.WriteString(fmt.Sprintf("  • Environment: %d setup utilities\n", len(EnvironmentComponents)))
	b.WriteString(fmt.Sprintf("  • Verification: %d diagnostic tools\n", len(VerificationComponents)))
	b.WriteString("\n")

	// Installation info
	b.WriteString(warningStyle.Render("  ⚠️  Estimated installation time: 30-120 minutes"))
	b.WriteString("\n")
	b.WriteString(warningStyle.Render("  ⚠️  Requires sudo privileges for system-level installations"))
	b.WriteString("\n\n")

	b.WriteString(helpStyle.Render("  Press Enter to begin hardware detection • Press q to quit"))

	return b.String()
}

// hardwareDetectView shows enhanced hardware detection screen
func (m Model) hardwareDetectView() string {
	var b strings.Builder

	b.WriteString("\n")
	b.WriteString(titleStyle.Render(" Hardware Detection "))
	b.WriteString("\n\n")

	b.WriteString(fmt.Sprintf("  %s Detecting AMD GPUs...\n\n", m.spinner.View()))

	// Detection progress indicators
	b.WriteString("  Scanning system components:\n")
	b.WriteString("    " + m.spinner.View() + " GPU detection...\n")
	b.WriteString("    " + m.spinner.View() + " Driver validation...\n")
	b.WriteString("    " + m.spinner.View() + " System compatibility check...\n\n")

	// Show hardware information if available
	if m.hardwareDetected {
		b.WriteString(hardwareHeaderStyle.Render("Detection Results:"))
		b.WriteString("\n")

		// GPU Information
		if m.gpuInfo.Model != "" {
			b.WriteString(gpuInfoStyle.Render(fmt.Sprintf("  🎯 GPU: %s\n", m.gpuInfo.Model)))
		}

		// System Information
		if m.systemInfo.Distribution != "" {
			b.WriteString(gpuInfoStyle.Render(fmt.Sprintf("  💻 System: %s\n", m.systemInfo.Distribution)))
		}

		// Memory Information
		if fmt.Sprintf("%.1f GB", m.systemInfo.Memory.TotalGB) != "" {
			b.WriteString(gpuInfoStyle.Render(fmt.Sprintf("  🧠 Memory: %s\n", fmt.Sprintf("%.1f GB", m.systemInfo.Memory.TotalGB))))
		}

		// Storage Information
		if "Available" != "" {
			b.WriteString(gpuInfoStyle.Render(fmt.Sprintf("  💾 Storage: %s\n", "Available")))
		}

		b.WriteString("\n")

		// Recommendations
		if len(m.hardwareRecommendations) > 0 {
			b.WriteString(subheaderStyle.Render("Hardware Recommendations:"))
			b.WriteString("\n")
			for _, rec := range m.hardwareRecommendations {
				b.WriteString(fmt.Sprintf("  💡 %s\n", rec))
			}
			b.WriteString("\n")
		}

		b.WriteString(helpStyle.Render("  Press Enter to continue to pre-flight checks • Press r to re-run detection"))
	} else {
		// Waiting for detection
		b.WriteString(dimStyle.Render("  Waiting for detection results..."))
		b.WriteString("\n")
		b.WriteString(helpStyle.Render("  Press Enter to retry detection • Press r to re-scan"))
	}

	return b.String()
}

// preflightCheckView shows pre-flight check results
func (m Model) preflightCheckView() string {
	var b strings.Builder

	b.WriteString("\n")
	b.WriteString(titleStyle.Render(" Pre-Flight Checks "))
	b.WriteString("\n\n")

	b.WriteString(fmt.Sprintf("  %s Running system validation...\n\n", m.spinner.View()))

	// Show check results if available
	if m.preflightResults != nil {
		b.WriteString(subheaderStyle.Render("Check Results:"))
		b.WriteString("\n")

		// Summary statistics
		b.WriteString(fmt.Sprintf("  Passed: %d | Failed: %d | Warnings: %d\n",
			m.preflightResults.PassedCount, m.preflightResults.FailedCount, m.preflightResults.WarningCount))
		b.WriteString("\n")

		// Individual checks
		for _, check := range m.preflightResults.Checks {
			var statusStyle lipgloss.Style
			var statusIcon string

			switch check.Status {
			case "passed":
				statusStyle = checkPassedStyle
				statusIcon = "✅"
			case "failed":
				statusStyle = checkFailedStyle
				statusIcon = "❌"
			case "warning":
				statusStyle = checkWarningStyle
				statusIcon = "⚠️"
			}

			b.WriteString(fmt.Sprintf("  %s %s - %s\n", statusIcon, statusStyle.Render(check.Name), check.Message))
		}

		b.WriteString("\n")

		// Auto-fixes available
		if len(m.preflightResults.AutoFixes) > 0 {
			b.WriteString(subheaderStyle.Render("Available Auto-Fixes:"))
			b.WriteString("\n")
			for i, fix := range m.preflightResults.AutoFixes {
				icon := "⚡"
				if fix.RequiresSudo {
					icon = "🔧"
				}
				b.WriteString(fmt.Sprintf("  %s [%d] %s - %s\n", icon, i, fix.Name, fix.Description))
			}
			b.WriteString("\n")
			b.WriteString(helpStyle.Render("  Press f to apply fixes • Enter to continue"))
		} else {
			b.WriteString(successStyle.Render("  ✅ All checks passed! System ready for installation.\n\n"))
			b.WriteString(helpStyle.Render("  Press Enter to continue to component selection"))
		}
	} else {
		// Waiting for checks
		b.WriteString(dimStyle.Render("  Running pre-flight checks..."))
		b.WriteString("\n")
		b.WriteString(helpStyle.Render("  Press Enter to retry checks"))
	}

	return b.String()
}

// configurationView shows configuration management interface with safe access
func (m Model) configurationView() string {
	var b strings.Builder

	b.WriteString("\n")
	b.WriteString(titleStyle.Render(" Configuration Management "))
	b.WriteString("\n\n")

	b.WriteString(subheaderStyle.Render("Current Configuration:"))
	b.WriteString("\n")

	// Installation settings with defensive checks
	b.WriteString(configSectionStyle.Render("Installation Settings:"))
	b.WriteString("\n")

	if m.config != nil {
		if m.config.InstallPath != "" {
			b.WriteString(fmt.Sprintf("  Installation Path: %s\n", configValueStyle.Render(m.config.InstallPath)))
		} else {
			b.WriteString(fmt.Sprintf("  Installation Path: %s\n", configValueStyle.Render("/opt/mlstack")))
		}

		if m.config.LogDir != "" {
			b.WriteString(fmt.Sprintf("  Log Directory: %s\n", configValueStyle.Render(m.config.LogDir)))
		} else {
			b.WriteString(fmt.Sprintf("  Log Directory: %s\n", configValueStyle.Render("~/.mlstack/logs")))
		}

		if m.config.EnvFile != "" {
			b.WriteString(fmt.Sprintf("  Environment File: %s\n", configValueStyle.Render(m.config.EnvFile)))
		} else {
			b.WriteString(fmt.Sprintf("  Environment File: %s\n", configValueStyle.Render("~/.mlstack_env")))
		}
	} else {
		b.WriteString(fmt.Sprintf("  Installation Path: %s\n", configValueStyle.Render("/opt/mlstack")))
		b.WriteString(fmt.Sprintf("  Log Directory: %s\n", configValueStyle.Render("~/.mlstack/logs")))
		b.WriteString(fmt.Sprintf("  Environment File: %s\n", configValueStyle.Render("~/.mlstack_env")))
	}
	b.WriteString("\n")

	// Component preferences with safe method calls
	b.WriteString(configSectionStyle.Render("Component Preferences:"))
	b.WriteString("\n")

	selected := m.GetSelectedComponents()
	if selected == nil {
		selected = []Component{}
	}

	b.WriteString(fmt.Sprintf("  Selected Components: %d/%d\n", len(selected), len(m.components)))
	b.WriteString(fmt.Sprintf("  Required Components: %d\n", len(m.GetRequiredComponents())))
	b.WriteString("\n")

	// Performance settings with safe access
	b.WriteString(configSectionStyle.Render("Performance Settings:"))
	b.WriteString("\n")
	if m.config != nil && len(m.config.PerformanceSettings) > 0 {
		for key, value := range m.config.PerformanceSettings {
			b.WriteString(fmt.Sprintf("  %s: %s\n", key, configValueStyle.Render(value)))
		}
	} else {
		b.WriteString(fmt.Sprintf("  %s\n", configValueStyle.Render("Default settings")))
	}
	b.WriteString("\n")

	// Network settings with safe access
	b.WriteString(configSectionStyle.Render("Network Settings:"))
	b.WriteString("\n")
	if m.config != nil {
		if m.config.NetworkSettings.Proxy.HTTPProxy != "" {
			b.WriteString(fmt.Sprintf("  HTTP Proxy: %s\n", configValueStyle.Render(m.config.NetworkSettings.Proxy.HTTPProxy)))
		} else {
			b.WriteString(fmt.Sprintf("  HTTP Proxy: %s\n", configValueStyle.Render("Not configured")))
		}

		if m.config.NetworkSettings.Proxy.HTTPSProxy != "" {
			b.WriteString(fmt.Sprintf("  HTTPS Proxy: %s\n", configValueStyle.Render(m.config.NetworkSettings.Proxy.HTTPSProxy)))
		} else {
			b.WriteString(fmt.Sprintf("  HTTPS Proxy: %s\n", configValueStyle.Render("Not configured")))
		}

		if m.config.NetworkSettings.TimeoutSeconds > 0 {
			b.WriteString(fmt.Sprintf("  Timeout: %ds\n", m.config.NetworkSettings.TimeoutSeconds))
		} else {
			b.WriteString(fmt.Sprintf("  Timeout: %s\n", configValueStyle.Render("Default (60s)")))
		}
	} else {
		b.WriteString(fmt.Sprintf("  HTTP Proxy: %s\n", configValueStyle.Render("Not configured")))
		b.WriteString(fmt.Sprintf("  HTTPS Proxy: %s\n", configValueStyle.Render("Not configured")))
		b.WriteString(fmt.Sprintf("  Timeout: %s\n", configValueStyle.Render("Default (60s)")))
	}
	b.WriteString("\n")

	// Security settings with safe access
	b.WriteString(configSectionStyle.Render("Security Settings:"))
	b.WriteString("\n")
	if m.config != nil {
		b.WriteString(fmt.Sprintf("  Script Validation: %v\n", m.config.SecuritySettings.ScriptValidation))
		b.WriteString(fmt.Sprintf("  Require Sudo: %v\n", m.config.SecuritySettings.RequireSudo))
		b.WriteString(fmt.Sprintf("  Backup System: %v\n", m.config.SecuritySettings.BackupSystem))
	} else {
		b.WriteString(fmt.Sprintf("  Script Validation: %s\n", configValueStyle.Render("Enabled")))
		b.WriteString(fmt.Sprintf("  Require Sudo: %s\n", configValueStyle.Render("Enabled")))
		b.WriteString(fmt.Sprintf("  Backup System: %s\n", configValueStyle.Render("Enabled")))
	}
	b.WriteString("\n")

	// Configuration actions
	b.WriteString(subheaderStyle.Render("Configuration Actions:"))
	b.WriteString("\n")
	b.WriteString(recoveryActionStyle.Render("  s - Save current configuration"))
	b.WriteString("\n")
	b.WriteString(recoveryActionStyle.Render("  r - Reset to defaults"))
	b.WriteString("\n")
	b.WriteString(recoveryActionStyle.Render("  e - Export configuration"))
	b.WriteString("\n")
	b.WriteString(recoveryActionStyle.Render("  i - Import configuration"))
	b.WriteString("\n\n")

	b.WriteString(helpStyle.Render("  Press Enter to continue to installation • Press c to modify configuration"))

	return b.String()
}

// componentSelectView shows enhanced component selection screen
func (m Model) componentSelectView() string {
	var b strings.Builder

	b.WriteString("\n")
	b.WriteString(titleStyle.Render(" Select Components "))
	b.WriteString("\n\n")

	// Defensive check for components initialization
	if m.components == nil {
		b.WriteString(errorStyle.Render("  ❌ Error: Components not initialized\n"))
		b.WriteString(helpStyle.Render("  Press r to restart installer • Press q to quit"))
		return b.String()
	}

	// Show detected hardware with safe access
	b.WriteString(headerStyle.Render("Detected Hardware:"))
	b.WriteString("\n")
	if m.gpuInfo.Model != "" {
		b.WriteString(fmt.Sprintf("  GPU:    %s\n", m.gpuInfo.Model))
	} else {
		b.WriteString("  GPU:    Not detected\n")
	}
	if m.systemInfo.Distribution != "" {
		b.WriteString(fmt.Sprintf("  System: %s\n", m.systemInfo.Distribution))
	} else {
		b.WriteString("  System: Not detected\n")
	}
	if m.systemInfo.Memory.TotalGB > 0 {
		b.WriteString(fmt.Sprintf("  Memory: %.1f GB\n", m.systemInfo.Memory.TotalGB))
	} else {
		b.WriteString("  Memory: Not detected\n")
	}
	b.WriteString("\n")

	// Show pre-flight check results summary with safe access
	if m.preflightResults != nil {
		b.WriteString(headerStyle.Render("Pre-Flight Status:"))
		b.WriteString("\n")
		b.WriteString(fmt.Sprintf("  Passed: %d | Failed: %d | Warnings: %d\n",
			m.preflightResults.PassedCount, m.preflightResults.FailedCount, m.preflightResults.WarningCount))
		if m.preflightResults.FailedCount > 0 {
			b.WriteString(warningStyle.Render("  ⚠️  Some critical checks failed - consider applying fixes\n"))
		}
		b.WriteString("\n")
	}

	// Show statistics with safe method calls
	selected := m.GetSelectedComponents()
	required := m.GetRequiredComponents()
	totalSize := m.GetTotalSize()

	// Additional safety checks
	if selected == nil {
		selected = []Component{}
	}
	if required == nil {
		required = []Component{}
	}

	b.WriteString(subheaderStyle.Render("Selection Summary:"))
	b.WriteString("\n")
	b.WriteString(fmt.Sprintf("  Components:  %d selected / %d total\n", len(selected), len(m.components)))
	b.WriteString(fmt.Sprintf("  Required:    %d required components\n", len(required)))
	b.WriteString(fmt.Sprintf("  Size:        ~%.1f GB total\n", float64(totalSize)/1024/1024))

	// Safe time estimation
	totalTime := m.GetTotalTime()
	if totalTime == "" {
		totalTime = "Unknown"
	}
	b.WriteString(fmt.Sprintf("  Time:        ~%s estimated\n", totalTime))
	b.WriteString("\n")

	// Component categories
	categories := []string{"foundation", "core", "extension", "environment", "verification"}
	categoryNames := map[string]string{
		"foundation":   "Foundation Components",
		"core":         "Core ML Stack",
		"extension":    "Advanced Features",
		"environment":  "Environment Setup",
		"verification": "Verification Tools",
	}

	for _, cat := range categories {
		components := GetComponentsByCategory(cat)
		if len(components) > 0 {
			b.WriteString(headerStyle.Render(categoryNames[cat] + ":"))
			b.WriteString("\n")

			for _, c := range components {
				checkbox := "☐"
				style := checkboxStyle
				if c.Selected {
					checkbox = "☑"
					style = selectedCheckboxStyle
				}

				componentStyle := componentStyle
				if c.Selected {
					componentStyle = selectedComponentStyle
				}

				b.WriteString(fmt.Sprintf("  %s %s\n", style.Render(checkbox), componentStyle.Render(c.Name)))
				b.WriteString(dimStyle.Render(fmt.Sprintf("      %s | %s\n", c.Description, c.Estimate)))
			}
			b.WriteString("\n")
		}
	}

	// Navigation help
	b.WriteString(helpStyle.Render("  Navigation:"))
	b.WriteString("\n")
	b.WriteString("  ↑/↓: Navigate components • Space: Toggle selection")
	b.WriteString("\n")
	b.WriteString("  ←/→: Switch categories • Tab: Next category • Enter: Continue")
	b.WriteString("\n")
	b.WriteString("  q: Quit installer")

	return b.String()
}

// confirmView shows enhanced confirmation screen
func (m Model) confirmView() string {
	var b strings.Builder

	b.WriteString("\n")
	b.WriteString(titleStyle.Render(" Confirm Installation "))
	b.WriteString("\n\n")

	// Defensive checks to prevent nil pointer dereference
	if m.components == nil {
		b.WriteString(errorStyle.Render("  ❌ Error: Components not initialized\n"))
		b.WriteString(helpStyle.Render("  Press r to restart installer • Press q to quit"))
		return b.String()
	}

	// Selected components summary with safe access
	selected := m.GetSelectedComponents()
	required := m.GetRequiredComponents()
	totalSize := m.GetTotalSize()

	// Additional safety checks
	if selected == nil {
		selected = []Component{}
	}
	if required == nil {
		required = []Component{}
	}

	b.WriteString(subheaderStyle.Render("Installation Summary:"))
	b.WriteString("\n")
	b.WriteString(fmt.Sprintf("  Components: %d selected (%d required, %d optional)\n",
		len(selected), len(required), len(selected)-len(required)))
	b.WriteString(fmt.Sprintf("  Size:       ~%.1f GB total disk space required\n",
		float64(totalSize)/1024/1024))

	// Safe time estimation
	totalTime := m.GetTotalTime()
	if totalTime == "" {
		totalTime = "Unknown"
	}
	b.WriteString(fmt.Sprintf("  Estimated:  ~%s total installation time\n", totalTime))
	b.WriteString("\n")

	// Selected components list with safe iteration
	b.WriteString(headerStyle.Render("Selected Components:"))
	b.WriteString("\n")

	if len(selected) == 0 {
		b.WriteString(warningStyle.Render("  ⚠️  No components selected\n"))
	} else {
		for _, c := range selected {
			// Defensive check for component validity
			if c.Name == "" {
				c.Name = "Unknown Component"
			}
			if c.Estimate == "" {
				c.Estimate = "Unknown time"
			}

			if c.Required {
				b.WriteString(fmt.Sprintf("  ⭐ %s (Required)\n", c.Name))
			} else {
				b.WriteString(fmt.Sprintf("  • %s (Optional)\n", c.Name))
			}
			if c.Size > 0 {
				b.WriteString(fmt.Sprintf("    Size: %.1fGB | Time: %s\n", float64(c.Size)/1024/1024, c.Estimate))
			} else {
				b.WriteString(fmt.Sprintf("    Time: %s\n", c.Estimate))
			}
		}
	}
	b.WriteString("\n")

	// System status with defensive checks
	b.WriteString(headerStyle.Render("System Status:"))
	b.WriteString("\n")

	// Hardware status with safe field access
	if m.hardwareDetected && m.gpuInfo.Model != "" {
		b.WriteString(successStyle.Render("  ✅ Hardware detected and compatible\n"))
	} else {
		b.WriteString(warningStyle.Render("  ⚠️  Hardware detection incomplete\n"))
	}

	// Pre-flight status with nil check
	if m.preflightResults != nil {
		if m.preflightResults.FailedCount == 0 {
			b.WriteString(successStyle.Render("  ✅ Pre-flight checks passed\n"))
		} else {
			b.WriteString(warningStyle.Render(fmt.Sprintf("  ⚠️  %d pre-flight checks failed\n", m.preflightResults.FailedCount)))
		}
	} else {
		b.WriteString(warningStyle.Render("  ⚠️  Pre-flight checks not completed\n"))
	}

	// Configuration status check
	if m.configLoaded {
		b.WriteString(successStyle.Render("  ✅ Configuration loaded\n"))
	} else {
		b.WriteString(warningStyle.Render("  ⚠️  Configuration not loaded\n"))
	}

	b.WriteString("\n")

	// Warning about system changes
	b.WriteString(warningStyle.Render("⚠️ System Warning:"))
	b.WriteString("\n")
	b.WriteString("  This installation will modify your system configuration.\n")
	b.WriteString("  Administrative privileges will be required.\n")
	b.WriteString("  Some components may require system restart.\n")
	b.WriteString("  Installation cannot be interrupted once started.\n\n")

	// Configuration snapshot info with safe access
	b.WriteString(infoStyle.Render("  Configuration snapshot will be created at: "))
	if m.config != nil && m.config.LogDir != "" {
		snapshotPath := m.config.LogDir + "/snapshots/"
		b.WriteString(configValueStyle.Render(snapshotPath))
	} else {
		b.WriteString(configValueStyle.Render("~/.mlstack/logs/snapshots/"))
	}
	b.WriteString("\n")

	// Start time
	b.WriteString(infoStyle.Render("  Start time: " + time.Now().Format("2006-01-02 15:04:05")))

	b.WriteString("\n\n")
	b.WriteString(helpStyle.Render("  Press Enter to start installation • Press q to cancel • Press r to restart • Press c to configure"))

	return b.String()
}

// installingView shows enhanced installation progress screen
func (m Model) installingView() string {
	var b strings.Builder

	b.WriteString("\n")
	b.WriteString(titleStyle.Render(" Installing ML Stack "))
	b.WriteString("\n\n")

	// Defensive checks
	selected := m.GetSelectedComponents()
	if selected == nil {
		selected = []Component{}
	}

	if m.currentComponent >= 0 && m.currentComponent < len(selected) {
		current := selected[m.currentComponent]

		// Safe spinner access
		spinnerView := "⠋"
		if m.spinner.View != nil {
			spinnerView = m.spinner.View()
		}

		// Safe component field access
		componentName := current.Name
		if componentName == "" {
			componentName = "Unknown Component"
		}

		scriptName := current.Script
		if scriptName == "" {
			scriptName = "Unknown script"
		}

		estimate := current.Estimate
		if estimate == "" {
			estimate = "Unknown time"
		}

		// Current installation status
		b.WriteString(fmt.Sprintf("  %s Installing %s...\n\n", spinnerView, componentName))
		b.WriteString(dimStyle.Render(fmt.Sprintf("  Script: %s | Estimated: %s\n", scriptName, estimate)))

		// Progress bar with percentage
		b.WriteString("\n")
		b.WriteString(progressStyle.Render("  Progress: "))
		b.WriteString(m.progress.View())
		b.WriteString("\n\n")

		// Completed components
		if m.currentComponent > 0 {
			b.WriteString(successStyle.Render("  Completed Components:\n"))
			for i := 0; i < m.currentComponent; i++ {
				comp := m.GetSelectedComponents()[i]
				progressIcon := "✅"
				if comp.Progress < 1.0 {
					progressIcon = "⏳"
				}
				b.WriteString(fmt.Sprintf("    %s %s (%.1f%%)\n", progressIcon, comp.Name, comp.Progress*100))
			}
			b.WriteString("\n")
		}

		// Queue of upcoming components
		if m.currentComponent+1 < len(selected) {
			b.WriteString(infoStyle.Render("  Next in queue:\n"))
			for i := m.currentComponent + 1; i < len(selected) && i < m.currentComponent+3; i++ {
				comp := selected[i]
				b.WriteString(fmt.Sprintf("    ⏳ %s (%s)\n", comp.Name, comp.Estimate))
			}
			if m.currentComponent+3 < len(selected) {
				b.WriteString(dimStyle.Render(fmt.Sprintf("    ... and %d more components\n", len(selected)-(m.currentComponent+3))))
			}
			b.WriteString("\n")
		}
	}

	// Real-time log (show last few entries)
	if len(m.installLog) > 0 {
		logEntries := m.installLog
		if len(logEntries) > 5 {
			logEntries = logEntries[len(logEntries)-5:]
		}

		b.WriteString(headerStyle.Render("Recent Activity:"))
		b.WriteString("\n")
		for _, entry := range logEntries {
			// Extract just the message part (without timestamp for display)
			parts := strings.SplitN(entry, "]", 3)
			if len(parts) >= 3 {
				msg := strings.TrimSpace(parts[2])
				b.WriteString(dimStyle.Render(fmt.Sprintf("  %s\n", msg)))
			}
		}
	}

	b.WriteString("\n")
	b.WriteString(warningStyle.Render("  ⚠️  Do not interrupt the installation process"))

	return b.String()
}

// completeView shows enhanced completion screen
func (m Model) completeView() string {
	var b strings.Builder

	b.WriteString("\n")
	b.WriteString(titleStyle.Render(" Installation Complete! "))
	b.WriteString("\n\n")

	// Summary statistics
	selected := m.GetSelectedComponents()
	successCount := 0
	for _, comp := range selected {
		if comp.Installed {
			successCount++
		}
	}

	// Overall status
	if len(m.errorLog) == 0 && successCount == len(selected) {
		b.WriteString(successStyle.Render("  ✅ All components installed successfully!\n\n"))
	} else {
		b.WriteString(warningStyle.Render("  ⚠️  Installation completed with warnings:\n\n"))
		if len(m.errorLog) > 0 {
			for _, err := range m.errorLog {
				b.WriteString(errorStyle.Render(fmt.Sprintf("    • %s\n", err)))
			}
		}
		b.WriteString("\n")

		// Summary
		b.WriteString(infoStyle.Render(fmt.Sprintf("  Success: %d/%d components (%.1f%%)\n",
			successCount, len(selected), float64(successCount)/float64(len(selected)*100))))
	}

	// Completion details
	b.WriteString("\n")
	b.WriteString(headerStyle.Render("Installation Summary:"))
	b.WriteString("\n")
	b.WriteString(fmt.Sprintf("  Start time:  %s\n", m.installLog[0]))
	if len(m.installLog) > 0 {
		endTime := m.installLog[len(m.installLog)-1]
		if strings.Contains(endTime, "[COMPLETE]") {
			parts := strings.SplitN(endTime, "]", 3)
			if len(parts) >= 3 {
				endTime = strings.TrimSpace(parts[2])
			}
		}
		b.WriteString(fmt.Sprintf("  End time:    %s\n", endTime))
	}
	b.WriteString(fmt.Sprintf("  Duration:    %s\n", time.Duration(120*time.Minute).Truncate(time.Minute)))
	b.WriteString(fmt.Sprintf("  Components:  %d selected, %d installed\n", len(selected), successCount))
	b.WriteString(fmt.Sprintf("  Errors:      %d encountered\n", len(m.errorLog)))
	b.WriteString("\n")

	// Log location
	b.WriteString(infoStyle.Render("  Installation logs saved to: ~/.mlstack/logs/\n"))
	b.WriteString(infoStyle.Render("  Log file format: component_timestamp.log\n\n"))

	// Next steps
	b.WriteString(headerStyle.Render("Next Steps:"))
	b.WriteString("\n")
	b.WriteString(successStyle.Render("  1. Source environment:\n"))
	b.WriteString("     source ~/.mlstack_env\n\n")
	b.WriteString(successStyle.Render("  2. Verify installation:\n"))
	b.WriteString("     ml-stack-verify\n\n")
	b.WriteString(successStyle.Render("  3. Test PyTorch:\n"))
	b.WriteString("     python -c 'import torch; print(torch.cuda.is_available())'\n\n")
	b.WriteString(successStyle.Render("  4. Check GPU status:\n"))
	b.WriteString("     roc-smi\n\n")

	b.WriteString(helpStyle.Render("  Press Enter to exit installer"))

	return b.String()
}

// recoveryView shows error recovery and rollback options
func (m Model) recoveryView() string {
	var b strings.Builder

	b.WriteString("\n")
	b.WriteString(titleStyle.Render(" Error Recovery "))
	b.WriteString("\n\n")

	if len(m.recoveryOptions) > 0 {
		b.WriteString(recoveryHeaderStyle.Render("Recovery Options:"))
		b.WriteString("\n")

		for i, option := range m.recoveryOptions {
			b.WriteString(fmt.Sprintf("  [%d] %s\n", i, option.Name))
			b.WriteString(dimStyle.Render(fmt.Sprintf("      %s\n", option.Description)))
		}
		b.WriteString("\n")

		// Available snapshots
		if len(m.availableSnapshots) > 0 {
			b.WriteString(recoveryHeaderStyle.Render("Available Snapshots:"))
			b.WriteString("\n")
			for _, snapshot := range m.availableSnapshots {
				b.WriteString(fmt.Sprintf("  📸 %s (%s)\n", snapshot.ID, snapshot.Timestamp.Format("2006-01-02 15:04:05")))
			}
			b.WriteString("\n")
			b.WriteString(helpStyle.Render("  Press s to restore snapshot • Press r to rollback to checkpoint"))
		}

		b.WriteString("\n")
		b.WriteString(helpStyle.Render("  Press Enter to retry installation • Press q to cancel"))
	} else {
		b.WriteString(dimStyle.Render("  No recovery options available"))
		b.WriteString("\n")
		b.WriteString(errorStyle.Render("  Installation encountered critical errors"))
		b.WriteString("\n")
		b.WriteString(helpStyle.Render("  Press r to restart installation • Press q to cancel"))
	}

	return b.String()
}
*/