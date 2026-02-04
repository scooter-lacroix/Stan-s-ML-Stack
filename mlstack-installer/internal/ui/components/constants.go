// internal/ui/components/constants.go
package components

// Enhanced AMD Brand Color Constants
const (
	// Primary AMD Brand Colors
	AMDRed      = "#ED1C24" // Primary AMD brand red
	AMDRedDark  = "#C41018" // Darker variant for depth
	AMDRedLight = "#FF4757" // Lighter variant for highlights
	AMDDarkRed  = "#8B0000" // Very dark red for accents

	// Secondary AMD Colors
	AMDOrange      = "#FF6B35" // AMD orange accent
	AMDOrangeDark  = "#E55100" // Darker orange
	AMDOrangeLight = "#FFAB40" // Lighter orange
	AMDYellow      = "#FFD600" // AMD yellow accent

	// Professional Grays
	AMDGray        = "#666666" // Standard gray
	AMDGrayDark    = "#404040" // Dark gray
	AMDGrayDarker  = "#2A2A2A" // Very dark gray
	AMDGrayLight   = "#909090" // Light gray
	AMDGrayLighter = "#BDBDBD" // Very light gray
	AMDSilver      = "#C0C0C0" // Silver accent

	// High Contrast Colors
	AMDBlack     = "#1A1A1A" // Professional black
	AMDBlackDark = "#0D0D0D" // Very dark black
	AMDWhite     = "#FFFFFF" // Pure white
	AMDOffWhite  = "#F5F5F5" // Off-white for reduced eye strain

	// Status Colors (AMD-themed)
	AMDSuccess     = "#00C853" // Success green
	AMDSuccessDark = "#00A846" // Darker success
	AMDInfo        = "#2196F3" // Information blue
	AMDInfoDark    = "#1976D2" // Darker info
	AMDError       = "#FF5252" // Error red
	AMDErrorDark   = "#D32F2F" // Darker error
	AMDWarning     = "#FFC107" // Warning yellow
	AMDWarningDark = "#FFA000" // Darker warning

	// AMD-specific ML/AI Colors
	AMDAccentBlue   = "#0078D4" // Microsoft-style blue for tech appeal
	AMDAccentPurple = "#6A1B9A" // Purple for AI/ML branding
	AMDAccentCyan   = "#00ACC1" // Cyan for data processing
	AMDAccentGreen  = "#00897B" // Green for optimization

	// Gradient Definitions
	AMDGradientStart = AMDRed
	AMDGradientEnd   = AMDOrange
	AMDGradientMid   = "#FF4757"
)

// AMD Typography and Spacing Constants
const (
	// Professional Spacing (based on 4px grid)
	AMDSpaceXSmall  = 4  // 0.25rem
	AMDSpaceSmall   = 8  // 0.5rem
	AMDSpaceMedium  = 16 // 1rem
	AMDSpaceLarge   = 24 // 1.5rem
	AMDSpaceXLarge  = 32 // 2rem
	AMDSpaceXXLarge = 48 // 3rem

	// Border Radius
	AMDBorderSmall  = 4
	AMDBorderMedium = 8
	AMDBorderLarge  = 12
	AMDBorderXLarge = 16

	// Component Heights
	AMDHeightSmall  = 32
	AMDHeightMedium = 40
	AMDHeightLarge  = 48
	AMDHeightXLarge = 56

	// Font Sizes (relative units)
	AMDFontXSmall   = "0.75rem"
	AMDFontSmall    = "0.875rem"
	AMDFontMedium   = "1rem"
	AMDFontLarge    = "1.125rem"
	AMDFontXLarge   = "1.25rem"
	AMDFontXXLarge  = "1.5rem"
	AMDFontXXXLarge = "2rem"
)

// AMD Component Patterns
const (
	// Border Patterns
	AMDBorderSolid  = "solid"
	AMDBorderDashed = "dashed"
	AMDBorderDotted = "dotted"

	// Shadow Effects (text-based representation)
	AMDShadowSoft   = "░"
	AMDShadowMedium = "▒"
	AMDShadowHard   = "▓"

	// Progress Bar Characters
	AMDProgressFull    = "█"
	AMDProgressEmpty   = "░"
	AMDProgressHalf    = "▄"
	AMDProgressQuarter = "▂"
)

// AMD Animation Timing (in milliseconds)
const (
	AMDAnimFast   = 150
	AMDAnimNormal = 300
	AMDAnimSlow   = 500
	AMDAnimSlower = 750
)

// AMD Accessibility Constants
const (
	// Minimum contrast ratios
	AMDContrastNormal   = 4.5 // WCAG AA standard
	AMDContrastLarge    = 3.0 // WCAG AA for large text
	AMDContrastEnhanced = 7.0 // WCAG AAA enhanced
)

// AMD Component State Mapping
var AMDStateColors = map[string]string{
	"default":  AMDGray,
	"hover":    AMDOrange,
	"focus":    AMDRed,
	"active":   AMDRedDark,
	"disabled": AMDGrayLight,
	"success":  AMDSuccess,
	"warning":  AMDWarning,
	"error":    AMDError,
	"info":     AMDInfo,
}

// AMD Component Level Colors (for hierarchical visual design)
var AMDLevelColors = map[string]string{
	"level1": AMDRed,       // Primary actions, titles
	"level2": AMDOrange,    // Secondary actions, subtitles
	"level3": AMDGrayDark,  // Tertiary content, descriptions
	"level4": AMDGray,      // Supporting text, metadata
	"level5": AMDGrayLight, // Disabled content, placeholders
}

// AMD-themed Status Indicators
var AMDStatusIndicators = map[string]string{
	"ready":   "✓",
	"loading": "⟳",
	"error":   "✗",
	"warning": "⚠",
	"info":    "ℹ",
	"success": "✓",
	"pending": "⏳",
	"running": "▶",
	"paused":  "⏸",
	"stopped": "⏹",
}

// AMD Hardware-specific Icons
var AMDHardwareIcons = map[string]string{
	"gpu":     "🔴",
	"cpu":     "🧠",
	"memory":  "💾",
	"storage": "💿",
	"network": "🌐",
	"thermal": "🌡",
	"power":   "⚡",
	"system":  "🖥",
}

// GetAMDColor retrieves an AMD color by name with fallback
func GetAMDColor(name string) string {
	if color, exists := AMDStateColors[name]; exists {
		return color
	}
	if color, exists := AMDLevelColors[name]; exists {
		return color
	}

	// Default fallbacks based on color category
	switch name {
	case "red":
		return AMDRed
	case "orange":
		return AMDOrange
	case "gray":
		return AMDGray
	case "black":
		return AMDBlack
	case "white":
		return AMDWhite
	case "success":
		return AMDSuccess
	case "error":
		return AMDError
	case "warning":
		return AMDWarning
	case "info":
		return AMDInfo
	default:
		return AMDGray
	}
}

// GetAMDGradient returns a gradient definition for AMD theming
func GetAMDGradient() (string, string, string) {
	return AMDGradientStart, AMDGradientMid, AMDGradientEnd
}

// GetAMDSpacing returns spacing value based on size identifier
func GetAMDSpacing(size string) int {
	switch size {
	case "xs":
		return AMDSpaceXSmall
	case "sm":
		return AMDSpaceSmall
	case "md":
		return AMDSpaceMedium
	case "lg":
		return AMDSpaceLarge
	case "xl":
		return AMDSpaceXLarge
	case "xxl":
		return AMDSpaceXXLarge
	default:
		return AMDSpaceMedium
	}
}

// IsAMDColor validates if a color string is a defined AMD color
func IsAMDColor(color string) bool {
	amdColors := []string{
		AMDRed, AMDRedDark, AMDRedLight, AMDDarkRed,
		AMDOrange, AMDOrangeDark, AMDOrangeLight, AMDYellow,
		AMDGray, AMDGrayDark, AMDGrayDarker, AMDGrayLight, AMDGrayLighter, AMDSilver,
		AMDBlack, AMDBlackDark, AMDWhite, AMDOffWhite,
		AMDSuccess, AMDSuccessDark, AMDInfo, AMDInfoDark,
		AMDError, AMDErrorDark, AMDWarning, AMDWarningDark,
		AMDAccentBlue, AMDAccentPurple, AMDAccentCyan, AMDAccentGreen,
	}

	for _, amdColor := range amdColors {
		if color == amdColor {
			return true
		}
	}
	return false
}
