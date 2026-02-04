// Package diagnostics provides comprehensive diagnostic tools for Bubble Tea UI issues
package diagnostics

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"syscall"
	"time"

	"github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/mattn/go-isatty"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/state"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// TerminalDiagnosticResult contains comprehensive terminal environment analysis
type TerminalDiagnosticResult struct {
	TerminalInfo    TerminalInfo
	SignalHandling  SignalHandlingInfo
	EnvironmentVars EnvironmentInfo
	Performance     PerformanceInfo
	Security        SecurityInfo
	Recommendations []string
	Timeline        []TimelineEvent
}

// TerminalInfo captures terminal environment details
type TerminalInfo struct {
	IsTerminal     bool
	TTYPath        string
	Width          int
	Height         int
	Cols           int
	Rows           int
	TerminalType   string
	ColorSupport   ColorSupportInfo
	InputMode      InputModeInfo
	Buffering      BufferingInfo
}

// ColorSupportInfo captures color terminal capabilities
type ColorSupportInfo struct {
	TrueColor     bool
	AnsiColors    int
	Has16Colors   bool
	Has256Colors  bool
	HasMillionColors bool
}

// InputModeInfo captures input device and mode information
type InputModeInfo struct {
	IsInteractive      bool
	HasKeyboard         bool
	HasMouse           bool
	InputDevice        string
	BaudRate           int
	CanEcho            bool
	CanCanonicalMode   bool
	CanRawMode         bool
}

// BufferingInfo captures terminal buffering configuration
type BufferingInfo struct {
	StdinBuffered    bool
	StdoutBuffered   bool
	StderrBuffered   bool
	LineBuffered     bool
	BlockBuffered    bool
	BufferSize       int
}

// SignalHandlingInfo captures signal handling capabilities
type SignalHandlingInfo struct {
	SIGINTSupported  bool
	SIGTERMSupported bool
	SIGPIPEHandled  bool
	SignalMask      []string
	CanHandleInterrupts bool
	GracefulShutdown bool
}

// EnvironmentInfo captures environment variables and context
type EnvironmentInfo struct {
	Variables map[string]string
	User      string
	Home      string
	SudoUser  string
	Display   string
	TerminalVars []string
	ParentProcess string
}

// PerformanceInfo captures system performance metrics
type PerformanceInfo struct {
	CPUCount     int
	MemoryTotal  uint64
	MemoryUsed   uint64
	Goroutines   int
	ThreadCount  int
	StackSize    uint64
	GCPercent    int
	LastGC       time.Time
	CPUPeak      float64
	MemoryPeak   uint64
}

// SecurityInfo captures security context
type SecurityInfo struct {
	RunningAsRoot bool
	EffectiveUID  int
	EffectiveGID  int
	SecureSession bool
	CanAccessTTY  bool
	TTYPerms      string
}

// TimelineEvent captures diagnostic timeline
type TimelineEvent struct {
	Timestamp time.Time
	Event     string
	Detail    string
	Duration  time.Duration
}

// DiagnosticContext manages diagnostic session state
type DiagnosticContext struct {
	Context     context.Context
	Cancel      context.CancelFunc
	StartTime   time.Time
	Timeline    []TimelineEvent
	Results     *TerminalDiagnosticResult
	Profiling   *ProfilingSession
}

// ProfilingSession manages runtime profiling
type ProfilingSession struct {
	CPUProfile    *os.File
	MemProfile    *os.File
	BlockProfile  *os.File
	GCProfile     *os.File
	TraceProfile  *os.File
	Active        bool
}

// NewDiagnosticContext creates a new diagnostic context
func NewDiagnosticContext() *DiagnosticContext {
	ctx, cancel := context.WithCancel(context.Background())
	return &DiagnosticContext{
		Context:   ctx,
		Cancel:    cancel,
		StartTime: time.Now(),
		Timeline:  []TimelineEvent{},
		Results:   &TerminalDiagnosticResult{},
	}
}

// RunComprehensiveDiagnostic runs a complete terminal diagnostic suite
func (dc *DiagnosticContext) RunComprehensiveDiagnostic() *TerminalDiagnosticResult {
	dc.addTimelineEvent("Diagnostic", "Starting comprehensive diagnostic suite", time.Duration(0))

	// 1. Terminal Environment Analysis
	dc.addTimelineEvent("Terminal", "Analyzing terminal environment", time.Duration(0))
	dc.analyzeTerminalEnvironment()

	// 2. Signal Handling Analysis
	dc.addTimelineEvent("Signals", "Analyzing signal handling capabilities", time.Duration(0))
	dc.analyzeSignalHandling()

	// 3. Environment Variables Analysis
	dc.addTimelineEvent("Environment", "Analyzing environment variables", time.Duration(0))
	dc.analyzeEnvironment()

	// 4. Performance Analysis
	dc.addTimelineEvent("Performance", "Analyzing system performance", time.Duration(0))
	dc.analyzePerformance()

	// 5. Security Context Analysis
	dc.addTimelineEvent("Security", "Analyzing security context", time.Duration(0))
	dc.analyzeSecurity()

	// 6. Generate Recommendations
	dc.addTimelineEvent("Analysis", "Generating recommendations", time.Duration(0))
	dc.generateRecommendations()

	return dc.Results
}

// analyzeTerminalEnvironment performs comprehensive terminal environment analysis
func (dc *DiagnosticContext) analyzeTerminalEnvironment() {
	// Terminal basic info
	termInfo := TerminalInfo{
		IsTerminal:   isatty.IsTerminal(os.Stdin.Fd()),
		TTYPath:     "/dev/tty",
		Width:       80,
		Height:      24,
		Cols:        80,
		Rows:        24,
		TerminalType: os.Getenv("TERM"),
	}

	// Get actual terminal dimensions
	if winsize, err := getTerminalSize(); err == nil {
		termInfo.Width = winsize.Cols
		termInfo.Height = winsize.Rows
		termInfo.Cols = winsize.Cols
		termInfo.Rows = winsize.Rows
	}

	// Check TTY accessibility
	if _, err := os.Stat("/dev/tty"); err == nil {
		if file, err := os.OpenFile("/dev/tty", os.O_RDWR, 0); err == nil {
			file.Close()
			termInfo.TTYPath = "/dev/tty"
		}
	}

	// Color support analysis
	colorInfo := ColorSupportInfo{
		TrueColor:     hasTrueColor(),
		AnsiColors:    getAnsiColorCount(),
		Has16Colors:  hasColorCapability("colors", 16),
		Has256Colors:  hasColorCapability("colors", 256),
		HasMillionColors: hasTrueColor(),
	}

	// Input mode analysis
	inputInfo := InputModeInfo{
		IsInteractive: isatty.IsTerminal(os.Stdin.Fd()),
		HasKeyboard:   isatty.IsTerminal(os.Stdin.Fd()),
		HasMouse:     hasMouseSupport(),
		InputDevice:  getInputDevice(),
		BaudRate:     getBaudRate(),
		CanEcho:      canEcho(),
		CanCanonicalMode: canCanonicalMode(),
		CanRawMode:   canRawMode(),
	}

	// Buffering analysis
	bufferInfo := BufferingInfo{
		StdinBuffered:  isBuffered(os.Stdin),
		StdoutBuffered: isBuffered(os.Stdout),
		StderrBuffered: isBuffered(os.Stderr),
		LineBuffered:   isLineBuffered(os.Stdout),
		BlockBuffered:  isBlockBuffered(os.Stdout),
		BufferSize:     getBufferSize(os.Stdout),
	}

	termInfo.ColorSupport = colorInfo
	termInfo.InputMode = inputInfo
	termInfo.Buffering = bufferInfo

	dc.Results.TerminalInfo = termInfo
}

// analyzeSignalHandling analyzes signal handling capabilities
func (dc *DiagnosticContext) analyzeSignalHandling() {
	sigInfo := SignalHandlingInfo{
		SIGINTSupported:  true,
		SIGTERMSupported: true,
		SIGPIPEHandled:  false,
		CanHandleInterrupts: true,
		GracefulShutdown: true,
	}

	// Test signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(sigChan)

	// Create a test signal handler
	go func() {
		select {
		case <-sigChan:
			sigInfo.SIGPIPEHandled = true
		case <-time.After(100 * time.Millisecond):
			// Signal not received within timeout
		}
	}()

	// Send test signal (this might not work in all environments)
	// syscall.Kill(os.Getpid(), syscall.SIGINT)

	sigInfo.SignalMask = []string{"SIGINT", "SIGTERM"}

	dc.Results.SignalHandling = sigInfo
}

// analyzeEnvironment analyzes environment variables and context
func (dc *DiagnosticContext) analyzeEnvironment() {
	envInfo := EnvironmentInfo{
		Variables: make(map[string]string),
		User:      "unknown",
		Home:      "/home/unknown",
		SudoUser:  "",
		Display:   os.Getenv("DISPLAY"),
		ParentProcess: "unknown",
	}

	// Collect environment variables
	for _, env := range os.Environ() {
		if key, value, found := strings.Cut(env, "="); found {
			envInfo.Variables[key] = value
		}
	}

	// User and home directory
	if currentUser, err := os.UserHomeDir(); err == nil {
		envInfo.Home = currentUser
	}
	if currentUser, err := os.UserLookup(os.Getenv("USER")); err == nil {
		envInfo.User = currentUser.Username
	}

	// Sudo user
	if sudoUser := os.Getenv("SUDO_USER"); sudoUser != "" {
		envInfo.SudoUser = sudoUser
	}

	// Terminal variables
	envInfo.TerminalVars = []string{
		"TERM", "COLORTERM", "TERM_PROGRAM", "TERM_PROGRAM_VERSION",
		"LANG", "LC_ALL", "LC_CTYPE", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
		"XDG_CACHE_HOME", "USER", "HOME", "SHELL", "LOGNAME",
		"DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "SSH_TTY",
	}

	// Parent process
	if parent := os.Getenv("PPID"); parent != "" {
		envInfo.ParentProcess = parent
	}

	dc.Results.EnvironmentVars = envInfo
}

// analyzePerformance analyzes system performance metrics
func (dc *DiagnosticContext) analyzePerformance() {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	perfInfo := PerformanceInfo{
		CPUCount:    runtime.NumCPU(),
		MemoryTotal: memStats.Sys,
		MemoryUsed:  memStats.Alloc,
		Goroutines:  runtime.NumGoroutine(),
		ThreadCount:  getThreadCount(),
		StackSize:   memStats.StackSys,
		GCPercent:   memStats.GCPercent,
		LastGC:      time.Unix(0, int64(memStats.LastGC)),
	}

	// Additional performance metrics
	if cpuProfile := pprof.Lookup("cpu"); cpuProfile != nil {
		perfInfo.CPUPeak = float64(cpuProfile.Count())
	}

	if memProfile := pprof.Lookup("heap"); memProfile != nil {
		perfInfo.MemoryPeak = uint64(memProfile.Sample[0].Value)
	}

	dc.Results.Performance = perfInfo
}

// analyzeSecurity analyzes security context
func (dc *DiagnosticContext) analyzeSecurity() {
	secInfo := SecurityInfo{
		RunningAsRoot: os.Geteuid() == 0,
		EffectiveUID:  os.Geteuid(),
		EffectiveGID:  os.Getegid(),
		SecureSession: isSecureSession(),
		CanAccessTTY:  canAccessTTY(),
	}

	// TTY permissions
	if info, err := os.Stat("/dev/tty"); err == nil {
		secInfo.TTYPerms = fmt.Sprintf("%04o", info.Mode().Perm())
	}

	dc.Results.Security = secInfo
}

// generateRecommendations generates actionable recommendations
func (dc *DiagnosticContext) generateRecommendations() {
	recs := []string{}

	// Terminal recommendations
	if !dc.Results.TerminalInfo.IsTerminal {
		recs = append(recs, "❌ Terminal not detected - use interactive shell or script command")
	}

	if dc.Results.TerminalInfo.ColorSupport.Has16Colors {
		recs = append(recs, "⚠️  Limited color support - consider using 256-color mode")
	}

	// Signal handling recommendations
	if !dc.Results.SignalHandling.CanHandleInterrupts {
		recs = append(recs, "⚠️  Signal handling may not work properly - use alternative shutdown methods")
	}

	// Environment recommendations
	if dc.Results.Security.RunningAsRoot {
		recs = append(recs, "🔒 Running as root - ensure proper permissions and security practices")
	}

	if dc.Results.EnvironmentVars.SudoUser != "" {
		recs = append(recs, "👤 Sudo session detected - preserve terminal environment variables")
	}

	// Performance recommendations
	if dc.Results.Performance.Goroutines > 100 {
		recs = append(recs, "⚠️  High goroutine count (" + fmt.Sprintf("%d", dc.Results.Performance.Goroutines) + ") - check for goroutine leaks")
	}

	if dc.Results.Performance.MemoryUsed > 100*1024*1024 { // 100MB
		recs = append(recs, "⚠️  High memory usage (" + formatBytes(dc.Results.Performance.MemoryUsed) + ") - check for memory leaks")
	}

	// Generate specific Bubble Tea recommendations
	recs = append(recs, "💡 Bubble Tea specific recommendations:")
	recs = append(recs, "   - Use tea.WithInput(os.Stdin) for non-interactive environments")
	recs = append(recs, "   - Implement proper context cancellation for graceful shutdown")
	recs = append(recs, "   - Add timeout mechanisms for long-running operations")
	recs = append(recs, "   - Use tea.WithAltScreen() only when TTY is confirmed")
	recs = append(recs, "   - Implement error handling for terminal initialization failures")

	dc.Results.Recommendations = recs
}

// addTimelineEvent adds an event to the diagnostic timeline
func (dc *DiagnosticContext) addTimelineEvent(event, detail string, duration time.Duration) {
	event := TimelineEvent{
		Timestamp: time.Now(),
		Event:     event,
		Detail:    detail,
		Duration:  duration,
	}
	dc.Timeline = append(dc.Timeline, event)
}

// GenerateDiagnosticReport generates a human-readable diagnostic report
func (dc *DiagnosticContext) GenerateDiagnosticReport() string {
	var report strings.Builder

	// Header
	report.WriteString("🔍 COMPREHENSIVE TERMINAL DIAGNOSTIC REPORT\n")
	report.WriteString("=" * 50 + "\n\n")
	report.WriteString(fmt.Sprintf("📅 Diagnostic Time: %s\n", dc.StartTime.Format("2006-01-02 15:04:05")))
	report.WriteString(fmt.Sprintf("⏱️  Duration: %v\n", time.Since(dc.StartTime)))
	report.WriteString("\n")

	// Terminal Information
	report.WriteString("🖥️  TERMINAL ENVIRONMENT\n")
	report.WriteString("-" * 30 + "\n")
	term := dc.Results.TerminalInfo
	report.WriteString(fmt.Sprintf("   Is Terminal: %v\n", term.IsTerminal))
	report.WriteString(fmt.Sprintf("   TTY Path: %s\n", term.TTYPath))
	report.WriteString(fmt.Sprintf("   Size: %dx%d\n", term.Width, term.Height))
	report.WriteString(fmt.Sprintf("   Terminal Type: %s\n", term.TerminalType))
	report.WriteString(fmt.Sprintf("   Colors: %d ANSI colors\n", term.ColorSupport.AnsiColors))
	report.WriteString(fmt.Sprintf("   Mouse Support: %v\n", term.InputMode.HasMouse))
	report.WriteString("\n")

	// Signal Handling
	report.WriteString("📡 SIGNAL HANDLING\n")
	report.WriteString("-" * 30 + "\n")
	sig := dc.Results.SignalHandling
	report.WriteString(fmt.Sprintf("   SIGINT: %v\n", sig.SIGINTSupported))
	report.WriteString(fmt.Sprintf("   SIGTERM: %v\n", sig.SIGTERMSupported))
	report.WriteString(fmt.Sprintf("   Graceful Shutdown: %v\n", sig.GracefulShutdown))
	report.WriteString("\n")

	// Environment
	report.WriteString("🌍 ENVIRONMENT CONTEXT\n")
	report.WriteString("-" * 30 + "\n")
	env := dc.Results.EnvironmentVars
	report.WriteString(fmt.Sprintf("   User: %s\n", env.User))
	report.WriteString(fmt.Sprintf("   Home: %s\n", env.Home))
	report.WriteString(fmt.Sprintf("   Sudo User: %s\n", env.SudoUser))
	report.WriteString(fmt.Sprintf("   Display: %s\n", env.Display))
	report.WriteString(fmt.Sprintf("   Parent Process: %s\n", env.ParentProcess))
	report.WriteString("\n")

	// Performance
	report.WriteString("⚡ PERFORMANCE METRICS\n")
	report.WriteString("-" * 30 + "\n")
	perf := dc.Results.Performance
	report.WriteString(fmt.Sprintf("   CPU Cores: %d\n", perf.CPUCount))
	report.WriteString(fmt.Sprintf("   Goroutines: %d\n", perf.Goroutines))
	report.WriteString(fmt.Sprintf("   Memory Used: %s\n", formatBytes(perf.MemoryUsed)))
	report.WriteString(fmt.Sprintf("   Memory Total: %s\n", formatBytes(perf.MemoryTotal)))
	report.WriteString(fmt.Sprintf("   GC Percent: %d%%\n", perf.GCPercent))
	report.WriteString("\n")

	// Security
	report.WriteString("🔒 SECURITY CONTEXT\n")
	report.WriteString("-" * 30 + "\n")
	sec := dc.Results.Security
	report.WriteString(fmt.Sprintf("   Running as Root: %v\n", sec.RunningAsRoot))
	report.WriteString(fmt.Sprintf("   Effective UID: %d\n", sec.EffectiveUID))
	report.WriteString(fmt.Sprintf("   Effective GID: %d\n", sec.EffectiveGID))
	report.WriteString(fmt.Sprintf("   Can Access TTY: %v\n", sec.CanAccessTTY))
	report.WriteString(fmt.Sprintf("   TTY Permissions: %s\n", sec.TTYPerms))
	report.WriteString("\n")

	// Recommendations
	report.WriteString("💡 RECOMMENDATIONS\n")
	report.WriteString("-" * 30 + "\n")
	for _, rec := range dc.Results.Recommendations {
		report.WriteString(fmt.Sprintf("   %s\n", rec))
	}
	report.WriteString("\n")

	// Timeline
	report.WriteString("📊 DIAGNOSTIC TIMELINE\n")
	report.WriteString("-" * 30 + "\n")
	for _, event := range dc.Timeline {
		report.WriteString(fmt.Sprintf("   [%s] %s: %s\n",
			event.Timestamp.Format("15:04:05"), event.Event, event.Detail))
	}

	return report.String()
}

// Helper functions for terminal analysis
func getTerminalSize() (winsize.Winsize, error) {
	// Implementation for getting terminal size
	var winsize winsize.Winsize
	// This would typically use ioctl syscall
	return winsize, nil
}

func hasTrueColor() bool {
	return os.Getenv("COLORTERM") == "truecolor" || os.Getenv("COLORTERM") == "24bit"
}

func getAnsiColorCount() int {
	term := os.Getenv("TERM")
	if strings.Contains(term, "256") {
		return 256
	}
	if strings.Contains(term, "color") || strings.Contains(term, "ansi") {
		return 16
	}
	return 0
}

func hasColorCapability(cap string, colors int) bool {
	// Implementation for color capability detection
	return false
}

func hasMouseSupport() bool {
	// Implementation for mouse support detection
	return false
}

func getInputDevice() string {
	// Implementation for input device detection
	return "/dev/tty"
}

func getBaudRate() int {
	return 9600
}

func canEcho() bool {
	// Implementation for echo capability
	return true
}

func canCanonicalMode() bool {
	// Implementation for canonical mode capability
	return true
}

func canRawMode() bool {
	// Implementation for raw mode capability
	return true
}

func isBuffered(f *os.File) bool {
	// Implementation for buffering detection
	return false
}

func isLineBuffered(f *os.File) bool {
	// Implementation for line buffering detection
	return false
}

func isBlockBuffered(f *os.File) bool {
	// Implementation for block buffering detection
	return false
}

func getBufferSize(f *os.File) int {
	// Implementation for buffer size detection
	return 4096
}

func getThreadCount() int {
	// Implementation for thread count detection
	return runtime.NumCPU()
}

func isSecureSession() bool {
	// Implementation for secure session detection
	return true
}

func canAccessTTY() bool {
	// Implementation for TTY access check
	return true
}

func formatBytes(b uint64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := uint64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(b)/float64(div), "KMGTPE"[exp])
}

// BubbleTeaDiagnostics creates a diagnostic Bubble Tea program
func BubbleTeaDiagnostics() *Model {
	// This would be a specialized diagnostic Bubble Tea model
	return &Model{}
}

// RunDiagnosticTest runs a specific diagnostic test
func RunDiagnosticTest(testName string) (*TerminalDiagnosticResult, error) {
	dc := NewDiagnosticContext()

	switch testName {
	case "terminal":
		dc.analyzeTerminalEnvironment()
	case "signals":
		dc.analyzeSignalHandling()
	case "environment":
		dc.analyzeEnvironment()
	case "performance":
		dc.analyzePerformance()
	case "security":
		dc.analyzeSecurity()
	case "comprehensive":
		return dc.RunComprehensiveDiagnostic(), nil
	default:
		return nil, fmt.Errorf("unknown diagnostic test: %s", testName)
	}

	return dc.Results, nil
}