// internal/ui/monitoring/dashboard_test.go
package monitoring

import (
	"fmt"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// TestDashboardCreation tests dashboard creation and initialization
func TestDashboardCreation(t *testing.T) {
	theme := NewAMDTheme()
	dashboard := NewDashboard(80, 24, theme)

	if dashboard == nil {
		t.Fatal("Failed to create dashboard")
	}

	if dashboard.width != 80 || dashboard.height != 24 {
		t.Errorf("Expected dimensions 80x24, got %dx%d", dashboard.width, dashboard.height)
	}

	if dashboard.theme == nil {
		t.Error("Theme not initialized")
	}

	if dashboard.systemMonitor == nil {
		t.Error("System monitor not initialized")
	}

	if dashboard.gpuMonitor == nil {
		t.Error("GPU monitor not initialized")
	}

	if dashboard.networkMonitor == nil {
		t.Error("Network monitor not initialized")
	}

	if dashboard.alertsPanel == nil {
		t.Error("Alerts panel not initialized")
	}
}

// TestDashboardUpdate tests dashboard message handling
func TestDashboardUpdate(t *testing.T) {
	theme := NewAMDTheme()
	dashboard := NewDashboard(80, 24, theme)

	// Test SystemStatusUpdateMsg
	msg := types.SystemStatusUpdateMsg{
		CPUUsage:    75.0,
		MemoryUsage: 60.0,
		DiskUsage:   45.0,
		GPUUsage:    []float64{80.0, 65.0},
	}

	model, cmd := dashboard.Update(msg)
	if model == nil {
		t.Error("Dashboard update returned nil model")
	}

	// Test key message
	keyMsg := tea.KeyMsg{Type: tea.KeyTab}
	model, cmd = dashboard.Update(keyMsg)
	if model == nil {
		t.Error("Dashboard update returned nil model for key message")
	}

	// Test window resize
	resizeMsg := tea.WindowSizeMsg{Width: 100, Height: 30}
	model, cmd = dashboard.Update(resizeMsg)
	if model == nil {
		t.Error("Dashboard update returned nil model for resize message")
	}

	dashboardModel, ok := model.(*Dashboard)
	if !ok {
		t.Error("Updated model is not a Dashboard")
	} else {
		if dashboardModel.width != 100 || dashboardModel.height != 30 {
			t.Errorf("Expected resized dimensions 100x30, got %dx%d",
				dashboardModel.width, dashboardModel.height)
		}
	}
}

// TestDashboardNavigation tests panel navigation
func TestDashboardNavigation(t *testing.T) {
	theme := NewAMDTheme()
	dashboard := NewDashboard(80, 24, theme)

	originalPanel := dashboard.activePanel

	// Test tab navigation
	tabMsg := tea.KeyMsg{Type: tea.KeyTab}
	dashboard.Update(tabMsg)

	if dashboard.activePanel == originalPanel {
		t.Error("Panel did not change after tab navigation")
	}

	// Test arrow navigation
	upMsg := tea.KeyMsg{Type: tea.KeyUp}
	dashboard.Update(upMsg)
	downMsg := tea.KeyMsg{Type: tea.KeyDown}
	dashboard.Update(downMsg)
	leftMsg := tea.KeyMsg{Type: tea.KeyLeft}
	dashboard.Update(leftMsg)
	rightMsg := tea.KeyMsg{Type: tea.KeyRight}
	dashboard.Update(rightMsg)

	// Test collapse toggle
	collapseMsg := tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'c'}}
	dashboard.Update(collapseMsg)

	if _, collapsed := dashboard.panelCollapsed[dashboard.activePanel]; !collapsed {
		t.Error("Panel was not collapsed after collapse command")
	}
}

// TestThemeSystem tests AMD theme functionality
func TestThemeSystem(t *testing.T) {
	theme := NewAMDTheme()

	// Test initial theme
	if theme.CurrentTheme != ThemeAMD {
		t.Errorf("Expected initial theme %s, got %s", ThemeAMD, theme.CurrentTheme)
	}

	// Test theme cycling
	originalTheme := theme.CurrentTheme
	theme.CycleTheme()

	if theme.CurrentTheme == originalTheme {
		t.Error("Theme did not change after cycling")
	}

	// Test specific theme setting
	theme.SetTheme(ThemeDark)
	if theme.CurrentTheme != ThemeDark {
		t.Errorf("Expected theme %s, got %s", ThemeDark, theme.CurrentTheme)
	}

	// Test accessibility options
	theme.SetAccessibilityOption("high_contrast", true)
	if !theme.HighContrast {
		t.Error("High contrast not enabled")
	}

	options := theme.GetAccessibilityOptions()
	if !options["high_contrast"] {
		t.Error("High contrast not reflected in accessibility options")
	}
}

// TestSystemMonitorWidget tests system monitoring widget
func TestSystemMonitorWidget(t *testing.T) {
	theme := NewAMDTheme()
	widget := NewSystemMonitorWidget(40, 15, theme)

	if widget == nil {
		t.Fatal("Failed to create system monitor widget")
	}

	// Test bounds setting
	widget.SetBounds(0, 0, 50, 20)
	x, y, width, height := widget.GetBounds()

	if x != 0 || y != 0 || width != 50 || height != 20 {
		t.Errorf("Expected bounds 0,0,50x20, got %d,%d,%dx%d", x, y, width, height)
	}

	// Test data update
	msg := types.SystemStatusUpdateMsg{
		CPUUsage:    85.0,
		MemoryUsage: 70.0,
		DiskUsage:   55.0,
		GPUUsage:    []float64{90.0, 75.0},
	}

	widget.UpdateData(msg)

	if widget.cpuUsage[len(widget.cpuUsage)-1] != 85.0 {
		t.Errorf("Expected CPU usage 85.0, got %f", widget.cpuUsage[len(widget.cpuUsage)-1])
	}

	if widget.memoryUsage[len(widget.memoryUsage)-1] != 70.0 {
		t.Errorf("Expected memory usage 70.0, got %f", widget.memoryUsage[len(widget.memoryUsage)-1])
	}

	// Test view rendering
	view := widget.View()
	if view == "" {
		t.Error("System monitor widget returned empty view")
	}

	if len(view) < 10 {
		t.Error("System monitor widget view too short")
	}
}

// TestGPUMonitorWidget tests GPU monitoring widget
func TestGPUMonitorWidget(t *testing.T) {
	theme := NewAMDTheme()
	widget := NewGPUMonitorWidget(30, 12, theme)

	if widget == nil {
		t.Fatal("Failed to create GPU monitor widget")
	}

	// Test data update
	msg := types.SystemStatusUpdateMsg{
		GPUUsage: []float64{95.0, 80.0},
	}

	widget.UpdateData(msg)

	if len(widget.gpuUsage) < 2 {
		t.Error("GPU usage data not updated correctly")
	}

	if widget.gpuUsage[0] != 95.0 {
		t.Errorf("Expected first GPU usage 95.0, got %f", widget.gpuUsage[0])
	}

	// Test view rendering
	view := widget.View()
	if view == "" {
		t.Error("GPU monitor widget returned empty view")
	}

	// Check for GPU-specific content
	if len(view) < 50 {
		t.Error("GPU monitor widget view too short")
	}
}

// TestDiagnosticEngine tests diagnostic functionality
func TestDiagnosticEngine(t *testing.T) {
	theme := NewAMDTheme()
	engine := NewDiagnosticEngine(theme)

	if engine == nil {
		t.Fatal("Failed to create diagnostic engine")
	}

	// Test system diagnostics
	results := engine.runSystemDiagnostics()
	if len(results) == 0 {
		t.Error("No diagnostic results returned")
	}

	// Check for expected diagnostic categories
	foundCPU := false
	foundMemory := false
	for _, result := range results {
		if result.ID == "cpu_health" {
			foundCPU = true
		}
		if result.ID == "memory_health" {
			foundMemory = true
		}
	}

	if !foundCPU {
		t.Error("CPU health diagnostic not found")
	}

	if !foundMemory {
		t.Error("Memory health diagnostic not found")
	}

	// Test GPU diagnostics
	gpuResults := engine.runGPUDiagnostics()
	if len(gpuResults) == 0 {
		t.Error("No GPU diagnostic results returned")
	}

	// Test overall score calculation
	engine.results = append(engine.results, results...)
	score := engine.GetOverallScore()
	if score < 0 || score > 100 {
		t.Errorf("Invalid overall score: %d", score)
	}
}

// TestCharts tests chart rendering functionality
func TestCharts(t *testing.T) {
	theme := NewAMDTheme()

	// Test line chart
	lineConfig := ChartConfig{
		Type:       ChartTypeLine,
		Width:      40,
		Height:     10,
		Title:      "Test Line Chart",
		ShowLegend: true,
		MaxPoints:  50,
	}

	lineChart := NewChart(lineConfig, theme)
	if lineChart == nil {
		t.Fatal("Failed to create line chart")
	}

	// Add test data
	for i := 0; i < 10; i++ {
		lineChart.AddData(float64(i)*10, fmt.Sprintf("Point %d", i))
	}

	view := lineChart.Render()
	if view == "" {
		t.Error("Line chart returned empty view")
	}

	// Test bar chart
	barConfig := ChartConfig{
		Type:      ChartTypeBar,
		Width:     30,
		Height:    8,
		Title:     "Test Bar Chart",
		MaxPoints: 5,
	}

	barChart := NewChart(barConfig, theme)
	for i := 0; i < 5; i++ {
		barChart.AddData(float64(i+1)*20, fmt.Sprintf("Bar %d", i))
	}

	barView := barChart.Render()
	if barView == "" {
		t.Error("Bar chart returned empty view")
	}

	// Test sparkline
	sparklineConfig := ChartConfig{
		Type:      ChartTypeSparkline,
		Width:     20,
		Height:    1,
		MaxPoints: 20,
	}

	sparkline := NewChart(sparklineConfig, theme)
	for i := 0; i < 20; i++ {
		sparkline.AddData(float64(i%10), "")
	}

	sparklineView := sparkline.Render()
	if sparklineView == "" {
		t.Error("Sparkline returned empty view")
	}
}

// TestAnimations tests animation functionality
func TestAnimations(t *testing.T) {
	theme := NewAMDTheme()

	// Test animation manager
	manager := NewAnimationManager(theme, 10)
	if manager == nil {
		t.Fatal("Failed to create animation manager")
	}

	// Test fade animation
	fadeConfig := AnimationConfig{
		Type:     AnimationTypeFade,
		Duration: 100 * time.Millisecond,
		Easing:   EaseOutCubic,
		From: AnimationValues{
			Opacity: 0.0,
		},
		To: AnimationValues{
			Opacity: 1.0,
		},
	}

	fadeAnimation := NewAnimation(fadeConfig, theme)
	if fadeAnimation == nil {
		t.Fatal("Failed to create fade animation")
	}

	// Test animation lifecycle
	fadeAnimation.Start()
	if !fadeAnimation.IsRunning() {
		t.Error("Animation not running after start")
	}

	// Test animation update
	fadeAnimation.Update(50 * time.Millisecond)
	currentOpacity := fadeAnimation.GetCurrentValue("opacity")
	if currentOpacity <= 0 || currentOpacity > 1 {
		t.Errorf("Invalid opacity during animation: %f", currentOpacity)
	}

	// Test adding animation to manager
	added := manager.Add(fadeAnimation)
	if !added {
		t.Error("Failed to add animation to manager")
	}

	// Test manager update
	manager.Update()
	if manager.GetActiveCount() == 0 {
		t.Error("No active animations in manager")
	}

	// Test loading animation
	loadingAnim := NewLoadingAnimation(LoadingTypeSpinner, "Loading...", theme)
	if loadingAnim == nil {
		t.Fatal("Failed to create loading animation")
	}

	loadingAnim.Start()
	if !loadingAnim.IsRunning {
		t.Error("Loading animation not running")
	}

	loadingView := loadingAnim.Render()
	if loadingView == "" {
		t.Error("Loading animation returned empty view")
	}
}

// TestPerformanceOptimizer tests performance optimization
func TestPerformanceOptimizer(t *testing.T) {
	theme := NewAMDTheme()
	optimizer := NewPerformanceOptimizer(theme)

	if optimizer == nil {
		t.Fatal("Failed to create performance optimizer")
	}

	// Test caching
	key := "test_key"
	content := "test content"
	width, height := 40, 10

	// Cache miss
	cached, found := optimizer.GetCachedRender(key, width, height)
	if found {
		t.Error("Unexpected cache hit on empty cache")
	}

	// Cache render
	optimizer.CacheRender(key, content, width, height)

	// Cache hit
	cached, found = optimizer.GetCachedRender(key, width, height)
	if !found {
		t.Error("Expected cache hit after caching")
	}
	if cached != content {
		t.Error("Cached content mismatch")
	}

	// Test frame rate limiting
	optimizer.frameRateLimit = 30
	shouldRender1 := optimizer.ShouldRender()
	shouldRender2 := optimizer.ShouldRender()

	if !shouldRender1 {
		t.Error("First render should be allowed")
	}

	// Second render might be skipped depending on timing
	// This is timing-dependent, so we just check it doesn't crash
	_ = shouldRender2

	// Test dirty regions
	optimizer.MarkDirtyRegion("test_region", 0, 0, 10, 10)
	dirtyRegions := optimizer.GetDirtyRegions()

	if len(dirtyRegions) == 0 {
		t.Error("No dirty regions found after marking")
	}

	optimizer.ClearDirtyRegions()
	dirtyRegions = optimizer.GetDirtyRegions()

	if len(dirtyRegions) != 0 {
		t.Error("Dirty regions not cleared")
	}

	// Test update throttling
	shouldUpdate1 := optimizer.ThrottleUpdate("test_key", "data1")
	if !shouldUpdate1 {
		t.Error("First update should be allowed")
	}

	// Second update might be throttled depending on timing
	shouldUpdate2 := optimizer.ThrottleUpdate("test_key", "data2")
	_ = shouldUpdate2

	// Test metrics update
	optimizer.UpdateMetrics(time.Now())
	metrics := optimizer.GetMetrics()

	if metrics.RenderCount == 0 {
		t.Error("No render metrics recorded")
	}
}

// TestAccessibilityManager tests accessibility features
func TestAccessibilityManager(t *testing.T) {
	theme := NewAMDTheme()
	manager := NewAccessibilityManager(theme)

	if manager == nil {
		t.Fatal("Failed to create accessibility manager")
	}

	// Test high contrast mode
	manager.SetHighContrast(true)
	if !manager.highContrast {
		t.Error("High contrast not enabled")
	}

	// Test large text mode
	manager.SetLargeText(true)
	if !manager.largeText {
		t.Error("Large text not enabled")
	}
	if manager.fontSize != 16 {
		t.Errorf("Expected font size 16, got %d", manager.fontSize)
	}

	// Test reduced motion
	manager.SetReducedMotion(true)
	if !manager.reducedMotion {
		t.Error("Reduced motion not enabled")
	}

	// Test screen reader mode
	manager.SetScreenReaderMode(true)
	if !manager.screenReaderMode {
		t.Error("Screen reader mode not enabled")
	}

	// Test color blind modes
	manager.SetColorBlindMode(ColorBlindProtanopia)
	if manager.colorBlindMode != ColorBlindProtanopia {
		t.Error("Color blind mode not set correctly")
	}

	// Test accessible text
	text := "Test text"
	accessibleText := manager.GetAccessibleText(text)
	if accessibleText == "" {
		t.Error("Accessible text is empty")
	}

	// Test focus styling
	focusStyle := manager.GetFocusStyle()
	if focusStyle == (lipgloss.Style{}) {
		t.Error("Focus style is empty")
	}

	// Test accessible colors
	fg, bg := manager.GetAccessibleColors()
	if fg == (lipgloss.Color{}) || bg == (lipgloss.Color{}) {
		t.Error("Accessible colors not set")
	}
}

// TestMonitoringSystemIntegration tests the complete monitoring system
func TestMonitoringSystemIntegration(t *testing.T) {
	config := MonitoringConfig{
		Enabled:         true,
		UpdateInterval:  1 * time.Second,
		Theme:           "amd",
		Optimization:    "medium",
		Accessibility:   true,
		HighContrast:    false,
		LargeText:       false,
		ReducedMotion:   false,
		ScreenReader:    false,
		ColorBlindMode:  "none",
		PerformanceMode: "balanced",
	}

	system := NewMonitoringSystem(config)
	if system == nil {
		t.Fatal("Failed to create monitoring system")
	}

	// Test initialization
	cmd := system.Initialize(80, 24)
	if cmd == nil {
		t.Error("Initialize returned no command")
	}

	if !system.initialized {
		t.Error("System not initialized after Initialize call")
	}

	// Test message handling
	keyMsg := tea.KeyMsg{Type: tea.KeyTab}
	model, cmd := system.Update(keyMsg)
	if model == nil {
		t.Error("Update returned nil model")
	}

	// Test view rendering
	view := system.View()
	if view == "" {
		t.Error("System view is empty")
	}

	// Test system status
	status := system.GetSystemStatus()
	if status == nil {
		t.Error("System status is nil")
	}

	if !status["initialized"].(bool) {
		t.Error("System not marked as initialized in status")
	}

	// Test performance metrics
	metrics := system.GetPerformanceMetrics()
	if metrics.RenderCount == 0 {
		t.Error("No performance metrics recorded")
	}

	// Test configuration
	newConfig := system.GetConfig()
	if newConfig.Enabled != config.Enabled {
		t.Error("Configuration not preserved")
	}

	// Test commands
	commands := system.GetMonitoringCommands()
	if len(commands) == 0 {
		t.Error("No monitoring commands available")
	}

	// Test diagnostic export
	report, err := system.ExportDiagnosticReport()
	if err != nil {
		t.Errorf("Failed to export diagnostic report: %v", err)
	}
	if report == "" {
		t.Error("Diagnostic report is empty")
	}

	// Test cleanup
	system.Cleanup()
	if system.initialized {
		t.Error("System still marked as initialized after cleanup")
	}
}

// BenchmarkDashboardRender benchmarks dashboard rendering performance
func BenchmarkDashboardRender(b *testing.B) {
	theme := NewAMDTheme()
	dashboard := NewDashboard(80, 24, theme)

	// Pre-populate with data
	for i := 0; i < 60; i++ {
		msg := types.SystemStatusUpdateMsg{
			CPUUsage:    float64(i % 100),
			MemoryUsage: float64(i % 100),
			DiskUsage:   float64(i % 100),
			GPUUsage:    []float64{float64(i % 100), float64((i + 50) % 100)},
		}
		dashboard.Update(msg)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		view := dashboard.View()
		if view == "" {
			b.Fatal("Empty view during benchmark")
		}
	}
}

// BenchmarkChartRendering benchmarks chart rendering performance
func BenchmarkChartRendering(b *testing.B) {
	theme := NewAMDTheme()

	config := ChartConfig{
		Type:       ChartTypeLine,
		Width:      60,
		Height:     15,
		ShowLegend: true,
		MaxPoints:  100,
	}

	chart := NewChart(config, theme)

	// Pre-populate with data
	for i := 0; i < 50; i++ {
		chart.AddData(float64(i)%50, fmt.Sprintf("Point %d", i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		view := chart.Render()
		if view == "" {
			b.Fatal("Empty chart view during benchmark")
		}
	}
}

// TestErrorHandling tests error handling and edge cases
func TestErrorHandling(t *testing.T) {
	theme := NewAMDTheme()

	// Test dashboard with zero dimensions
	dashboard := NewDashboard(0, 0, theme)
	if dashboard == nil {
		t.Error("Dashboard creation failed with zero dimensions")
	}

	// Test chart with invalid data
	config := ChartConfig{
		Type:   ChartTypeLine,
		Width:  10,
		Height: 5,
	}

	chart := NewChart(config, theme)
	if chart == nil {
		t.Fatal("Failed to create chart")
	}

	// Test with nil data
	view := chart.Render()
	if view == "" {
		t.Error("Chart returned empty view with no data")
	}

	// Test animation with invalid duration
	animConfig := AnimationConfig{
		Type:     AnimationTypeFade,
		Duration: 0,
	}

	animation := NewAnimation(animConfig, theme)
	if animation == nil {
		t.Fatal("Failed to create animation with zero duration")
	}

	animation.Start()
	if !animation.IsRunning() {
		t.Error("Animation not running with zero duration")
	}

	// Test performance optimizer with disabled state
	optimizer := NewPerformanceOptimizer(theme)
	optimizer.SetEnabled(false)

	cached, found := optimizer.GetCachedRender("test", 10, 10)
	if found {
		t.Error("Unexpected cache hit when optimizer disabled")
	}

	// Test accessibility manager with invalid settings
	manager := NewAccessibilityManager(theme)
	manager.SetBrightness(-1.0) // Invalid brightness
	if manager.brightness < 0.1 {
		t.Error("Brightness not clamped to minimum value")
	}

	manager.SetContrast(10.0) // Invalid contrast
	if manager.contrast > 2.0 {
		t.Error("Contrast not clamped to maximum value")
	}
}
