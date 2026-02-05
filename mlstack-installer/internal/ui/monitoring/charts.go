// internal/ui/monitoring/charts.go
package monitoring

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
)

// ChartType represents different types of charts
type ChartType int

const (
	ChartTypeLine ChartType = iota
	ChartTypeBar
	ChartTypeArea
	ChartTypeSparkline
	ChartTypeGauge
	ChartTypeHeatmap
)

// ChartData represents data points for charts
type ChartData struct {
	Values  []float64
	Labels  []string
	Times   []time.Time
	Max     float64
	Min     float64
	Average float64
	Last    float64
	Trend   string // "up", "down", "stable"
}

// ChartConfig represents chart configuration
type ChartConfig struct {
	Type        ChartType
	Width       int
	Height      int
	Title       string
	ShowLegend  bool
	ShowAxes    bool
	ShowGrid    bool
	ColorScheme string
	Animate     bool
	RefreshRate time.Duration
	MaxPoints   int
}

// Chart represents a generic chart component
type Chart struct {
	config    ChartConfig
	data      ChartData
	theme     *AMDTheme
	animation *ChartAnimation
	renderer  ChartRenderer
}

// ChartAnimation handles chart animations
type ChartAnimation struct {
	Enabled      bool
	Duration     time.Duration
	StartTime    time.Time
	Progress     float64
	Easing       EasingFunction
	CurrentValue float64
	TargetValue  float64
	IsAnimating  bool
}

// EasingFunction represents different easing functions for animations
type EasingFunction func(float64) float64

// ChartRenderer interface for different chart rendering strategies
type ChartRenderer interface {
	Render(chart *Chart) string
	UpdateData(chart *Chart, newData ChartData)
	Animate(chart *Chart, deltaTime time.Duration)
}

// NewChart creates a new chart instance
func NewChart(config ChartConfig, theme *AMDTheme) *Chart {
	chart := &Chart{
		config: config,
		data: ChartData{
			Values:  make([]float64, 0, config.MaxPoints),
			Labels:  make([]string, 0, config.MaxPoints),
			Times:   make([]time.Time, 0, config.MaxPoints),
			Max:     0,
			Min:     0,
			Average: 0,
			Last:    0,
			Trend:   "stable",
		},
		theme: theme,
		animation: &ChartAnimation{
			Enabled:     config.Animate,
			Duration:    500 * time.Millisecond,
			Easing:      EaseOutCubic,
			IsAnimating: false,
		},
	}

	// Create appropriate renderer
	chart.createRenderer()

	return chart
}

// createRenderer creates the appropriate chart renderer based on chart type
func (c *Chart) createRenderer() {
	switch c.config.Type {
	case ChartTypeLine:
		c.renderer = NewLineChartRenderer(c.theme)
	case ChartTypeBar:
		c.renderer = NewBarChartRenderer(c.theme)
	case ChartTypeArea:
		c.renderer = NewAreaChartRenderer(c.theme)
	case ChartTypeSparkline:
		c.renderer = NewSparklineRenderer(c.theme)
	case ChartTypeGauge:
		c.renderer = NewGaugeRenderer(c.theme)
	case ChartTypeHeatmap:
		c.renderer = NewHeatmapRenderer(c.theme)
	default:
		c.renderer = NewLineChartRenderer(c.theme)
	}
}

// AddData adds a new data point to the chart
func (c *Chart) AddData(value float64, label string) {
	// Add timestamp
	timestamp := time.Now()

	// Add new data point
	c.data.Values = append(c.data.Values, value)
	c.data.Labels = append(c.data.Labels, label)
	c.data.Times = append(c.data.Times, timestamp)

	// Limit data points
	if len(c.data.Values) > c.config.MaxPoints {
		c.data.Values = c.data.Values[1:]
		c.data.Labels = c.data.Labels[1:]
		c.data.Times = c.data.Times[1:]
	}

	// Update statistics
	c.updateStatistics()

	// Trigger animation if enabled
	if c.animation.Enabled && !c.animation.IsAnimating {
		c.startAnimation(c.data.Last)
	}

	// Update renderer
	c.renderer.UpdateData(c, c.data)
}

// updateStatistics updates chart statistics
func (c *Chart) updateStatistics() {
	if len(c.data.Values) == 0 {
		return
	}

	// Calculate min, max, average
	min, max := c.data.Values[0], c.data.Values[0]
	sum := 0.0
	for _, value := range c.data.Values {
		if value < min {
			min = value
		}
		if value > max {
			max = value
		}
		sum += value
	}

	c.data.Min = min
	c.data.Max = max
	c.data.Average = sum / float64(len(c.data.Values))
	c.data.Last = c.data.Values[len(c.data.Values)-1]

	// Calculate trend
	if len(c.data.Values) >= 2 {
		recent := c.data.Values[len(c.data.Values)-1]
		previous := c.data.Values[len(c.data.Values)-2]

		if recent > previous*1.05 {
			c.data.Trend = "up"
		} else if recent < previous*0.95 {
			c.data.Trend = "down"
		} else {
			c.data.Trend = "stable"
		}
	}
}

// startAnimation starts a chart animation
func (c *Chart) startAnimation(targetValue float64) {
	c.animation.StartTime = time.Now()
	c.animation.TargetValue = targetValue
	c.animation.CurrentValue = c.data.Last
	c.animation.Progress = 0
	c.animation.IsAnimating = true
}

// Update updates the chart state
func (c *Chart) Update(deltaTime time.Duration) {
	// Update animation
	if c.animation.IsAnimating {
		c.animation.Progress = float64(time.Since(c.animation.StartTime)) / float64(c.animation.Duration)

		if c.animation.Progress >= 1.0 {
			c.animation.Progress = 1.0
			c.animation.IsAnimating = false
			c.animation.CurrentValue = c.animation.TargetValue
		} else {
			// Apply easing function
			easedProgress := c.animation.Easing(c.animation.Progress)
			c.animation.CurrentValue = c.animation.CurrentValue + (c.animation.TargetValue-c.animation.CurrentValue)*easedProgress
		}

		// Update renderer with animation
		c.renderer.Animate(c, deltaTime)
	}
}

// Render renders the chart
func (c *Chart) Render() string {
	if c.renderer == nil {
		return c.theme.Text.Render("Chart renderer not initialized")
	}

	return c.renderer.Render(c)
}

// LineChartRenderer renders line charts
type LineChartRenderer struct {
	theme *AMDTheme
}

// NewLineChartRenderer creates a new line chart renderer
func NewLineChartRenderer(theme *AMDTheme) *LineChartRenderer {
	return &LineChartRenderer{theme: theme}
}

// Render implements ChartRenderer interface
func (r *LineChartRenderer) Render(chart *Chart) string {
	if len(chart.data.Values) < 2 {
		return r.theme.Text.Render("Insufficient data for line chart")
	}

	var lines []string

	// Title
	if chart.config.Title != "" {
		lines = append(lines, r.theme.Header.Render(chart.config.Title))
	}

	// Calculate chart dimensions
	charWidth := chart.config.Width - 4   // Account for borders
	charHeight := chart.config.Height - 4 // Account for borders and title
	if chart.config.Title != "" {
		charHeight -= 2
	}

	// Create line chart
	lineChart := r.createLineChart(chart.data.Values, charWidth, charHeight)
	lines = append(lines, lineChart)

	// X-axis labels (if enabled and space permits)
	if chart.config.ShowAxes && chart.config.Height > 8 {
		labels := r.createTimeLabels(chart.data.Times, charWidth)
		lines = append(lines, labels)
	}

	// Legend (if enabled)
	if chart.config.ShowLegend {
		legend := r.createLegend(chart.data)
		lines = append(lines, legend)
	}

	content := strings.Join(lines, "\n")
	return r.theme.Panel.Width(chart.config.Width).Height(chart.config.Height).Render(content)
}

// createLineChart creates a line chart from data points
func (r *LineChartRenderer) createLineChart(values []float64, width, height int) string {
	if len(values) == 0 {
		return strings.Repeat(" ", width*height)
	}

	// Find min and max values
	min, max := values[0], values[0]
	for _, value := range values {
		if value < min {
			min = value
		}
		if value > max {
			max = value
		}
	}

	// Prevent division by zero
	if max == min {
		max = min + 1
	}

	// Create chart grid
	chart := make([][]rune, height)
	for i := range chart {
		chart[i] = make([]rune, width)
		for j := range chart[i] {
			chart[i][j] = ' '
		}
	}

	// Plot data points
	for i, value := range values {
		if i >= width {
			break // Don't exceed chart width
		}

		// Calculate y position
		normalized := (value - min) / (max - min)
		y := height - 1 - int(normalized*float64(height-1))
		if y < 0 {
			y = 0
		}
		if y >= height {
			y = height - 1
		}

		chart[y][i] = '●'
	}

	// Draw connecting lines
	for i := 1; i < len(values) && i < width; i++ {
		value1 := values[i-1]
		value2 := values[i]

		normalized1 := (value1 - min) / (max - min)
		normalized2 := (value2 - min) / (max - min)

		y1 := height - 1 - int(normalized1*float64(height-1))
		y2 := height - 1 - int(normalized2*float64(height-1))

		if y1 < 0 {
			y1 = 0
		}
		if y1 >= height {
			y1 = height - 1
		}
		if y2 < 0 {
			y2 = 0
		}
		if y2 >= height {
			y2 = height - 1
		}

		// Draw line between points
		r.drawLine(chart, i-1, y1, i, y2)
	}

	// Convert to string
	var result strings.Builder
	for _, row := range chart {
		result.WriteString(string(row))
		result.WriteString("\n")
	}

	return result.String()
}

// drawLine draws a line between two points
func (r *LineChartRenderer) drawLine(chart [][]rune, x1, y1, x2, y2 int) {
	dx := x2 - x1
	dy := y2 - y1
	steps := maxInt(abs(dx), abs(dy))

	if steps == 0 {
		return
	}

	for i := 0; i <= steps; i++ {
		t := float64(i) / float64(steps)
		x := int(float64(x1) + float64(dx)*t)
		y := int(float64(y1) + float64(dy)*t)

		if y >= 0 && y < len(chart) && x >= 0 && x < len(chart[0]) {
			if chart[y][x] == ' ' {
				chart[y][x] = '•'
			}
		}
	}
}

// createTimeLabels creates time labels for x-axis
func (r *LineChartRenderer) createTimeLabels(times []time.Time, width int) string {
	if len(times) == 0 {
		return ""
	}

	// Create labels at intervals
	labelInterval := maxInt(1, len(times)/width)
	var labels []string

	for i := 0; i < len(times); i += labelInterval {
		if i < width {
			t := times[i]
			label := t.Format("15:04")
			if len(labels) > 0 && i < len(times)-labelInterval {
				label = "" // Only show first and last labels to avoid crowding
			}
			labels = append(labels, fmt.Sprintf("%-8s", label))
		}
	}

	return strings.Join(labels, "")
}

// createLegend creates a legend for the chart
func (r *LineChartRenderer) createLegend(data ChartData) string {
	trendIcon := "→"
	switch data.Trend {
	case "up":
		trendIcon = "↑"
	case "down":
		trendIcon = "↓"
	}

	return fmt.Sprintf("Current: %.1f | Avg: %.1f | %s", data.Last, data.Average, trendIcon)
}

// UpdateData implements ChartRenderer interface
func (r *LineChartRenderer) UpdateData(chart *Chart, newData ChartData) {
	// Line chart renderer updates are handled in the main chart
}

// Animate implements ChartRenderer interface
func (r *LineChartRenderer) Animate(chart *Chart, deltaTime time.Duration) {
	// Animation updates are handled in the main chart
}

// BarChartRenderer renders bar charts
type BarChartRenderer struct {
	theme *AMDTheme
}

// NewBarChartRenderer creates a new bar chart renderer
func NewBarChartRenderer(theme *AMDTheme) *BarChartRenderer {
	return &BarChartRenderer{theme: theme}
}

// Render implements ChartRenderer interface
func (r *BarChartRenderer) Render(chart *Chart) string {
	if len(chart.data.Values) == 0 {
		return r.theme.Text.Render("No data for bar chart")
	}

	var lines []string

	// Title
	if chart.config.Title != "" {
		lines = append(lines, r.theme.Header.Render(chart.config.Title))
	}

	// Calculate bar dimensions
	barWidth := maxInt(1, (chart.config.Width-4)/len(chart.data.Values))
	maxHeight := chart.config.Height - 6 // Account for borders, title, and labels

	// Find max value for scaling
	maxValue := 0.0
	for _, value := range chart.data.Values {
		if value > maxValue {
			maxValue = value
		}
	}

	if maxValue == 0 {
		maxValue = 1
	}

	// Create bars
	var barLines []string
	for i, value := range chart.data.Values {
		barHeight := int((value / maxValue) * float64(maxHeight))
		if barHeight < 1 {
			barHeight = 1
		}

		// Create bar with color based on value
		barColor := r.theme.Success
		if value > maxValue*0.8 {
			barColor = r.theme.Error
		} else if value > maxValue*0.6 {
			barColor = r.theme.Warning
		}

		bar := strings.Repeat("█", barHeight)
		styledBar := lipgloss.NewStyle().Foreground(barColor).Render(bar)

		label := chart.data.Labels[i]
		if label == "" {
			label = fmt.Sprintf("%d", i)
		}

		barLine := fmt.Sprintf("%s %s", styledBar, label)
		barLines = append(barLines, barLine)
	}

	// Combine bars horizontally
	for row := maxHeight - 1; row >= 0; row-- {
		var rowChars []string
		for _, barLine := range barLines {
			if row < len(strings.Split(barLine, "█"))-1 {
				rowChars = append(rowChars, "█")
			} else {
				rowChars = append(rowChars, " ")
			}
		}
		lines = append(lines, strings.Join(rowChars, " "))
	}

	// Add labels
	if chart.config.ShowAxes {
		var labelLine []string
		for _, label := range chart.data.Labels {
			if label == "" {
				label = fmt.Sprintf("%d", len(labelLine))
			}
			labelLine = append(labelLine, fmt.Sprintf("%-*s", barWidth, label[:minInt(barWidth, len(label))]))
		}
		lines = append(lines, strings.Join(labelLine, " "))
	}

	content := strings.Join(lines, "\n")
	return r.theme.Panel.Width(chart.config.Width).Height(chart.config.Height).Render(content)
}

// UpdateData implements ChartRenderer interface
func (r *BarChartRenderer) UpdateData(chart *Chart, newData ChartData) {
	// Bar chart renderer updates are handled in the main chart
}

// Animate implements ChartRenderer interface
func (r *BarChartRenderer) Animate(chart *Chart, deltaTime time.Duration) {
	// Animation updates are handled in the main chart
}

// SparklineRenderer renders sparkline charts
type SparklineRenderer struct {
	theme *AMDTheme
}

// NewSparklineRenderer creates a new sparkline renderer
func NewSparklineRenderer(theme *AMDTheme) *SparklineRenderer {
	return &SparklineRenderer{theme: theme}
}

// Render implements ChartRenderer interface
func (r *SparklineRenderer) Render(chart *Chart) string {
	if len(chart.data.Values) == 0 {
		return r.theme.Text.Render("No data")
	}

	// Create sparkline
	sparklineChars := []rune(" ▁▂▃▄▅▆▇█")
	var sparkline strings.Builder

	// Find min and max for scaling
	min, max := chart.data.Min, chart.data.Max
	if max == min {
		max = min + 1
	}

	for _, value := range chart.data.Values {
		normalized := (value - min) / (max - min)
		index := int(normalized * float64(len(sparklineChars)-1))
		if index < 0 {
			index = 0
		}
		if index >= len(sparklineChars) {
			index = len(sparklineChars) - 1
		}
		sparkline.WriteRune(sparklineChars[index])
	}

	// Add value indicator
	indicator := r.theme.Metric.Render(fmt.Sprintf("%.1f", chart.data.Last))

	// Add trend arrow
	trendIcon := "→"
	switch chart.data.Trend {
	case "up":
		trendIcon = "↑"
	case "down":
		trendIcon = "↓"
	}

	return fmt.Sprintf("%s %s %s", sparkline.String(), indicator, trendIcon)
}

// UpdateData implements ChartRenderer interface
func (r *SparklineRenderer) UpdateData(chart *Chart, newData ChartData) {
	// Sparkline renderer updates are handled in the main chart
}

// Animate implements ChartRenderer interface
func (r *SparklineRenderer) Animate(chart *Chart, deltaTime time.Duration) {
	// Animation updates are handled in the main chart
}

// GaugeRenderer renders gauge charts
type GaugeRenderer struct {
	theme *AMDTheme
}

// NewGaugeRenderer creates a new gauge renderer
func NewGaugeRenderer(theme *AMDTheme) *GaugeRenderer {
	return &GaugeRenderer{theme: theme}
}

// Render implements ChartRenderer interface
func (r *GaugeRenderer) Render(chart *Chart) string {
	if len(chart.data.Values) == 0 {
		return r.theme.Text.Render("No data")
	}

	value := chart.data.Last
	maxValue := chart.data.Max
	if maxValue == 0 {
		maxValue = 100
	}

	percentage := value / maxValue

	// Determine color
	color := r.theme.Success
	if percentage > 0.8 {
		color = r.theme.Error
	} else if percentage > 0.6 {
		color = r.theme.Warning
	}

	// Create gauge arc
	gaugeWidth := chart.config.Width - 4
	filledWidth := int(float64(gaugeWidth) * percentage)

	filled := strings.Repeat("█", filledWidth)
	empty := strings.Repeat("░", gaugeWidth-filledWidth)

	gauge := lipgloss.NewStyle().
		Foreground(color).
		Render(filled) + lipgloss.NewStyle().
		Foreground(r.theme.Muted).
		Render(empty)

	// Add title and value
	title := chart.config.Title
	if title == "" {
		title = "Gauge"
	}

	valueText := fmt.Sprintf("%.1f/%.1f", value, maxValue)

	return fmt.Sprintf("%s\n%s\n%s",
		r.theme.Header.Render(title),
		gauge,
		r.theme.Text.Render(valueText))
}

// UpdateData implements ChartRenderer interface
func (r *GaugeRenderer) UpdateData(chart *Chart, newData ChartData) {
	// Gauge renderer updates are handled in the main chart
}

// Animate implements ChartRenderer interface
func (r *GaugeRenderer) Animate(chart *Chart, deltaTime time.Duration) {
	// Animation updates are handled in the main chart
}

// AreaChartRenderer renders area charts
type AreaChartRenderer struct {
	theme *AMDTheme
}

// NewAreaChartRenderer creates a new area chart renderer
func NewAreaChartRenderer(theme *AMDTheme) *AreaChartRenderer {
	return &AreaChartRenderer{theme: theme}
}

// Render implements ChartRenderer interface
func (r *AreaChartRenderer) Render(chart *Chart) string {
	if len(chart.data.Values) < 2 {
		return r.theme.Text.Render("Insufficient data for area chart")
	}

	// Similar to line chart but fills area under the line
	// This is a simplified implementation
	lineRenderer := NewLineChartRenderer(r.theme)

	// Get line chart representation
	lineChart := lineRenderer.createLineChart(chart.data.Values, chart.config.Width-4, chart.config.Height-6)

	// Fill area under the line (simplified)
	lines := strings.Split(lineChart, "\n")
	var filledLines []string

	for i, line := range lines {
		if i == len(lines)-1 {
			// Bottom line - fill completely
			filledLine := strings.ReplaceAll(line, " ", "▄")
			filledLines = append(filledLines, filledLine)
		} else {
			filledLines = append(filledLines, line)
		}
	}

	content := strings.Join(filledLines, "\n")
	return r.theme.Panel.Width(chart.config.Width).Height(chart.config.Height).Render(content)
}

// UpdateData implements ChartRenderer interface
func (r *AreaChartRenderer) UpdateData(chart *Chart, newData ChartData) {
	// Area chart renderer updates are handled in the main chart
}

// Animate implements ChartRenderer interface
func (r *AreaChartRenderer) Animate(chart *Chart, deltaTime time.Duration) {
	// Animation updates are handled in the main chart
}

// HeatmapRenderer renders heatmap charts
type HeatmapRenderer struct {
	theme *AMDTheme
}

// NewHeatmapRenderer creates a new heatmap renderer
func NewHeatmapRenderer(theme *AMDTheme) *HeatmapRenderer {
	return &HeatmapRenderer{theme: theme}
}

// Render implements ChartRenderer interface
func (r *HeatmapRenderer) Render(chart *Chart) string {
	if len(chart.data.Values) == 0 {
		return r.theme.Text.Render("No data for heatmap")
	}

	// Simple heatmap implementation
	heatmapChars := []rune(" ░▒▓█")

	// Find min and max
	min, max := chart.data.Min, chart.data.Max
	if max == min {
		max = min + 1
	}

	// Create heatmap grid
	gridWidth := chart.config.Width - 4
	gridHeight := chart.config.Height - 6

	var heatmap strings.Builder

	for row := 0; row < gridHeight; row++ {
		for col := 0; col < gridWidth; col++ {
			// Get value or use default
			index := row*gridWidth + col
			var value float64
			if index < len(chart.data.Values) {
				value = chart.data.Values[index]
			} else {
				value = chart.data.Average
			}

			// Normalize and select character
			normalized := (value - min) / (max - min)
			charIndex := int(normalized * float64(len(heatmapChars)-1))
			if charIndex < 0 {
				charIndex = 0
			}
			if charIndex >= len(heatmapChars) {
				charIndex = len(heatmapChars) - 1
			}

			// Select color based on intensity
			char := heatmapChars[charIndex]
			var color lipgloss.Color

			switch {
			case normalized < 0.25:
				color = r.theme.Success
			case normalized < 0.5:
				color = r.theme.Info
			case normalized < 0.75:
				color = r.theme.Warning
			default:
				color = r.theme.Error
			}

			styledChar := lipgloss.NewStyle().Foreground(color).Render(string(char))
			heatmap.WriteString(styledChar)
		}
		heatmap.WriteString("\n")
	}

	return r.theme.Panel.Width(chart.config.Width).Height(chart.config.Height).Render(heatmap.String())
}

// UpdateData implements ChartRenderer interface
func (r *HeatmapRenderer) UpdateData(chart *Chart, newData ChartData) {
	// Heatmap renderer updates are handled in the main chart
}

// Animate implements ChartRenderer interface
func (r *HeatmapRenderer) Animate(chart *Chart, deltaTime time.Duration) {
	// Animation updates are handled in the main chart
}

// Easing functions for smooth animations
var (
	EaseLinear    EasingFunction = func(t float64) float64 { return t }
	EaseInQuad    EasingFunction = func(t float64) float64 { return t * t }
	EaseOutQuad   EasingFunction = func(t float64) float64 { return t * (2 - t) }
	EaseInOutQuad EasingFunction = func(t float64) float64 {
		if t < 0.5 {
			return 2 * t * t
		}
		return -1 + (4-2*t)*t
	}
	EaseInCubic  EasingFunction = func(t float64) float64 { return t * t * t }
	EaseOutCubic EasingFunction = func(t float64) float64 {
		t--
		return t*t*t + 1
	}
	EaseInOutCubic EasingFunction = func(t float64) float64 {
		if t < 0.5 {
			return 4 * t * t * t
		}
		t--
		return (t*t*t+1)/2 + 0.5
	}
)



func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
