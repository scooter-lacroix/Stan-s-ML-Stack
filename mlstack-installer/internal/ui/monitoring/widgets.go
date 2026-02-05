// internal/ui/monitoring/widgets.go
package monitoring

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// SystemMonitorWidget displays system-wide monitoring information
type SystemMonitorWidget struct {
	width, height int
	x, y          int
	theme         *AMDTheme

	// Data
	cpuUsage    []float64
	memoryUsage []float64
	diskUsage   float64
	loadAverage []float64
	uptime      time.Duration

	// UI components
	cpuProgress    progress.Model
	memoryProgress progress.Model
	diskProgress   progress.Model
	spinner        spinner.Model

	// History (for charts)
	maxHistoryPoints int
	lastUpdate       time.Time
}

// GPUMonitorWidget displays GPU-specific monitoring information
type GPUMonitorWidget struct {
	width, height int
	x, y          int
	theme         *AMDTheme

	// GPU data
	gpuCount    int
	gpuUsage    []float64
	temperature []float64
	memoryUsed  []float64
	memoryTotal []float64
	powerUsage  []float64
	clockSpeed  []float64

	// UI components
	progressBars []progress.Model
	spinner      spinner.Model

	// AMD GPU specific
	rocmVersion   string
	driverVersion string
	gpuModel      string
	computeUnits  []int
}

// NetworkMonitorWidget displays network monitoring information
type NetworkMonitorWidget struct {
	width, height int
	x, y          int
	theme         *AMDTheme

	// Network data
	interfaces []NetworkInterface
	rxBytes    []int64
	txBytes    []int64
	rxRate     []float64
	txRate     []float64

	// History
	maxHistoryPoints int
	lastUpdate       time.Time
}

// NetworkInterface represents a network interface
type NetworkInterface struct {
	Name      string
	State     string
	IP        string
	MAC       string
	Speed     int64
	RxBytes   int64
	TxBytes   int64
	Connected bool
}

// ProcessMonitorWidget displays process monitoring information
type ProcessMonitorWidget struct {
	width, height int
	x, y          int
	theme         *AMDTheme

	// Process data
	processes    []Process
	selected     int
	scrollOffset int
	sortBy       string
	sortOrder    string

	// Filtering
	filterText   string
	showOnlyUser bool
}

// Process represents a system process
type Process struct {
	PID       int
	Name      string
	User      string
	CPU       float64
	Memory    float64
	Status    string
	Command   string
	StartTime time.Time
	RunTime   time.Duration
}

// AlertsPanelWidget displays system alerts and notifications
type AlertsPanelWidget struct {
	width, height int
	x, y          int
	theme         *AMDTheme

	// Alert data
	alerts    []Alert
	selected  int
	maxAlerts int

	// Filtering
	severityFilter map[string]bool
	categoryFilter map[string]bool
}

// Alert represents a system alert
type Alert struct {
	ID        string
	Timestamp time.Time
	Severity  string // "info", "warning", "error", "critical"
	Category  string // "system", "gpu", "network", "security"
	Title     string
	Message   string
	Source    string
	Read      bool
}

// PerformanceBarWidget displays a compact performance overview
type PerformanceBarWidget struct {
	width, height int
	x, y          int
	theme         *AMDTheme

	// Performance metrics
	cpuUsage    float64
	memoryUsage float64
	gpuUsage    []float64
	networkRate float64
	diskIO      float64

	// Status indicators
	systemStatus string
	lastUpdate   time.Time
}

// NewSystemMonitorWidget creates a new system monitor widget
func NewSystemMonitorWidget(width, height int, theme *AMDTheme) *SystemMonitorWidget {
	widget := &SystemMonitorWidget{
		width:            width,
		height:           height,
		theme:            theme,
		maxHistoryPoints: 60, // Keep 60 data points (1 minute at 1-second intervals)
		lastUpdate:       time.Now(),
	}

	// Initialize progress bars
	widget.cpuProgress = progress.New(progress.WithDefaultGradient())
	widget.memoryProgress = progress.New(progress.WithDefaultGradient())
	widget.diskProgress = progress.New(progress.WithDefaultGradient())

	// Initialize spinner
	widget.spinner = spinner.New(spinner.WithSpinner(spinner.Dot))

	// Initialize data
	widget.initializeData()

	return widget
}

// initializeData initializes the system monitor with sample data
func (w *SystemMonitorWidget) initializeData() {
	// Initialize with empty history
	w.cpuUsage = make([]float64, w.maxHistoryPoints)
	w.memoryUsage = make([]float64, w.maxHistoryPoints)
	w.loadAverage = make([]float64, 3)

	// Sample initial data
	for i := 0; i < w.maxHistoryPoints; i++ {
		w.cpuUsage[i] = 0
		w.memoryUsage[i] = 0
	}

	w.diskUsage = 0
	w.uptime = 0
	w.loadAverage = []float64{0, 0, 0}
}

// SetBounds implements the LayoutComponent interface
func (w *SystemMonitorWidget) SetBounds(x, y, width, height int) {
	w.x = x
	w.y = y
	w.width = width
	w.height = height
}

// GetBounds implements the LayoutComponent interface
func (w *SystemMonitorWidget) GetBounds() (int, int, int, int) {
	return w.x, w.y, w.width, w.height
}

// Update implements the tea.Model interface
func (w *SystemMonitorWidget) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case types.SystemStatusUpdateMsg:
		w.updateData(msg)
	case spinner.TickMsg:
		var cmd tea.Cmd
		w.spinner, cmd = w.spinner.Update(msg)
		return w, cmd
	}
	return w, nil
}

// updateData updates the widget with new system data
func (w *SystemMonitorWidget) UpdateData(msg types.SystemStatusUpdateMsg) {
	w.updateData(msg)
}

// updateData internal method to update system data
func (w *SystemMonitorWidget) updateData(msg types.SystemStatusUpdateMsg) {
	// Shift history and add new data
	w.cpuUsage = append(w.cpuUsage[1:], msg.CPUUsage)
	w.memoryUsage = append(w.memoryUsage[1:], msg.MemoryUsage)
	w.diskUsage = msg.DiskUsage

	// Update load average (simulated)
	w.loadAverage[0] = msg.CPUUsage / 100.0
	w.loadAverage[1] = msg.CPUUsage / 100.0 * 0.8
	w.loadAverage[2] = msg.CPUUsage / 100.0 * 0.6

	// Update uptime
	w.lastUpdate = time.Now()
}

// View implements the tea.Model interface
func (w *SystemMonitorWidget) View() string {
	if w.width < 20 || w.height < 5 {
		return w.theme.Text.Render("System Monitor (too small)")
	}

	// Build content sections
	var sections []string

	// Header
	header := w.theme.Header.Render("🖥️  System Monitor")
	sections = append(sections, header)

	// CPU Section
	cpuSection := w.renderCPUSection()
	sections = append(sections, cpuSection)

	// Memory Section
	memorySection := w.renderMemorySection()
	sections = append(sections, memorySection)

	// Disk Section
	diskSection := w.renderDiskSection()
	sections = append(sections, diskSection)

	// System Info Section
	systemSection := w.renderSystemInfo()
	sections = append(sections, systemSection)

	// Combine all sections
	content := strings.Join(sections, "\n\n")

	return w.theme.Panel.Width(w.width).Height(w.height).Render(content)
}

// renderCPUSection renders the CPU monitoring section
func (w *SystemMonitorWidget) renderCPUSection() string {
	currentCPU := w.cpuUsage[len(w.cpuUsage)-1]
	cpuPercent := int(currentCPU)

	// Update progress bar
	w.cpuProgress.SetPercent(float64(cpuPercent) / 100.0)

	// Create CPU graph
	graph := w.createSparkline(w.cpuUsage, 20)

	cpuInfo := fmt.Sprintf("CPU: %s%% %s",
		w.theme.GetMetricStyle(currentCPU, 80).Render(fmt.Sprintf("%d", cpuPercent)),
		graph)

	// Load average
	loadAvg := fmt.Sprintf("Load: %.2f, %.2f, %.2f",
		w.loadAverage[0], w.loadAverage[1], w.loadAverage[2])

	// CPU cores info
	cores := "Cores: 8 (16 threads)" // This would be dynamic

	cpuContent := lipgloss.JoinHorizontal(lipgloss.Top, cpuInfo, "  ", loadAvg, "\n", cores)

	return w.theme.SubTitle.Render("📊 CPU Usage") + "\n" + cpuContent
}

// renderMemorySection renders the memory monitoring section
func (w *SystemMonitorWidget) renderMemorySection() string {
	currentMemory := w.memoryUsage[len(w.memoryUsage)-1]
	memoryPercent := int(currentMemory)

	// Update progress bar
	w.memoryProgress.SetPercent(float64(memoryPercent) / 100.0)

	// Create memory graph
	graph := w.createSparkline(w.memoryUsage, 20)

	memInfo := fmt.Sprintf("RAM: %s%% %s",
		w.theme.GetMetricStyle(currentMemory, 90).Render(fmt.Sprintf("%d", memoryPercent)),
		graph)

	// Memory details
	totalMem := "32.0 GB" // This would be dynamic
	usedMem := fmt.Sprintf("%.1f GB", float64(memoryPercent)*0.32)
	memDetails := fmt.Sprintf("Used: %s / %s", usedMem, totalMem)

	memContent := lipgloss.JoinHorizontal(lipgloss.Top, memInfo, "  ", memDetails)

	return w.theme.SubTitle.Render("💾 Memory Usage") + "\n" + memContent
}

// renderDiskSection renders the disk monitoring section
func (w *SystemMonitorWidget) renderDiskSection() string {
	diskPercent := int(w.diskUsage)

	// Update progress bar
	w.diskProgress.SetPercent(float64(diskPercent) / 100.0)

	diskInfo := fmt.Sprintf("Disk: %s%%",
		w.theme.GetMetricStyle(w.diskUsage, 85).Render(fmt.Sprintf("%d", diskPercent)))

	// Disk details
	totalDisk := "1.0 TB"
	usedDisk := fmt.Sprintf("%.1f GB", float64(diskPercent)*10.0)
	diskDetails := fmt.Sprintf("Used: %s / %s", usedDisk, totalDisk)

	diskContent := lipgloss.JoinHorizontal(lipgloss.Top, diskInfo, "  ", diskDetails)

	return w.theme.SubTitle.Render("💿 Disk Usage") + "\n" + diskContent
}

// renderSystemInfo renders general system information
func (w *SystemMonitorWidget) renderSystemInfo() string {
	uptime := w.formatDuration(w.uptime)
	systemInfo := fmt.Sprintf("Uptime: %s", uptime)

	status := "🟢 Healthy"
	if w.cpuUsage[len(w.cpuUsage)-1] > 80 || w.memoryUsage[len(w.memoryUsage)-1] > 90 {
		status = "🟡 Elevated"
	}
	if w.cpuUsage[len(w.cpuUsage)-1] > 95 || w.memoryUsage[len(w.memoryUsage)-1] > 95 {
		status = "🔴 Critical"
	}

	content := lipgloss.JoinHorizontal(lipgloss.Top, systemInfo, "  Status: ", status)

	return w.theme.SubTitle.Render("ℹ️  System") + "\n" + content
}

// createSparkline creates a simple sparkline graph
func (w *SystemMonitorWidget) createSparkline(data []float64, width int) string {
	if len(data) < width {
		width = len(data)
	}

	// Take the last 'width' points
	start := len(data) - width
	end := len(data)
	points := data[start:end]

	if len(points) == 0 {
		return "▁▂▃▄▅▆▇█"
	}

	// Find min and max for scaling
	min, max := points[0], points[0]
	for _, p := range points {
		if p < min {
			min = p
		}
		if p > max {
			max = p
		}
	}

	// Create sparkline
	sparklineChars := []rune(" ▁▂▃▄▅▆▇█")
	var sparkline strings.Builder

	for _, p := range points {
		if max == min {
			sparkline.WriteRune(sparklineChars[len(sparklineChars)/2])
		} else {
			normalized := (p - min) / (max - min)
			index := int(normalized * float64(len(sparklineChars)-1))
			if index < 0 {
				index = 0
			}
			if index >= len(sparklineChars) {
				index = len(sparklineChars) - 1
			}
			sparkline.WriteRune(sparklineChars[index])
		}
	}

	return sparkline.String()
}

// formatDuration formats a duration in a human-readable way
func (w *SystemMonitorWidget) formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	} else if d < time.Hour {
		return fmt.Sprintf("%dm %ds", int(d.Minutes()), int(d.Seconds())%60)
	} else if d < 24*time.Hour {
		return fmt.Sprintf("%dh %dm", int(d.Hours()), int(d.Minutes())%60)
	} else {
		days := int(d.Hours()) / 24
		hours := int(d.Hours()) % 24
		return fmt.Sprintf("%dd %dh", days, hours)
	}
}

// NewGPUMonitorWidget creates a new GPU monitor widget
func NewGPUMonitorWidget(width, height int, theme *AMDTheme) *GPUMonitorWidget {
	widget := &GPUMonitorWidget{
		width:        width,
		height:       height,
		theme:        theme,
		gpuCount:     2, // This would be detected dynamically
		gpuUsage:     make([]float64, 2),
		temperature:  make([]float64, 2),
		memoryUsed:   make([]float64, 2),
		memoryTotal:  make([]float64, 2),
		powerUsage:   make([]float64, 2),
		clockSpeed:   make([]float64, 2),
		computeUnits: []int{60, 60}, // This would be detected dynamically
	}

	// Initialize progress bars
	widget.progressBars = make([]progress.Model, widget.gpuCount)
	for i := range widget.progressBars {
		widget.progressBars[i] = progress.New(progress.WithDefaultGradient())
	}

	// Initialize spinner
	widget.spinner = spinner.New(spinner.WithSpinner(spinner.Dot))

	// Initialize with sample data
	for i := 0; i < widget.gpuCount; i++ {
		widget.gpuUsage[i] = 0
		widget.temperature[i] = 45.0
		widget.memoryUsed[i] = 8.0
		widget.memoryTotal[i] = 16.0
		widget.powerUsage[i] = 250.0
		widget.clockSpeed[i] = 2100.0
	}

	widget.rocmVersion = "6.4.43482"
	widget.driverVersion = "23.40.4"
	widget.gpuModel = "AMD Radeon RX 7900 XTX"

	return widget
}

// SetBounds implements the LayoutComponent interface
func (w *GPUMonitorWidget) SetBounds(x, y, width, height int) {
	w.x = x
	w.y = y
	w.width = width
	w.height = height
}

// GetBounds implements the LayoutComponent interface
func (w *GPUMonitorWidget) GetBounds() (int, int, int, int) {
	return w.x, w.y, w.width, w.height
}

// Update implements the tea.Model interface
func (w *GPUMonitorWidget) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case types.SystemStatusUpdateMsg:
		w.updateData(msg)
	case spinner.TickMsg:
		var cmd tea.Cmd
		w.spinner, cmd = w.spinner.Update(msg)
		return w, cmd
	}
	return w, nil
}

// updateData updates the widget with new GPU data
func (w *GPUMonitorWidget) UpdateData(msg types.SystemStatusUpdateMsg) {
	// Update GPU usage from system status message
	if len(msg.GPUUsage) > 0 {
		for i := 0; i < w.gpuCount && i < len(msg.GPUUsage); i++ {
			w.gpuUsage[i] = msg.GPUUsage[i]
			// Update progress bars
			w.progressBars[i].SetPercent(msg.GPUUsage[i] / 100.0)
		}
	}

	// Simulate other GPU metrics
	for i := 0; i < w.gpuCount; i++ {
		// Temperature increases with usage
		w.temperature[i] = 45.0 + (w.gpuUsage[i] * 0.35)

		// Memory usage correlates with GPU usage
		w.memoryUsed[i] = 8.0 + (w.gpuUsage[i] * 0.08)

		// Power usage correlates with GPU usage
		w.powerUsage[i] = 150.0 + (w.gpuUsage[i] * 2.5)

		// Clock speed varies with load
		w.clockSpeed[i] = 1800.0 + (w.gpuUsage[i] * 3.0)
	}
}

// View implements the tea.Model interface
func (w *GPUMonitorWidget) View() string {
	if w.width < 20 || w.height < 5 {
		return w.theme.Text.Render("GPU Monitor (too small)")
	}

	var sections []string

	// Header
	header := w.theme.Header.Render("🎮 GPU Monitor")
	sections = append(sections, header)

	// GPU Model and ROCm info
	modelInfo := fmt.Sprintf("%s\nROCm: %s | Driver: %s",
		w.gpuModel, w.rocmVersion, w.driverVersion)
	sections = append(sections, w.theme.MutedText.Render(modelInfo))

	// GPU information for each GPU
	for i := 0; i < w.gpuCount; i++ {
		gpuSection := w.renderGPUInfo(i)
		sections = append(sections, gpuSection)
	}

	content := strings.Join(sections, "\n\n")

	return w.theme.Panel.Width(w.width).Height(w.height).Render(content)
}

// renderGPUInfo renders information for a specific GPU
func (w *GPUMonitorWidget) renderGPUInfo(gpuID int) string {
	usage := w.gpuUsage[gpuID]
	temp := w.temperature[gpuID]
	memUsed := w.memoryUsed[gpuID]
	memTotal := w.memoryTotal[gpuID]
	power := w.powerUsage[gpuID]
	clock := w.clockSpeed[gpuID]

	// Usage with color coding
	usageStyle := w.theme.GetMetricStyle(usage, 80)
	usageText := fmt.Sprintf("GPU %d: %s%%", gpuID, usageStyle.Render(fmt.Sprintf("%.1f", usage)))

	// Temperature with warning for high temps
	tempStyle := w.theme.Text
	if temp > 80 {
		tempStyle = w.theme.WarningAlert
	} else if temp > 70 {
		tempStyle = w.theme.GetStatusStyle("warning")
	}
	tempText := fmt.Sprintf("Temp: %s°C", tempStyle.Render(fmt.Sprintf("%.1f", temp)))

	// Memory usage
	memPercent := (memUsed / memTotal) * 100
	memStyle := w.theme.GetMetricStyle(memPercent, 90)
	memText := fmt.Sprintf("VRAM: %s/%.0fGB (%.1f%%)",
		memStyle.Render(fmt.Sprintf("%.1f", memUsed)), memTotal, memPercent)

	// Power usage
	powerText := fmt.Sprintf("Power: %.0fW", power)

	// Clock speed
	clockText := fmt.Sprintf("Clock: %.0fMHz", clock)

	// Combine metrics
	line1 := lipgloss.JoinHorizontal(lipgloss.Top, usageText, "  ", tempText)
	line2 := lipgloss.JoinHorizontal(lipgloss.Top, memText, "  ", powerText, "  ", clockText)

	return fmt.Sprintf("GPU %d\n%s\n%s", gpuID, line1, line2)
}

// NewNetworkMonitorWidget creates a new network monitor widget
func NewNetworkMonitorWidget(width, height int, theme *AMDTheme) *NetworkMonitorWidget {
	widget := &NetworkMonitorWidget{
		width:            width,
		height:           height,
		theme:            theme,
		maxHistoryPoints: 60,
		lastUpdate:       time.Now(),
		interfaces: []NetworkInterface{
			{
				Name:      "eth0",
				State:     "up",
				IP:        "192.168.1.100",
				MAC:       "00:1A:2B:3C:4D:5E",
				Speed:     1000000000, // 1 Gbps
				Connected: true,
			},
		},
		rxBytes: make([]int64, 60),
		txBytes: make([]int64, 60),
		rxRate:  make([]float64, 60),
		txRate:  make([]float64, 60),
	}

	return widget
}

// SetBounds implements the LayoutComponent interface
func (w *NetworkMonitorWidget) SetBounds(x, y, width, height int) {
	w.x = x
	w.y = y
	w.width = width
	w.height = height
}

// GetBounds implements the LayoutComponent interface
func (w *NetworkMonitorWidget) GetBounds() (int, int, int, int) {
	return w.x, w.y, w.width, w.height
}

// Update implements the tea.Model interface
func (w *NetworkMonitorWidget) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	return w, nil
}

// View implements the tea.Model interface
func (w *NetworkMonitorWidget) View() string {
	if w.width < 20 || w.height < 5 {
		return w.theme.Text.Render("Network Monitor (too small)")
	}

	var sections []string

	// Header
	header := w.theme.Header.Render("🌐 Network Monitor")
	sections = append(sections, header)

	// Network interface info
	for _, iface := range w.interfaces {
		ifaceSection := w.renderInterface(iface)
		sections = append(sections, ifaceSection)
	}

	content := strings.Join(sections, "\n\n")

	return w.theme.Panel.Width(w.width).Height(w.height).Render(content)
}

// renderInterface renders information for a network interface
func (w *NetworkMonitorWidget) renderInterface(iface NetworkInterface) string {
	status := "🟢"
	if !iface.Connected {
		status = "🔴"
	}

	// Interface name and status
	nameLine := fmt.Sprintf("%s %s (%s)", status, iface.Name, iface.State)

	// IP and speed
	ipLine := fmt.Sprintf("IP: %s | Speed: %s", iface.IP, w.formatSpeed(iface.Speed))

	// Traffic rates (simulated)
	rxRate := 1024.5 * 1024 // 1 MB/s
	txRate := 512.3 * 1024  // 512 KB/s
	trafficLine := fmt.Sprintf("↓ %s/s | ↑ %s/s",
		w.formatBytes(rxRate), w.formatBytes(txRate))

	return lipgloss.JoinVertical(lipgloss.Left, nameLine, ipLine, trafficLine)
}

// formatSpeed formats network speed in human-readable format
func (w *NetworkMonitorWidget) formatSpeed(bps int64) string {
	if bps >= 1000000000 {
		return fmt.Sprintf("%.1f Gbps", float64(bps)/1000000000)
	} else if bps >= 1000000 {
		return fmt.Sprintf("%.1f Mbps", float64(bps)/1000000)
	} else if bps >= 1000 {
		return fmt.Sprintf("%.1f Kbps", float64(bps)/1000)
	}
	return fmt.Sprintf("%d bps", bps)
}

// formatBytes formats bytes in human-readable format
func (w *NetworkMonitorWidget) formatBytes(bytes float64) string {
	units := []string{"B", "KB", "MB", "GB", "TB"}

	for _, unit := range units {
		if bytes < 1024 {
			return fmt.Sprintf("%.1f %s", bytes, unit)
		}
		bytes /= 1024
	}

	return fmt.Sprintf("%.1f PB", bytes)
}

// NewProcessMonitorWidget creates a new process monitor widget
func NewProcessMonitorWidget(width, height int, theme *AMDTheme) *ProcessMonitorWidget {
	widget := &ProcessMonitorWidget{
		width:        width,
		height:       height,
		theme:        theme,
		selected:     0,
		scrollOffset: 0,
		sortBy:       "CPU",
		sortOrder:    "desc",
		filterText:   "",
		showOnlyUser: true,
	}

	// Initialize with sample processes
	widget.initializeProcesses()

	return widget
}

// initializeProcesses initializes the process monitor with sample data
func (w *ProcessMonitorWidget) initializeProcesses() {
	w.processes = []Process{
		{
			PID:       1234,
			Name:      "python3",
			User:      "user",
			CPU:       25.5,
			Memory:    2.8,
			Status:    "Running",
			Command:   "python3 train_model.py",
			StartTime: time.Now().Add(-2 * time.Hour),
			RunTime:   2 * time.Hour,
		},
		{
			PID:       5678,
			Name:      "chrome",
			User:      "user",
			CPU:       15.2,
			Memory:    1.5,
			Status:    "Running",
			Command:   "/usr/bin/chrome",
			StartTime: time.Now().Add(-1 * time.Hour),
			RunTime:   1 * time.Hour,
		},
		{
			PID:       9012,
			Name:      "rocm-smi",
			User:      "user",
			CPU:       0.1,
			Memory:    0.05,
			Status:    "Running",
			Command:   "/opt/rocm/bin/rocm-smi",
			StartTime: time.Now().Add(-5 * time.Minute),
			RunTime:   5 * time.Minute,
		},
	}
}

// SetBounds implements the LayoutComponent interface
func (w *ProcessMonitorWidget) SetBounds(x, y, width, height int) {
	w.x = x
	w.y = y
	w.width = width
	w.height = height
}

// GetBounds implements the LayoutComponent interface
func (w *ProcessMonitorWidget) GetBounds() (int, int, int, int) {
	return w.x, w.y, w.width, w.height
}

// Update implements the tea.Model interface
func (w *ProcessMonitorWidget) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	return w, nil
}

// View implements the tea.Model interface
func (w *ProcessMonitorWidget) View() string {
	if w.width < 20 || w.height < 5 {
		return w.theme.Text.Render("Process Monitor (too small)")
	}

	var sections []string

	// Header
	header := w.theme.Header.Render("⚙️  Process Monitor")
	sections = append(sections, header)

	// Process table
	tableHeader := w.renderTableHeader()
	sections = append(sections, tableHeader)

	// Process rows (limited by height)
	maxRows := w.height - 6 // Account for header, borders, etc.
	start := w.scrollOffset
	end := start + maxRows
	if end > len(w.processes) {
		end = len(w.processes)
	}

	for i := start; i < end; i++ {
		row := w.renderProcessRow(w.processes[i], i)
		sections = append(sections, row)
	}

	content := strings.Join(sections, "\n")

	return w.theme.Panel.Width(w.width).Height(w.height).Render(content)
}

// renderTableHeader renders the process table header
func (w *ProcessMonitorWidget) renderTableHeader() string {
	return fmt.Sprintf("%-8s %-15s %-8s %-8s %-10s %s",
		"PID", "NAME", "CPU%", "MEM%", "USER", "STATUS")
}

// renderProcessRow renders a single process row
func (w *ProcessMonitorWidget) renderProcessRow(process Process, index int) string {
	style := w.theme.Text
	if index == w.selected {
		style = w.theme.Highlight
	}

	cpuStyle := w.theme.GetMetricStyle(process.CPU, 20)
	memStyle := w.theme.GetMetricStyle(process.Memory, 5)

	return fmt.Sprintf("%-8d %-15.15s %-8s %-8s %-10.10s %s",
		process.PID,
		process.Name,
		cpuStyle.Render(fmt.Sprintf("%.1f", process.CPU)),
		memStyle.Render(fmt.Sprintf("%.1f", process.Memory)),
		process.User,
		style.Render(process.Status))
}

// NewAlertsPanelWidget creates a new alerts panel widget
func NewAlertsPanelWidget(width, height int, theme *AMDTheme) *AlertsPanelWidget {
	widget := &AlertsPanelWidget{
		width:          width,
		height:         height,
		theme:          theme,
		selected:       0,
		maxAlerts:      10,
		severityFilter: map[string]bool{"info": true, "warning": true, "error": true, "critical": true},
		categoryFilter: map[string]bool{"system": true, "gpu": true, "network": true, "security": true},
	}

	// Initialize with sample alerts
	widget.initializeAlerts()

	return widget
}

// initializeAlerts initializes the alerts panel with sample data
func (w *AlertsPanelWidget) initializeAlerts() {
	w.alerts = []Alert{
		{
			ID:        "1",
			Timestamp: time.Now().Add(-5 * time.Minute),
			Severity:  "warning",
			Category:  "system",
			Title:     "High CPU Usage",
			Message:   "CPU usage has exceeded 80% for more than 5 minutes",
			Source:    "system-monitor",
			Read:      false,
		},
		{
			ID:        "2",
			Timestamp: time.Now().Add(-10 * time.Minute),
			Severity:  "info",
			Category:  "gpu",
			Title:     "GPU Temperature Normal",
			Message:   "All GPU temperatures are within normal operating range",
			Source:    "rocm-smi",
			Read:      true,
		},
	}
}

// SetBounds implements the LayoutComponent interface
func (w *AlertsPanelWidget) SetBounds(x, y, width, height int) {
	w.x = x
	w.y = y
	w.width = width
	w.height = height
}

// GetBounds implements the LayoutComponent interface
func (w *AlertsPanelWidget) GetBounds() (int, int, int, int) {
	return w.x, w.y, w.width, w.height
}

// Update implements the tea.Model interface
func (w *AlertsPanelWidget) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	return w, nil
}

// View implements the tea.Model interface
func (w *AlertsPanelWidget) View() string {
	if w.width < 20 || w.height < 5 {
		return w.theme.Text.Render("Alerts Panel (too small)")
	}

	var sections []string

	// Header
	header := w.theme.Header.Render("🚨 System Alerts")
	sections = append(sections, header)

	// Alert count
	unreadCount := 0
	for _, alert := range w.alerts {
		if !alert.Read {
			unreadCount++
		}
	}
	countText := fmt.Sprintf("%d alerts (%d unread)", len(w.alerts), unreadCount)
	sections = append(sections, w.theme.MutedText.Render(countText))

	// Alert items
	maxRows := w.height - 8 // Account for header, count, borders
	for i := 0; i < len(w.alerts) && i < maxRows; i++ {
		alert := w.alerts[i]
		alertItem := w.renderAlert(alert, i)
		sections = append(sections, alertItem)
	}

	content := strings.Join(sections, "\n")

	return w.theme.Panel.Width(w.width).Height(w.height).Render(content)
}

// renderAlert renders a single alert item
func (w *AlertsPanelWidget) renderAlert(alert Alert, index int) string {
	// Determine icon based on severity
	icon := "ℹ️"
	style := w.theme.Alert
	switch alert.Severity {
	case "warning":
		icon = "⚠️"
		style = w.theme.WarningAlert
	case "error":
		icon = "❌"
		style = w.theme.ErrorAlert
	case "critical":
		icon = "🔴"
		style = w.theme.ErrorAlert
	}

	// Unread indicator
	unread := ""
	if !alert.Read {
		unread = "● "
	}

	// Time ago
	timeAgo := w.formatTimeAgo(time.Since(alert.Timestamp))

	// Alert title and time
	title := fmt.Sprintf("%s%s%s (%s)", unread, icon, alert.Title, timeAgo)

	// Alert message (truncated)
	message := alert.Message
	if len(message) > 50 {
		message = message[:47] + "..."
	}

	return style.Render(title + "\n" + message)
}

// formatTimeAgo formats time duration in a human-readable way
func (w *AlertsPanelWidget) formatTimeAgo(d time.Duration) string {
	if d < time.Minute {
		return "now"
	} else if d < time.Hour {
		return fmt.Sprintf("%dm", int(d.Minutes()))
	} else if d < 24*time.Hour {
		return fmt.Sprintf("%dh", int(d.Hours()))
	} else {
		return fmt.Sprintf("%dd", int(d.Hours())/24)
	}
}

// AddAlert adds a new alert to the panel
func (w *AlertsPanelWidget) AddAlert(severity, message, category string) {
	alert := Alert{
		ID:        fmt.Sprintf("%d", time.Now().UnixNano()),
		Timestamp: time.Now(),
		Severity:  severity,
		Category:  category,
		Title:     strings.Title(severity),
		Message:   message,
		Source:    "monitor",
		Read:      false,
	}

	w.alerts = append([]Alert{alert}, w.alerts...)

	// Keep only the most recent alerts
	if len(w.alerts) > w.maxAlerts {
		w.alerts = w.alerts[:w.maxAlerts]
	}
}

// NewPerformanceBarWidget creates a new performance bar widget
func NewPerformanceBarWidget(width, height int, theme *AMDTheme) *PerformanceBarWidget {
	widget := &PerformanceBarWidget{
		width:      width,
		height:     height,
		theme:      theme,
		lastUpdate: time.Now(),
	}

	return widget
}

// SetBounds implements the LayoutComponent interface
func (w *PerformanceBarWidget) SetBounds(x, y, width, height int) {
	w.x = x
	w.y = y
	w.width = width
	w.height = height
}

// GetBounds implements the LayoutComponent interface
func (w *PerformanceBarWidget) GetBounds() (int, int, int, int) {
	return w.x, w.y, w.width, w.height
}

// Update implements the tea.Model interface
func (w *PerformanceBarWidget) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case types.SystemStatusUpdateMsg:
		w.UpdateData(msg)
	}
	return w, nil
}

// UpdateData updates the widget with new performance data
func (w *PerformanceBarWidget) UpdateData(msg types.SystemStatusUpdateMsg) {
	w.cpuUsage = msg.CPUUsage
	w.memoryUsage = msg.MemoryUsage
	w.gpuUsage = msg.GPUUsage
	w.lastUpdate = time.Now()

	// Update system status
	if w.cpuUsage > 80 || w.memoryUsage > 90 {
		w.systemStatus = "degraded"
	} else if w.cpuUsage > 95 || w.memoryUsage > 95 {
		w.systemStatus = "critical"
	} else {
		w.systemStatus = "healthy"
	}
}

// View implements the tea.Model interface
func (w *PerformanceBarWidget) View() string {
	if w.width < 20 {
		return ""
	}

	// Create progress indicators
	cpuStyle := w.theme.GetProgressBarStyle(w.cpuUsage)
	memoryStyle := w.theme.GetProgressBarStyle(w.memoryUsage)

	cpuBar := cpuStyle.Render(fmt.Sprintf("CPU %3.0f%%", w.cpuUsage))
	memoryBar := memoryStyle.Render(fmt.Sprintf("RAM %3.0f%%", w.memoryUsage))

	// GPU usage (if available)
	var gpuBar string
	if len(w.gpuUsage) > 0 {
		gpuStyle := w.theme.GetProgressBarStyle(w.gpuUsage[0])
		gpuBar = gpuStyle.Render(fmt.Sprintf("GPU %3.0f%%", w.gpuUsage[0]))
	}

	// System status indicator
	statusIcon := "🟢"
	statusText := "Healthy"
	switch w.systemStatus {
	case "degraded":
		statusIcon = "🟡"
		statusText = "Elevated"
	case "critical":
		statusIcon = "🔴"
		statusText = "Critical"
	}

	status := fmt.Sprintf("%s %s", statusIcon, statusText)

	// Combine all elements
	var elements []string
	elements = append(elements, cpuBar, memoryBar)
	if gpuBar != "" {
		elements = append(elements, gpuBar)
	}
	elements = append(elements, status)

	// Create horizontal layout with spacing
	return w.theme.Container.
		Width(w.width).
		Height(w.height).
		Render(lipgloss.JoinHorizontal(lipgloss.Left, elements...))
}
