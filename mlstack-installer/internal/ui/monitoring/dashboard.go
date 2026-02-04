// internal/ui/monitoring/dashboard.go
package monitoring

import (
	"fmt"
	"time"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// Dashboard represents the main monitoring dashboard component
type Dashboard struct {
	// Layout and dimensions
	width, height int
	x, y          int

	// Components
	systemMonitor  *SystemMonitorWidget
	gpuMonitor     *GPUMonitorWidget
	networkMonitor *NetworkMonitorWidget
	processMonitor *ProcessMonitorWidget
	alertsPanel    *AlertsPanelWidget
	performanceBar *PerformanceBarWidget

	// Layout configuration
	gridLayout     GridLayout
	activePanel    string
	panelCollapsed map[string]bool

	// Theme and styling
	theme *AMDTheme

	// Help and navigation
	help   help.Model
	keyMap DashboardKeyMap

	// State
	lastUpdate      time.Time
	refreshInterval time.Duration
	isRefreshing    bool
}

// GridLayout defines the dashboard grid layout
type GridLayout struct {
	Columns int
	Rows    int
	Cells   []GridCell
}

// GridCell represents a cell in the grid layout
type GridCell struct {
	Row         int
	Col         int
	RowSpan     int
	ColSpan     int
	Component   string
	MinWidth    int
	MinHeight   int
	Priority    int
	Resizable   bool
	Collapsible bool
}

// DashboardKeyMap defines key bindings for the dashboard
type DashboardKeyMap struct {
	Refresh    key.Binding
	Quit       key.Binding
	Navigation struct {
		Up    key.Binding
		Down  key.Binding
		Left  key.Binding
		Right key.Binding
	}
	Panels struct {
		Toggle   key.Binding
		Collapse key.Binding
		Maximize key.Binding
		Reset    key.Binding
	}
	Theme      key.Binding
	Help       key.Binding
}

// FullHelp returns all keybindings for the dashboard
func (k DashboardKeyMap) FullHelp() [][]key.Binding {
	return [][]key.Binding{
		{k.Refresh, k.Quit, k.Theme, k.Help},
		{k.Navigation.Up, k.Navigation.Down, k.Navigation.Left, k.Navigation.Right},
		{k.Panels.Toggle, k.Panels.Collapse, k.Panels.Maximize, k.Panels.Reset},
	}
}

// NewDashboard creates a new monitoring dashboard instance
func NewDashboard(width, height int) *Dashboard {
	theme := NewAMDTheme()

	dashboard := &Dashboard{
		width:           width,
		height:          height,
		x:               0,
		y:               0,
		theme:           theme,
		help:            help.New(),
		keyMap:          defaultDashboardKeyMap(),
		lastUpdate:      time.Now(),
		refreshInterval: 1 * time.Second,
		activePanel:     "system",
		panelCollapsed:  make(map[string]bool),
	}

	// Initialize grid layout
	dashboard.initializeGridLayout()

	// Initialize components
	dashboard.initializeComponents()

	return dashboard
}

// defaultDashboardKeyMap returns the default key bindings
func defaultDashboardKeyMap() DashboardKeyMap {
	return DashboardKeyMap{
		Refresh: key.NewBinding(
			key.WithKeys("r"),
			key.WithHelp("r", "refresh"),
		),
		Quit: key.NewBinding(
			key.WithKeys("q", "esc"),
			key.WithHelp("q/esc", "quit"),
		),
		Navigation: struct {
			Up    key.Binding
			Down  key.Binding
			Left  key.Binding
			Right key.Binding
		}{
			Up: key.NewBinding(
				key.WithKeys("up", "k"),
				key.WithHelp("↑/k", "navigate up"),
			),
			Down: key.NewBinding(
				key.WithKeys("down", "j"),
				key.WithHelp("↓/j", "navigate down"),
			),
			Left: key.NewBinding(
				key.WithKeys("left", "h"),
				key.WithHelp("←/h", "navigate left"),
			),
			Right: key.NewBinding(
				key.WithKeys("right", "l"),
				key.WithHelp("→/l", "navigate right"),
			),
		},
		Panels: struct {
			Toggle   key.Binding
			Collapse key.Binding
			Maximize key.Binding
			Reset    key.Binding
		}{
			Toggle: key.NewBinding(
				key.WithKeys("tab"),
				key.WithHelp("tab", "toggle panel"),
			),
			Collapse: key.NewBinding(
				key.WithKeys("c"),
				key.WithHelp("c", "collapse panel"),
			),
			Maximize: key.NewBinding(
				key.WithKeys("m"),
				key.WithHelp("m", "maximize panel"),
			),
			Reset: key.NewBinding(
				key.WithKeys("x"),
				key.WithHelp("x", "reset layout"),
			),
		},
		Theme: struct {
			Cycle key.Binding
			Dark  key.Binding
			Light key.Binding
		}{
			Cycle: key.NewBinding(
				key.WithKeys("t"),
				key.WithHelp("t", "cycle theme"),
			),
			Dark: key.NewBinding(
				key.WithKeys("d"),
				key.WithHelp("d", "dark theme"),
			),
			Light: key.NewBinding(
				key.WithKeys("l"),
				key.WithHelp("l", "light theme"),
			),
		},
		Help: struct {
			Show   key.Binding
			Toggle key.Binding
		}{
			Show: key.NewBinding(
				key.WithKeys("?"),
				key.WithHelp("?", "show help"),
			),
			Toggle: key.NewBinding(
				key.WithKeys("f1"),
				key.WithHelp("F1", "toggle help"),
			),
		},
	}
}

// initializeGridLayout sets up the default grid layout
func (d *Dashboard) initializeGridLayout() {
	d.gridLayout = GridLayout{
		Columns: 3,
		Rows:    3,
		Cells: []GridCell{
			// System Monitor (top-left, 2x2)
			{
				Row: 0, Col: 0, RowSpan: 2, ColSpan: 2,
				Component: "system", MinWidth: 40, MinHeight: 15,
				Priority: 1, Resizable: true, Collapsible: false,
			},
			// GPU Monitor (top-right, 1x1)
			{
				Row: 0, Col: 2, RowSpan: 1, ColSpan: 1,
				Component: "gpu", MinWidth: 30, MinHeight: 8,
				Priority: 2, Resizable: true, Collapsible: true,
			},
			// Network Monitor (middle-right, 1x1)
			{
				Row: 1, Col: 2, RowSpan: 1, ColSpan: 1,
				Component: "network", MinWidth: 30, MinHeight: 8,
				Priority: 3, Resizable: true, Collapsible: true,
			},
			// Process Monitor (bottom-left, 1x2)
			{
				Row: 2, Col: 0, RowSpan: 1, ColSpan: 2,
				Component: "process", MinWidth: 40, MinHeight: 10,
				Priority: 4, Resizable: true, Collapsible: true,
			},
			// Alerts Panel (bottom-right, 1x1)
			{
				Row: 2, Col: 2, RowSpan: 1, ColSpan: 1,
				Component: "alerts", MinWidth: 30, MinHeight: 10,
				Priority: 5, Resizable: true, Collapsible: true,
			},
		},
	}
}

// initializeComponents creates all dashboard components
func (d *Dashboard) initializeComponents() {
	// Calculate component dimensions based on grid layout
	dimensions := d.calculateComponentDimensions()

	// System Monitor
	d.systemMonitor = NewSystemMonitorWidget(
		dimensions["system"].width,
		dimensions["system"].height,
		d.theme,
	)

	// GPU Monitor
	d.gpuMonitor = NewGPUMonitorWidget(
		dimensions["gpu"].width,
		dimensions["gpu"].height,
		d.theme,
	)

	// Network Monitor
	d.networkMonitor = NewNetworkMonitorWidget(
		dimensions["network"].width,
		dimensions["network"].height,
		d.theme,
	)

	// Process Monitor
	d.processMonitor = NewProcessMonitorWidget(
		dimensions["process"].width,
		dimensions["process"].height,
		d.theme,
	)

	// Alerts Panel
	d.alertsPanel = NewAlertsPanelWidget(
		dimensions["alerts"].width,
		dimensions["alerts"].height,
		d.theme,
	)

	// Performance Bar (always visible at bottom)
	d.performanceBar = NewPerformanceBarWidget(
		d.width,
		3, // Fixed height for performance bar
		d.theme,
	)
}

// calculateComponentDimensions calculates dimensions for each component based on grid layout
func (d *Dashboard) calculateComponentDimensions() map[string]struct{ width, height int } {
	dimensions := make(map[string]struct{ width, height int })

	cellWidth := d.width / d.gridLayout.Columns
	cellHeight := (d.height - 3) / d.gridLayout.Columns // Reserve space for performance bar

	for _, cell := range d.gridLayout.Cells {
		if d.panelCollapsed[cell.Component] {
			dimensions[cell.Component] = struct{ width, height int }{
				width:  cell.MinWidth,
				height: 3, // Collapsed height
			}
			continue
		}

		width := cellWidth * cell.ColSpan
		height := cellHeight * cell.RowSpan

		// Ensure minimum dimensions
		if width < cell.MinWidth {
			width = cell.MinWidth
		}
		if height < cell.MinHeight {
			height = cell.MinHeight
		}

		dimensions[cell.Component] = struct{ width, height int }{
			width:  width,
			height: height,
		}
	}

	return dimensions
}

// SetBounds implements the LayoutComponent interface
func (d *Dashboard) SetBounds(x, y, width, height int) {
	d.x = x
	d.y = y
	d.width = width
	d.height = height

	// Recalculate component dimensions
	dimensions := d.calculateComponentDimensions()

	// Update component bounds
	if d.systemMonitor != nil {
		d.systemMonitor.SetBounds(0, 0, dimensions["system"].width, dimensions["system"].height)
	}
	if d.gpuMonitor != nil {
		d.gpuMonitor.SetBounds(0, 0, dimensions["gpu"].width, dimensions["gpu"].height)
	}
	if d.networkMonitor != nil {
		d.networkMonitor.SetBounds(0, 0, dimensions["network"].width, dimensions["network"].height)
	}
	if d.processMonitor != nil {
		d.processMonitor.SetBounds(0, 0, dimensions["process"].width, dimensions["process"].height)
	}
	if d.alertsPanel != nil {
		d.alertsPanel.SetBounds(0, 0, dimensions["alerts"].width, dimensions["alerts"].height)
	}
	if d.performanceBar != nil {
		d.performanceBar.SetBounds(0, 0, d.width, 3)
	}
}

// GetBounds implements the LayoutComponent interface
func (d *Dashboard) GetBounds() (int, int, int, int) {
	return d.x, d.y, d.width, d.height
}

// Init implements the tea.Model interface
func (d *Dashboard) Init() tea.Cmd {
	// Start with a refresh command
	return tea.Batch(
		d.refreshCommand(),
		d.tickCommand(),
	)
}

// Update implements the tea.Model interface
func (d *Dashboard) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd
	var updatedModel tea.Model // Declare updatedModel here

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch {
		case key.Matches(msg, d.keyMap.Quit):
			return d, tea.Quit
		case key.Matches(msg, d.keyMap.Refresh):
			// Handle refresh
		case key.Matches(msg, d.keyMap.Navigation.Up):
			// Handle navigation up
		case key.Matches(msg, d.keyMap.Navigation.Down):
			// Handle navigation down
		case key.Matches(msg, d.keyMap.Navigation.Left):
			// Handle navigation left
		case key.Matches(msg, d.keyMap.Navigation.Right):
			// Handle navigation right
		case key.Matches(msg, d.keyMap.Panels.Toggle):
			// Handle panel toggle
		case key.Matches(msg, d.keyMap.Panels.Collapse):
			// Handle panel collapse
		}

	case types.PerformanceWarningMsg:
		if d.alertsPanel != nil {
			d.alertsPanel.AddAlert("performance", msg.String(), "warning")
		}
	}

	// Update active component
	var cmd tea.Cmd
	switch d.activePanel {
	case "system":
		if d.systemMonitor != nil {
			updatedModel, cmd = d.systemMonitor.Update(msg)
			d.systemMonitor = updatedModel.(*SystemMonitorWidget)
			cmds = append(cmds, cmd)
		}
	case "gpu":
		if d.gpuMonitor != nil {
			updatedModel, cmd = d.gpuMonitor.Update(msg)
			d.gpuMonitor = updatedModel.(*GPUMonitorWidget)
			cmds = append(cmds, cmd)
		}
	case "network":
		if d.networkMonitor != nil {
			updatedModel, cmd = d.networkMonitor.Update(msg)
			d.networkMonitor = updatedModel.(*NetworkMonitorWidget)
			cmds = append(cmds, cmd)
		}
	case "processes":
		if d.processMonitor != nil {
			updatedModel, cmd = d.processMonitor.Update(msg)
			d.processMonitor = updatedModel.(*ProcessMonitorWidget)
			cmds = append(cmds, cmd)
		}
	case "alerts":
		if d.alertsPanel != nil {
			updatedModel, cmd = d.alertsPanel.Update(msg)
			d.alertsPanel = updatedModel.(*AlertsPanelWidget)
			cmds = append(cmds, cmd)
		}
	case "performance_bar":
		if d.performanceBar != nil {
			updatedModel, cmd = d.performanceBar.Update(msg)
			d.performanceBar = updatedModel.(*PerformanceBarWidget)
			cmds = append(cmds, cmd)
		}
	}

	return d, tea.Batch(cmds...)
}
// View implements the tea.Model interface
func (d *Dashboard) View() string {
	// Create layout grid
	grid := d.renderGrid()

	// Add performance bar at bottom
	perfBar := ""
	if d.performanceBar != nil {
		perfBar = d.performanceBar.View()
	}

	// Add help section if visible
	helpSection := ""
	if d.help.ShowAll {
		helpSection = d.theme.Help.Render(d.help.View(d.keyMap))
	}

	// Combine all sections
	content := lipgloss.JoinVertical(
		lipgloss.Left,
		grid,
		perfBar,
		helpSection,
	)

	// Apply container style
	return d.theme.Container.Width(d.width).Height(d.height).Render(content)
}

// renderGrid renders the dashboard grid layout
func (d *Dashboard) renderGrid() string {
	// Create grid rows
	var rows []string

	// Sort cells by position
	sortedCells := make([]GridCell, len(d.gridLayout.Cells))
	copy(sortedCells, d.gridLayout.Cells)

	// Group cells by row
	maxRow := 0
	for _, cell := range sortedCells {
		if cell.Row > maxRow {
			maxRow = cell.Row
		}
	}

	for row := 0; row <= maxRow; row++ {
		var rowCells []string

		// Find cells in this row
		for _, cell := range sortedCells {
			if cell.Row == row {
				componentView := d.renderComponent(cell.Component)
				if componentView != "" {
					if d.activePanel == cell.Component {
						componentView = d.theme.ActiveBorder.Render(componentView)
					} else {
						componentView = d.theme.Border.Render(componentView)
					}
					rowCells = append(rowCells, componentView)
				}
			}
		}

		if len(rowCells) > 0 {
			rowView := lipgloss.JoinHorizontal(lipgloss.Top, rowCells...)
			rows = append(rows, rowView)
		}
	}

	return lipgloss.JoinVertical(lipgloss.Left, rows...)
}

// renderComponent renders a specific component
func (d *Dashboard) renderComponent(componentName string) string {
	switch componentName {
	case "system":
		if d.systemMonitor != nil {
			return d.systemMonitor.View()
		}
	case "gpu":
		if d.gpuMonitor != nil {
			return d.gpuMonitor.View()
		}
	case "network":
		if d.networkMonitor != nil {
			return d.networkMonitor.View()
		}
	case "process":
		if d.processMonitor != nil {
			return d.processMonitor.View()
		}
	case "alerts":
		if d.alertsPanel != nil {
			return d.alertsPanel.View()
		}
	}
	return ""
}

// togglePanel cycles through available panels
func (d *Dashboard) togglePanel() {
	panels := []string{"system", "gpu", "network", "process", "alerts"}

	for i, panel := range panels {
		if panel == d.activePanel {
			d.activePanel = panels[(i+1)%len(panels)]
			return
		}
	}
	d.activePanel = panels[0]
}

// toggleCollapsePanel toggles collapse state of current panel
func (d *Dashboard) toggleCollapsePanel() {
	d.panelCollapsed[d.activePanel] = !d.panelCollapsed[d.activePanel]
	d.initializeComponents() // Reinitialize with new dimensions
}

// resetLayout resets the dashboard layout to default
func (d *Dashboard) resetLayout() {
	d.panelCollapsed = make(map[string]bool)
	d.activePanel = "system"
	d.initializeGridLayout()
	d.initializeComponents()
}

// navigatePanel handles panel navigation
func (d *Dashboard) navigatePanel(direction string) {
	panels := []string{"system", "gpu", "network", "process", "alerts"}

	for i, panel := range panels {
		if panel == d.activePanel {
			var newIndex int

			switch direction {
			case "up":
				newIndex = (i - 1 + len(panels)) % len(panels)
			case "down":
				newIndex = (i + 1) % len(panels)
			case "left":
				newIndex = (i - 1 + len(panels)) % len(panels)
			case "right":
				newIndex = (i + 1) % len(panels)
			default:
				return
			}

			d.activePanel = panels[newIndex]
			return
		}
	}
}

// refreshCommand returns a command to refresh all dashboard data
func (d *Dashboard) refreshCommand() tea.Cmd {
	return tea.Tick(d.refreshInterval, func(t time.Time) tea.Msg {
		return types.SystemStatusUpdateMsg{
			CPUUsage:    getCPUUsage(),
			MemoryUsage: getMemoryUsage(),
			DiskUsage:   getDiskUsage(),
			GPUUsage:    getGPUUsage(),
		}
	})
}

// tickCommand returns a command for periodic updates
func (d *Dashboard) tickCommand() tea.Cmd {
	return tea.Tick(d.refreshInterval, func(t time.Time) tea.Msg {
		return types.TickMsg{Time: t}
	})
}

// updateSystemData updates all components with new system data
func (d *Dashboard) updateSystemData(msg types.SystemStatusUpdateMsg) {
	if d.systemMonitor != nil {
		d.systemMonitor.UpdateData(msg)
	}
	if d.gpuMonitor != nil {
		d.gpuMonitor.UpdateData(msg)
	}
	if d.performanceBar != nil {
		d.performanceBar.UpdateData(msg)
	}
}

// Placeholder functions for system data collection (would be implemented with real system calls)
func getCPUUsage() float64 {
	// Simulate CPU usage
	return 25.0 + (float64(time.Now().UnixNano()%10000) / 100.0)
}

func getMemoryUsage() float64 {
	// Simulate memory usage
	return 40.0 + (float64(time.Now().UnixNano()%20000) / 100.0)
}

func getDiskUsage() float64 {
	// Simulate disk usage
	return 60.0
}

func getGPUUsage() []float64 {
	// Simulate GPU usage
	return []float64{
		45.0 + (float64(time.Now().UnixNano()%15000) / 100.0),
		30.0 + (float64(time.Now().UnixNano()%10000) / 100.0),
	}
}
