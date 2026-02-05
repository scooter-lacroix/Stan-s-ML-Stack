// internal/ui/components/component_select.go
package components

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// ComponentSelectComponent implements component selection with AMD theming
type ComponentSelectComponent struct {
	// Dimensions
	Width  int
	Height int
	X, Y   int

	// State
	Components         []types.Component
	SelectedCategories map[string]bool
	Ready              bool
	Loading            bool
	Focused            bool
	CurrentCategory    string
	SearchQuery        string
	ShowDetails        bool

	// UI elements
	List           list.Model
	SearchInput    textinput.Model
	Spinner        spinner.Model
	FilterKeys     map[string]key.Binding
	NavigationKeys map[string]key.Binding

	// AMD styling
	TitleStyle        lipgloss.Style
	SubtitleStyle     lipgloss.Style
	ListStyle         lipgloss.Style
	SelectedItemStyle lipgloss.Style
	InfoStyle         lipgloss.Style
	SearchStyle       lipgloss.Style
	CategoryStyle     lipgloss.Style

	// Performance tracking
	startTime time.Time
}

// NewComponentSelectComponent creates a new component selection component
func NewComponentSelectComponent(components []types.Component, selectedCategories map[string]bool, width, height int) *ComponentSelectComponent {
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

	listStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(AMDGray)).
		Padding(1).
		Width(width - 4)

	selectedItemStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDWhite)).
		Background(lipgloss.Color(AMDRed)).
		Bold(true)

	infoStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDGray)).
		Width(width).
		Align(lipgloss.Left)

	searchStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDOrange)).
		Background(lipgloss.Color(AMDBlack)).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(AMDGray)).
		Padding(0, 1)

	categoryStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(AMDWhite)).
		Background(lipgloss.Color(AMDGray)).
		Padding(0, 2).
		Margin(0, 1)

	// Initialize search input
	searchInput := textinput.New()
	searchInput.Placeholder = "Search components..."
	searchInput.Focus()
	searchInput.Width = width - 10

	// Initialize spinner
	spinner := spinner.New(spinner.WithSpinner(spinner.Dot))
	spinner.Style = lipgloss.NewStyle().Foreground(lipgloss.Color(AMDOrange))

	// Create delegate for list
	delegate := list.NewDefaultDelegate()
	delegate.Styles.SelectedTitle = selectedItemStyle
	delegate.Styles.SelectedDesc = selectedItemStyle

	// Initialize list
	listModel := list.New([]list.Item{}, delegate, width-10, height-20)
	listModel.Title = "ML Stack Components"
	listModel.SetShowStatusBar(true)
	listModel.SetFilteringEnabled(true)

	// Initialize component
	component := &ComponentSelectComponent{
		Width:              width,
		Height:             height,
		X:                  0,
		Y:                  0,
		Components:         components,
		SelectedCategories: selectedCategories,
		Ready:              false,
		Loading:            true,
		Focused:            true,
		CurrentCategory:    "all",
		SearchQuery:        "",
		ShowDetails:        false,
		List:               listModel,
		SearchInput:        searchInput,
		Spinner:            spinner,
		TitleStyle:         titleStyle,
		SubtitleStyle:      subtitleStyle,
		ListStyle:          listStyle,
		SelectedItemStyle:  selectedItemStyle,
		InfoStyle:          infoStyle,
		SearchStyle:        searchStyle,
		CategoryStyle:      categoryStyle,
		startTime:          time.Now(),
	}

	// Initialize key bindings
	component.initializeKeyBindings()

	// Load components
	component.loadComponents()

	return component
}

// initializeKeyBindings sets up keyboard shortcuts
func (c *ComponentSelectComponent) initializeKeyBindings() {
	c.FilterKeys = map[string]key.Binding{
		"all": key.NewBinding(
			key.WithKeys("a"),
			key.WithHelp("A", "All Components"),
		),
		"core": key.NewBinding(
			key.WithKeys("c"),
			key.WithHelp("C", "Core Components"),
		),
		"extensions": key.NewBinding(
			key.WithKeys("e"),
			key.WithHelp("E", "Extensions"),
		),
		"optimization": key.NewBinding(
			key.WithKeys("o"),
			key.WithHelp("O", "Optimization"),
		),
	}

	c.NavigationKeys = map[string]key.Binding{
		"up": key.NewBinding(
			key.WithKeys("up", "k"),
			key.WithHelp("↑/K", "Move Up"),
		),
		"down": key.NewBinding(
			key.WithKeys("down", "j"),
			key.WithHelp("↓/J", "Move Down"),
		),
		"select": key.NewBinding(
			key.WithKeys("enter", " "),
			key.WithHelp("Enter", "Toggle Selection"),
		),
		"details": key.NewBinding(
			key.WithKeys("d", "i"),
			key.WithHelp("D/I", "Show Details"),
		),
		"search": key.NewBinding(
			key.WithKeys("/", "s"),
			key.WithHelp("/S", "Search"),
		),
		"back": key.NewBinding(
			key.WithKeys("esc", "q"),
			key.WithHelp("Esc/Q", "Back"),
		),
	}
}

// loadComponents loads components into the list
func (c *ComponentSelectComponent) loadComponents() {
	items := make([]list.Item, len(c.Components))
	for i, component := range c.Components {
		items[i] = ComponentItem{Component: component}
	}
	c.List.SetItems(items)
}

// ComponentItem implements list.Item for components
type ComponentItem struct {
	Component types.Component
}

func (i ComponentItem) Title() string {
	status := "□"
	if i.Component.Selected {
		status = "✓"
	}
	if i.Component.Installed {
		status = "✓"
	}
	return fmt.Sprintf("%s %s", status, i.Component.Name)
}

func (i ComponentItem) Description() string {
	return fmt.Sprintf("%s (%s) - %s", i.Component.Category, i.Component.Estimate, i.Component.Description)
}

func (i ComponentItem) FilterValue() string {
	return fmt.Sprintf("%s %s %s", i.Component.Name, i.Component.Category, i.Component.Description)
}

// Init initializes the component selection component
func (c *ComponentSelectComponent) Init() tea.Cmd {
	c.Ready = true
	c.Loading = false

	// Start spinner if loading
	if c.Loading {
		return c.Spinner.Tick
	}

	return nil
}

// Update handles messages for the component selection component
func (c *ComponentSelectComponent) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		// Handle search input focus
		if c.SearchInput.Focused() {
			c.SearchInput, cmd = c.SearchInput.Update(msg)
			if c.SearchInput.Value() != c.SearchQuery {
				c.SearchQuery = c.SearchInput.Value()
				c.filterComponents()
			}

			// Exit search mode on Escape
			if key.Matches(msg, c.NavigationKeys["back"]) {
				c.SearchInput.Blur()
				return c, nil
			}

			break
		}

		// Handle normal navigation
		switch {
		case key.Matches(msg, c.FilterKeys["all"]):
			c.CurrentCategory = "all"
			c.filterComponents()

		case key.Matches(msg, c.FilterKeys["core"]):
			c.CurrentCategory = "core"
			c.filterComponents()

		case key.Matches(msg, c.FilterKeys["extensions"]):
			c.CurrentCategory = "extensions"
			c.filterComponents()

		case key.Matches(msg, c.FilterKeys["optimization"]):
			c.CurrentCategory = "optimization"
			c.filterComponents()

		case key.Matches(msg, c.NavigationKeys["up"], c.NavigationKeys["down"]):
			c.List, cmd = c.List.Update(msg)

		case key.Matches(msg, c.NavigationKeys["select"]):
			c.toggleSelection()

		case key.Matches(msg, c.NavigationKeys["details"]):
			c.ShowDetails = !c.ShowDetails

		case key.Matches(msg, c.NavigationKeys["search"]):
			c.SearchInput.Focus()
			c.SearchInput.CursorEnd()

		case key.Matches(msg, c.NavigationKeys["back"]):
			return c, func() tea.Msg {
				return types.NavigateBackMsg{}
			}
		}

	case spinner.TickMsg:
		if c.Loading {
			c.Spinner, cmd = c.Spinner.Update(msg)
		}

	default:
		// Let list handle other messages
		c.List, cmd = c.List.Update(msg)
	}

	return c, cmd
}

// View renders the component selection component
func (c *ComponentSelectComponent) View() string {
	if !c.Ready {
		return "Initializing components..."
	}

	var builder strings.Builder

	// Title
	builder.WriteString(c.TitleStyle.Render("Select ML Stack Components") + "\n\n")

	// Subtitle
	builder.WriteString(c.SubtitleStyle.Render("Choose components for your AMD-powered ML environment") + "\n\n")

	// Category filters
	builder.WriteString(c.renderCategoryFilters() + "\n")

	// Search input
	if c.SearchInput.Focused() {
		builder.WriteString(c.SearchStyle.Render("Search: "+c.SearchInput.View()) + "\n\n")
	} else {
		builder.WriteString(c.InfoStyle.Render("Press / to search") + "\n\n")
	}

	// Component list
	builder.WriteString(c.ListStyle.Render(c.List.View()))

	// Component details
	if c.ShowDetails && c.List.SelectedItem() != nil {
		builder.WriteString("\n\n" + c.renderComponentDetails())
	}

	// Help text
	builder.WriteString("\n\n" + c.renderHelp())

	return builder.String()
}

// SetBounds updates the component dimensions
func (c *ComponentSelectComponent) SetBounds(x, y, width, height int) {
	c.X = x
	c.Y = y
	c.Width = width
	c.Height = height

	// Update styles with new dimensions
	c.TitleStyle = c.TitleStyle.Width(width)
	c.SubtitleStyle = c.SubtitleStyle.Width(width)
	c.ListStyle = c.ListStyle.Width(width - 4)
	c.InfoStyle = c.InfoStyle.Width(width)
	c.SearchInput.Width = width - 10

	// Update list dimensions
	c.List.SetSize(width-10, height-20)
}

// GetBounds returns the current component bounds
func (c *ComponentSelectComponent) GetBounds() (x, y, width, height int) {
	return c.X, c.Y, c.Width, c.Height
}

// toggleSelection toggles the selection of the current component
func (c *ComponentSelectComponent) toggleSelection() {
	if selectedItem, ok := c.List.SelectedItem().(ComponentItem); ok {
		for i, component := range c.Components {
			if component.ID == selectedItem.Component.ID {
				c.Components[i].Selected = !c.Components[i].Selected
				c.loadComponents() // Refresh list
				break
			}
		}
	}
}

// filterComponents filters the component list based on current category and search
func (c *ComponentSelectComponent) filterComponents() {
	var filteredItems []list.Item

	for _, component := range c.Components {
		// Category filter
		if c.CurrentCategory != "all" && component.Category != c.CurrentCategory {
			continue
		}

		// Search filter
		if c.SearchQuery != "" {
			searchLower := strings.ToLower(c.SearchQuery)
			componentText := strings.ToLower(fmt.Sprintf("%s %s %s",
				component.Name, component.Category, component.Description))
			if !strings.Contains(componentText, searchLower) {
				continue
			}
		}

		filteredItems = append(filteredItems, ComponentItem{Component: component})
	}

	c.List.SetItems(filteredItems)
}

// renderCategoryFilters renders the category filter buttons
func (c *ComponentSelectComponent) renderCategoryFilters() string {
	categories := []struct {
		key  string
		name string
	}{
		{"all", "All"},
		{"core", "Core"},
		{"extensions", "Extensions"},
		{"optimization", "Optimization"},
	}

	var filters []string
	for _, cat := range categories {
		style := c.CategoryStyle
		if c.CurrentCategory == cat.key {
			style = style.Background(lipgloss.Color(AMDRed))
		}
		filters = append(filters, style.Render(fmt.Sprintf("%s [%s]", cat.name, c.FilterKeys[cat.key].Help().Key)))
	}

	return lipgloss.NewStyle().Align(lipgloss.Center).Width(c.Width).Render(strings.Join(filters, " "))
}

// renderComponentDetails renders details for the selected component
func (c *ComponentSelectComponent) renderComponentDetails() string {
	if selectedItem, ok := c.List.SelectedItem().(ComponentItem); ok {
		details := fmt.Sprintf(
			"Component: %s\nCategory: %s\nDescription: %s\nSize: %s\nEstimate: %s\nRequired: %t",
			selectedItem.Component.Name,
			selectedItem.Component.Category,
			selectedItem.Component.Description,
			fmt.Sprintf("%.1f MB", float64(selectedItem.Component.Size)/1024/1024),
			selectedItem.Component.Estimate,
			selectedItem.Component.Required,
		)

		if selectedItem.Component.Installed {
			details += "\nStatus: ✓ Installed"
		}

		return c.InfoStyle.Render(details)
	}
	return ""
}

// renderHelp renders the help text
func (c *ComponentSelectComponent) renderHelp() string {
	helpItems := []string{
		fmt.Sprintf("%s: Move Up/Down", c.NavigationKeys["up"].Help().Key),
		fmt.Sprintf("%s: Toggle Selection", c.NavigationKeys["select"].Help().Key),
		fmt.Sprintf("%s: Show Details", c.NavigationKeys["details"].Help().Key),
		fmt.Sprintf("%s: Search", c.NavigationKeys["search"].Help().Key),
		fmt.Sprintf("%s: Back", c.NavigationKeys["back"].Help().Key),
	}

	return c.InfoStyle.Render(strings.Join(helpItems, " • "))
}

// GetSelectedComponents returns the list of selected components
func (c *ComponentSelectComponent) GetSelectedComponents() []types.Component {
	var selected []types.Component
	for _, component := range c.Components {
		if component.Selected {
			selected = append(selected, component)
		}
	}
	return selected
}

// GetState returns the component state
func (c *ComponentSelectComponent) GetState() map[string]interface{} {
	return map[string]interface{}{
		"ready":            c.Ready,
		"loading":          c.Loading,
		"current_category": c.CurrentCategory,
		"search_query":     c.SearchQuery,
		"show_details":     c.ShowDetails,
		"selected_count":   len(c.GetSelectedComponents()),
		"total_components": len(c.Components),
		"width":            c.Width,
		"height":           c.Height,
		"start_time":       c.startTime,
	}
}
