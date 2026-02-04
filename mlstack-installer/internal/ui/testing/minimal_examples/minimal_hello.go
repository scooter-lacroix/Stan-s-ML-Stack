// Package minimal_examples provides minimal Bubble Tea reproducible examples for testing
package minimal_examples

import (
	"fmt"
	"time"

	"github.com/charmbracelet/bubbletea"
)

// MinimalHello represents the simplest possible Bubble Tea program
type MinimalHello struct {
	message string
}

// NewMinimalHello creates a minimal hello world example
func NewMinimalHello() *MinimalHello {
	return &MinimalHello{
		message: "Hello, World!",
	}
}

// Init initializes the program
func (m *MinimalHello) Init() tea.Cmd {
	return nil // No commands needed
}

// Update handles messages
func (m *MinimalHello) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit
		case tea.KeyEnter:
			m.message = "Hello, World! (Updated)"
		}
	}
	return m, nil
}

// View renders the UI
func (m *MinimalHello) View() string {
	return m.message + "\n\nPress Enter to update, Ctrl+C to quit"
}

// TestMinimalHello runs the minimal hello example
func TestMinimalHello() error {
	model := NewMinimalHello()
	program := tea.NewProgram(model)

	if _, err := program.Run(); err != nil {
		return fmt.Errorf("minimal hello test failed: %w", err)
	}

	return nil
}

// MinimalKeyHandling demonstrates minimal key handling
type MinimalKeyHandling struct {
	keysPressed []string
}

// NewMinimalKeyHandling creates a minimal key handling example
func NewMinimalKeyHandling() *MinimalKeyHandling {
	return &MinimalKeyHandling{
		keysPressed: []string{},
	}
}

// Init initializes the program
func (m *MinimalKeyHandling) Init() tea.Cmd {
	return nil
}

// Update handles messages
func (m *MinimalKeyHandling) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		m.keysPressed = append(m.keysPressed, msg.Type.String())
		if len(m.keysPressed) > 10 {
			m.keysPressed = m.keysPressed[1:] // Keep last 10 keys
		}
		return m, nil
	}
	return m, nil
}

// View renders the UI
func (m *MinimalKeyHandling) View() string {
	result := "Key Presses:\n"
	for i, key := range m.keysPressed {
		result += fmt.Sprintf("%d. %s\n", i+1, key)
	}
	result += "\nPress any key to record, Ctrl+C to quit"
	return result
}

// TestMinimalKeyHandling runs the minimal key handling example
func TestMinimalKeyHandling() error {
	model := NewMinimalKeyHandling()
	program := tea.NewProgram(model)

	if _, err := program.Run(); err != nil {
		return fmt.Errorf("minimal key handling test failed: %w", err)
	}

	return nil
}

// MinimalSpinner demonstrates minimal spinner usage
type MinimalSpinner struct {
	spinner   string
	spinnerPos int
	count     int
}

// NewMinimalSpinner creates a minimal spinner example
func NewMinimalSpinner() *MinimalSpinner {
	return &MinimalSpinner{
		spinner:   "|/-\\",
		spinnerPos: 0,
		count:    0,
	}
}

// Init initializes the program
func (m *MinimalSpinner) Init() tea.Cmd {
	return m.tickCommand()
}

// Update handles messages
func (m *MinimalSpinner) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case spinnerMsg:
		m.spinnerPos = (m.spinnerPos + 1) % len(m.spinner)
		m.count++
		return m, m.tickCommand()
	case tea.KeyMsg:
		if msg.Type == tea.KeyCtrlC {
			return m, tea.Quit
		}
	}
	return m, nil
}

// View renders the UI
func (m *MinimalSpinner) View() string {
	return fmt.Sprintf("Spinning: %s | Count: %d\nPress Ctrl+C to quit",
		string(m.spinner[m.spinnerPos]), m.count)
}

// tickCommand returns a tick command for spinner
func (m *MinimalSpinner) tickCommand() tea.Cmd {
	return tea.Tick(100*time.Millisecond, func(t time.Time) tea.Msg {
		return spinnerMsg{}
	})
}

// spinnerMsg represents a spinner tick message
type spinnerMsg struct{}

// TestMinimalSpinner runs the minimal spinner example
func TestMinimalSpinner() error {
	model := NewMinimalSpinner()
	program := tea.NewProgram(model)

	if _, err := program.Run(); err != nil {
		return fmt.Errorf("minimal spinner test failed: %w", err)
	}

	return nil
}

// MinimalList demonstrates minimal list functionality
type MinimalList struct {
	items     []string
	selected  int
}

// NewMinimalList creates a minimal list example
func NewMinimalList() *MinimalList {
	return &MinimalList{
		items:    []string{"Item 1", "Item 2", "Item 3", "Item 4", "Item 5"},
		selected: 0,
	}
}

// Init initializes the program
func (m *MinimalList) Init() tea.Cmd {
	return nil
}

// Update handles messages
func (m *MinimalList) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyUp:
			m.selected = max(0, m.selected-1)
		case tea.KeyDown:
			m.selected = min(len(m.items)-1, m.selected+1)
		case tea.KeyEnter:
			m.items[m.selected] += " (Selected)"
		case tea.KeyCtrlC:
			return m, tea.Quit
		}
	}
	return m, nil
}

// View renders the UI
func (m *MinimalList) View() string {
	result := "Minimal List:\n\n"
	for i, item := range m.items {
		if i == m.selected {
			result += "> " + item + "\n"
		} else {
			result += "  " + item + "\n"
		}
	}
	result += "\nUse ↑/↓ to navigate, Enter to select, Ctrl+C to quit"
	return result
}

// TestMinimalList runs the minimal list example
func TestMinimalList() error {
	model := NewMinimalList()
	program := tea.NewProgram(model)

	if _, err := program.Run(); err != nil {
		return fmt.Errorf("minimal list test failed: %w", err)
	}

	return nil
}

// MinimalInput demonstrates minimal input handling
type MinimalInput struct {
	input     string
	cursorPos int
}

// NewMinimalInput creates a minimal input example
func NewMinimalInput() *MinimalInput {
	return &MinimalInput{
		input:     "",
		cursorPos: 0,
	}
}

// Init initializes the program
func (m *MinimalInput) Init() tea.Cmd {
	return nil
}

// Update handles messages
func (m *MinimalInput) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit
		case tea.KeyEnter:
			// Just echo the input
			return m, nil
		case tea.KeyBackspace:
			if m.cursorPos > 0 {
				m.input = m.input[:m.cursorPos-1] + m.input[m.cursorPos:]
				m.cursorPos--
			}
		case tea.KeyDelete:
			if m.cursorPos < len(m.input) {
				m.input = m.input[:m.cursorPos] + m.input[m.cursorPos+1:]
			}
		default:
			// Add character to input
			if msg.Type >= 32 && msg.Type <= 126 { // Printable ASCII
				m.input = m.input[:m.cursorPos] + string(msg.Type) + m.input[m.cursorPos:]
				m.cursorPos++
			}
		}
	}
	return m, nil
}

// View renders the UI
func (m *MinimalInput) View() string {
	// Show cursor position
	display := m.input
	if m.cursorPos >= 0 && m.cursorPos <= len(m.input) {
		display = display[:m.cursorPos] + "|" + display[m.cursorPos:]
	}

	return fmt.Sprintf("Input: %s\n\nType to input, Backspace/Delete to remove, Ctrl+C to quit", display)
}

// TestMinimalInput runs the minimal input example
func TestMinimalInput() error {
	model := NewMinimalInput()
	program := tea.NewProgram(model)

	if _, err := program.Run(); err != nil {
		return fmt.Errorf("minimal input test failed: %w", err)
	}

	return nil
}

// MinimalProgressBar demonstrates minimal progress bar
type MinimalProgressBar struct {
	progress   float64
	maxValue   float64
}

// NewMinimalProgressBar creates a minimal progress bar example
func NewMinimalProgressBar() *MinimalProgressBar {
	return &MinimalProgressBar{
		progress: 0,
		maxValue: 100,
	}
}

// Init initializes the program
func (m *MinimalProgressBar) Init() tea.Cmd {
	return m.progressCommand()
}

// Update handles messages
func (m *MinimalProgressBar) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case progressMsg:
		m.progress += 2
		if m.progress >= m.maxValue {
			m.progress = m.maxValue
			return m, tea.Quit
		}
		return m, m.progressCommand()
	case tea.KeyMsg:
		if msg.Type == tea.KeyCtrlC {
			return m, tea.Quit
		}
	}
	return m, nil
}

// View renders the UI
func (m *MinimalProgressBar) View() string {
	barWidth := 30
	filled := int((m.progress / m.maxValue) * float64(barWidth))
	bar := ""
	for i := 0; i < barWidth; i++ {
		if i < filled {
			bar += "="
		} else {
			bar += " "
		}
	}

	return fmt.Sprintf("Progress: [%s] %.1f%%\n\nPress Ctrl+C to quit", bar, m.progress)
}

// progressMsg represents a progress update message
type progressMsg struct{}

// progressCommand returns a progress command
func (m *MinimalProgressBar) progressCommand() tea.Cmd {
	return tea.Tick(100*time.Millisecond, func(t time.Time) tea.Msg {
		return progressMsg{}
	})
}

// TestMinimalProgressBar runs the minimal progress bar example
func TestMinimalProgressBar() error {
	model := NewMinimalProgressBar()
	program := tea.NewProgram(model)

	if _, err := program.Run(); err != nil {
		return fmt.Errorf("minimal progress bar test failed: %w", err)
	}

	return nil
}

// MinimalForm demonstrates minimal form handling
type MinimalForm struct {
	fields    []Field
	selected  int
	submitted bool
}

// Field represents a form field
type Field struct {
	Label  string
	Value  string
	Active bool
}

// NewMinimalForm creates a minimal form example
func NewMinimalForm() *MinimalForm {
	return &MinimalForm{
		fields: []Field{
			{Label: "Name:", Value: "", Active: true},
			{Label: "Email:", Value: "", Active: false},
			{Label: "Message:", Value: "", Active: false},
		},
		selected:  0,
		submitted: false,
	}
}

// Init initializes the program
func (m *MinimalForm) Init() tea.Cmd {
	return nil
}

// Update handles messages
func (m *MinimalForm) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		if m.submitted {
			if msg.Type == tea.KeyCtrlC {
				return m, tea.Quit
			}
			return m, nil
		}

		switch msg.Type {
		case tea.KeyUp:
			m.selected = max(0, m.selected-1)
			m.updateFieldActivity()
		case tea.KeyDown:
			m.selected = min(len(m.fields)-1, m.selected+1)
			m.updateFieldActivity()
		case tea.KeyEnter:
			if m.selected == len(m.fields)-1 {
				m.submitted = true
			}
		case tea.KeyCtrlC:
			return m, tea.Quit
		default:
			// Add character to current field
			if msg.Type >= 32 && msg.Type <= 126 {
				m.fields[m.selected].Value += string(msg.Type)
			}
		}
	case tea.KeyMsg:
		// Handle backspace
		if msg.Type == tea.KeyBackspace && len(m.fields[m.selected].Value) > 0 {
			m.fields[m.selected].Value = m.fields[m.selected].Value[:len(m.fields[m.selected].Value)-1]
		}
	}
	return m, nil
}

// View renders the UI
func (m *MinimalForm) View() string {
	if m.submitted {
		return fmt.Sprintf("Form Submitted!\n\nName: %s\nEmail: %s\nMessage: %s\n\nPress Ctrl+C to quit",
			m.fields[0].Value, m.fields[1].Value, m.fields[2].Value)
	}

	result := "Minimal Form:\n\n"
	for i, field := range m.fields {
		if i == m.selected {
			result += "> " + field.Label + " " + field.Value + "\n"
		} else {
			result += "  " + field.Label + " " + field.Value + "\n"
		}
	}
	result += "\nUse ↑/↓ to navigate, Enter to submit, Backspace to delete, Ctrl+C to quit"
	return result
}

// updateFieldActivity updates field active state
func (m *MinimalForm) updateFieldActivity() {
	for i := range m.fields {
		m.fields[i].Active = i == m.selected
	}
}

// TestMinimalForm runs the minimal form example
func TestMinimalForm() error {
	model := NewMinimalForm()
	program := tea.NewProgram(model)

	if _, err := program.Run(); err != nil {
		return fmt.Errorf("minimal form test failed: %w", err)
	}

	return nil
}

// Helper functions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// MinimalTimer demonstrates minimal timer functionality
type MinimalTimer struct {
	seconds   int
	startTime time.Time
	running   bool
}

// NewMinimalTimer creates a minimal timer example
func NewMinimalTimer() *MinimalTimer {
	return &MinimalTimer{
		seconds:   0,
		startTime: time.Now(),
		running:   true,
	}
}

// Init initializes the program
func (m *MinimalTimer) Init() tea.Cmd {
	return m.timerCommand()
}

// Update handles messages
func (m *MinimalTimer) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case timerMsg:
		m.seconds++
		return m, m.timerCommand()
	case tea.KeyMsg:
		if msg.Type == tea.KeyCtrlC {
			return m, tea.Quit
		}
		if msg.Type == tea.KeySpace {
			m.running = !m.running
			if m.running {
				return m, m.timerCommand()
			}
		}
	}
	return m, nil
}

// View renders the UI
func (m *MinimalTimer) View() string {
	status := "Running"
	if !m.running {
		status = "Paused"
	}

	return fmt.Sprintf("Timer: %d seconds\nStatus: %s\n\nPress Space to %s, Ctrl+C to quit",
		m.seconds, status, status)
}

// timerMsg represents a timer tick message
type timerMsg struct{}

// timerCommand returns a timer command
func (m *MinimalTimer) timerCommand() tea.Cmd {
	if m.running {
		return tea.Tick(1*time.Second, func(t time.Time) tea.Msg {
			return timerMsg{}
		})
	}
	return nil
}

// TestMinimalTimer runs the minimal timer example
func TestMinimalTimer() error {
	model := NewMinimalTimer()
	program := tea.NewProgram(model)

	if _, err := program.Run(); err != nil {
		return fmt.Errorf("minimal timer test failed: %w", err)
	}

	return nil
}

// MinimalModal demonstrates minimal modal dialog
type MinimalModal struct {
	showModal bool
	message   string
}

// NewMinimalModal creates a minimal modal example
func NewMinimalModal() *MinimalModal {
	return &MinimalModal{
		showModal: false,
		message:   "This is a modal dialog!",
	}
}

// Init initializes the program
func (m *MinimalModal) Init() tea.Cmd {
	return nil
}

// Update handles messages
func (m *MinimalModal) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyEnter:
			m.showModal = true
		case tea.KeyEscape:
			m.showModal = false
		case tea.KeyCtrlC:
			return m, tea.Quit
		}
	}
	return m, nil
}

// View renders the UI
func (m *MinimalModal) View() string {
	result := "Minimal Modal Example:\n\n"
	result += "Press Enter to show modal, Esc to close, Ctrl+C to quit\n\n"

	if m.showModal {
		// Create a simple modal
		modalWidth := 30
		modalHeight := 5

		// Create modal border
		border := "+" + strings.Repeat("-", modalWidth-2) + "+\n"
		modal := border
		modal += "|" + strings.Repeat(" ", modalWidth-2) + "|\n"
		modal += "|" + centerText(m.message, modalWidth-2) + "|\n"
		modal += "|" + strings.Repeat(" ", modalWidth-2) + "|\n"
		modal += border

		result += "\n" + modal + "\n"
		result += "Press Esc to close modal\n"
	}

	return result
}

// centerText centers text within given width
func centerText(text string, width int) string {
	if len(text) >= width {
		return text[:width]
	}
	leftPadding := (width - len(text)) / 2
	rightPadding := width - len(text) - leftPadding
	return strings.Repeat(" ", leftPadding) + text + strings.Repeat(" ", rightPadding)
}

// TestMinimalModal runs the minimal modal example
func TestMinimalModal() error {
	model := NewMinimalModal()
	program := tea.NewProgram(model)

	if _, err := program.Run(); err != nil {
		return fmt.Errorf("minimal modal test failed: %w", err)
	}

	return nil
}

// MinimalErrorHandling demonstrates minimal error handling
type MinimalErrorHandling struct {
	errors    []error
	lastError error
}

// NewMinimalErrorHandling creates a minimal error handling example
func NewMinimalErrorHandling() *MinimalErrorHandling {
	return &MinimalErrorHandling{
		errors:    []error{},
		lastError: nil,
	}
}

// Init initializes the program
func (m *MinimalErrorHandling) Init() tea.Cmd {
	return m.errorCommand()
}

// Update handles messages
func (m *MinimalErrorHandling) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case errorMsg:
		m.lastError = msg.err
		m.errors = append(m.errors, msg.err)
		return m, m.errorCommand()
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit
		case tea.KeyEnter:
			// Simulate an error
			return m, func() tea.Msg {
				return errorMsg{err: fmt.Errorf("simulated error at %s", time.Now().Format("15:04:05"))}
			}
		}
	}
	return m, nil
}

// View renders the UI
func (m *MinimalErrorHandling) View() string {
	result := "Minimal Error Handling:\n\n"
	result += "Press Enter to simulate error, Ctrl+C to quit\n\n"

	if m.lastError != nil {
		result += fmt.Sprintf("Last Error: %s\n", m.lastError.Error())
	}

	result += fmt.Sprintf("Total Errors: %d\n", len(m.errors))

	if len(m.errors) > 0 {
		result += "\nRecent Errors:\n"
		for i := len(m.errors) - 1; i >= max(0, len(m.errors)-5); i-- {
			result += fmt.Sprintf("- %s\n", m.errors[i].Error())
		}
	}

	return result
}

// errorMsg represents an error message
type errorMsg struct {
	err error
}

// errorCommand returns an error command
func (m *MinimalErrorHandling) errorCommand() tea.Cmd {
	return tea.Tick(3*time.Second, func(t time.Time) tea.Msg {
		return errorMsg{err: fmt.Errorf("periodic error at %s", t.Format("15:04:05"))}
	})
}

// TestMinimalErrorHandling runs the minimal error handling example
func TestMinimalErrorHandling() error {
	model := NewMinimalErrorHandling()
	program := tea.NewProgram(model)

	if _, err := program.Run(); err != nil {
		return fmt.Errorf("minimal error handling test failed: %w", err)
	}

	return nil
}

// TestSuite runs all minimal examples
func TestSuite() error {
	tests := []struct {
		name string
		test func() error
	}{
		{"MinimalHello", TestMinimalHello},
		{"MinimalKeyHandling", TestMinimalKeyHandling},
		{"MinimalSpinner", TestMinimalSpinner},
		{"MinimalList", TestMinimalList},
		{"MinimalInput", TestMinimalInput},
		{"MinimalProgressBar", TestMinimalProgressBar},
		{"MinimalForm", TestMinimalForm},
		{"MinimalTimer", TestMinimalTimer},
		{"MinimalModal", TestMinimalModal},
		{"MinimalErrorHandling", TestMinimalErrorHandling},
	}

	fmt.Println("Running minimal Bubble Tea test suite...")

	for _, test := range tests {
		fmt.Printf("Running %s... ", test.name)
		err := test.test()
		if err != nil {
			fmt.Printf("FAILED: %v\n", err)
			return err
		}
		fmt.Printf("PASSED\n")
	}

	fmt.Println("All tests passed!")
	return nil
}