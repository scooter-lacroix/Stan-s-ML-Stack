// main.go
package main

import (
	"bufio"
	"fmt"
	"os"
	"os/user"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/charmbracelet/bubbletea"
	"github.com/mattn/go-isatty"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui"
)

func main() {
	// Check for help flag
	if len(os.Args) > 1 && (os.Args[1] == "--help" || os.Args[1] == "-h") {
		fmt.Println("Stan's ML Stack Installer")
		fmt.Println("A modern TUI installer for Stan's ML Stack with Pure MVU Architecture")
		fmt.Println("")
		fmt.Println("Usage:")
		fmt.Printf("  sudo %s [options]\n", os.Args[0])
		fmt.Println("")
		fmt.Println("Options:")
		fmt.Println("  -h, --help     Show this help message")
		fmt.Println("  --version      Show version information")
		fmt.Println("")
		fmt.Println("This installer requires root privileges for system installations.")
		return
	}

	// Check for version flag
	if len(os.Args) > 1 && os.Args[1] == "--version" {
		fmt.Println("Stan's ML Stack Installer v0.1.5")
		return
	}

	// Check if running as root for system installations
	if os.Geteuid() != 0 {
		fmt.Println("⚠️  Running in demo mode without root privileges")
		fmt.Println("Some features may not work correctly")
		fmt.Println("Continue anyway? (y/n): ")

		// Use non-blocking input with timeout to prevent blank screen hangs
		if !promptForUserConfirmation() {
			fmt.Printf("  sudo %s\n", os.Args[0])
			os.Exit(1)
		}
	}

	// If we reach here, we're already running as root due to sudo.
	// Preserve terminal environment for Bubble Tea to work properly
	preserveTerminalEnvironment()

	// Check TTY environment validation before Bubble Tea initialization
	if !isInteractiveEnvironment() {
		displayNonInteractiveHelp()
		return
	}

	// Initialize the Pure MVU architecture model
	model := ui.NewModel()

	// Set up signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Create the program with enhanced terminal-safe options
	var opts []tea.ProgramOption
	opts = append(opts,
		tea.WithAltScreen(),       // Use alternate screen buffer
		tea.WithMouseCellMotion(), // Enable mouse support
	)

	// Choose appropriate output stream based on environment
	outputStream := os.Stdout
	if os.Geteuid() == 0 {
		// When running with sudo, try stderr first for better TTY compatibility
		outputStream = os.Stderr
	}

	// Detect if we need special TTY handling
	if !isatty.IsTerminal(os.Stdout.Fd()) && isatty.IsTerminal(os.Stderr.Fd()) {
		outputStream = os.Stderr
	}

	opts = append(opts, tea.WithOutput(outputStream))

	// Additional safety options for problematic environments
	if !isatty.IsTerminal(os.Stdin.Fd()) {
		// If stdin is not a terminal, disable some features that might fail
		opts = append(opts, tea.WithInput(os.Stdin))
	}

	p := tea.NewProgram(model, opts...)

	// Handle graceful shutdown
	go func() {
		<-sigChan
		fmt.Println("\nShutting down gracefully...")
		p.Kill()
	}()

	if _, err := p.Run(); err != nil {
		// Handle TTY-related errors gracefully with enhanced diagnostics
		if isTTYError(err) {
			fmt.Printf("⚠️  Terminal access error detected\n")
			fmt.Printf("Error details: %v\n\n", err)
			fmt.Printf("Enhanced troubleshooting:\n")
			fmt.Printf("Environment diagnostics:\n")
			fmt.Printf("- stdin is TTY: %v\n", isatty.IsTerminal(os.Stdin.Fd()))
			fmt.Printf("- stdout is TTY: %v\n", isatty.IsTerminal(os.Stdout.Fd()))
			fmt.Printf("- stderr is TTY: %v\n", isatty.IsTerminal(os.Stderr.Fd()))
			fmt.Printf("- /dev/tty accessible: %v\n", checkTTYAccessible())
			fmt.Printf("- Running as root: %v\n", os.Geteuid() == 0)
			fmt.Printf("\nTry one of these solutions:\n")
			fmt.Printf("1. Run with environment preservation:\n")
			fmt.Printf("   sudo -E %s\n", os.Args[0])
			fmt.Printf("\n2. Use script command to create pseudo-terminal:\n")
			fmt.Printf("   script -q -c \"sudo %s\" /dev/null\n", os.Args[0])
			fmt.Printf("\n3. Ensure TTY device permissions:\n")
			fmt.Printf("   sudo chmod 666 /dev/tty\n")
			fmt.Printf("\n4. Force TTY allocation via SSH:\n")
			fmt.Printf("   ssh -t user@host sudo -E %s\n", os.Args[0])
			fmt.Printf("\n5. For Docker containers:\n")
			fmt.Printf("   docker run -it --device=/dev/tty --privileged ... %s\n", os.Args[0])
			fmt.Printf("\nIf the issue persists, the installer cannot run in this environment.\n")
			os.Exit(1)
		} else {
			fmt.Printf("❌ Unexpected error occurred:\n")
			fmt.Printf("Error: %v\n", err)
			fmt.Printf("\nThis may be related to terminal access issues.\n")
			fmt.Printf("Try running with: sudo -E %s\n", os.Args[0])
			os.Exit(1)
		}
	}
}

// isInteractiveEnvironment checks if we're in an interactive TTY environment
func isInteractiveEnvironment() bool {
	// Check if stdin is a terminal
	if !isatty.IsTerminal(os.Stdin.Fd()) {
		return false
	}

	// Check if we can access /dev/tty
	if _, err := os.Stat("/dev/tty"); err != nil {
		return false
	}

	// Check if stdout is a terminal for interactive output
	if !isatty.IsTerminal(os.Stdout.Fd()) {
		return false
	}

	// Additional validation: check if we can read from stdin
	if !canReadInteractiveInput() {
		return false
	}

	return true
}

// canReadInteractiveInput checks if we can read interactive input from user
func canReadInteractiveInput() bool {
	return isatty.IsTerminal(os.Stdin.Fd()) && isatty.IsTerminal(os.Stdout.Fd())
}

// checkTTYAccessible verifies if /dev/tty is accessible
func checkTTYAccessible() bool {
	if _, err := os.Stat("/dev/tty"); err != nil {
		return false
	}

	// Try to open /dev/tty for reading/writing
	file, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		return false
	}
	file.Close()
	return true
}

// promptForUserConfirmation handles user input with timeout to prevent hanging
func promptForUserConfirmation() bool {
	// If not in interactive mode, default to false (don't continue)
	if !canReadInteractiveInput() {
		fmt.Println("\n⚠️  Non-interactive environment detected")
		fmt.Println("Cannot read user input safely in this environment.")
		fmt.Println("Please run with root privileges or in an interactive terminal.")
		return false
	}

	reader := bufio.NewReader(os.Stdin)
	resultChan := make(chan string, 1)

	// Start goroutine to read input
	go func() {
		input, err := reader.ReadString('\n')
		if err != nil {
			resultChan <- ""
			return
		}
		resultChan <- strings.TrimSpace(input)
	}()

	// Wait for input with timeout
	select {
	case input := <-resultChan:
		return input == "y" || input == "Y" || input == "yes" || input == "YES"
	case <-time.After(30 * time.Second):
		fmt.Println("\n⚠️  Input timeout after 30 seconds")
		fmt.Println("Assuming non-interactive environment. Exiting...")
		return false
	}
}

// displayNonInteractiveHelp shows helpful message when TTY is not available
func displayNonInteractiveHelp() {
	fmt.Println("⚠️  TTY Terminal Access Error Detected")
	fmt.Println("")
	fmt.Println("The installer requires an interactive terminal environment to run.")
	fmt.Println("Current environment does not support TTY operations.")
	fmt.Println("")
	fmt.Println("Possible solutions:")
	fmt.Println("1. Run in a real terminal (not in a script or pipe):")
	fmt.Printf("   sudo -E %s\n", os.Args[0])
	fmt.Println("")
	fmt.Println("2. Use script command to create a pseudo-terminal:")
	fmt.Printf("   script -q -c \"sudo %s\" /dev/null\n", os.Args[0])
	fmt.Println("")
	fmt.Println("3. Ensure TTY device is accessible:")
	fmt.Println("   sudo chmod 666 /dev/tty")
	fmt.Println("")
	fmt.Println("4. For SSH sessions, ensure pseudo-tty allocation:")
	fmt.Println("   ssh -t user@host sudo -E %s")
	fmt.Println("")
	fmt.Println("5. Check if running in Docker/Container:")
	fmt.Println("   docker run -it --device=/dev/tty --privileged ...")
	fmt.Println("")
	fmt.Println("Error details: Could not open /dev/tty - no such device or address")
}

// preserveTerminalEnvironment ensures terminal variables are preserved when running with sudo
func preserveTerminalEnvironment() {
	// Preserve important terminal environment variables
	terminalVars := []string{
		"TERM", "COLORTERM", "TERM_PROGRAM", "TERM_PROGRAM_VERSION",
		"LANG", "LC_ALL", "LC_CTYPE",
		"XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
		"USER", "HOME", "SHELL", "LOGNAME",
		"DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY",
		"SSH_TTY", "SSH_CLIENT", "SSH_CONNECTION",
	}

	for _, varName := range terminalVars {
		if value := os.Getenv(varName); value != "" {
			os.Setenv(varName, value)
		}
	}

	// Fix TTY device access for sudo sessions
	if os.Geteuid() == 0 {
		// Get original user information
		if sudoUser := os.Getenv("SUDO_USER"); sudoUser != "" {
			if origUser, err := user.Lookup(sudoUser); err == nil {
				uid, _ := strconv.Atoi(origUser.Uid)
				_, _ = strconv.Atoi(origUser.Gid) // gid not used but parsed for completeness

				// Try to access the original user's terminal
				if tty := os.Getenv("SSH_TTY"); tty != "" {
					// SSH session - use SSH_TTY
					os.Setenv("GOTTY", tty)
				} else if tty := findTTYDevice(uid); tty != "" {
					// Find accessible TTY device
					os.Setenv("GOTTY", tty)
				}

				// Ensure we can access the terminal
				if tty := os.Getenv("GOTTY"); tty != "" {
					if file, err := os.OpenFile(tty, os.O_RDWR, 0); err == nil {
						file.Close()
						// Successfully accessed TTY
					}
				}
			}
		}
	}
}

// findTTYDevice attempts to find an accessible TTY device
func findTTYDevice(uid int) string {
	// Common TTY device locations
	ttyPaths := []string{
		"/dev/tty",
		"/dev/console",
		"/dev/pts/0",
		"/dev/pts/1",
		"/dev/pts/2",
	}

	for _, path := range ttyPaths {
		if file, err := os.OpenFile(path, os.O_RDWR, 0); err == nil {
			file.Close()
			return path
		}
	}

	return ""
}

// isTTYError checks if an error is related to TTY access issues
func isTTYError(err error) bool {
	errStr := err.Error()
	ttyErrorStrings := []string{
		"could not open a new TTY",
		"/dev/tty",
		"no such device or address",
		"operation not permitted",
		"inappropriate ioctl for device",
		"not a tty",
		"unable to access terminal",
	}

	for _, ttyErr := range ttyErrorStrings {
		if contains(errStr, ttyErr) {
			return true
		}
	}

	return false
}

// contains checks if a string contains a substring (case-insensitive)
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr ||
		(len(s) > len(substr) &&
			(s[:len(substr)] == substr ||
			 s[len(s)-len(substr):] == substr ||
			 findSubstring(s, substr))))
}

// findSubstring implements simple substring search
func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
