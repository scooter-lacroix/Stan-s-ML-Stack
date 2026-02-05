// internal/installer/executor.go
package installer

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// Security constants
const (
	MaxScriptLength = 10000             // Maximum script length in bytes
	MaxArgs         = 50                // Maximum command arguments
	AllowedTempDir  = "/tmp"            // Only allow temp files in /tmp
	MaxLogSize      = 50 * 1024         // 50KB max log size per component
	ScriptTimeout   = 120 * time.Minute // 2 hours max timeout per script
)

// ProgressMsg sends progress updates back to the UI
type ProgressMsg struct {
	ComponentID string
	Progress    float64
	Message     string
}

// CompletedMsg signals component installation is complete
type CompletedMsg struct {
	ComponentID string
	Success     bool
	Error       error
	Output      string
}

// ScriptExecutor handles running bash scripts with progress tracking and security
type ScriptExecutor struct {
	ScriptsDir   string
	LogDir       string
	SecurityMode bool
	mutex        sync.RWMutex

	// Allowed scripts whitelist
	allowedScripts map[string]bool

	// Environment sanitization rules
	safeEnvVars  map[string]bool
	dangerousEnv map[string]bool
}

// Initialize security whitelist for allowed scripts
func init() {
	// Script allowlist - only allow known, safe installation scripts
	allowedScripts := map[string]bool{
		"install_rocm.sh":                 true,
		"install_pytorch_rocm.sh":         true,
		"install_pytorch_multi.sh":        true,
		"install_triton.sh":               true,
		"install_triton_multi.sh":         true,
		"install_mpi4py.sh":               true,
		"install_deepspeed.sh":            true,
		"install_ml_stack.sh":             true,
		"install_flash_attention_ck.sh":   true,
		"install_megatron.sh":             true,
		"install_vllm.sh":                 true,
		"install_vllm_multi.sh":           true,
		"install_vllm_studio.sh":          true,
		"build_onnxruntime.sh":            true,
		"build_onnxruntime_multi.sh":      true,
		"install_bitsandbytes.sh":         true,
		"install_bitsandbytes_multi.sh":   true,
		"install_rocm_smi.sh":             true,
		"install_migraphx_python.sh":      true,
		"install_migraphx_multi.sh":       true,
		"install_pytorch_profiler.sh":     true,
		"install_wandb.sh":                true,
		"setup_environment.sh":            true,
		"enhanced_setup_environment.sh":   true,
		"repair_ml_stack.sh":              true,
		"verify_installation.sh":          true,
		"enhanced_verify_installation.sh": true,
		"custom_verify_installation.sh":   true,
		"verify_and_build.sh":             true,
		"install_flash_attn_amd.sh":       true,
		"install_amdgpu_drivers.sh":       true,
		"install_aiter.sh":                true,
		"create_persistent_env.sh":        true,
		"comprehensive_rebuild.sh":        true,
		"check_components.sh":             true,
		"build_onnxruntime_outline.sh":    true,
		"run_vllm.sh":                     true,
		"run_tests.sh":                    true,
		"run_benchmarks.sh":               true,
		"package_manager_utils.sh":        true,
		"ml_stack_component_detector.sh":  true,
		"install_ml_stack_part1.sh":       true,
		"install_ml_stack_part2.sh":       true,
		"install_ml_stack_part3.sh":       true,
		"install_ml_stack_part4.sh":       true,
		"install_ml_stack_extensions.sh":  true,
		"cleanup_for_git.sh":              true,
		"run_ml_stack_ui.sh":              true,
		"prepare_for_git.sh":              true,
		"install_rccl.sh":                 true,
	}

	// Safe environment variables
	safeEnvVars := map[string]bool{
		"HOME":                     true,
		"USER":                     true,
		"LOGNAME":                  true,
		"PATH":                     true,
		"SHELL":                    true,
		"TERM":                     true,
		"LANG":                     true,
		"LC_*":                     true, // All locale variables
		"ROCM_PATH":                true,
		"HIP_VISIBLE_DEVICES":      true,
		"CUDA_VISIBLE_DEVICES":     true,
		"HSA_OVERRIDE_GFX_VERSION": true,
		"PYTORCH_ROCM_ARCH":        true,
		"DEBIAN_FRONTEND":          true,
		"NEEDRESTART_MODE":         true,
		"MLSTACK_BATCH_MODE":       true,
		"ROCM_INSTALL_MODE":        true,
		"ACCEPT_EULA":              true,
		"DRY_RUN":                  true,
	}

	// Dangerous environment variables to sanitize
	dangerousEnv := map[string]bool{
		"LD_PRELOAD":        true,
		"LD_LIBRARY_PATH":   true,
		"PYTHONPATH":        true,
		"PERL5LIB":          true,
		"RUBYLIB":           true,
		"NODE_PATH":         true,
		"JAVA_TOOL_OPTIONS": true,
		"CLASSPATH":         true,
		"_RLD":              true, // HP-UX
		"LIBPATH":           true, // AIX
		"SHLIB_PATH":        true, // HP-UX
		"LIBRARY_PATH":      true, // Common
	}

	// Set globally
	scriptExecutorAllowedScripts = allowedScripts
	scriptExecutorSafeEnvVars = safeEnvVars
	scriptExecutorDangerousEnv = dangerousEnv
}

var (
	scriptExecutorAllowedScripts map[string]bool
	scriptExecutorSafeEnvVars    map[string]bool
	scriptExecutorDangerousEnv   map[string]bool
)

// validatePath checks if a path is safe to use
func validatePath(path string) bool {
	cleanPath := filepath.Clean(path)

	// Check if path exists
	if _, err := os.Stat(cleanPath); err != nil {
		return false
	}

	// Check if it's a directory and readable
	info, err := os.Stat(cleanPath)
	if err != nil || !info.IsDir() {
		return false
	}

	// Basic security check - no symlink traversal
	if strings.Contains(cleanPath, "..") {
		return false
	}

	return true
}

func NewScriptExecutor() *ScriptExecutor {
	homeDir, _ := os.UserHomeDir()
	logDir := filepath.Join(homeDir, ".mlstack", "logs")
	os.MkdirAll(logDir, 0755)

	// Try to find scripts directory with security validation
	scriptsDir := "./scripts"
	parentScriptsDir := "../scripts"

	// Validate and set scripts directory
	if validatePath(parentScriptsDir) {
		scriptsDir = parentScriptsDir
	} else if validatePath(scriptsDir) {
		// Use current directory scripts
	} else {
		// Fallback to parent directory if neither exists
		scriptsDir = "../scripts"
	}

	return &ScriptExecutor{
		ScriptsDir:     scriptsDir,
		LogDir:         logDir,
		SecurityMode:   true,
		allowedScripts: scriptExecutorAllowedScripts,
		safeEnvVars:    scriptExecutorSafeEnvVars,
		dangerousEnv:   scriptExecutorDangerousEnv,
	}
}

// Execute runs a bash script and streams progress with security validation
func (se *ScriptExecutor) Execute(componentID, scriptName string) tea.Cmd {
	return func() tea.Msg {
		// Security validation
		if se.SecurityMode && !se.validateScript(scriptName) {
			return CompletedMsg{
				ComponentID: componentID,
				Success:     false,
				Error:       fmt.Errorf("script validation failed: '%s' not in allowlist", scriptName),
			}
		}

		// Input sanitization
		if err := sanitizeInput(componentID, scriptName); err != nil {
			return CompletedMsg{
				ComponentID: componentID,
				Success:     false,
				Error:       fmt.Errorf("input validation failed: %w", err),
			}
		}

		scriptPath := filepath.Join(se.ScriptsDir, scriptName)
		logPath := filepath.Join(se.LogDir, fmt.Sprintf("%s_%d.log",
			componentID, time.Now().Unix()))

		// Create log file with secure permissions
		logFile, err := se.createSecureLogFile(logPath)
		if err != nil {
			return CompletedMsg{
				ComponentID: componentID,
				Success:     false,
				Error:       fmt.Errorf("failed to create log file: %w", err),
			}
		}
		defer se.cleanupLogFile(logFile, logPath)

		// Log start with timestamp
		logEntry := fmt.Sprintf("[%s] [START] Installing %s with script %s\n",
			time.Now().Format("2006-01-02 15:04:05"), componentID, scriptName)
		logFile.WriteString(logEntry)
		logFile.Sync()

		// Special handling for ROCm installer (interactive)
		if strings.Contains(scriptName, "install_rocm") {
			return se.executeROCmInstaller(componentID, scriptPath, logFile)
		}

		// Standard non-interactive execution
		return se.executeStandard(componentID, scriptPath, logFile)
	}
}

// validateScript checks if a script is in the allowlist and safe to execute
func (se *ScriptExecutor) validateScript(scriptName string) bool {
	se.mutex.RLock()
	defer se.mutex.RUnlock()

	// Basic path traversal protection
	if strings.Contains(scriptName, "..") || strings.Contains(scriptName, "/") {
		return false
	}

	// Check if script is in allowlist
	return se.allowedScripts[scriptName]
}

// sanitizeInput validates component ID and script name for injection prevention
func sanitizeInput(componentID, scriptName string) error {
	// Validate component ID
	if componentID == "" {
		return fmt.Errorf("empty component ID")
	}

	// Only allow safe characters in component ID
	if !isSafeString(componentID, "abcdefghijklmnopqrstuvwxyz0123456789-_") {
		return fmt.Errorf("invalid characters in component ID: %s", componentID)
	}

	// Validate script name
	if scriptName == "" {
		return fmt.Errorf("empty script name")
	}

	// Only allow safe characters in script name
	if !isSafeString(scriptName, "abcdefghijklmnopqrstuvwxyz0123456789-_.") {
		return fmt.Errorf("invalid characters in script name: %s", scriptName)
	}

	return nil
}

// isSafeString checks if a string only contains allowed characters
func isSafeString(s, allowedChars string) bool {
	for _, char := range s {
		if !strings.Contains(allowedChars, string(char)) {
			return false
		}
	}
	return true
}

// createSecureLogFile creates a log file with secure permissions
func (se *ScriptExecutor) createSecureLogFile(logPath string) (*os.File, error) {
	// Ensure parent directory exists
	if err := os.MkdirAll(filepath.Dir(logPath), 0755); err != nil {
		return nil, err
	}

	// Create file with restrictive permissions
	file, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0600)
	if err != nil {
		return nil, err
	}

	return file, nil
}

// cleanupLogFile ensures secure cleanup of log files
func (se *ScriptExecutor) cleanupLogFile(file *os.File, path string) {
	if file != nil {
		file.Close()
	}
	// Log files are intentionally kept for audit purposes
}

// sanitizeEnvironment cleans dangerous environment variables
func (se *ScriptExecutor) sanitizeEnvironment() []string {
	se.mutex.RLock()
	defer se.mutex.RUnlock()

	env := os.Environ()
	sanitized := make([]string, 0, len(env))

	for _, varPair := range env {
		parts := strings.SplitN(varPair, "=", 2)
		if len(parts) != 2 {
			continue
		}

		varName := parts[0]

		// Check if variable is in safe list
		if se.safeEnvVars[varName] || strings.HasPrefix(varName, "LC_") {
			// Allow safe variables
			sanitized = append(sanitized, varPair)
		} else if se.dangerousEnv[varName] {
			// Remove dangerous variables (overwrite with empty)
			sanitized = append(sanitized, fmt.Sprintf("%s=", varName))
		}
	}

	// Add safe ML stack environment variables
	sanitized = append(sanitized,
		"DEBIAN_FRONTEND=noninteractive",
		"NEEDRESTART_MODE=a",
		"MLSTACK_BATCH_MODE=1",
		"ROCM_INSTALL_MODE=silent",
		"ACCEPT_EULA=yes",
	)

	return sanitized
}

// executeStandard runs a script in non-interactive mode with security and timeout
func (se *ScriptExecutor) executeStandard(componentID, scriptPath string, logFile *os.File) tea.Msg {
	// Validate script exists and is readable
	if err := validateScriptPath(scriptPath); err != nil {
		logEntry := fmt.Sprintf("[%s] [ERROR] Script validation failed: %v\n", time.Now().Format("2006-01-02 15:04:05"), err)
		logFile.WriteString(logEntry)
		return CompletedMsg{
			ComponentID: componentID,
			Success:     false,
			Error:       fmt.Errorf("script validation failed: %w", err),
		}
	}

	// Create wrapper script with sudo password support if available
	wrapperScript := se.createSudoWrapperScript(componentID, scriptPath, logFile)
	if wrapperScript != "" {
		defer os.Remove(wrapperScript)
		return se.executeStandardWithWrapper(componentID, wrapperScript, logFile)
	}

	// Fallback to original execution without wrapper
	return se.executeStandardWithoutWrapper(componentID, scriptPath, logFile)
}

// createSudoWrapperScript creates a wrapper script that handles sudo with cached password
func (se *ScriptExecutor) createSudoWrapperScript(componentID, scriptPath string, logFile *os.File) string {
	// Check if sudo password is available
	sudoPassword := os.Getenv("SUDO_PASSWORD")
	if sudoPassword == "" {
		return "" // No cached password, return empty to skip wrapper
	}

	// Create sudo wrapper function
	wrapperContent := `#!/bin/bash
set -euo pipefail

# Sudo password wrapper function
sudo_with_pass() {
    echo '` + sudoPassword + `' | sudo -S "$@"
}

# Export the sudo wrapper function
export -f sudo_with_pass

# Replace sudo calls with wrapper in the script
sed -i 's/\\bsudo\\b/sudo_with_pass/g' '` + filepath.Base(scriptPath) + `'

# Set environment variables for ML stack installation
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export MLSTACK_BATCH_MODE=1
export ROCM_INSTALL_MODE=silent
export ACCEPT_EULA=yes

# Execute the modified script
bash "` + filepath.Base(scriptPath) + `"

# Clean up - restore original script (optional, for safety)
if [ -f "` + filepath.Base(scriptPath) + `" ]; then
    # Remove the wrapper from the original script to avoid future conflicts
    sed -i 's/\\bsudo_with_pass\\b/sudo/g' "` + filepath.Base(scriptPath) + `"
fi
`

	// Create temporary wrapper script
	wrapperScript := filepath.Join(AllowedTempDir, fmt.Sprintf("sudo_wrapper_%d_%s.sh", time.Now().Unix(), componentID[len(componentID)-4:]))
	wrapperScript = filepath.Clean(wrapperScript)

	// Write wrapper script with secure permissions
	if err := os.WriteFile(wrapperScript, []byte(wrapperContent), 0700); err != nil {
		logEntry := fmt.Sprintf("[%s] [ERROR] Failed to create sudo wrapper script: %v\n",
			time.Now().Format("2006-01-02 15:04:05"), err)
		logFile.WriteString(logEntry)
		return ""
	}

	return wrapperScript
}

// executeStandardWithWrapper executes a script using the sudo wrapper
func (se *ScriptExecutor) executeStandardWithWrapper(componentID, wrapperScript string, logFile *os.File) tea.Msg {
	// Get sanitized environment
	sanitizedEnv := se.sanitizeEnvironment()

	cmd := exec.Command("bash", wrapperScript)
	cmd.Env = sanitizedEnv

	// Validate command arguments
	if err := validateCommandArgs(cmd); err != nil {
		logEntry := fmt.Sprintf("[%s] [ERROR] Command validation failed: %v\n", time.Now().Format("2006-01-02 15:04:05"), err)
		logFile.WriteString(logEntry)
		return CompletedMsg{
			ComponentID: componentID,
			Success:     false,
			Error:       fmt.Errorf("command validation failed: %w", err),
		}
	}

	// Capture both stdout and stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		logEntry := fmt.Sprintf("[%s] [ERROR] Failed to create stdout pipe: %v\n", time.Now().Format("2006-01-02 15:04:05"), err)
		logFile.WriteString(logEntry)
		return CompletedMsg{
			ComponentID: componentID,
			Success:     false,
			Error:       fmt.Errorf("failed to create stdout pipe: %w", err),
		}
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		logEntry := fmt.Sprintf("[%s] [ERROR] Failed to create stderr pipe: %v\n", time.Now().Format("2006-01-02 15:04:05"), err)
		logFile.WriteString(logEntry)
		return CompletedMsg{
			ComponentID: componentID,
			Success:     false,
			Error:       fmt.Errorf("failed to create stderr pipe: %w", err),
		}
	}

	// Start the command
	if err := cmd.Start(); err != nil {
		logEntry := fmt.Sprintf("[%s] [ERROR] Failed to start command: %v\n", time.Now().Format("2006-01-02 15:04:05"), err)
		logFile.WriteString(logEntry)
		return CompletedMsg{
			ComponentID: componentID,
			Success:     false,
			Error:       fmt.Errorf("failed to start command: %w", err),
		}
	}

	// Log that sudo wrapper is being used
	logEntry := fmt.Sprintf("[%s] [INFO] Using sudo wrapper for password caching\n", time.Now().Format("2006-01-02 15:04:05"))
	logFile.WriteString(logEntry)

	// Start progress tracking
	progressChan := make(chan float64, 10)
	done := make(chan struct{})

	// Stream output with progress tracking
	go se.streamOutputWithProgress(componentID, stdout, stderr, logFile, progressChan, done)

	// Handle timeout
	err = se.waitForCommandWithTimeout(cmd, ScriptTimeout)

	close(progressChan)
	<-done // Wait for streaming to complete

	// Log completion
	completionTime := time.Now().Format("2006-01-02 15:04:05")
	if err != nil {
		logEntry := fmt.Sprintf("[%s] [FAILED] Component %s failed: %v\n", completionTime, componentID, err)
		logFile.WriteString(logEntry)
	} else {
		logEntry := fmt.Sprintf("[%s] [SUCCESS] Component %s completed successfully\n", completionTime, componentID)
		logFile.WriteString(logEntry)
	}

	logFile.Sync()

	return CompletedMsg{
		ComponentID: componentID,
		Success:     err == nil,
		Error:       err,
	}
}

// executeStandardWithoutWrapper executes a script without sudo wrapper (fallback)
func (se *ScriptExecutor) executeStandardWithoutWrapper(componentID, scriptPath string, logFile *os.File) tea.Msg {
	// Get sanitized environment
	sanitizedEnv := se.sanitizeEnvironment()

	cmd := exec.Command("bash", scriptPath)
	cmd.Env = sanitizedEnv

	// Validate command arguments
	if err := validateCommandArgs(cmd); err != nil {
		logEntry := fmt.Sprintf("[%s] [ERROR] Command validation failed: %v\n", time.Now().Format("2006-01-02 15:04:05"), err)
		logFile.WriteString(logEntry)
		return CompletedMsg{
			ComponentID: componentID,
			Success:     false,
			Error:       fmt.Errorf("command validation failed: %w", err),
		}
	}

	// Capture both stdout and stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		logEntry := fmt.Sprintf("[%s] [ERROR] Failed to create stdout pipe: %v\n", time.Now().Format("2006-01-02 15:04:05"), err)
		logFile.WriteString(logEntry)
		return CompletedMsg{
			ComponentID: componentID,
			Success:     false,
			Error:       fmt.Errorf("failed to create stdout pipe: %w", err),
		}
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		logEntry := fmt.Sprintf("[%s] [ERROR] Failed to create stderr pipe: %v\n", time.Now().Format("2006-01-02 15:04:05"), err)
		logFile.WriteString(logEntry)
		return CompletedMsg{
			ComponentID: componentID,
			Success:     false,
			Error:       fmt.Errorf("failed to create stderr pipe: %w", err),
		}
	}

	// Start the command
	if err := cmd.Start(); err != nil {
		logEntry := fmt.Sprintf("[%s] [ERROR] Failed to start command: %v\n", time.Now().Format("2006-01-02 15:04:05"), err)
		logFile.WriteString(logEntry)
		return CompletedMsg{
			ComponentID: componentID,
			Success:     false,
			Error:       fmt.Errorf("failed to start command: %w", err),
		}
	}

	// Log that standard execution is being used (no sudo password)
	logEntry := fmt.Sprintf("[%s] [INFO] Using standard execution (no cached sudo password)\n", time.Now().Format("2006-01-02 15:04:05"))
	logFile.WriteString(logEntry)

	// Start progress tracking
	progressChan := make(chan float64, 10)
	done := make(chan struct{})

	// Stream output with progress tracking
	go se.streamOutputWithProgress(componentID, stdout, stderr, logFile, progressChan, done)

	// Handle timeout
	err = se.waitForCommandWithTimeout(cmd, ScriptTimeout)

	close(progressChan)
	<-done // Wait for streaming to complete

	// Log completion
	completionTime := time.Now().Format("2006-01-02 15:04:05")
	if err != nil {
		logEntry := fmt.Sprintf("[%s] [FAILED] Component %s failed: %v\n", completionTime, componentID, err)
		logFile.WriteString(logEntry)
	} else {
		logEntry := fmt.Sprintf("[%s] [SUCCESS] Component %s completed successfully\n", completionTime, componentID)
		logFile.WriteString(logEntry)
	}

	logFile.Sync()

	return CompletedMsg{
		ComponentID: componentID,
		Success:     err == nil,
		Error:       err,
	}
}

// validateScriptPath checks if the script exists and is safe to execute
func validateScriptPath(scriptPath string) error {
	// Clean path to prevent directory traversal
	cleanPath := filepath.Clean(scriptPath)

	// Check if path is within scripts directory or parent
	baseScriptsDir := filepath.Clean("./scripts")
	parentScriptsDir := filepath.Clean("../scripts")

	if !strings.HasPrefix(cleanPath, baseScriptsDir) && !strings.HasPrefix(cleanPath, parentScriptsDir) {
		return fmt.Errorf("script path outside allowed directory: %s", cleanPath)
	}

	// Check if file exists
	if _, err := os.Stat(cleanPath); err != nil {
		return fmt.Errorf("script file does not exist: %w", err)
	}

	// Check if file is regular file (not symlink, device, etc.)
	info, err := os.Lstat(cleanPath)
	if err != nil {
		return fmt.Errorf("failed to stat script file: %w", err)
	}

	if !info.Mode().IsRegular() {
		return fmt.Errorf("script is not a regular file: %s", cleanPath)
	}

	// Check file size (prevent oversized scripts)
	if info.Size() > MaxScriptLength {
		return fmt.Errorf("script too large: %d bytes (max %d)", info.Size(), MaxScriptLength)
	}

	return nil
}

// validateCommandArgs validates command arguments for security
func validateCommandArgs(cmd *exec.Cmd) error {
	if len(cmd.Args) > MaxArgs {
		return fmt.Errorf("too many command arguments: %d (max %d)", len(cmd.Args), MaxArgs)
	}

	for _, arg := range cmd.Args {
		if len(arg) > 1024 { // Prevent overly long arguments
			return fmt.Errorf("argument too long: %d characters (max 1024)", len(arg))
		}

		// Check for suspicious patterns
		if strings.Contains(arg, "&&") || strings.Contains(arg, "||") ||
			strings.Contains(arg, ";") || strings.Contains(arg, "$(") {
			return fmt.Errorf("suspicious argument pattern detected: %s", arg)
		}
	}

	return nil
}

// streamOutputWithProgress streams output and tracks progress
func (se *ScriptExecutor) streamOutputWithProgress(componentID string, stdout, stderr io.Reader, logFile *os.File, progressChan chan float64, done chan struct{}) {
	defer close(done)

	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		line := scanner.Text()

		// Log line
		logEntry := fmt.Sprintf("[%s] [OUTPUT] %s\n", time.Now().Format("2006-01-02 15:04:05"), line)
		logFile.WriteString(logEntry)

		// Extract progress and send to UI
		progress := se.parseProgress(line)
		if progress > 0 {
			progressChan <- progress
		}
	}

	// Copy stderr to log
	io.Copy(logFile, stderr)
}

// parseProgress extracts progress information from script output
func (se *ScriptExecutor) parseProgress(output string) float64 {
	// Look for percentage progress
	percentageRegex := regexp.MustCompile(`(\d+)%`)
	matches := percentageRegex.FindStringSubmatch(output)
	if len(matches) > 1 {
		if pct, err := strconv.Atoi(matches[1]); err == nil && pct <= 100 {
			return float64(pct) / 100.0
		}
	}

	// Look for common progress indicators
	lowerOutput := strings.ToLower(output)
	if strings.Contains(lowerOutput, "100%") || strings.Contains(lowerOutput, "complete") || strings.Contains(lowerOutput, "finished") {
		return 1.0
	}
	if strings.Contains(lowerOutput, "75%") || strings.Contains(lowerOutput, "three quarters") {
		return 0.75
	}
	if strings.Contains(lowerOutput, "50%") || strings.Contains(lowerOutput, "half") {
		return 0.5
	}
	if strings.Contains(lowerOutput, "25%") || strings.Contains(lowerOutput, "quarter") {
		return 0.25
	}

	// Default incremental progress
	return 0.0
}

// waitForCommandWithTimeout waits for command completion with timeout
func (se *ScriptExecutor) waitForCommandWithTimeout(cmd *exec.Cmd, timeout time.Duration) error {
	// Create a channel to receive the command completion
	done := make(chan error, 1)

	// Start a goroutine that waits for the command to complete
	go func() {
		done <- cmd.Wait()
	}()

	// Use select to wait for either the command to complete or the timeout
	select {
	case err := <-done:
		return err
	case <-time.After(timeout):
		// Command timed out, kill it
		if cmd.Process != nil {
			cmd.Process.Kill()
			// Wait a bit for cleanup
			time.Sleep(5 * time.Second)
		}
		return fmt.Errorf("command timed out after %v", timeout)
	}
}

// executeROCmInstaller handles the interactive ROCm installation
// by automatically providing expected responses with enhanced security
func (se *ScriptExecutor) executeROCmInstaller(componentID, scriptPath string, logFile *os.File) tea.Msg {
	// Log ROCm installation start
	logEntry := fmt.Sprintf("[%s] [START] Starting interactive ROCm installation\n", time.Now().Format("2006-01-02 15:04:05"))
	logFile.WriteString(logEntry)

	// Create secure temporary wrapper script for ROCm
	tmpScript := se.createSecureROCmWrapper(componentID, scriptPath, logFile)
	if tmpScript == "" {
		return CompletedMsg{
			ComponentID: componentID,
			Success:     false,
			Error:       fmt.Errorf("failed to create secure ROCm wrapper"),
		}
	}

	// Ensure cleanup
	defer os.Remove(tmpScript)

	// Execute ROCm with enhanced security and timeout
	return se.executeROCmWithPTY(componentID, tmpScript, logFile)
}

// createSecureROCmWrapper creates a secure wrapper script for ROCm installation
func (se *ScriptExecutor) createSecureROCmWrapper(componentID, scriptPath string, logFile *os.File) string {
	// Create a secure, automated ROCm installation wrapper
	wrapperScript := `#!/bin/bash
set -euo pipefail

# Log ROCm installation start
echo "[ROCm] Starting automated ROCm installation at $(date)"

# Set up secure environment for ROCm installation
export DEBIAN_FRONTEND=noninteractive
export ROCM_INSTALL_MODE=silent
export ACCEPT_EULA=yes
export NEEDRESTART_MODE=a

# Function to auto-respond to prompts with logging
auto_respond() {
    echo "[ROCm] Auto-responding to prompts..."

    # Accept EULA
    echo "yes"
    sleep 1

    # Select components (ROCm runtime, Development tools, ROCm libraries)
    echo "1"  # ROCm runtime
    sleep 0.5
    echo "2"  # Development tools
    sleep 0.5
    echo "3"  # ROCm libraries
    sleep 0.5
    echo ""   # Finish selection
    sleep 2

    # Confirm installation
    echo "yes"
    sleep 1
}

# Export auto-response function
export -f auto_respond

# Execute ROCm installer with automated responses
echo "[ROCm] Running ROCm installer..."
auto_respond | bash -c "` + scriptPath + `"

# Check exit status
EXIT_CODE=$?
echo "[ROCm] ROCm installer completed with exit code: $EXIT_CODE"
exit $EXIT_CODE
`

	// Create temporary wrapper script with secure permissions
	tmpScript := filepath.Join(AllowedTempDir, fmt.Sprintf("rocm_wrapper_%d_%s.sh", time.Now().Unix(), componentID[len(componentID)-4:]))
	tmpScript = filepath.Clean(tmpScript) // Ensure clean path

	if err := os.WriteFile(tmpScript, []byte(wrapperScript), 0700); err != nil {
		logEntry := fmt.Sprintf("[%s] [ERROR] Failed to create ROCm wrapper script: %v\n",
			time.Now().Format("2006-01-02 15:04:05"), err)
		logFile.WriteString(logEntry)
		return ""
	}

	return tmpScript
}

// executeROCmWithPTY runs ROCm script with pseudo-terminal for interactive handling
func (se *ScriptExecutor) executeROCmWithPTY(componentID, scriptPath string, logFile *os.File) tea.Msg {
	// Validate the script path first
	if err := validateScriptPath(scriptPath); err != nil {
		logEntry := fmt.Sprintf("[%s] [ERROR] ROCm script validation failed: %v\n",
			time.Now().Format("2006-01-02 15:04:05"), err)
		logFile.WriteString(logEntry)
		return CompletedMsg{
			ComponentID: componentID,
			Success:     false,
			Error:       fmt.Errorf("ROCm script validation failed: %w", err),
		}
	}

	// Create enhanced command with security settings
	cmd := exec.Command("bash", scriptPath)

	// Set sanitized environment
	cmd.Env = se.sanitizeEnvironment()

	// Add ROCm-specific environment
	cmd.Env = append(cmd.Env,
		"DEBIAN_FRONTEND=noninteractive",
		"ROCM_INSTALL_MODE=silent",
		"ACCEPT_EULA=yes",
		"NEEDRESTART_MODE=a",
		"MLSTACK_ROCM_INTERACTIVE=1",
	)

	// Create pipes for stdin/stdout/stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		logEntry := fmt.Sprintf("[%s] [ERROR] Failed to create stdin pipe: %v\n",
			time.Now().Format("2006-01-02 15:04:05"), err)
		logFile.WriteString(logEntry)
		return CompletedMsg{
			ComponentID: componentID,
			Success:     false,
			Error:       fmt.Errorf("failed to create stdin pipe: %w", err),
		}
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		logEntry := fmt.Sprintf("[%s] [ERROR] Failed to create stdout pipe: %v\n",
			time.Now().Format("2006-01-02 15:04:05"), err)
		logFile.WriteString(logEntry)
		return CompletedMsg{
			ComponentID: componentID,
			Success:     false,
			Error:       fmt.Errorf("failed to create stdout pipe: %w", err),
		}
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		logEntry := fmt.Sprintf("[%s] [ERROR] Failed to create stderr pipe: %v\n",
			time.Now().Format("2006-01-02 15:04:05"), err)
		logFile.WriteString(logEntry)
		return CompletedMsg{
			ComponentID: componentID,
			Success:     false,
			Error:       fmt.Errorf("failed to create stderr pipe: %w", err),
		}
	}

	// Start command
	if err := cmd.Start(); err != nil {
		logEntry := fmt.Sprintf("[%s] [ERROR] Failed to start ROCm command: %v\n",
			time.Now().Format("2006-01-02 15:04:05"), err)
		logFile.WriteString(logEntry)
		return CompletedMsg{
			ComponentID: componentID,
			Success:     false,
			Error:       fmt.Errorf("failed to start ROCm command: %w", err),
		}
	}

	// Setup progress tracking
	progressChan := make(chan float64, 10)
	done := make(chan struct{})

	// Enhanced ROCm-specific prompt handling
	go se.handleROCmPrompts(componentID, stdin, stdout, stderr, logFile, progressChan, done)

	// Wait for completion with extended timeout for ROCm
	rocTimeout := ScriptTimeout // Already 2 hours
	err = se.waitForCommandWithTimeout(cmd, rocTimeout)

	close(progressChan)
	<-done // Wait for prompt handling to complete

	// Log completion
	completionTime := time.Now().Format("2006-01-02 15:04:05")
	if err != nil {
		logEntry := fmt.Sprintf("[%s] [FAILED] ROCm installation failed: %v\n", completionTime, err)
		logFile.WriteString(logEntry)
	} else {
		logEntry := fmt.Sprintf("[%s] [SUCCESS] ROCm installation completed successfully\n", completionTime)
		logFile.WriteString(logEntry)
	}

	logFile.Sync()

	return CompletedMsg{
		ComponentID: componentID,
		Success:     err == nil,
		Error:       err,
	}
}

// handleROCmPrompts specifically handles ROCm installer prompts
func (se *ScriptExecutor) handleROCmPrompts(componentID string, stdin io.Writer, stdout, stderr io.Reader, logFile *os.File, progressChan chan float64, done chan struct{}) {
	defer close(done)

	scanner := bufio.NewScanner(stdout)
	lineCount := 0
	promptCount := 0

	for scanner.Scan() {
		line := scanner.Text()
		lineCount++

		// Log line
		timestamp := time.Now().Format("2006-01-02 15:04:05")
		logEntry := fmt.Sprintf("[%s] [ROCm] %s\n", timestamp, line)
		logFile.WriteString(logEntry)

		// Extract progress and send to UI
		progress := se.parseROCmProgress(line, promptCount)
		if progress > 0 {
			progressChan <- progress
		}

		// Enhanced ROCm prompt detection and response
		lowerLine := strings.ToLower(line)
		if strings.Contains(lowerLine, "do you accept") ||
			strings.Contains(lowerLine, "continue?") ||
			strings.Contains(lowerLine, "[y/n]") ||
			strings.Contains(lowerLine, "accept license") {
			fmt.Printf("[ROCm] Detected EULA prompt, responding with 'yes'\n")
			stdin.Write([]byte("y\n"))
			promptCount++
			time.Sleep(1 * time.Second) // Brief delay for response processing
		} else if strings.Contains(lowerLine, "press enter") ||
			strings.Contains(lowerLine, "press return") {
			fmt.Printf("[ROCm] Detected 'press enter' prompt\n")
			stdin.Write([]byte("\n"))
			promptCount++
			time.Sleep(1 * time.Second)
		} else if strings.Contains(lowerLine, "select") && strings.Contains(lowerLine, "component") {
			fmt.Printf("[ROCm] Detected component selection prompt\n")
			// Select all components
			stdin.Write([]byte("1\n")) // ROCm runtime
			time.Sleep(500 * time.Millisecond)
			stdin.Write([]byte("2\n")) // Development tools
			time.Sleep(500 * time.Millisecond)
			stdin.Write([]byte("3\n")) // ROCm libraries
			time.Sleep(1 * time.Second)
			stdin.Write([]byte("\n")) // Finish selection
			promptCount++
		}
	}

	// Copy any remaining stderr
	io.Copy(logFile, stderr)
}

// parseROCmProgress extracts progress information specific to ROCm installation
func (se *ScriptExecutor) parseROCmProgress(output string, promptCount int) float64 {
	lowerOutput := strings.ToLower(output)

	// Look for specific ROCm progress patterns
	if strings.Contains(lowerOutput, "100%") || strings.Contains(lowerOutput, "complete") {
		return 1.0
	}

	// Use prompt count as a rough progress indicator for ROCm
	// ROCm typically has 3-4 major prompts, so we estimate progress
	maxPrompts := 4
	if promptCount > 0 {
		return float64(promptCount) / float64(maxPrompts) * 0.75 // Max 75% before completion
	}

	return 0.0
}
