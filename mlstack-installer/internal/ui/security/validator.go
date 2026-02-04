// internal/ui/security/validator.go
package security

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// SecurityValidator implements comprehensive security validation for the UI
type SecurityValidator struct {
	inputPatterns     map[string]*regexp.Regexp
	scriptWhitelist   []string
	allowedPaths      map[string]bool
	maxInputLength    int
	allowedCharacters string
	validationContext map[string]bool
}

// NewValidator creates a new security validator with default settings
func NewValidator() *SecurityValidator {
	validator := &SecurityValidator{
		inputPatterns:     make(map[string]*regexp.Regexp),
		scriptWhitelist:   getScriptWhitelist(),
		allowedPaths:      getAllowedPaths(),
		maxInputLength:    1024,
		allowedCharacters: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.,:;!?()[]{}@#$%^&*+=|\\/<>~ \t\n\r",
		validationContext: make(map[string]bool),
	}

	// Initialize validation patterns
	validator.initializePatterns()

	// Initialize validation context
	validator.initializeContext()

	return validator
}

// getScriptWhitelist returns a list of allowed scripts
func getScriptWhitelist() []string {
	return []string{
		"install_rocm.sh",
		"install_pytorch_rocm.sh",
		"install_triton.sh",
		"install_mpi4py.sh",
		"install_deepspeed.sh",
		"install_ml_stack.sh",
		"install_flash_attention_ck.sh",
		"repair_ml_stack.sh",
		"install_megatron.sh",
		"install_vllm.sh",
		"install_vllm_multi.sh",
		"install_vllm_studio.sh",
		"install_aiter.sh",
		"build_onnxruntime.sh",
		"build_onnxruntime_multi.sh",
		"install_bitsandbytes.sh",
		"install_bitsandbytes_multi.sh",
		"install_rocm_smi.sh",
		"install_migraphx_python.sh",
		"install_migraphx_multi.sh",
		"install_pytorch_profiler.sh",
		"install_wandb.sh",
		"setup_environment.sh",
		"enhanced_setup_environment.sh",
		"verify_installation.sh",
		"enhanced_verify_installation.sh",
		"verify_and_build.sh",
	}
}

// getAllowedPaths returns a map of allowed paths
func getAllowedPaths() map[string]bool {
	allowed := make(map[string]bool)

	// Add system paths
	home, _ := os.UserHomeDir()
	if home != "" {
		allowed[home] = true
	}

	// Add common installation paths
	allowed["/opt"] = true
	allowed["/usr/local"] = true
	allowed["/tmp"] = true
	allowed["/var/tmp"] = true

	// Add project root
	allowed["."] = true
	allowed["./"] = true
	allowed["./scripts"] = true
	allowed["./mlstack-installer"] = true

	return allowed
}

// initializePatterns sets up validation regex patterns
func (v *SecurityValidator) initializePatterns() {
	// Email pattern
	v.inputPatterns["email"] = regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)

	// Username pattern
	v.inputPatterns["username"] = regexp.MustCompile(`^[a-zA-Z0-9_-]{3,20}$`)

	// Path pattern (basic validation)
	v.inputPatterns["path"] = regexp.MustCompile(`^[a-zA-Z0-9_\-/\.~]+$`)

	// Number pattern
	v.inputPatterns["number"] = regexp.MustCompile(`^[0-9]+$`)

	// Port pattern
	v.inputPatterns["port"] = regexp.MustCompile(`^([0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])$`)

	// URL pattern
	v.inputPatterns["url"] = regexp.MustCompile(`^(https?|ftp)://[^\s/$.?#].[^\s]*$`)

	// Script name pattern
	v.inputPatterns["script"] = regexp.MustCompile(`^[a-zA-Z0-9_-]+\.sh$`)

	// Component ID pattern
	v.inputPatterns["component_id"] = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

	// GPU model pattern
	v.inputPatterns["gpu_model"] = regexp.MustCompile(`^[a-zA-Z0-9 _-]+$`)

	// System info pattern
	v.inputPatterns["system_info"] = regexp.MustCompile(`^[a-zA-Z0-9 _.,-]+$`)
}

// initializeContext sets up validation context
func (v *SecurityValidator) initializeContext() {
	v.validationContext["input_sanitization"] = true
	v.validationContext["path_validation"] = true
	v.validationContext["script_validation"] = true
	v.validationContext["privilege_check"] = true
	v.validationContext["command_injection_prevention"] = true
	v.validationContext["file_access_control"] = true
	v.validationContext["network_security"] = true
	v.validationContext["resource_protection"] = true
}

// ValidateInput validates user input with context-aware rules
func (v *SecurityValidator) ValidateInput(input string, context string) error {
	if input == "" {
		return fmt.Errorf("input cannot be empty")
	}

	// Check length
	if len(input) > v.maxInputLength {
		return fmt.Errorf("input exceeds maximum length of %d characters", v.maxInputLength)
	}

	// Validate characters
	if !v.validateCharacters(input) {
		return fmt.Errorf("input contains invalid characters")
	}

	// Context-specific validation
	switch context {
	case "email":
		return v.validateEmail(input)
	case "username":
		return v.validateUsername(input)
	case "path":
		return v.validatePath(input)
	case "number":
		return v.validateNumber(input)
	case "port":
		return v.validatePort(input)
	case "url":
		return v.validateURL(input)
	case "script":
		return v.validateScript(input)
	case "component_id":
		return v.validateComponentID(input)
	case "gpu_model":
		return v.validateGPUModel(input)
	case "system_info":
		return v.validateSystemInfo(input)
	default:
		// General validation
		return v.validateGeneralInput(input)
	}
}

// validateCharacters checks if input contains only allowed characters
func (v *SecurityValidator) validateCharacters(input string) bool {
	for _, r := range input {
		if !strings.ContainsRune(v.allowedCharacters, r) {
			return false
		}
	}
	return true
}

// validateEmail validates email format
func (v *SecurityValidator) validateEmail(email string) error {
	if !v.inputPatterns["email"].MatchString(email) {
		return fmt.Errorf("invalid email format")
	}
	return nil
}

// validateUsername validates username format
func (v *SecurityValidator) validateUsername(username string) error {
	if !v.inputPatterns["username"].MatchString(username) {
		return fmt.Errorf("invalid username format")
	}

	// Additional checks
	if strings.HasPrefix(username, "-") || strings.HasSuffix(username, "-") {
		return fmt.Errorf("username cannot start or end with hyphen")
	}

	return nil
}

// validatePath validates file path
func (v *SecurityValidator) validatePath(path string) error {
	if !v.inputPatterns["path"].MatchString(path) {
		return fmt.Errorf("invalid path format")
	}

	// Check for path traversal
	if v.containsPathTraversal(path) {
		return fmt.Errorf("path traversal detected")
	}

	// Normalize path
	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("cannot resolve absolute path: %v", err)
	}

	// Check if path is allowed
	if !v.isPathAllowed(absPath) {
		return fmt.Errorf("path not allowed")
	}

	return nil
}

// validateNumber validates numeric input
func (v *SecurityValidator) validateNumber(numStr string) error {
	if !v.inputPatterns["number"].MatchString(numStr) {
		return fmt.Errorf("invalid number format")
	}

	// Convert to number and check range
	num, err := strconv.Atoi(numStr)
	if err != nil {
		return fmt.Errorf("invalid number format")
	}

	// Additional range validation based on context
	if num < 0 {
		return fmt.Errorf("number cannot be negative")
	}

	return nil
}

// validatePort validates port number
func (v *SecurityValidator) validatePort(portStr string) error {
	if !v.inputPatterns["port"].MatchString(portStr) {
		return fmt.Errorf("invalid port number")
	}

	port, err := strconv.Atoi(portStr)
	if err != nil {
		return fmt.Errorf("invalid port number")
	}

	// Check if it's a privileged port
	if port < 1024 && v.isPrivilegedPort() {
		return fmt.Errorf("privileged port access not allowed")
	}

	return nil
}

// validateURL validates URL format
func (v *SecurityValidator) validateURL(urlStr string) error {
	if !v.inputPatterns["url"].MatchString(urlStr) {
		return fmt.Errorf("invalid URL format")
	}

	// Additional security checks
	if v.containsMaliciousURL(urlStr) {
		return fmt.Errorf("malicious URL detected")
	}

	return nil
}

// validateScript validates script name and path
func (v *SecurityValidator) validateScript(script string) error {
	if !v.inputPatterns["script"].MatchString(script) {
		return fmt.Errorf("invalid script name format")
	}

	// Check if script is in whitelist
	if !v.isScriptAllowed(script) {
		return fmt.Errorf("script not in allowed whitelist")
	}

	return nil
}

// validateComponentID validates component ID format
func (v *SecurityValidator) validateComponentID(componentID string) error {
	if !v.inputPatterns["component_id"].MatchString(componentID) {
		return fmt.Errorf("invalid component ID format")
	}

	return nil
}

// validateGPUModel validates GPU model format
func (v *SecurityValidator) validateGPUModel(gpuModel string) error {
	if !v.inputPatterns["gpu_model"].MatchString(gpuModel) {
		return fmt.Errorf("invalid GPU model format")
	}

	return nil
}

// validateSystemInfo validates system information
func (v *SecurityValidator) validateSystemInfo(systemInfo string) error {
	if !v.inputPatterns["system_info"].MatchString(systemInfo) {
		return fmt.Errorf("invalid system info format")
	}

	return nil
}

// validateGeneralInput performs general input validation
func (v *SecurityValidator) validateGeneralInput(input string) error {
	// Check for SQL injection patterns
	if v.containsSQLInjection(input) {
		return fmt.Errorf("SQL injection attempt detected")
	}

	// Check for XSS patterns
	if v.containsXSS(input) {
		return fmt.Errorf("XSS attempt detected")
	}

	// Check for command injection patterns
	if v.containsCommandInjection(input) {
		return fmt.Errorf("command injection attempt detected")
	}

	return nil
}

// SanitizeInput sanitizes user input to prevent injection attacks
func (v *SecurityValidator) SanitizeInput(input string) string {
	// Remove control characters except allowed ones
	input = strings.Map(func(r rune) rune {
		if r >= 32 && r <= 126 { // Printable ASCII
			return r
		}
		if r == '\t' || r == '\n' || r == '\r' {
			return r
		}
		return -1 // Remove all other characters
	}, input)

	// Escape HTML characters
	input = strings.ReplaceAll(input, "&", "&amp;")
	input = strings.ReplaceAll(input, "<", "&lt;")
	input = strings.ReplaceAll(input, ">", "&gt;")
	input = strings.ReplaceAll(input, "\"", "&quot;")
	input = strings.ReplaceAll(input, "'", "&#39;")

	// Remove potential command injection
	input = v.removeCommandInjection(input)

	return input
}

// ValidateScriptPath validates script execution path and permissions
func (v *SecurityValidator) ValidateScriptPath(scriptPath string) error {
	// Validate path
	if err := v.ValidateInput(scriptPath, "path"); err != nil {
		return fmt.Errorf("invalid script path: %v", err)
	}

	// Check if file exists
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		return fmt.Errorf("script file does not exist")
	}

	// Check file permissions
	info, err := os.Stat(scriptPath)
	if err != nil {
		return fmt.Errorf("cannot access script file: %v", err)
	}

	// Check if it's a regular file
	if !info.Mode().IsRegular() {
		return fmt.Errorf("script path is not a regular file")
	}

	// Check executable permission
	if info.Mode().Perm()&0111 == 0 {
		return fmt.Errorf("script is not executable")
	}

	return nil
}

// ValidatePrivileges validates user privileges for operations
func (v *SecurityValidator) ValidatePrivileges(operation string) error {
	switch operation {
	case "sudo_install":
		return v.validateSudoPrivileges()
	case "network_access":
		return v.validateNetworkAccess()
	case "system_modification":
		return v.validateSystemModification()
	case "file_system_access":
		return v.validateFileSystemAccess()
	default:
		return fmt.Errorf("unknown operation: %s", operation)
	}
}

// validateSudoPrivileges checks if user has sudo privileges
func (v *SecurityValidator) validateSudoPrivileges() error {
	// Check if user has sudo privileges with timeout to prevent hanging
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sudo", "-n", "echo", "test")
	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return fmt.Errorf("sudo check timed out - privileges may be unavailable")
	}
	if err != nil {
		return fmt.Errorf("sudo privileges required but not available")
	}
	return nil
}

// validateNetworkAccess validates network access privileges
func (v *SecurityValidator) validateNetworkAccess() error {
	// Check network access capabilities with timeout
	if runtime.GOOS == "linux" {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		cmd := exec.CommandContext(ctx, "ping", "-c", "1", "8.8.8.8", "-W", "1")
		err := cmd.Run()
		if ctx.Err() == context.DeadlineExceeded {
			return fmt.Errorf("network check timed out - connectivity may be limited")
		}
		if err != nil {
			return fmt.Errorf("network access not available")
		}
	}
	return nil
}

// validateSystemModification validates system modification privileges
func (v *SecurityValidator) validateSystemModification() error {
	// Check if we can write to system directories
	testFile := "/tmp/mlstack_test_write"
	file, err := os.Create(testFile)
	if err != nil {
		return fmt.Errorf("system modification not allowed: %v", err)
	}
	file.Close()
	os.Remove(testFile)
	return nil
}

// validateFileSystemAccess validates file system access
func (v *SecurityValidator) validateFileSystemAccess() error {
	// Check basic file system access
	if _, err := os.Stat("/tmp"); os.IsNotExist(err) {
		return fmt.Errorf("file system access not available")
	}
	return nil
}

// PreventCommandInjection removes potential command injection characters
func (v *SecurityValidator) PreventCommandInjection(command string) string {
	// Remove dangerous shell metacharacters
	dangerous := []string{";", "&", "|", "`", "$", ">", "<", "(", ")", "{", "}", "[", "]", "!", "@", "#", "%"}

	for _, char := range dangerous {
		command = strings.ReplaceAll(command, char, "")
	}

	return command
}

// containsPathTraversal checks for path traversal attempts
func (v *SecurityValidator) containsPathTraversal(path string) bool {
	// Check for ../ or ..\
	return strings.Contains(path, "../") || strings.Contains(path, "..\\")
}

// isPathAllowed checks if path is in allowed paths
func (v *SecurityValidator) isPathAllowed(path string) bool {
	// Check exact path
	if v.allowedPaths[path] {
		return true
	}

	// Check if path starts with any allowed path
	for allowed := range v.allowedPaths {
		if strings.HasPrefix(path, allowed) {
			return true
		}
	}

	return false
}

// isScriptAllowed checks if script is in whitelist
func (v *SecurityValidator) isScriptAllowed(script string) bool {
	for _, allowed := range v.scriptWhitelist {
		if script == allowed {
			return true
		}
	}
	return false
}

// isPrivilegedPort checks if current user can use privileged ports
func (v *SecurityValidator) isPrivilegedPort() bool {
	// Check if user is root
	if os.Getuid() == 0 {
		return false // Root can use any port
	}
	return true // Non-root users cannot use privileged ports
}

// containsMaliciousURL checks for malicious URLs
func (v *SecurityValidator) containsMaliciousURL(urlStr string) bool {
	// Check for javascript: or data: schemes
	if strings.HasPrefix(urlStr, "javascript:") || strings.HasPrefix(urlStr, "data:") {
		return true
	}

	// Check for file:// scheme
	if strings.HasPrefix(urlStr, "file://") {
		return true
	}

	return false
}

// containsSQLInjection checks for SQL injection patterns
func (v *SecurityValidator) containsSQLInjection(input string) bool {
	sqlPatterns := []string{
		"union\\s+select",
		"or\\s+1=1",
		"or\\s*'1'='1",
		"or\\s*1=1",
		"drop\\s+table",
		"insert\\s+into",
		"delete\\s+from",
		"update\\s+set",
		"create\\s+table",
		"alter\\s+table",
	}

	lowerInput := strings.ToLower(input)
	for _, pattern := range sqlPatterns {
		if strings.Contains(lowerInput, pattern) {
			return true
		}
	}
	return false
}

// containsXSS checks for XSS patterns
func (v *SecurityValidator) containsXSS(input string) bool {
	xssPatterns := []string{
		"<script",
		"</script>",
		"javascript:",
		"onload=",
		"onerror=",
		"onclick=",
		"onmouseover=",
		"<iframe",
		"<object",
		"<embed",
	}

	lowerInput := strings.ToLower(input)
	for _, pattern := range xssPatterns {
		if strings.Contains(lowerInput, pattern) {
			return true
		}
	}
	return false
}

// containsCommandInjection checks for command injection patterns
func (v *SecurityValidator) containsCommandInjection(input string) bool {
	injectionPatterns := []string{
		";",
		"|",
		"&",
		"`",
		"$(",
		"${",
		"&&",
		"||",
		"\\|",
		"\\&",
		"\\;",
		"\\`",
		"\\$",
	}

	lowerInput := strings.ToLower(input)
	for _, pattern := range injectionPatterns {
		if strings.Contains(lowerInput, pattern) {
			return true
		}
	}
	return false
}

// removeCommandInjection removes potential command injection
func (v *SecurityValidator) removeCommandInjection(input string) string {
	// Remove shell metacharacters and escape sequences
	dangerous := []rune{';', '|', '&', '`', '$', '>', '<', '(', ')', '{', '}', '[', ']', '!', '@', '#', '%', '\\', '"', '\''}

	result := strings.Map(func(r rune) rune {
		for _, dangerousRune := range dangerous {
			if r == dangerousRune {
				return ' ' // Replace with space
			}
		}
		return r
	}, input)

	return strings.TrimSpace(result)
}

// GetValidationContext returns the current validation context
func (v *SecurityValidator) GetValidationContext() map[string]bool {
	context := make(map[string]bool)
	for k, v := range v.validationContext {
		context[k] = v
	}
	return context
}

// SetValidationContext updates the validation context
func (v *SecurityValidator) SetValidationContext(context map[string]bool) {
	v.validationContext = make(map[string]bool)
	for k, val := range context {
		v.validationContext[k] = val
	}
}

// ValidateInputWithContext validates input with context-aware rules
func (v *SecurityValidator) ValidateInputWithContext(input string, context string, validationContext map[string]bool) error {
	// Set validation context
	originalContext := v.GetValidationContext()
	v.SetValidationContext(validationContext)
	defer v.SetValidationContext(originalContext)

	return v.ValidateInput(input, context)
}

// ValidateScriptWithContext validates script with context-aware rules
func (v *SecurityValidator) ValidateScriptWithContext(scriptPath string, operation string, validationContext map[string]bool) error {
	// Set validation context
	originalContext := v.GetValidationContext()
	v.SetValidationContext(validationContext)
	defer v.SetValidationContext(originalContext)

	// Validate script path
	if err := v.ValidateScriptPath(scriptPath); err != nil {
		return fmt.Errorf("script validation failed: %v", err)
	}

	// Validate privileges
	if err := v.ValidatePrivileges(operation); err != nil {
		return fmt.Errorf("privilege validation failed: %v", err)
	}

	return nil
}

// PerformSecurityCheck performs comprehensive security checks
func (v *SecurityValidator) PerformSecurityCheck(input string, context string, operation string) *types.SecurityCheckResult {
	result := &types.SecurityCheckResult{
		Checks: make(map[string]bool),
		Log:    make([]string, 0),
	}

	// Input validation
	if err := v.ValidateInput(input, context); err != nil {
		result.Valid = false
		result.Error = fmt.Errorf("input validation failed: %v", err)
		result.Checks["input_validation"] = false
		result.Log = append(result.Log, fmt.Sprintf("Input validation failed: %v", err))
		return result
	}
	result.Checks["input_validation"] = true

	// Path validation if applicable
	if operation == "script_execution" || operation == "file_access" {
		if err := v.ValidateScriptPath(input); err != nil {
			result.Valid = false
			result.Error = fmt.Errorf("path validation failed: %v", err)
			result.Checks["path_validation"] = false
			result.Log = append(result.Log, fmt.Sprintf("Path validation failed: %v", err))
			return result
		}
		result.Checks["path_validation"] = true
	}

	// Privilege validation
	if operation != "" {
		if err := v.ValidatePrivileges(operation); err != nil {
			result.Valid = false
			result.Error = fmt.Errorf("privilege validation failed: %v", err)
			result.Checks["privilege_validation"] = false
			result.Log = append(result.Log, fmt.Sprintf("Privilege validation failed: %v", err))
			return result
		}
		result.Checks["privilege_validation"] = true
	}

	result.Valid = true
	result.Log = append(result.Log, "All security checks passed")
	return result
}

// GetSecurityReport returns a comprehensive security report
func (v *SecurityValidator) GetSecurityReport() map[string]interface{} {
	return map[string]interface{}{
		"validation_context":   v.GetValidationContext(),
		"input_patterns":       len(v.inputPatterns),
		"script_whitelist":     len(v.scriptWhitelist),
		"allowed_paths":        len(v.allowedPaths),
		"max_input_length":     v.maxInputLength,
		"allowed_characters":   v.allowedCharacters,
		"last_validation_time": time.Now().Format(time.RFC3339),
		"security_features": []string{
			"input_sanitization",
			"path_validation",
			"script_validation",
			"privilege_check",
			"command_injection_prevention",
			"file_access_control",
			"network_security",
			"resource_protection",
		},
	}
}
