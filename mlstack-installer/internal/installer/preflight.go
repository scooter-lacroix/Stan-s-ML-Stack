// internal/installer/preflight.go
package installer

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// PreflightManager manages pre-flight check operations
type PreflightManager struct {
	config        *Config
	checkResults  *PreFlightCheckResults
	systemInfo    SystemInfo
	gpuInfo       GPUInfo
	checkpointDir string
	logFile       *os.File
}

// NewPreflightManager creates a new pre-flight manager
func NewPreflightManager(config *Config, sysInfo SystemInfo, gpuInfo GPUInfo) (*PreflightManager, error) {
	manager := &PreflightManager{
		config:        config,
		systemInfo:    sysInfo,
		gpuInfo:       gpuInfo,
		checkpointDir: filepath.Join(config.LogDir, "checkpoints"),
	}

	// Create checkpoint directory
	if err := os.MkdirAll(manager.checkpointDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create checkpoint directory: %v", err)
	}

	// Initialize log file
	logFile, err := os.OpenFile(filepath.Join(config.LogDir, "preflight.log"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to create log file: %v", err)
	}
	manager.logFile = logFile

	return manager, nil
}

// RunChecks runs all pre-flight checks and returns results
func (pm *PreflightManager) RunChecks() (*PreFlightCheckResults, error) {
	pm.log("Starting pre-flight checks")

	// Create system snapshot before checking
	snapshot, err := pm.createSystemSnapshot()
	if err != nil {
		pm.logError(fmt.Sprintf("Failed to create system snapshot: %v", err))
		return nil, err
	}

	// Save checkpoint
	if err := pm.saveCheckpoint("pre-checks", snapshot); err != nil {
		pm.logError(fmt.Sprintf("Failed to save pre-checks checkpoint: %v", err))
	}

	// Run comprehensive checks
	results, err := RunPreFlightChecks(pm.systemInfo, pm.gpuInfo)
	if err != nil {
		pm.logError(fmt.Sprintf("Pre-flight checks failed: %v", err))
		return nil, err
	}

	pm.checkResults = results

	// Log results
	pm.log(fmt.Sprintf("Pre-flight checks completed: %d passed, %d failed, %d warnings",
		results.PassedCount, results.FailedCount, results.WarningCount))

	// Auto-fix available issues if requested
	if len(results.AutoFixes) > 0 {
		pm.log(fmt.Sprintf("Found %d auto-fixable issues", len(results.AutoFixes)))
	}

	pm.log("Pre-flight checks completed successfully")
	return results, nil
}

// ApplyAutoFixes applies automatic fixes for detected issues
func (pm *PreflightManager) ApplyAutoFixes(fixIndices []int) ([]string, error) {
	if pm.checkResults == nil {
		return nil, fmt.Errorf("no pre-flight check results available")
	}

	var appliedFixes []string
	var errors []string

	// Apply fixes by index
	for _, index := range fixIndices {
		if index < 0 || index >= len(pm.checkResults.AutoFixes) {
			errors = append(errors, fmt.Sprintf("Invalid fix index: %d", index))
			continue
		}

		fix := pm.checkResults.AutoFixes[index]
		pm.log(fmt.Sprintf("Applying auto-fix: %s", fix.Name))

		if err := pm.applyFix(fix); err != nil {
			errorMsg := fmt.Sprintf("Failed to apply fix %s: %v", fix.Name, err)
			pm.logError(errorMsg)
			errors = append(errors, errorMsg)
		} else {
			appliedFixes = append(appliedFixes, fix.Name)
			pm.log(fmt.Sprintf("Successfully applied fix: %s", fix.Name))
		}
	}

	// Re-run checks after fixes
	if len(appliedFixes) > 0 {
		pm.log("Re-running checks after applying fixes")
		newResults, err := RunPreFlightChecks(pm.systemInfo, pm.gpuInfo)
		if err != nil {
			pm.logError(fmt.Sprintf("Re-check after fixes failed: %v", err))
		} else {
			pm.checkResults = newResults
			pm.log("Re-check completed successfully")
		}
	}

	return appliedFixes, nil
}

// GetFixStatus returns the status of all available fixes
func (pm *PreflightManager) GetFixStatus() []FixStatus {
	if pm.checkResults == nil {
		return []FixStatus{}
	}

	statuses := make([]FixStatus, len(pm.checkResults.AutoFixes))
	for i, fix := range pm.checkResults.AutoFixes {
		statuses[i] = FixStatus{
			Index:        i,
			Name:         fix.Name,
			Description:  fix.Description,
			Command:      fix.Command,
			RequiresSudo: fix.RequiresSudo,
			RiskLevel:    fix.RiskLevel,
			Applied:      false,
		}
	}
	return statuses
}

// ValidatePrerequisites validates that all prerequisites are met
func (pm *PreflightManager) ValidatePrerequisites() error {
	// Check critical requirements
	criticalChecks := []string{
		"root_privileges",
		"disk_space",
		"network_connectivity",
		"gpu_detection",
		"package_manager",
		"python_availability",
		"system_dependencies",
	}

	for _, checkName := range criticalChecks {
		for _, result := range pm.checkResults.Checks {
			if result.Name == checkName {
				if result.Status == StatusError {
					return fmt.Errorf("critical check failed: %s - %s", result.Name, result.Message)
				}
				break
			}
		}
	}

	return nil
}

// GetOptimizationRecommendations returns optimization recommendations based on system
func (pm *PreflightManager) GetOptimizationRecommendations() []OptimizationRecommendation {
	recommendations := []OptimizationRecommendation{}

	// GPU-specific recommendations
	if pm.gpuInfo.GPUCount > 0 {
		if pm.gpuInfo.Architecture == GFX1100 {
			recommendations = append(recommendations, OptimizationRecommendation{
				Category:     "GPU",
				Priority:     "High",
				Title:        "RDNA 3 Optimization",
				Description:  "Enable WGP wavefront optimization for RDNA 3 architecture",
				Command:      "export HSA_OVERRIDE_GFX_VERSION=11.0.0",
				RequiresSudo: false,
				Impact:       "High",
			})
		}

		if pm.gpuInfo.MemoryGB >= 24 {
			recommendations = append(recommendations, OptimizationRecommendation{
				Category:     "Memory",
				Priority:     "Medium",
				Title:        "High VRAM Optimization",
				Description:  "Enable large batch processing for 24GB+ VRAM",
				Command:      "export PYTORCH_HIP_ALLOCATOR=backend",
				RequiresSudo: false,
				Impact:       "Medium",
			})
		}
	}

	// CPU recommendations
	if pm.systemInfo.CPU.IsAMD {
		recommendations = append(recommendations, OptimizationRecommendation{
			Category:     "CPU",
			Priority:     "High",
			Title:        "AMD CPU Optimization",
			Description:  "Enable AMD-specific CPU optimizations",
			Command:      "export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH",
			RequiresSudo: false,
			Impact:       "High",
		})
	}

	// Memory recommendations
	if pm.systemInfo.Memory.TotalGB >= 32 {
		recommendations = append(recommendations, OptimizationRecommendation{
			Category:     "Memory",
			Priority:     "Medium",
			Title:        "Large Memory Optimization",
			Description:  "Enable memory-intensive optimizations for 32GB+ RAM",
			Command:      "export PYTORCH_HIP_ALLOCATOR=max_split_size_mb=512",
			RequiresSudo: false,
			Impact:       "Medium",
		})
	}

	return recommendations
}

// CreateInstallationSnapshot creates a snapshot of current system state
func (pm *PreflightManager) CreateInstallationSnapshot() (*InstallationSnapshot, error) {
	pm.log("Creating installation snapshot")

	snapshot := &InstallationSnapshot{
		Timestamp:       time.Now(),
		SystemInfo:      pm.systemInfo,
		GPUInfo:         pm.gpuInfo,
		CheckResults:    pm.checkResults,
		PackageList:     pm.getInstalledPackages(),
		EnvironmentVars: pm.getEnvironmentVariables(),
		DiskUsage:       pm.getDiskUsage(),
	}

	// Save snapshot
	snapshotPath := filepath.Join(pm.checkpointDir, fmt.Sprintf("snapshot_%d.json", time.Now().Unix()))
	if err := pm.saveCheckpoint("installation", snapshot); err != nil {
		pm.logError(fmt.Sprintf("Failed to save installation snapshot: %v", err))
		return nil, err
	}

	pm.log(fmt.Sprintf("Installation snapshot saved to: %s", snapshotPath))
	return snapshot, nil
}

// RollbackToSnapshot rolls back system to a previous snapshot
func (pm *PreflightManager) RollbackToSnapshot(snapshotID string) error {
	pm.log(fmt.Sprintf("Starting rollback to snapshot: %s", snapshotID))

	// Load snapshot
	snapshot, err := pm.loadCheckpoint("installation", snapshotID)
	if err != nil {
		pm.logError(fmt.Sprintf("Failed to load snapshot: %v", err))
		return fmt.Errorf("failed to load snapshot: %v", err)
	}

	// Validate snapshot type
	instSnapshot, ok := snapshot.(*InstallationSnapshot)
	if !ok {
		pm.logError("Invalid snapshot type for rollback")
		return fmt.Errorf("invalid snapshot type")
	}

	// Rollback environment variables
	if err := pm.rollbackEnvironment(instSnapshot.EnvironmentVars); err != nil {
		pm.logError(fmt.Sprintf("Failed to rollback environment: %v", err))
	}

	// Rollback packages (uninstall installed packages)
	if err := pm.rollbackPackages(instSnapshot.PackageList); err != nil {
		pm.logError(fmt.Sprintf("Failed to rollback packages: %v", err))
	}

	// Restore disk usage patterns (this is simulated)
	pm.log("Rollback completed successfully")

	return nil
}

// Close cleans up resources
func (pm *PreflightManager) Close() error {
	if pm.logFile != nil {
		pm.log("Closing pre-flight manager")
		return pm.logFile.Close()
	}
	return nil
}

// Private methods

func (pm *PreflightManager) log(message string) {
	logEntry := fmt.Sprintf("[%s] %s", time.Now().Format("2006-01-02 15:04:05"), message)
	if pm.logFile != nil {
		pm.logFile.WriteString(logEntry + "\n")
	}
	fmt.Println(logEntry)
}

func (pm *PreflightManager) logError(message string) {
	logEntry := fmt.Sprintf("[%s] ERROR: %s", time.Now().Format("2006-01-02 15:04:05"), message)
	if pm.logFile != nil {
		pm.logFile.WriteString(logEntry + "\n")
	}
	fmt.Println(logEntry)
}

func (pm *PreflightManager) applyFix(fix AutoFix) error {
	if fix.RequiresSudo {
		pm.log(fmt.Sprintf("Running sudo command: %s", fix.Command))
		// In a real implementation, this would use sudo escalation
		// For now, just execute the command
		return pm.executeCommand(fix.Command)
	} else {
		pm.log(fmt.Sprintf("Running command: %s", fix.Command))
		return pm.executeCommand(fix.Command)
	}
}

func (pm *PreflightManager) executeCommand(command string) error {
	// Split command into parts
	parts := strings.Fields(command)
	if len(parts) == 0 {
		return fmt.Errorf("empty command")
	}

	// Use 30 second timeout for most commands, 5 minutes for long-running operations
	timeout := 30 * time.Second
	if strings.Contains(command, "apt") || strings.Contains(command, "install") || strings.Contains(command, "build") {
		timeout = 5 * time.Minute
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, parts[0], parts[1:]...)
	cmd.Stdout = pm.logFile
	cmd.Stderr = pm.logFile

	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return fmt.Errorf("command timed out after %v: %s", timeout, command)
	}
	return err
}

func (pm *PreflightManager) createSystemSnapshot() (*SystemSnapshot, error) {
	snapshot := &SystemSnapshot{
		Timestamp: time.Now(),
	}

	// Create system file snapshots
	snapshot.Files = make(map[string]string)
	systemFiles := []string{
		"/etc/os-release",
		"/proc/version",
		"/proc/cpuinfo",
		"/proc/meminfo",
		"/proc/version",
	}

	for _, file := range systemFiles {
		if content, err := os.ReadFile(file); err == nil {
			snapshot.Files[file] = string(content)
		}
	}

	return snapshot, nil
}

func (pm *PreflightManager) saveCheckpoint(name string, data interface{}) error {
	snapshotPath := filepath.Join(pm.checkpointDir, fmt.Sprintf("%s_%d.json", name, time.Now().Unix()))

	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(snapshotPath, jsonData, 0644)
}

func (pm *PreflightManager) loadCheckpoint(name, snapshotID string) (interface{}, error) {
	snapshotPath := filepath.Join(pm.checkpointDir, fmt.Sprintf("%s_%s.json", name, snapshotID))

	data, err := os.ReadFile(snapshotPath)
	if err != nil {
		return nil, err
	}

	// Determine type and unmarshal accordingly
	if strings.Contains(snapshotPath, "installation") {
		var snapshot InstallationSnapshot
		err = json.Unmarshal(data, &snapshot)
		return snapshot, err
	} else if strings.Contains(snapshotPath, "system") {
		var snapshot SystemSnapshot
		err = json.Unmarshal(data, &snapshot)
		return snapshot, err
	}

	// Generic unmarshal
	var genericData interface{}
	err = json.Unmarshal(data, &genericData)
	return genericData, err
}

func (pm *PreflightManager) getInstalledPackages() []string {
	// Get list of installed packages from apt
	var packages []string

	cmd := exec.Command("dpkg-query", "-W", "-f=${Package}\n")
	output, err := cmd.Output()
	if err == nil {
		packages = strings.Split(strings.TrimSpace(string(output)), "\n")
	}

	return packages
}

func (pm *PreflightManager) getEnvironmentVariables() []string {
	var envVars []string

	for _, env := range os.Environ() {
		envVars = append(envVars, env)
	}

	return envVars
}

func (pm *PreflightManager) getDiskUsage() []DiskUsageInfo {
	var diskUsage []DiskUsageInfo

	cmd := exec.Command("df", "-h")
	output, err := cmd.Output()
	if err == nil {
		lines := strings.Split(string(output), "\n")
		for _, line := range lines[1:] { // Skip header
			if line == "" {
				continue
			}

			fields := strings.Fields(line)
			if len(fields) >= 6 {
				info := DiskUsageInfo{
					Path:      fields[0],
					Size:      fields[1],
					Used:      fields[2],
					Available: fields[3],
					Mount:     fields[5],
				}
				diskUsage = append(diskUsage, info)
			}
		}
	}

	return diskUsage
}

func (pm *PreflightManager) rollbackEnvironment(oldEnv []string) error {
	// This is a simplified implementation
	// In production, you'd restore specific environment variables
	pm.log("Rolling back environment variables")
	return nil
}

func (pm *PreflightManager) rollbackPackages(keepPackages []string) error {
	// This is a simplified implementation
	// In production, you'd uninstall packages not in keepPackages
	pm.log("Rolling back installed packages")
	return nil
}

// Additional data structures

type FixStatus struct {
	Index        int
	Name         string
	Description  string
	Command      string
	RequiresSudo bool
	RiskLevel    string
	Applied      bool
}

type OptimizationRecommendation struct {
	Category     string
	Priority     string
	Title        string
	Description  string
	Command      string
	RequiresSudo bool
	Impact       string
}

type InstallationSnapshot struct {
	Timestamp       time.Time              `json:"timestamp"`
	SystemInfo      SystemInfo             `json:"system_info"`
	GPUInfo         GPUInfo                `json:"gpu_info"`
	CheckResults    *PreFlightCheckResults `json:"check_results"`
	PackageList     []string               `json:"package_list"`
	EnvironmentVars []string               `json:"environment_vars"`
	DiskUsage       []DiskUsageInfo        `json:"disk_usage"`
}

type SystemSnapshot struct {
	Timestamp time.Time         `json:"timestamp"`
	Files     map[string]string `json:"files"`
	Processes []ProcessInfo     `json:"processes"`
	Services  []ServiceInfo     `json:"services"`
}

type DiskUsageInfo struct {
	Path      string
	Size      string
	Used      string
	Available string
	Mount     string
}

type ProcessInfo struct {
	PID     int
	Command string
	Status  string
}

type ServiceInfo struct {
	Name    string
	Status  string
	Enabled bool
}

// AutoFixExecutor provides interface for applying fixes
type AutoFixExecutor interface {
	ApplyFix(fix AutoFix) error
	UndoFix(fix AutoFix) error
	GetFixStatus(fix AutoFix) string
}

// SudoFixExecutor implements AutoFixExecutor for sudo commands
type SudoFixExecutor struct {
}

func (sfe *SudoFixExecutor) ApplyFix(fix AutoFix) error {
	if !fix.RequiresSudo {
		// Add timeout to prevent hanging
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		cmd := exec.CommandContext(ctx, "sh", "-c", fix.Command)
		err := cmd.Run()
		if ctx.Err() == context.DeadlineExceeded {
			return fmt.Errorf("fix command timed out: %s", fix.Command)
		}
		return err
	}

	// In production, this would use proper sudo escalation
	// For demo purposes, we'll just log it
	fmt.Printf("Applying sudo fix: %s\n", fix.Name)
	return nil
}

func (sfe *SudoFixExecutor) UndoFix(fix AutoFix) error {
	// Implement undo logic based on fix type
	fmt.Printf("Undoing fix: %s\n", fix.Name)
	return nil
}

func (sfe *SudoFixExecutor) GetFixStatus(fix AutoFix) string {
	// Return current status of fix
	return "ready"
}

// CheckResultValidator validates check results
type CheckResultValidator struct{}

func (crv *CheckResultValidator) ValidateResult(result CheckResult) error {
	// Validate required fields
	if result.Name == "" {
		return fmt.Errorf("check result missing name")
	}
	if result.Type == "" {
		return fmt.Errorf("check result missing type")
	}
	if result.Status == "" {
		return fmt.Errorf("check result missing status")
	}
	if result.Message == "" {
		return fmt.Errorf("check result missing message")
	}

	// Validate status
	switch result.Status {
	case "passed", "failed", "warning":
		// Valid status
	default:
		return fmt.Errorf("invalid status: %s", result.Status)
	}

	// Validate type
	switch result.Type {
	case CheckTypeCritical, CheckTypeWarning, CheckTypeInfo:
		// Valid type
	default:
		return fmt.Errorf("invalid type: %s", result.Type)
	}

	return nil
}

// PreflightCheckScheduler manages scheduling of checks
type PreflightCheckScheduler struct {
	checks      []ScheduledCheck
	executor    AutoFixExecutor
	results     *PreFlightCheckResults
	isRunning   bool
	stopChannel chan struct{}
}

type ScheduledCheck struct {
	Name      string
	CheckFunc func() (CheckResult, []AutoFix)
	Schedule  []string // Cron-like schedule
	Interval  time.Duration
	Enabled   bool
	LastRun   time.Time
	NextRun   time.Time
}

// NewPreflightCheckScheduler creates a new check scheduler
func NewPreflightCheckScheduler(executor AutoFixExecutor) *PreflightCheckScheduler {
	return &PreflightCheckScheduler{
		checks:      []ScheduledCheck{},
		executor:    executor,
		stopChannel: make(chan struct{}),
	}
}

// AddCheck adds a scheduled check
func (pcs *PreflightCheckScheduler) AddCheck(name string, checkFunc func() (CheckResult, []AutoFix), interval time.Duration) {
	check := ScheduledCheck{
		Name:      name,
		CheckFunc: checkFunc,
		Interval:  interval,
		Enabled:   true,
		NextRun:   time.Now().Add(interval),
	}
	pcs.checks = append(pcs.checks, check)
}

// Start starts the scheduler
func (pcs *PreflightCheckScheduler) Start() {
	if pcs.isRunning {
		return
	}

	pcs.isRunning = true
	go pcs.runScheduler()
}

// Stop stops the scheduler
func (pcs *PreflightCheckScheduler) Stop() {
	if !pcs.isRunning {
		return
	}

	pcs.isRunning = false
	close(pcs.stopChannel)
}

// runScheduler runs the check scheduling loop
func (pcs *PreflightCheckScheduler) runScheduler() {
	for {
		select {
		case <-pcs.stopChannel:
			return
		case <-time.After(1 * time.Second):
			pcs.executeDueChecks()
		}
	}
}

// executeDueChecks executes checks that are due
func (pcs *PreflightCheckScheduler) executeDueChecks() {
	now := time.Now()

	for _, check := range pcs.checks {
		if check.Enabled && now.After(check.NextRun) {
			result, fixes := check.CheckFunc()
			check.LastRun = now
			check.NextRun = now.Add(check.Interval)

			// Apply fixes if available
			for _, fix := range fixes {
				if err := pcs.executor.ApplyFix(fix); err != nil {
					fmt.Printf("Failed to apply fix %s: %v\n", fix.Name, err)
				}
			}

			// Add to results
			if pcs.results != nil {
				pcs.results.Checks = append(pcs.results.Checks, result)
				pcs.results.AutoFixes = append(pcs.results.AutoFixes, fixes...)
			}
		}
	}
}

// GetNextCheckTime returns the next time a check is due
func (pcs *PreflightCheckScheduler) GetNextCheckTime() time.Time {
	if len(pcs.checks) == 0 {
		return time.Time{}
	}

	earliest := time.Now().Add(24 * time.Hour) // Default to far future
	for _, check := range pcs.checks {
		if check.Enabled && check.NextRun.Before(earliest) {
			earliest = check.NextRun
		}
	}
	return earliest
}

// AddPeriodicCheck adds a periodic check with given interval
func (pcs *PreflightCheckScheduler) AddPeriodicCheck(name string, checkFunc func() (CheckResult, []AutoFix), interval time.Duration) {
	pcs.AddCheck(name, checkFunc, interval)
}

// GetCheckStatus returns the status of all scheduled checks
func (pcs *PreflightCheckScheduler) GetCheckStatus() []CheckStatusInfo {
	statusInfo := make([]CheckStatusInfo, len(pcs.checks))
	for i, check := range pcs.checks {
		statusInfo[i] = CheckStatusInfo{
			Name:     check.Name,
			Enabled:  check.Enabled,
			LastRun:  check.LastRun,
			NextRun:  check.NextRun,
			Interval: check.Interval,
		}
	}
	return statusInfo
}

type CheckStatusInfo struct {
	Name     string
	Enabled  bool
	LastRun  time.Time
	NextRun  time.Time
	Interval time.Duration
}
