// internal/installer/recovery.go
package installer

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// RecoveryManager handles error recovery and rollback operations
type RecoveryManager struct {
	config             *Config
	checkpointDir      string
	recoveryLog        *os.File
	availableSnapshots []SnapshotInfo
}

// RecoveryOption represents a recovery action that can be taken
type RecoveryOption struct {
	ID           string
	Name         string
	Description  string
	Command      string
	RequiresSudo bool
	RiskLevel    string
	Category     string
}

// SnapshotInfo represents information about available snapshots
type SnapshotInfo struct {
	ID          string
	Timestamp   time.Time
	Type        string // "pre-installation", "post-component", "checkpoint"
	Components  []string
	Size        int64
	Description string
}

// RecoveryStep represents a single step in recovery process
type RecoveryStep struct {
	Name         string
	Command      string
	Description  string
	RequiresSudo bool
	Rollback     string // Command to undo this step
	Success      bool
	Error        error
}

// NewRecoveryManager creates a new recovery manager
func NewRecoveryManager(config *Config) (*RecoveryManager, error) {
	manager := &RecoveryManager{
		config:        config,
		checkpointDir: filepath.Join(config.LogDir, "checkpoints"),
	}

	// Create checkpoint directory if it doesn't exist
	if err := os.MkdirAll(manager.checkpointDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create checkpoint directory: %v", err)
	}

	// Initialize recovery log
	logFile, err := os.OpenFile(filepath.Join(config.LogDir, "recovery.log"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to create recovery log: %v", err)
	}
	manager.recoveryLog = logFile

	// Load available snapshots
	if err := manager.loadAvailableSnapshots(); err != nil {
		log.Printf("Warning: Could not load available snapshots: %v", err)
	}

	return manager, nil
}

// CreateRecoveryCheckpoint creates a recovery checkpoint at current state
func (rm *RecoveryManager) CreateRecoveryCheckpoint(name string, component string) error {
	rm.log(fmt.Sprintf("Creating recovery checkpoint: %s", name))

	checkpoint := RecoveryCheckpoint{
		ID:              fmt.Sprintf("ckpt_%s_%d", name, time.Now().Unix()),
		Name:            name,
		Timestamp:       time.Now(),
		Component:       component,
		SystemInfo:      rm.getCurrentSystemInfo(),
		PackageList:     rm.getInstalledPackages(),
		EnvironmentVars: rm.getEnvironmentVariables(),
		DiskUsage:       rm.getDiskUsage(),
		Config:          *rm.config,
	}

	// Save checkpoint
	if err := rm.saveCheckpoint(checkpoint); err != nil {
		rm.logError(fmt.Sprintf("Failed to save checkpoint %s: %v", name, err))
		return err
	}

	rm.log(fmt.Sprintf("Checkpoint %s created successfully", name))
	return nil
}

// CreateRecoverySnapshot creates a comprehensive recovery snapshot
func (rm *RecoveryManager) CreateRecoverySnapshot(name string, components []string) error {
	rm.log(fmt.Sprintf("Creating recovery snapshot: %s", name))

	snapshot := RecoverySnapshot{
		ID:              fmt.Sprintf("snap_%s_%d", name, time.Now().Unix()),
		Name:            name,
		Timestamp:       time.Now(),
		Type:            "checkpoint",
		Components:      components,
		SystemInfo:      rm.getCurrentSystemInfo(),
		PackageList:     rm.getInstalledPackages(),
		EnvironmentVars: rm.getEnvironmentVariables(),
		DiskUsage:       rm.getDiskUsage(),
		Config:          *rm.config,
		BackupFiles:     rm.identifyCriticalFiles(),
	}

	// Save snapshot
	if err := rm.saveSnapshot(snapshot); err != nil {
		rm.logError(fmt.Sprintf("Failed to save snapshot %s: %v", name, err))
		return err
	}

	rm.log(fmt.Sprintf("Snapshot %s created successfully", name))
	return nil
}

// GetRecoveryOptions returns available recovery options for current state
func (rm *RecoveryManager) GetRecoveryOptions(currentStage string, lastComponent string, errors []string) []RecoveryOption {
	options := []RecoveryOption{}

	// Generic recovery options
	options = append(options, RecoveryOption{
		ID:           "retry_last",
		Name:         "Retry Last Component",
		Description:  "Retry the last failed component installation",
		Command:      "retry-component",
		RequiresSudo: false,
		RiskLevel:    "Low",
		Category:     "installation",
	})

	options = append(options, RecoveryOption{
		ID:           "restart_install",
		Name:         "Restart Installation",
		Description:  "Restart the entire installation process",
		Command:      "restart-install",
		RequiresSudo: false,
		RiskLevel:    "Medium",
		Category:     "installation",
	})

	// Stage-specific recovery options
	switch currentStage {
	case "hardware":
		options = append(options, RecoveryOption{
			ID:           "re detect_hardware",
			Name:         "Re-run Hardware Detection",
			Description:  "Detect hardware components again",
			Command:      "redetect-hardware",
			RequiresSudo: false,
			RiskLevel:    "Low",
			Category:     "hardware",
		})

	case "preflight":
		options = append(options, RecoveryOption{
			ID:           "fix_issues",
			Name:         "Fix Pre-flight Issues",
			Description:  "Apply automatic fixes for pre-flight check failures",
			Command:      "fix-preflight",
			RequiresSudo: true,
			RiskLevel:    "Medium",
			Category:     "preflight",
		})

	case "component":
		if lastComponent != "" {
			options = append(options, RecoveryOption{
				ID:           "uninstall_last",
				Name:         "Uninstall Last Component",
				Description:  "Remove the last failed component and try again",
				Command:      "uninstall-component",
				RequiresSudo: true,
				RiskLevel:    "High",
				Category:     "component",
			})
		}

		options = append(options, RecoveryOption{
			ID:           "skip_component",
			Name:         "Skip Problematic Component",
			Description:  "Continue installation without the problematic component",
			Command:      "skip-component",
			RequiresSudo: false,
			RiskLevel:    "Medium",
			Category:     "component",
		})

	case "installation":
		options = append(options, RecoveryOption{
			ID:           "rollback_checkpoint",
			Name:         "Rollback to Last Checkpoint",
			Description:  "Rollback system to last stable checkpoint",
			Command:      "rollback-checkpoint",
			RequiresSudo: true,
			RiskLevel:    "High",
			Category:     "rollback",
		})

		options = append(options, RecoveryOption{
			ID:           "restore_snapshot",
			Name:         "Restore from Snapshot",
			Description:  "Restore system from previous snapshot",
			Command:      "restore-snapshot",
			RequiresSudo: true,
			RiskLevel:    "High",
			Category:     "rollback",
		})
	}

	// Error-specific recovery options
	for _, err := range errors {
		if strings.Contains(strings.ToLower(err), "permission") {
			options = append(options, RecoveryOption{
				ID:           "fix_permissions",
				Name:         "Fix File Permissions",
				Description:  "Correct file permission issues",
				Command:      "fix-permissions",
				RequiresSudo: true,
				RiskLevel:    "Low",
				Category:     "system",
			})
		}

		if strings.Contains(strings.ToLower(err), "disk") || strings.Contains(strings.ToLower(err), "space") {
			options = append(options, RecoveryOption{
				ID:           "cleanup_disk",
				Name:         "Clean Disk Space",
				Description:  "Clean up unnecessary files to free disk space",
				Command:      "cleanup-disk",
				RequiresSudo: true,
				RiskLevel:    "Low",
				Category:     "system",
			})
		}

		if strings.Contains(strings.ToLower(err), "dependency") {
			options = append(options, RecoveryOption{
				ID:           "install_deps",
				Name:         "Install Dependencies",
				Description:  "Install missing system dependencies",
				Command:      "install-deps",
				RequiresSudo: true,
				RiskLevel:    "Medium",
				Category:     "system",
			})
		}
	}

	return options
}

// ExecuteRecoveryOption executes a recovery option with proper error handling
func (rm *RecoveryManager) ExecuteRecoveryOption(option RecoveryOption) error {
	rm.log(fmt.Sprintf("Executing recovery option: %s", option.Name))

	// Create checkpoint before attempting recovery
	if err := rm.CreateRecoveryCheckpoint("pre_recovery", ""); err != nil {
		rm.logError(fmt.Sprintf("Failed to create pre-recovery checkpoint: %v", err))
	}

	// Execute recovery steps
	steps := rm.getRecoverySteps(option)

	for _, step := range steps {
		if err := rm.executeRecoveryStep(step); err != nil {
			rm.logError(fmt.Sprintf("Recovery step '%s' failed: %v", step.Name, err))

			// Attempt rollback of this step
			if step.Rollback != "" {
				if rollbackErr := rm.executeCommand(step.Rollback); rollbackErr != nil {
					rm.logError(fmt.Sprintf("Rollback of step '%s' failed: %v", step.Name, rollbackErr))
				}
			}

			return fmt.Errorf("recovery failed at step '%s': %v", step.Name, err)
		}
	}

	// Create checkpoint after successful recovery
	if err := rm.CreateRecoveryCheckpoint("post_recovery", ""); err != nil {
		rm.logError(fmt.Sprintf("Failed to create post-recovery checkpoint: %v", err))
	}

	rm.log(fmt.Sprintf("Recovery option '%s' completed successfully", option.Name))
	return nil
}

// RollbackToCheckpoint rolls back system to specified checkpoint
func (rm *RecoveryManager) RollbackToCheckpoint(checkpointID string) error {
	rm.log(fmt.Sprintf("Starting rollback to checkpoint: %s", checkpointID))

	// Load checkpoint
	checkpoint, err := rm.loadCheckpoint(checkpointID)
	if err != nil {
		rm.logError(fmt.Sprintf("Failed to load checkpoint %s: %v", checkpointID, err))
		return fmt.Errorf("failed to load checkpoint: %v", err)
	}

	// Execute rollback steps
	if err := rm.executeRollback(*checkpoint); err != nil {
		rm.logError(fmt.Sprintf("Rollback failed: %v", err))
		return fmt.Errorf("rollback failed: %v", err)
	}

	rm.log(fmt.Sprintf("Rollback to checkpoint %s completed successfully", checkpointID))
	return nil
}

// RestoreFromSnapshot restores system from specified snapshot
func (rm *RecoveryManager) RestoreFromSnapshot(snapshotID string) error {
	rm.log(fmt.Sprintf("Starting restoration from snapshot: %s", snapshotID))

	// Load snapshot
	snapshot, err := rm.loadSnapshot(snapshotID)
	if err != nil {
		rm.logError(fmt.Sprintf("Failed to load snapshot %s: %v", snapshotID, err))
		return fmt.Errorf("failed to load snapshot: %v", err)
	}

	// Execute restoration
	if err := rm.executeSnapshotRestore(*snapshot); err != nil {
		rm.logError(fmt.Sprintf("Snapshot restoration failed: %v", err))
		return fmt.Errorf("restoration failed: %v", err)
	}

	rm.log(fmt.Sprintf("Restoration from snapshot %s completed successfully", snapshotID))
	return nil
}

// GetAvailableSnapshots returns list of available snapshots
func (rm *RecoveryManager) GetAvailableSnapshots() []SnapshotInfo {
	return rm.availableSnapshots
}

// Close cleans up recovery manager resources
func (rm *RecoveryManager) Close() error {
	if rm.recoveryLog != nil {
		rm.log("Closing recovery manager")
		return rm.recoveryLog.Close()
	}
	return nil
}

// Private methods

func (rm *RecoveryManager) log(message string) {
	logEntry := fmt.Sprintf("[%s] RECOVERY: %s", time.Now().Format("2006-01-02 15:04:05"), message)
	if rm.recoveryLog != nil {
		rm.recoveryLog.WriteString(logEntry + "\n")
	}
	fmt.Println(logEntry)
}

func (rm *RecoveryManager) logError(message string) {
	logEntry := fmt.Sprintf("[%s] ERROR: %s", time.Now().Format("2006-01-02 15:04:05"), message)
	if rm.recoveryLog != nil {
		rm.recoveryLog.WriteString(logEntry + "\n")
	}
	fmt.Println(logEntry)
}

func (rm *RecoveryManager) saveCheckpoint(checkpoint RecoveryCheckpoint) error {
	checkpointPath := filepath.Join(rm.checkpointDir, checkpoint.ID+".json")

	jsonData, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(checkpointPath, jsonData, 0644)
}

func (rm *RecoveryManager) saveSnapshot(snapshot RecoverySnapshot) error {
	snapshotPath := filepath.Join(rm.checkpointDir, snapshot.ID+".json")

	jsonData, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(snapshotPath, jsonData, 0644)
}

func (rm *RecoveryManager) loadAvailableSnapshots() error {
	rm.availableSnapshots = []SnapshotInfo{}

	entries, err := os.ReadDir(rm.checkpointDir)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".json") {
			snapshotID := strings.TrimSuffix(entry.Name(), ".json")
			info, err := entry.Info()
			if err != nil {
				continue
			}

			// Load basic info without full content
			snapshot := SnapshotInfo{
				ID:        snapshotID,
				Timestamp: info.ModTime(),
				Size:      info.Size(),
			}

			rm.availableSnapshots = append(rm.availableSnapshots, snapshot)
		}
	}

	return nil
}

func (rm *RecoveryManager) getRecoverySteps(option RecoveryOption) []RecoveryStep {
	switch option.Command {
	case "retry-component":
		return []RecoveryStep{
			{
				Name:         "Retry Component",
				Command:      "retry-last-component",
				Description:  "Retry the last failed component installation",
				RequiresSudo: false,
				Rollback:     "",
			},
		}

	case "fix-permissions":
		return []RecoveryStep{
			{
				Name:         "Fix File Permissions",
				Command:      "chmod -R 755 " + rm.config.UserPreferences.InstallationPath,
				Description:  "Correct file permissions in installation directory",
				RequiresSudo: true,
				Rollback:     "chmod -R 644 " + rm.config.UserPreferences.InstallationPath,
			},
		}

	case "cleanup-disk":
		return []RecoveryStep{
			{
				Name:         "Clean APT Cache",
				Command:      "apt clean",
				Description:  "Clean apt package cache",
				RequiresSudo: true,
				Rollback:     "",
			},
			{
				Name:         "Clean Temporary Files",
				Command:      "rm -rf /tmp/*",
				Description:  "Clean temporary files",
				RequiresSudo: true,
				Rollback:     "",
			},
		}

	case "install-deps":
		return []RecoveryStep{
			{
				Name:         "Update Package Lists",
				Command:      "apt update",
				Description:  "Update package lists",
				RequiresSudo: true,
				Rollback:     "",
			},
			{
				Name:         "Install Dependencies",
				Command:      "apt install -y build-essential cmake git",
				Description:  "Install build dependencies",
				RequiresSudo: true,
				Rollback:     "apt remove -y build-essential cmake git",
			},
		}

	default:
		return []RecoveryStep{
			{
				Name:         "Execute Recovery",
				Command:      option.Command,
				Description:  option.Description,
				RequiresSudo: option.RequiresSudo,
				Rollback:     "",
			},
		}
	}
}

func (rm *RecoveryManager) executeRecoveryStep(step RecoveryStep) error {
	rm.log(fmt.Sprintf("Executing recovery step: %s", step.Name))

	if step.RequiresSudo {
		rm.log(fmt.Sprintf("Running with sudo: %s", step.Command))
		// In production, this would use proper sudo escalation
		return rm.executeCommand(step.Command)
	} else {
		rm.log(fmt.Sprintf("Running command: %s", step.Command))
		return rm.executeCommand(step.Command)
	}
}

func (rm *RecoveryManager) executeRollback(checkpoint RecoveryCheckpoint) error {
	rm.log("Starting rollback process")

	// Rollback environment variables
	if err := rm.rollbackEnvironment(checkpoint.EnvironmentVars); err != nil {
		rm.logError(fmt.Sprintf("Failed to rollback environment: %v", err))
	}

	// Rollback packages (remove packages not in original state)
	if err := rm.rollbackPackages(checkpoint.PackageList); err != nil {
		rm.logError(fmt.Sprintf("Failed to rollback packages: %v", err))
	}

	// Restore critical files
	// Checkpoints don't have BackupFiles field, so identify current critical files
	backupFiles := rm.identifyCriticalFiles()

	if err := rm.restoreCriticalFiles(backupFiles); err != nil {
		rm.logError(fmt.Sprintf("Failed to restore critical files: %v", err))
	}

	rm.log("Rollback completed successfully")
	return nil
}

func (rm *RecoveryManager) executeSnapshotRestore(snapshot RecoverySnapshot) error {
	rm.log("Starting snapshot restoration")

	// Restore environment variables
	if err := rm.rollbackEnvironment(snapshot.EnvironmentVars); err != nil {
		rm.logError(fmt.Sprintf("Failed to restore environment: %v", err))
	}

	// Restore packages
	if err := rm.restorePackages(snapshot.PackageList); err != nil {
		rm.logError(fmt.Sprintf("Failed to restore packages: %v", err))
	}

	// Restore critical files
	// Handle BackupFiles field that might not exist in older snapshots
	backupFiles := []string{}
	if snapshot.BackupFiles != nil {
		backupFiles = snapshot.BackupFiles
	} else {
		backupFiles = rm.identifyCriticalFiles()
	}

	if err := rm.restoreCriticalFiles(backupFiles); err != nil {
		rm.logError(fmt.Sprintf("Failed to restore critical files: %v", err))
	}

	rm.log("Snapshot restoration completed successfully")
	return nil
}

func (rm *RecoveryManager) executeCommand(command string) error {
	// Split command into parts
	parts := strings.Fields(command)
	if len(parts) == 0 {
		return fmt.Errorf("empty command")
	}

	// Use appropriate timeout based on command type
	timeout := 30 * time.Second
	if strings.Contains(command, "apt") || strings.Contains(command, "install") {
		timeout = 5 * time.Minute
	} else if strings.Contains(command, "rm") || strings.Contains(command, "clean") {
		timeout = 2 * time.Minute
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, parts[0], parts[1:]...)
	cmd.Stdout = rm.recoveryLog
	cmd.Stderr = rm.recoveryLog

	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return fmt.Errorf("recovery command timed out after %v: %s", timeout, command)
	}
	return err
}

func (rm *RecoveryManager) getCurrentSystemInfo() SystemInfo {
	// This would call the existing hardware detection
	// For now, return empty struct
	return SystemInfo{}
}

func (rm *RecoveryManager) getInstalledPackages() []string {
	// Get list of installed packages from apt
	var packages []string

	cmd := exec.Command("dpkg-query", "-W", "-f=${Package}\n")
	output, err := cmd.Output()
	if err == nil {
		packages = strings.Split(strings.TrimSpace(string(output)), "\n")
	}

	return packages
}

func (rm *RecoveryManager) getEnvironmentVariables() []string {
	var envVars []string

	for _, env := range os.Environ() {
		envVars = append(envVars, env)
	}

	return envVars
}

func (rm *RecoveryManager) getDiskUsage() []DiskUsageInfo {
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

func (rm *RecoveryManager) identifyCriticalFiles() []string {
	criticalFiles := []string{
		"/etc/os-release",
		"/proc/version",
		"/proc/cpuinfo",
		"/proc/meminfo",
		"/proc/version",
		filepath.Join(rm.config.UserPreferences.InstallationPath, "bin"),
		filepath.Join(rm.config.UserPreferences.InstallationPath, "lib"),
		filepath.Join(rm.config.UserPreferences.InstallationPath, "share"),
	}

	return criticalFiles
}

func (rm *RecoveryManager) restoreCriticalFiles(fileList []string) error {
	// This would implement file restoration from backup
	rm.log("Restoring critical files")
	return nil
}

func (rm *RecoveryManager) rollbackEnvironment(oldEnv []string) error {
	// This is a simplified implementation
	rm.log("Rolling back environment variables")
	return nil
}

func (rm *RecoveryManager) rollbackPackages(originalPackages []string) error {
	// This would remove packages not in original list
	rm.log("Rolling back installed packages")
	return nil
}

func (rm *RecoveryManager) restorePackages(targetPackages []string) error {
	// This would ensure target packages are installed
	rm.log("Restoring target packages")
	return nil
}

// Additional data structures

type RecoveryCheckpoint struct {
	ID              string          `json:"id"`
	Name            string          `json:"name"`
	Timestamp       time.Time       `json:"timestamp"`
	Component       string          `json:"component"`
	SystemInfo      SystemInfo      `json:"system_info"`
	PackageList     []string        `json:"package_list"`
	EnvironmentVars []string        `json:"environment_vars"`
	DiskUsage       []DiskUsageInfo `json:"disk_usage"`
	Config          Config          `json:"config"`
}

type RecoverySnapshot struct {
	ID              string          `json:"id"`
	Name            string          `json:"name"`
	Timestamp       time.Time       `json:"timestamp"`
	Type            string          `json:"type"`
	Components      []string        `json:"components"`
	SystemInfo      SystemInfo      `json:"system_info"`
	PackageList     []string        `json:"package_list"`
	EnvironmentVars []string        `json:"environment_vars"`
	DiskUsage       []DiskUsageInfo `json:"disk_usage"`
	Config          Config          `json:"config"`
	BackupFiles     []string        `json:"backup_files"`
}

func (rm *RecoveryManager) loadCheckpoint(checkpointID string) (*RecoveryCheckpoint, error) {
	checkpointPath := filepath.Join(rm.checkpointDir, checkpointID+".json")

	data, err := os.ReadFile(checkpointPath)
	if err != nil {
		return nil, err
	}

	var checkpoint RecoveryCheckpoint
	err = json.Unmarshal(data, &checkpoint)
	return &checkpoint, err
}

func (rm *RecoveryManager) loadSnapshot(snapshotID string) (*RecoverySnapshot, error) {
	snapshotPath := filepath.Join(rm.checkpointDir, snapshotID+".json")

	data, err := os.ReadFile(snapshotPath)
	if err != nil {
		return nil, err
	}

	var snapshot RecoverySnapshot
	err = json.Unmarshal(data, &snapshot)
	return &snapshot, err
}
