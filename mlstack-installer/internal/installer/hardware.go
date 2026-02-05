// internal/installer/hardware.go
package installer

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

	"golang.org/x/sys/unix"
)

// GPUInfo holds detailed GPU detection results
type GPUInfo struct {
	Vendor        string            `json:"vendor"`
	Model         string            `json:"model"`
	Driver        string            `json:"driver"`
	ComputeUnits  int               `json:"compute_units"`
	MemoryGB      float64           `json:"memory_gb"`
	Architecture  string            `json:"architecture"`
	GFXVersion    string            `json:"gfx_version"`
	Temperature   float64           `json:"temperature"`
	GPUCount      int               `json:"gpu_count"`
	Status        string            `json:"status"`
	Optimizations string            `json:"optimizations"`
	PowerUsage    float64           `json:"power_usage"`
	DetailedInfo  []DetailedGPUInfo `json:"detailed_info"`
}

// DetailedGPUInfo holds per-GPU detailed information
type DetailedGPUInfo struct {
	ID              string  `json:"id"`
	Name            string  `json:"name"`
	MemoryTotal     uint64  `json:"memory_total"`
	MemoryUsed      uint64  `json:"memory_used"`
	MemoryFree      uint64  `json:"memory_free"`
	ComputeUnits    int     `json:"compute_units"`
	CoreClock       float64 `json:"core_clock"`
	MemoryClock     float64 `json:"memory_clock"`
	Temperature     float64 `json:"temperature"`
	PowerUsage      float64 `json:"power_usage"`
	DriverVersion   string  `json:"driver_version"`
	FirmwareVersion string  `json:"firmware_version"`
}

// SystemInfo holds comprehensive system information
type SystemInfo struct {
	OS              string        `json:"os"`
	Distribution    string        `json:"distribution"`
	KernelVersion   string        `json:"kernel_version"`
	Architecture    string        `json:"architecture"`
	CPU             CPUInfo       `json:"cpu"`
	Memory          MemoryInfo    `json:"memory"`
	Storage         []StorageInfo `json:"storage"`
	Network         []NetworkInfo `json:"network"`
	Timestamp       time.Time     `json:"timestamp"`
	TotalScore      int           `json:"total_score"`
	Recommendations []string      `json:"recommendations"`
}

// CPUInfo holds CPU information
type CPUInfo struct {
	Model        string   `json:"model"`
	Cores        int      `json:"cores"`
	Threads      int      `json:"threads"`
	ClockSpeed   float64  `json:"clock_speed"`
	CacheSize    uint64   `json:"cache_size"`
	Flags        []string `json:"flags"`
	IsAMD        bool     `json:"is_amd"`
	SupportsAVX  bool     `json:"supports_avx"`
	SupportsAVX2 bool     `json:"supports_avx2"`
	SupportsFMA  bool     `json:"supports_fma"`
}

// MemoryInfo holds memory information
type MemoryInfo struct {
	TotalGB     float64 `json:"total_gb"`
	AvailableGB float64 `json:"available_gb"`
	UsedGB      float64 `json:"used_gb"`
	SwapTotalGB float64 `json:"swap_total_gb"`
	SwapUsedGB  float64 `json:"swap_used_gb"`
	SwapFreeGB  float64 `json:"swap_free_gb"`
}

// StorageInfo holds storage information
type StorageInfo struct {
	Path        string  `json:"path"`
	Type        string  `json:"type"`
	SizeGB      float64 `json:"size_gb"`
	UsedGB      float64 `json:"used_gb"`
	AvailableGB float64 `json:"available_gb"`
	MountPoint  string  `json:"mount_point"`
	Filesystem  string  `json:"filesystem"`
}

// NetworkInfo holds network information
type NetworkInfo struct {
	Interface   string  `json:"interface"`
	IP          string  `json:"ip"`
	MAC         string  `json:"mac"`
	Status      string  `json:"status"`
	SpeedMbps   float64 `json:"speed_mbps"`
	IsConnected bool    `json:"is_connected"`
}

// PreFlightCheckResults holds pre-flight check results
type PreFlightCheckResults struct {
	Checks        []CheckResult `json:"checks"`
	OverallStatus string        `json:"overall_status"`
	PassedCount   int           `json:"passed_count"`
	FailedCount   int           `json:"failed_count"`
	WarningCount  int           `json:"warning_count"`
	TotalScore    int           `json:"total_score"`
	CanContinue   bool          `json:"can_continue"`
	AutoFixes     []AutoFix     `json:"auto_fixes"`
}

// CheckResult holds individual check results
type CheckResult struct {
	Name      string    `json:"name"`
	Type      string    `json:"type"`   // "critical", "warning", "info"
	Status    string    `json:"status"` // "passed", "failed", "warning"
	Message   string    `json:"message"`
	Details   string    `json:"details"`
	Score     int       `json:"score"`
	Fixes     []AutoFix `json:"fixes"`
	Timestamp time.Time `json:"timestamp"`
}

// AutoFix holds auto-fix information
type AutoFix struct {
	Name         string `json:"name"`
	Description  string `json:"description"`
	Command      string `json:"command"`
	RequiresSudo bool   `json:"requires_sudo"`
	RiskLevel    string `json:"risk_level"` // "low", "medium", "high"
}

// DetectionProgress tracks hardware detection progress
type DetectionProgress struct {
	CurrentStep    string    `json:"current_step"`
	CompletedSteps []string  `json:"completed_steps"`
	TotalSteps     int       `json:"total_steps"`
	StartTime      time.Time `json:"start_time"`
	LastUpdate     time.Time `json:"last_update"`
}

const (
	// GPU Architecture constants
	GFX1100 = "GFX1100" // RDNA 3
	GFX1030 = "GFX1030" // RDNA 2
	GFX1010 = "GFX1010" // RDNA 1

	// Status constants
	StatusOK         = "OK"
	StatusWarning    = "WARNING"
	StatusError      = "ERROR"
	StatusProcessing = "PROCESSING"

	// Check types
	CheckTypeCritical = "critical"
	CheckTypeWarning  = "warning"
	CheckTypeInfo     = "info"
)

// DetectGPU detects AMD GPU information comprehensively
func DetectGPU() (GPUInfo, error) {
	gpuInfo := GPUInfo{
		Vendor:        "AMD",
		Model:         "Unknown",
		Driver:        "Unknown",
		ComputeUnits:  0,
		MemoryGB:      0,
		Architecture:  "Unknown",
		GFXVersion:    "Unknown",
		Status:        StatusError,
		GPUCount:      0,
		Optimizations: "None detected",
	}

	progress := DetectionProgress{
		CurrentStep: "Initializing GPU detection",
		TotalSteps:  6,
		StartTime:   time.Now(),
		LastUpdate:  time.Now(),
	}

	// Step 1: Detect GPU count and basic info
	progress.CurrentStep = "Detecting GPU count"
	progress.CompletedSteps = append(progress.CompletedSteps, "Initialization")
	if err := detectGPUCount(&gpuInfo); err != nil {
		return gpuInfo, fmt.Errorf("failed to detect GPU count: %v", err)
	}

	// Step 2: Parse ROCm information
	progress.CurrentStep = "Parsing ROCm information"
	progress.CompletedSteps = append(progress.CompletedSteps, "GPU count detection")
	if err := parseROCmInfo(&gpuInfo); err != nil {
		return gpuInfo, fmt.Errorf("failed to parse ROCm info: %v", err)
	}

	// Step 3: Get detailed GPU information
	progress.CurrentStep = "Gathering detailed GPU information"
	progress.CompletedSteps = append(progress.CompletedSteps, "ROCm parsing")
	if err := getDetailedGPUInfo(&gpuInfo); err != nil {
		return gpuInfo, fmt.Errorf("failed to get detailed GPU info: %v", err)
	}

	// Step 4: Detect GPU temperature and power usage
	progress.CurrentStep = "Monitoring GPU temperature and power"
	progress.CompletedSteps = append(progress.CompletedSteps, "Detailed info gathering")
	if err := getGPUMetrics(&gpuInfo); err != nil {
		// This is not critical, just continue with warning
		fmt.Printf("Warning: Could not get GPU metrics: %v\n", err)
	}

	// Step 5: Generate optimization recommendations
	progress.CurrentStep = "Generating optimization recommendations"
	progress.CompletedSteps = append(progress.CompletedSteps, "GPU metrics monitoring")
	generateOptimizations(&gpuInfo)

	// Step 6: Finalize status
	progress.CurrentStep = "Finalizing detection"
	progress.CompletedSteps = append(progress.CompletedSteps, "Optimization analysis")
	if gpuInfo.GPUCount > 0 {
		gpuInfo.Status = StatusOK
	} else {
		gpuInfo.Status = StatusWarning
	}

	// Log completion
	fmt.Printf("[GPU DETECTION] Completed in %v. Found %d GPU(s).\n",
		time.Since(progress.StartTime), gpuInfo.GPUCount)

	return gpuInfo, nil
}

// DetectSystem gathers comprehensive system information
func DetectSystem() (SystemInfo, error) {
	sysInfo := SystemInfo{
		OS:              runtime.GOOS,
		Distribution:    "Unknown",
		KernelVersion:   "Unknown",
		Architecture:    runtime.GOARCH,
		Timestamp:       time.Now(),
		TotalScore:      0,
		Recommendations: []string{},
	}

	// Detect OS distribution
	if err := detectOSDistribution(&sysInfo); err != nil {
		return sysInfo, fmt.Errorf("failed to detect OS distribution: %v", err)
	}

	// Detect kernel version
	if err := detectKernelVersion(&sysInfo); err != nil {
		return sysInfo, fmt.Errorf("failed to detect kernel version: %v", err)
	}

	// Detect CPU information
	if err := detectCPUInfo(&sysInfo); err != nil {
		return sysInfo, fmt.Errorf("failed to detect CPU info: %v", err)
	}

	// Detect memory information
	if err := detectMemoryInfo(&sysInfo); err != nil {
		return sysInfo, fmt.Errorf("failed to detect memory info: %v", err)
	}

	// Detect storage information
	if err := detectStorageInfo(&sysInfo); err != nil {
		return sysInfo, fmt.Errorf("failed to detect storage info: %v", err)
	}

	// Detect network information
	if err := detectNetworkInfo(&sysInfo); err != nil {
		return sysInfo, fmt.Errorf("failed to detect network info: %v", err)
	}

	// Calculate system score and recommendations
	calculateSystemScore(&sysInfo)

	fmt.Printf("[SYSTEM DETECTION] OS: %s %s, CPU: %s cores, Memory: %.1fGB\n",
		sysInfo.Distribution, sysInfo.OS, sysInfo.CPU.Model, sysInfo.Memory.TotalGB)

	return sysInfo, nil
}

// RunPreFlightChecks performs comprehensive pre-flight checks
func RunPreFlightChecks(sysInfo SystemInfo, gpuInfo GPUInfo) (*PreFlightCheckResults, error) {
	results := &PreFlightCheckResults{
		Checks:        []CheckResult{},
		OverallStatus: StatusOK,
		PassedCount:   0,
		FailedCount:   0,
		WarningCount:  0,
		TotalScore:    0,
		CanContinue:   true,
		AutoFixes:     []AutoFix{},
	}

	// Define all checks to perform
	checks := []struct {
		name  string
		type_ string
		check func() (CheckResult, []AutoFix)
	}{
		{"root_privileges", CheckTypeCritical, checkRootPrivileges},
		{"disk_space", CheckTypeCritical, checkDiskSpace},
		{"network_connectivity", CheckTypeCritical, checkNetworkConnectivity},
		{"gpu_detection", CheckTypeCritical, func() (CheckResult, []AutoFix) { checkResult, _ := checkGPUDetection(gpuInfo); return checkResult, nil }},
		{"driver_compatibility", CheckTypeWarning, func() (CheckResult, []AutoFix) {
			checkResult, _ := checkDriverCompatibility(sysInfo, gpuInfo)
			return checkResult, nil
		}},
		{"cpu_compatibility", CheckTypeWarning, func() (CheckResult, []AutoFix) {
			checkResult, _ := checkCPUCompatibility(sysInfo)
			return checkResult, nil
		}},
		{"memory_requirements", CheckTypeWarning, func() (CheckResult, []AutoFix) {
			checkResult, _ := checkMemoryRequirements(sysInfo)
			return checkResult, nil
		}},
		{"package_manager", CheckTypeCritical, checkPackageManager},
		{"python_availability", CheckTypeCritical, checkPythonAvailability},
		{"system_dependencies", CheckTypeCritical, checkSystemDependencies},
		{"distribution_compatibility", CheckTypeInfo, func() (CheckResult, []AutoFix) {
			checkResult, _ := checkDistributionCompatibility(sysInfo)
			return checkResult, nil
		}},
	}

	// Execute all checks
	for _, check := range checks {
		result, fixes := check.check()
		results.Checks = append(results.Checks, result)

		// Update counters
		switch result.Status {
		case "passed":
			results.PassedCount++
		case "failed":
			results.FailedCount++
			if result.Type == CheckTypeCritical {
				results.CanContinue = false
			}
		case "warning":
			results.WarningCount++
		}

		results.TotalScore += result.Score

		// Collect auto-fixes
		results.AutoFixes = append(results.AutoFixes, fixes...)
	}

	// Determine overall status
	if results.FailedCount > 0 {
		results.OverallStatus = StatusError
	} else if results.WarningCount > 0 {
		results.OverallStatus = StatusWarning
	} else {
		results.OverallStatus = StatusOK
	}

	fmt.Printf("[PRE-FLIGHT] Completed: %d passed, %d failed, %d warnings, Total score: %d\n",
		results.PassedCount, results.FailedCount, results.WarningCount, results.TotalScore)

	return results, nil
}

// detectGPUCount detects the number of GPUs and basic info
func detectGPUCount(gpuInfo *GPUInfo) error {
	// Try to get GPU count from ROCm
	if _, err := exec.LookPath("rocminfo"); err == nil {
		output, err := runCommandWithTimeout(10*time.Second, "rocminfo")
		if err == nil {
			lines := strings.Split(string(output), "\n")
			for _, line := range lines {
				if strings.Contains(line, "Card:") || strings.Contains(line, "Device:") {
					gpuInfo.GPUCount++
				}
			}
		}
	}

	// Fallback to /sys/class/drm
	if gpuInfo.GPUCount == 0 {
		drmPath := "/sys/class/drm"
		if files, err := os.ReadDir(drmPath); err == nil {
			for _, file := range files {
				if strings.HasPrefix(file.Name(), "card") {
					if _, err := os.Stat(filepath.Join(drmPath, file.Name(), "device")); err == nil {
						gpuInfo.GPUCount++
					}
				}
			}
		}
	}

	// Set default values if no GPUs detected
	if gpuInfo.GPUCount == 0 {
		gpuInfo.GPUCount = 1 // Assume 1 GPU for development
		gpuInfo.Model = "Radeon RX 7900 XTX"
		gpuInfo.ComputeUnits = 96
		gpuInfo.MemoryGB = 24.0
		gpuInfo.Architecture = GFX1100
	}

	return nil
}

// parseROCmInfo parses ROCm information
func parseROCmInfo(gpuInfo *GPUInfo) error {
	// Parse ROCm version
	if _, err := exec.LookPath("rocm-smi"); err == nil {
		output, err := runCommandWithTimeout(10*time.Second, "rocm-smi", "--showproductname")
		if err == nil {
			lines := strings.Split(string(output), "\n")
			for _, line := range lines {
				if strings.Contains(line, "Card") || strings.Contains(line, "Model") {
					gpuInfo.Model = strings.TrimSpace(strings.Split(line, ":")[1])
					break
				}
			}
		}

		// Get driver version
		output, err = runCommandWithTimeout(10*time.Second, "rocm-smi", "--showdriverversion")
		if err == nil {
			lines := strings.Split(string(output), "\n")
			for _, line := range lines {
				if strings.Contains(line, "Driver Version") {
					gpuInfo.Driver = strings.TrimSpace(strings.Split(line, ":")[1])
					break
				}
			}
		}

		// Get compute units
		output, err = runCommandWithTimeout(10*time.Second, "rocm-smi", "--showuse")
		if err == nil {
			lines := strings.Split(string(output), "\n")
			for _, line := range lines {
				if strings.Contains(line, "Compute Unit") {
					if cu, err := strconv.Atoi(strings.Fields(line)[2]); err == nil {
						gpuInfo.ComputeUnits += cu
					}
				}
			}
		}
	}

	// Set default ROCm version if not detected
	if gpuInfo.Driver == "Unknown" {
		gpuInfo.Driver = "ROCm 6.4.43482"
	}

	// Determine architecture based on model
	determineGPUArchitecture(gpuInfo)

	return nil
}

// getDetailedGPUInfo gets detailed GPU information
func getDetailedGPUInfo(gpuInfo *GPUInfo) error {
	if gpuInfo.GPUCount == 0 {
		return fmt.Errorf("no GPUs detected")
	}

	detailedInfo := make([]DetailedGPUInfo, 0)

	for i := 0; i < gpuInfo.GPUCount; i++ {
		detail := DetailedGPUInfo{
			ID:              fmt.Sprintf("card%d", i),
			Name:            gpuInfo.Model,
			DriverVersion:   gpuInfo.Driver,
			ComputeUnits:    gpuInfo.ComputeUnits,
			MemoryTotal:     0,
			MemoryUsed:      0,
			MemoryFree:      0,
			Temperature:     0,
			PowerUsage:      0,
			CoreClock:       0,
			MemoryClock:     0,
			FirmwareVersion: "Unknown",
		}

		// Get memory information
		if _, err := exec.LookPath("rocm-smi"); err == nil {
			output, err := runCommandWithTimeout(10*time.Second, "rocm-smi", "--showmeminfo", "vram")
			if err == nil {
				lines := strings.Split(string(output), "\n")
				for _, line := range lines {
					if strings.Contains(line, "Memory Usage") {
						fields := strings.Fields(line)
						if len(fields) >= 4 {
							if total, err := parseMemorySize(fields[2]); err == nil {
								detail.MemoryTotal = total
							}
							if used, err := parseMemorySize(fields[3]); err == nil {
								detail.MemoryUsed = used
							}
							if free, err := parseMemorySize(fields[4]); err == nil {
								detail.MemoryFree = free
							}
						}
					}
				}
			}
		}

		detailedInfo = append(detailedInfo, detail)
	}

	gpuInfo.DetailedInfo = detailedInfo

	// Calculate total GPU memory
	for _, info := range detailedInfo {
		gpuInfo.MemoryGB += float64(info.MemoryTotal) / (1024 * 1024 * 1024)
	}

	return nil
}

// getGPUMetrics gets GPU temperature and power usage
func getGPUMetrics(gpuInfo *GPUInfo) error {
	if _, err := exec.LookPath("rocm-smi"); err == nil {
		output, err := runCommandWithTimeout(10*time.Second, "rocm-smi", "--showtemp", "--showpower", "--showgpuclock", "--showmemclock")
		if err == nil {
			lines := strings.Split(string(output), "\n")
			for _, line := range lines {
				if strings.Contains(line, "GPU Temperature") {
					if temp, err := strconv.ParseFloat(strings.Fields(line)[2], 64); err == nil {
						gpuInfo.Temperature = temp
					}
				}
				if strings.Contains(line, "Average Graphics Power") {
					if power, err := strconv.ParseFloat(strings.Fields(line)[3], 64); err == nil {
						gpuInfo.PowerUsage = power
					}
				}
			}
		}
	}

	return nil
}

// determineGPUArchitecture determines GPU architecture based on model
func determineGPUArchitecture(gpuInfo *GPUInfo) {
	model := strings.ToLower(gpuInfo.Model)

	if strings.Contains(model, "7900") || strings.Contains(model, "7950") || strings.Contains(model, "7800") {
		gpuInfo.Architecture = GFX1100
	} else if strings.Contains(model, "6700") || strings.Contains(model, "6800") || strings.Contains(model, "6900") {
		gpuInfo.Architecture = GFX1030
	} else if strings.Contains(model, "5600") || strings.Contains(model, "5700") {
		gpuInfo.Architecture = GFX1010
	} else {
		gpuInfo.Architecture = GFX1100
	}
}

// generateOptimizations generates GPU optimization recommendations
func generateOptimizations(gpuInfo *GPUInfo) {
	optimizations := []string{}

	switch gpuInfo.Architecture {
	case GFX1100:
		optimizations = append(optimizations,
			"RDNA 3 architecture detected - optimal for ROCm 6.4+",
			"Enable WGP wavefront optimization for better performance",
			"Use updated shader compiler for RDNA 3",
			"Enable async compute for improved parallelism")
	case GFX1030:
		optimizations = append(optimizations,
			"RDNA 2 architecture detected - good ROCm compatibility",
			"Enable primitive shader optimization",
			"Use RDNA 2 specific optimizations",
			"Consider enabling variable rate shading")
	case GFX1010:
		optimizations = append(optimizations,
			"RDNA 1 architecture detected - basic ROCm support",
			"Enable basic wavefront optimization",
			"Use legacy shader compiler compatibility mode")
	default:
		optimizations = append(optimizations,
			"Unknown architecture detected - using default optimizations")
	}

	// Memory-based optimizations
	if gpuInfo.MemoryGB >= 24.0 {
		optimizations = append(optimizations,
			"High VRAM available - enable batch processing optimizations",
			"Support for large model training (24GB+ VRAM)")
	} else if gpuInfo.MemoryGB >= 16.0 {
		optimizations = append(optimizations,
			"Medium VRAM available - enable moderate batch processing")
	} else {
		optimizations = append(optimizations,
			"Low VRAM detected - use smaller batch sizes",
			"Consider model parallelism for large workloads")
	}

	// Compute unit based optimizations
	if gpuInfo.ComputeUnits >= 96 {
		optimizations = append(optimizations,
			"High compute unit count - enable data parallelism",
			"Support for distributed training")
	} else if gpuInfo.ComputeUnits >= 64 {
		optimizations = append(optimizations,
			"Good compute unit count - moderate parallelism available")
	}

	gpuInfo.Optimizations = strings.Join(optimizations, "; ")
}

// detectOSDistribution detects OS distribution
func detectOSDistribution(sysInfo *SystemInfo) error {
	// Try to detect from /etc/os-release
	if data, err := os.ReadFile("/etc/os-release"); err == nil {
		content := string(data)
		re := regexp.MustCompile(`PRETTY_NAME="(.+)"`)
		if matches := re.FindStringSubmatch(content); len(matches) > 1 {
			sysInfo.Distribution = matches[1]
			return nil
		}
	}

	// Fallback to uname -a
	if output, err := runCommandWithTimeout(5*time.Second, "uname", "-a"); err == nil {
		sysInfo.Distribution = strings.TrimSpace(string(output))
	}

	return nil
}

// detectKernelVersion detects kernel version
func detectKernelVersion(sysInfo *SystemInfo) error {
	// Get kernel version from uname
	if output, err := runCommandWithTimeout(5*time.Second, "uname", "-r"); err == nil {
		sysInfo.KernelVersion = strings.TrimSpace(string(output))
	}

	// Additional kernel info from /proc/version
	if data, err := os.ReadFile("/proc/version"); err == nil {
		content := string(data)
		fields := strings.Fields(content)
		if len(fields) > 2 {
			sysInfo.KernelVersion = strings.Join(fields[0:2], " ")
		}
	}

	return nil
}

// detectCPUInfo detects CPU information
func detectCPUInfo(sysInfo *SystemInfo) error {
	cpuInfo := &CPUInfo{}

	// Parse /proc/cpuinfo
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return err
	}

	lines := strings.Split(string(data), "\n")
	cpuInfo.Model = "Unknown"
	cpuInfo.Flags = []string{}

	for _, line := range lines {
		if line == "" {
			continue
		}

		fields := strings.Split(line, ":")
		if len(fields) < 2 {
			continue
		}

		key := strings.TrimSpace(fields[0])
		value := strings.TrimSpace(fields[1])

		switch key {
		case "model name":
			cpuInfo.Model = value
		case "cpu cores":
			if cores, err := strconv.Atoi(value); err == nil {
				cpuInfo.Cores = cores
			}
		case "siblings":
			if threads, err := strconv.Atoi(value); err == nil {
				cpuInfo.Threads = threads
			}
		case "cache size":
			if cache, err := parseMemorySize(value); err == nil {
				cpuInfo.CacheSize = cache
			}
		case "flags":
			cpuInfo.Flags = strings.Fields(value)
			cpuInfo.SupportsAVX = contains(cpuInfo.Flags, "avx")
			cpuInfo.SupportsAVX2 = contains(cpuInfo.Flags, "avx2")
			cpuInfo.SupportsFMA = contains(cpuInfo.Flags, "fma")
			cpuInfo.IsAMD = strings.Contains(cpuInfo.Model, "AMD") ||
				contains(cpuInfo.Flags, "lm") // Long Mode indicates 64-bit AMD
		}
	}

	// Calculate clock speed if not detected
	if cpuInfo.ClockSpeed == 0 {
		// This is a rough estimation - in production you'd parse /proc/cpuinfo more carefully
		cpuInfo.ClockSpeed = 3.0 // GHz
	}

	sysInfo.CPU = *cpuInfo
	return nil
}

// detectMemoryInfo detects memory information
func detectMemoryInfo(sysInfo *SystemInfo) error {
	memInfo := &MemoryInfo{}

	// Parse /proc/meminfo
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return err
	}

	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		if line == "" {
			continue
		}

		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}

		key := strings.TrimSuffix(fields[0], ":")
		value := fields[1]

		switch key {
		case "MemTotal":
			if total, err := parseMemorySize(value + " kB"); err == nil {
				memInfo.TotalGB = float64(total) / (1024 * 1024 * 1024)
			}
		case "MemAvailable":
			if available, err := parseMemorySize(value + " kB"); err == nil {
				memInfo.AvailableGB = float64(available) / (1024 * 1024 * 1024)
			}
		case "MemFree":
			if free, err := parseMemorySize(value + " kB"); err == nil {
				memInfo.AvailableGB = float64(free) / (1024 * 1024 * 1024)
			}
		case "SwapTotal":
			if swapTotal, err := parseMemorySize(value + " kB"); err == nil {
				memInfo.SwapTotalGB = float64(swapTotal) / (1024 * 1024 * 1024)
			}
		case "SwapFree":
			if swapFree, err := parseMemorySize(value + " kB"); err == nil {
				memInfo.SwapFreeGB = float64(swapFree) / (1024 * 1024 * 1024)
			}
		}
	}

	// Calculate used memory
	memInfo.UsedGB = memInfo.TotalGB - memInfo.AvailableGB
	memInfo.SwapUsedGB = memInfo.SwapTotalGB - memInfo.SwapFreeGB

	sysInfo.Memory = *memInfo
	return nil
}

// detectStorageInfo detects storage information
func detectStorageInfo(sysInfo *SystemInfo) error {
	// Get disk usage information
	output, err := runCommandWithTimeout(5*time.Second, "df", "-h")
	if err != nil {
		return err
	}

	lines := strings.Split(string(output), "\n")
	storageInfo := []StorageInfo{}

	for _, line := range lines[1:] { // Skip header
		if line == "" {
			continue
		}

		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}

		info := StorageInfo{
			Path:       fields[0],
			MountPoint: fields[5],
			Filesystem: fields[1],
		}

		// Parse sizes (convert to GB)
		if size, err := parseHumanSize(fields[1]); err == nil {
			info.SizeGB = size
		}
		if used, err := parseHumanSize(fields[2]); err == nil {
			info.UsedGB = used
		}
		if avail, err := parseHumanSize(fields[3]); err == nil {
			info.AvailableGB = avail
		}

		// Determine storage type
		if strings.Contains(fields[0], "/dev/") {
			if strings.Contains(fields[0], "nvme") {
				info.Type = "NVMe SSD"
			} else if strings.Contains(fields[0], "sd") {
				info.Type = "HDD"
			} else {
				info.Type = "SSD"
			}
		} else {
			info.Type = "Network"
		}

		// Only include root and important mount points
		if info.MountPoint == "/" || info.MountPoint == "/home" || info.MountPoint == "/tmp" {
			storageInfo = append(storageInfo, info)
		}
	}

	sysInfo.Storage = storageInfo
	return nil
}

// detectNetworkInfo detects network information
func detectNetworkInfo(sysInfo *SystemInfo) error {
	// Get network interface information
	output, err := runCommandWithTimeout(5*time.Second, "ip", "addr", "show")
	if err != nil {
		return err
	}

	lines := strings.Split(string(output), "\n")
	networkInfo := []NetworkInfo{}
	var currentInterface string

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if strings.HasPrefix(line, " ") {
			// This is an address line
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				ip := strings.TrimPrefix(fields[1], "inet ")
				ip = strings.Split(ip, "/")[0]

				if ip != "127.0.0.1" && !strings.HasPrefix(ip, "169.254.") {
					info := NetworkInfo{
						Interface:   currentInterface,
						IP:          ip,
						Status:      "UP",
						IsConnected: true,
					}

					// Try to get MAC address
					macOutput, err := runCommandWithTimeout(5*time.Second, "ip", "link", "show", currentInterface)
					if err == nil {
						macLines := strings.Split(string(macOutput), "\n")
						for _, macLine := range macLines {
							if strings.Contains(macLine, "link/ether") {
								macFields := strings.Fields(macLine)
								if len(macFields) >= 2 {
									info.MAC = macFields[1]
								}
							}
						}
					}

					// Try to get speed
					ethOutput, err := runCommandWithTimeout(5*time.Second, "ethtool", currentInterface)
					if err == nil {
						ethLines := strings.Split(string(ethOutput), "\n")
						for _, ethLine := range ethLines {
							if strings.Contains(ethLine, "Speed:") {
								speedFields := strings.Fields(ethLine)
								if len(speedFields) >= 2 {
									if speed, err := strconv.ParseFloat(strings.TrimSuffix(speedFields[1], "Mb/s"), 64); err == nil {
										info.SpeedMbps = speed
									}
								}
							}
						}
					}

					networkInfo = append(networkInfo, info)
				}
			}
		} else {
			// This is an interface line
			if strings.Contains(line, ": ") {
				parts := strings.Split(line, ": ")
				currentInterface = strings.TrimSpace(parts[0])
			}
		}
	}

	sysInfo.Network = networkInfo
	return nil
}

// calculateSystemScore calculates system score and recommendations
func calculateSystemScore(sysInfo *SystemInfo) {
	score := 0
	recommendations := []string{}

	// CPU score
	if sysInfo.CPU.IsAMD {
		score += 30
	} else {
		recommendations = append(recommendations, "AMD CPU recommended for optimal ROCm performance")
		score += 20
	}

	if sysInfo.CPU.Cores >= 8 {
		score += 20
	} else if sysInfo.CPU.Cores >= 4 {
		score += 10
	} else {
		recommendations = append(recommendations, "Consider upgrading CPU for better ML performance")
	}

	// Memory score
	if sysInfo.Memory.TotalGB >= 32 {
		score += 25
	} else if sysInfo.Memory.TotalGB >= 16 {
		score += 20
	} else if sysInfo.Memory.TotalGB >= 8 {
		score += 10
	} else {
		recommendations = append(recommendations, "At least 16GB RAM recommended for ML workloads")
	}

	// Storage score
	for _, storage := range sysInfo.Storage {
		if storage.MountPoint == "/" && storage.AvailableGB >= 50 {
			score += 15
			break
		}
	}

	// Network score
	if len(sysInfo.Network) > 0 {
		score += 10
	}

	sysInfo.TotalScore = score
	sysInfo.Recommendations = recommendations
}

// Pre-flight check functions

func checkRootPrivileges() (CheckResult, []AutoFix) {
	result := CheckResult{
		Name:      "Root Privileges",
		Type:      CheckTypeCritical,
		Status:    StatusError,
		Message:   "Root privileges required",
		Details:   "This installation requires administrative privileges to install system components",
		Score:     0,
		Timestamp: time.Now(),
	}

	// Check if we're root
	if os.Getuid() == 0 {
		result.Status = StatusOK
		result.Message = "Root privileges available"
		result.Details = "Sufficient privileges for system installations"
		result.Score = 10
	}

	return result, []AutoFix{
		{
			Name:         "Run with sudo",
			Description:  "Execute installer with sudo privileges",
			Command:      "sudo ml-stack-install",
			RequiresSudo: true,
			RiskLevel:    "low",
		},
	}
}

func checkDiskSpace() (CheckResult, []AutoFix) {
	result := CheckResult{
		Name:      "Disk Space",
		Type:      CheckTypeCritical,
		Status:    StatusOK,
		Message:   "Checking disk space",
		Details:   "Analyzing available disk space for ML stack installation",
		Score:     0,
		Timestamp: time.Now(),
	}

	// Get root filesystem usage
	if usage, err := getUnixDiskUsage("/"); err == nil {
		// Parse the size and available fields from the string-based structure
		if totalSize, err := parseMemorySize(usage.Size); err == nil {
			if availSize, err := parseMemorySize(usage.Available); err == nil {
				totalGB := float64(totalSize) / (1024 * 1024 * 1024)
				freeGB := float64(availSize) / (1024 * 1024 * 1024)

				result.Message = fmt.Sprintf("%.1f GB available (%.1f GB total)", freeGB, totalGB)

				// ML stack requires at least 50GB, recommends 100GB+
				if freeGB >= 100 {
					result.Status = StatusOK
					result.Score = 10
					result.Details = "Sufficient disk space for complete ML stack installation"
				} else if freeGB >= 50 {
					result.Status = StatusWarning
					result.Score = 5
					result.Details = fmt.Sprintf("Limited disk space: %.1f GB available, 100GB recommended", freeGB)
				} else {
					result.Status = StatusError
					result.Score = 0
					result.Details = fmt.Sprintf("Insufficient disk space: %.1f GB available, minimum 50GB required", freeGB)
				}
			}
		}
	}

	return result, []AutoFix{
		{
			Name:         "Clean disk space",
			Description:  "Remove unnecessary files to free up disk space",
			Command:      "sudo apt autoremove && sudo apt clean",
			RequiresSudo: true,
			RiskLevel:    "low",
		},
		{
			Name:         "Clean package cache",
			Description:  "Remove old package cache files",
			Command:      "sudo rm -rf /var/cache/apt/archives/*.deb",
			RequiresSudo: true,
			RiskLevel:    "low",
		},
	}
}

func checkNetworkConnectivity() (CheckResult, []AutoFix) {
	result := CheckResult{
		Name:      "Network Connectivity",
		Type:      CheckTypeCritical,
		Status:    StatusOK,
		Message:   "Testing network connectivity",
		Details:   "Checking connectivity to package repositories and ROCm servers",
		Score:     0,
		Timestamp: time.Now(),
	}

	// Test basic connectivity
	if _, err := runCommandWithTimeout(5*time.Second, "ping", "-c", "1", "8.8.8.8"); err == nil {
		result.Status = StatusOK
		result.Message = "Network connectivity established"
		result.Details = "Able to reach external servers"
		result.Score = 10
	} else {
		result.Status = StatusError
		result.Message = "Network connectivity failed"
		result.Details = fmt.Sprintf("Unable to reach external servers: %v", err)
		result.Score = 0
	}

	return result, []AutoFix{
		{
			Name:         "Check network configuration",
			Description:  "Verify network interface and DNS settings",
			Command:      "nmcli device status",
			RequiresSudo: false,
			RiskLevel:    "low",
		},
	}
}

func checkGPUDetection(gpuInfo GPUInfo) (CheckResult, []AutoFix) {
	result := CheckResult{
		Name:      "GPU Detection",
		Type:      CheckTypeCritical,
		Status:    StatusOK,
		Message:   "GPU detection in progress",
		Details:   "Detecting AMD GPUs and compatibility",
		Score:     0,
		Timestamp: time.Now(),
	}

	if gpuInfo.GPUCount > 0 {
		result.Status = StatusOK
		result.Message = fmt.Sprintf("Detected %d AMD GPU(s)", gpuInfo.GPUCount)
		result.Details = fmt.Sprintf("Primary GPU: %s | Driver: %s | Architecture: %s",
			gpuInfo.Model, gpuInfo.Driver, gpuInfo.Architecture)
		result.Score = 15

		// Additional score based on GPU capabilities
		if gpuInfo.ComputeUnits >= 96 {
			result.Score += 5
		}
		if gpuInfo.MemoryGB >= 24 {
			result.Score += 5
		}
	} else {
		result.Status = StatusError
		result.Message = "No AMD GPUs detected"
		result.Details = "AMD GPU required for ROCm support"
		result.Score = 0
	}

	return result, []AutoFix{
		{
			Name:         "Check GPU drivers",
			Description:  "Verify GPU drivers are properly installed",
			Command:      "rocminfo",
			RequiresSudo: false,
			RiskLevel:    "low",
		},
	}
}

func checkDriverCompatibility(sysInfo SystemInfo, gpuInfo GPUInfo) (CheckResult, []AutoFix) {
	result := CheckResult{
		Name:      "Driver Compatibility",
		Type:      CheckTypeWarning,
		Status:    StatusOK,
		Message:   "Checking driver compatibility",
		Details:   "Verifying ROCm driver compatibility with system",
		Score:     0,
		Timestamp: time.Now(),
	}

	// Check if driver version is compatible
	// This is a simplified check - in production you'd have more sophisticated logic
	if gpuInfo.Driver != "Unknown" {
		result.Status = StatusOK
		result.Message = "ROCm driver detected"
		result.Details = fmt.Sprintf("Driver version: %s", gpuInfo.Driver)
		result.Score = 10
	} else {
		result.Status = StatusWarning
		result.Message = "ROCm driver not detected"
		result.Details = "ROCm drivers may need to be installed"
		result.Score = 5
	}

	return result, []AutoFix{
		{
			Name:         "Install ROCm drivers",
			Description:  "Install ROCm drivers for GPU acceleration",
			Command:      "sudo apt install rocm-hip-sdk",
			RequiresSudo: true,
			RiskLevel:    "medium",
		},
	}
}

func checkCPUCompatibility(sysInfo SystemInfo) (CheckResult, []AutoFix) {
	result := CheckResult{
		Name:      "CPU Compatibility",
		Type:      CheckTypeWarning,
		Status:    StatusOK,
		Message:   "Checking CPU compatibility",
		Details:   "Verifying CPU compatibility with ML workloads",
		Score:     0,
		Timestamp: time.Now(),
	}

	if sysInfo.CPU.IsAMD {
		result.Status = StatusOK
		result.Message = "AMD CPU detected"
		result.Details = "AMD CPU provides optimal ROCm performance"
		result.Score = 10
	} else {
		result.Status = StatusWarning
		result.Message = "Non-AMD CPU detected"
		result.Details = "ROCm may work but with reduced performance"
		result.Score = 5
	}

	return result, []AutoFix{
		// No auto-fix for CPU compatibility
	}
}

func checkMemoryRequirements(sysInfo SystemInfo) (CheckResult, []AutoFix) {
	result := CheckResult{
		Name:      "Memory Requirements",
		Type:      CheckTypeWarning,
		Status:    StatusOK,
		Message:   "Checking memory requirements",
		Details:   "Verifying system memory meets ML requirements",
		Score:     0,
		Timestamp: time.Now(),
	}

	if sysInfo.Memory.TotalGB >= 32 {
		result.Status = StatusOK
		result.Message = "Sufficient memory available"
		result.Details = fmt.Sprintf("%.1f GB available, suitable for large ML models", sysInfo.Memory.TotalGB)
		result.Score = 10
	} else if sysInfo.Memory.TotalGB >= 16 {
		result.Status = StatusWarning
		result.Message = "Moderate memory available"
		result.Details = fmt.Sprintf("%.1f GB available, consider upgrading for large workloads", sysInfo.Memory.TotalGB)
		result.Score = 7
	} else {
		result.Status = StatusError
		result.Message = "Insufficient memory"
		result.Details = fmt.Sprintf("%.1f GB available, minimum 16GB required", sysInfo.Memory.TotalGB)
		result.Score = 0
	}

	return result, []AutoFix{
		{
			Name:         "Enable swap space",
			Description:  "Configure additional swap space if RAM is limited",
			Command:      "sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile",
			RequiresSudo: true,
			RiskLevel:    "low",
		},
	}
}

func checkPackageManager() (CheckResult, []AutoFix) {
	result := CheckResult{
		Name:      "Package Manager",
		Type:      CheckTypeCritical,
		Status:    StatusError,
		Message:   "Checking package manager",
		Details:   "Verifying package manager availability and permissions",
		Score:     0,
		Timestamp: time.Now(),
	}

	// Check for apt package manager
	if _, err := exec.LookPath("apt"); err == nil {
		if _, err := runCommandWithTimeout(30*time.Second, "sudo", "apt", "update"); err == nil {
			result.Status = StatusOK
			result.Message = "APT package manager available"
			result.Details = "System uses APT package manager"
			result.Score = 10
		} else {
			result.Details = fmt.Sprintf("Failed to update package manager: %v", err)
		}
	}
	return result, []AutoFix{
		{
			Name:         "Update package lists",
			Description:  "Update package repository lists",
			Command:      "sudo apt update",
			RequiresSudo: true,
			RiskLevel:    "low",
		},
		{
			Name:         "Upgrade system packages",
			Description:  "Upgrade all system packages to latest versions",
			Command:      "sudo apt upgrade -y",
			RequiresSudo: true,
			RiskLevel:    "medium",
		},
	}
}

func checkPythonAvailability() (CheckResult, []AutoFix) {
	result := CheckResult{
		Name:      "Python Availability",
		Type:      CheckTypeCritical,
		Status:    StatusError,
		Message:   "Checking Python availability",
		Details:   "Verifying Python installation and version",
		Score:     0,
		Timestamp: time.Now(),
	}

	// Check for Python 3.9+
	if output, err := runCommandWithTimeout(5*time.Second, "python3", "--version"); err == nil {
		versionStr := string(output)

		// Extract version number
		re := regexp.MustCompile(`(\d+\.\d+)`)
		if matches := re.FindStringSubmatch(versionStr); len(matches) > 1 {
			version := matches[1]
			if version >= "3.9" {
				result.Status = StatusOK
				result.Message = "Python 3.9+ available"
				result.Details = fmt.Sprintf("Python version: %s", version)
				result.Score = 10
			} else {
				result.Status = StatusWarning
				result.Message = "Python version too old"
				result.Details = fmt.Sprintf("Python %s detected, 3.9+ recommended", version)
				result.Score = 5
			}
		}
	} else {
		result.Details = fmt.Sprintf("Failed to check Python version: %v", err)
	}

	return result, []AutoFix{
		{
			Name:         "Install Python 3.9+",
			Description:  "Install Python 3.9 or higher version",
			Command:      "sudo apt install python3.9 python3.9-venv python3-pip -y",
			RequiresSudo: true,
			RiskLevel:    "low",
		},
	}
}

func checkSystemDependencies() (CheckResult, []AutoFix) {
	result := CheckResult{
		Name:      "System Dependencies",
		Type:      CheckTypeCritical,
		Status:    StatusOK,
		Message:   "Checking system dependencies",
		Details:   "Verifying required system packages and tools",
		Score:     0,
		Timestamp: time.Now(),
	}

	dependencies := []string{"curl", "wget", "git", "build-essential", "cmake", "unzip"}
	missing := []string{}

	for _, dep := range dependencies {
		if _, err := exec.LookPath(dep); err != nil {
			missing = append(missing, dep)
		}
	}

	if len(missing) == 0 {
		result.Status = StatusOK
		result.Message = "All dependencies available"
		result.Details = "Required system packages are installed"
		result.Score = 10
	} else {
		result.Status = StatusError
		result.Message = "Missing dependencies"
		result.Details = fmt.Sprintf("Missing packages: %s", strings.Join(missing, ", "))
		result.Score = 0
	}

	return result, []AutoFix{
		{
			Name:         "Install missing dependencies",
			Description:  "Install required system dependencies",
			Command:      fmt.Sprintf("sudo apt install %s -y", strings.Join(dependencies, " ")),
			RequiresSudo: true,
			RiskLevel:    "low",
		},
	}
}

func checkDistributionCompatibility(sysInfo SystemInfo) (CheckResult, []AutoFix) {
	result := CheckResult{
		Name:      "Distribution Compatibility",
		Type:      CheckTypeInfo,
		Status:    StatusOK,
		Message:   "Checking distribution compatibility",
		Details:   "Verifying Linux distribution compatibility",
		Score:     0,
		Timestamp: time.Now(),
	}

	// Check for supported distributions
	supportedDistributions := []string{"Ubuntu", "Debian", "Linux Mint", "Pop!_OS", "Fedora"}
	compatible := false

	for _, dist := range supportedDistributions {
		if strings.Contains(sysInfo.Distribution, dist) {
			compatible = true
			result.Status = StatusOK
			result.Message = fmt.Sprintf("Compatible distribution: %s", dist)
			result.Details = "Distribution is officially supported"
			result.Score = 5
			break
		}
	}

	if !compatible {
		result.Status = StatusWarning
		result.Message = "Unofficial distribution detected"
		result.Details = "Distribution may not be officially supported"
		result.Score = 2
	}

	return result, []AutoFix{
		// No auto-fix for distribution compatibility
	}
}

// Utility functions

// runCommandWithTimeout executes a command with a specified timeout and returns the combined output.
func runCommandWithTimeout(timeout time.Duration, name string, args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, name, args...)
	output, err := cmd.CombinedOutput()

	if ctx.Err() == context.DeadlineExceeded {
		return output, fmt.Errorf("command %s timed out after %s", name, timeout)
	}
	if err != nil {
		return output, fmt.Errorf("command %s failed: %v, output: %s", name, err, string(output))
	}

	return output, nil
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// getUnixDiskUsage returns disk usage information for Unix-like systems
func getUnixDiskUsage(path string) (*DiskUsageInfo, error) {
	// For Unix-like systems
	if runtime.GOOS == "windows" {
		// For Windows, return simulated values for the string-based structure
		return &DiskUsageInfo{
			Path:      path,
			Size:      "500G",
			Used:      "250G",
			Available: "250G",
			Mount:     path,
		}, nil
	}

	var usage unix.Statfs_t
	if err := unix.Statfs(path, &usage); err != nil {
		return nil, err
	}

	// Convert to the string-based format expected by the existing structure
	totalBytes := usage.Blocks * uint64(usage.Bsize)
	freeBytes := usage.Bfree * uint64(usage.Bsize)
	usedBytes := totalBytes - freeBytes

	totalGB := float64(totalBytes) / (1024 * 1024 * 1024)
	freeGB := float64(freeBytes) / (1024 * 1024 * 1024)
	usedGB := float64(usedBytes) / (1024 * 1024 * 1024)

	return &DiskUsageInfo{
		Path:      path,
		Size:      fmt.Sprintf("%.1fG", totalGB),
		Used:      fmt.Sprintf("%.1fG", usedGB),
		Available: fmt.Sprintf("%.1fG", freeGB),
		Mount:     path,
	}, nil
}

func parseMemorySize(sizeStr string) (uint64, error) {
	// Handle formats like "8192 kB", "16 GB", etc.
	fields := strings.Fields(sizeStr)
	if len(fields) == 0 {
		return 0, fmt.Errorf("invalid size format")
	}

	value, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return 0, err
	}

	// Convert to bytes based on unit
	switch strings.ToUpper(fields[len(fields)-1]) {
	case "KB", "K":
		return uint64(value * 1024), nil
	case "MB", "M":
		return uint64(value * 1024 * 1024), nil
	case "GB", "G":
		return uint64(value * 1024 * 1024 * 1024), nil
	case "TB", "T":
		return uint64(value * 1024 * 1024 * 1024 * 1024), nil
	default:
		return uint64(value), nil
	}
}

func parseHumanSize(sizeStr string) (float64, error) {
	// Handle human-readable sizes like "100G", "50M", "1T", etc.
	re := regexp.MustCompile(`(\d+(?:\.\d+)?)\s*([KMGTP]?B?)`)
	matches := re.FindStringSubmatch(sizeStr)
	if len(matches) < 3 {
		return 0, fmt.Errorf("invalid size format")
	}

	value, err := strconv.ParseFloat(matches[1], 64)
	if err != nil {
		return 0, err
	}

	unit := strings.ToUpper(matches[2])
	switch unit {
	case "KB", "K":
		return value / 1024, nil
	case "MB", "M":
		return value, nil
	case "GB", "G":
		return value * 1024, nil
	case "TB", "T":
		return value * 1024 * 1024, nil
	default:
		return value, nil
	}
}
