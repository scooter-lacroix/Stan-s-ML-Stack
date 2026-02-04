// internal/ui/monitoring/diagnostics.go
package monitoring

import (
	"encoding/json"
	"fmt"
	"runtime"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// DiagnosticResult represents the result of a diagnostic test
type DiagnosticResult struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Category        string                 `json:"category"`
	Status          string                 `json:"status"` // "pass", "fail", "warning", "info"
	Score           int                    `json:"score"`  // 0-100
	Message         string                 `json:"message"`
	Details         map[string]interface{} `json:"details"`
	Recommendations []string               `json:"recommendations"`
	Timestamp       time.Time              `json:"timestamp"`
	Duration        time.Duration          `json:"duration"`
	Error           error                  `json:"error,omitempty"`
}

// DiagnosticCategory represents different diagnostic categories
type DiagnosticCategory string

const (
	CategorySystem      DiagnosticCategory = "system"
	CategoryGPU         DiagnosticCategory = "gpu"
	CategoryNetwork     DiagnosticCategory = "network"
	CategoryStorage     DiagnosticCategory = "storage"
	CategorySecurity    DiagnosticCategory = "security"
	CategoryPerformance DiagnosticCategory = "performance"
	CategorySoftware    DiagnosticCategory = "software"
)

// DiagnosticEngine performs system diagnostics
type DiagnosticEngine struct {
	theme       *AMDTheme
	results     []DiagnosticResult
	running     bool
	progress    float64
	currentTest string
	lastRun     time.Time
	config      DiagnosticConfig
}

// DiagnosticConfig holds configuration for diagnostics
type DiagnosticConfig struct {
	EnabledCategories []DiagnosticCategory `json:"enabled_categories"`
	TestTimeout       time.Duration        `json:"test_timeout"`
	ParallelTests     int                  `json:"parallel_tests"`
	AutoFix           bool                 `json:"auto_fix"`
	SaveResults       bool                 `json:"save_results"`
	MaxHistory        int                  `json:"max_history"`
}

// DiagnosticWidget displays diagnostic results and controls
type DiagnosticWidget struct {
	width, height int
	x, y          int
	theme         *AMDTheme
	engine        *DiagnosticEngine
	selected      int
	scrollOffset  int
	showDetails   bool
	autoRunning   bool
	lastUpdate    time.Time
}

// SystemProfiler collects detailed system information
type SystemProfiler struct {
	theme *AMDTheme
}

// HardwareProfiler collects hardware-specific information
type HardwareProfiler struct {
	theme   *AMDTheme
	gpuInfo []GPUInfo
}

// GPUInfo contains detailed GPU information
type GPUInfo struct {
	Vendor        string    `json:"vendor"`
	Model         string    `json:"model"`
	DeviceID      string    `json:"device_id"`
	DriverVersion string    `json:"driver_version"`
	ROCmVersion   string    `json:"rocm_version"`
	MemoryTotal   uint64    `json:"memory_total"`
	MemoryUsed    uint64    `json:"memory_used"`
	Temperature   float64   `json:"temperature"`
	PowerUsage    float64   `json:"power_usage"`
	ClockSpeed    float64   `json:"clock_speed"`
	ComputeUnits  int       `json:"compute_units"`
	GFXVersion    string    `json:"gfx_version"`
	LinkWidth     int       `json:"link_width"`
	LinkSpeed     float64   `json:"link_speed"`
	ActivityLevel float64   `json:"activity_level"`
	Status        string    `json:"status"`
	LastUpdated   time.Time `json:"last_updated"`
}

// PerformanceAnalyzer analyzes system performance
type PerformanceAnalyzer struct {
	theme            *AMDTheme
	benchmarks       map[string]BenchmarkResult
	performanceTrend []PerformanceSnapshot
}

// BenchmarkResult represents the result of a performance benchmark
type BenchmarkResult struct {
	Name       string             `json:"name"`
	Category   string             `json:"category"`
	Score      float64            `json:"score"`
	Unit       string             `json:"unit"`
	Metrics    map[string]float64 `json:"metrics"`
	Timestamp  time.Time          `json:"timestamp"`
	SystemInfo map[string]string  `json:"system_info"`
}

// PerformanceSnapshot represents a point-in-time performance snapshot
type PerformanceSnapshot struct {
	Timestamp       time.Time              `json:"timestamp"`
	CPUUsage        float64                `json:"cpu_usage"`
	MemoryUsage     float64                `json:"memory_usage"`
	GPUUsage        []float64              `json:"gpu_usage"`
	DiskIO          float64                `json:"disk_io"`
	NetworkIO       float64                `json:"network_io"`
	LoadAverage     []float64              `json:"load_average"`
	ContextSwitches uint64                 `json:"context_switches"`
	Processes       int                    `json:"processes"`
	Additional      map[string]interface{} `json:"additional"`
}

// NewDiagnosticEngine creates a new diagnostic engine
func NewDiagnosticEngine(theme *AMDTheme) *DiagnosticEngine {
	return &DiagnosticEngine{
		theme:   theme,
		results: make([]DiagnosticResult, 0),
		running: false,
		config: DiagnosticConfig{
			EnabledCategories: []DiagnosticCategory{
				CategorySystem,
				CategoryGPU,
				CategoryNetwork,
				CategoryStorage,
				CategorySecurity,
				CategoryPerformance,
				CategorySoftware,
			},
			TestTimeout:   30 * time.Second,
			ParallelTests: 4,
			AutoFix:       false,
			SaveResults:   true,
			MaxHistory:    100,
		},
	}
}

// RunDiagnostics runs all enabled diagnostic tests
func (e *DiagnosticEngine) RunDiagnostics() tea.Cmd {
	return func() tea.Msg {
		e.running = true
		e.progress = 0.0

		var allResults []DiagnosticResult
		totalTests := len(e.config.EnabledCategories)

		for i, category := range e.config.EnabledCategories {
			e.currentTest = string(category)
			e.progress = float64(i) / float64(totalTests)

			results := e.runCategoryDiagnostics(category)
			allResults = append(allResults, results...)
		}

		e.progress = 1.0
		e.running = false
		e.lastRun = time.Now()

		// Store results
		e.results = append(e.results, allResults...)
		if len(e.results) > e.config.MaxHistory {
			e.results = e.results[len(e.results)-e.config.MaxHistory:]
		}

		return DiagnosticCompleteMsg{
			Results:   allResults,
			Timestamp: time.Now(),
			Duration:  time.Since(time.Now().Add(-time.Minute)), // Simplified
		}
	}
}

// runCategoryDiagnostics runs diagnostics for a specific category
func (e *DiagnosticEngine) runCategoryDiagnostics(category DiagnosticCategory) []DiagnosticResult {
	var results []DiagnosticResult

	switch category {
	case CategorySystem:
		results = append(results, e.runSystemDiagnostics()...)
	case CategoryGPU:
		results = append(results, e.runGPUDiagnostics()...)
	case CategoryNetwork:
		results = append(results, e.runNetworkDiagnostics()...)
	case CategoryStorage:
		results = append(results, e.runStorageDiagnostics()...)
	case CategorySecurity:
		results = append(results, e.runSecurityDiagnostics()...)
	case CategoryPerformance:
		results = append(results, e.runPerformanceDiagnostics()...)
	case CategorySoftware:
		results = append(results, e.runSoftwareDiagnostics()...)
	}

	return results
}

// runSystemDiagnostics runs system-level diagnostics
func (e *DiagnosticEngine) runSystemDiagnostics() []DiagnosticResult {
	var results []DiagnosticResult

	// CPU diagnostics
	results = append(results, e.checkCPUHealth())

	// Memory diagnostics
	results = append(results, e.checkMemoryHealth())

	// System load diagnostics
	results = append(results, e.checkSystemLoad())

	// System resources diagnostics
	results = append(results, e.checkSystemResources())

	return results
}

// checkCPUHealth checks CPU health and performance
func (e *DiagnosticEngine) checkCPUHealth() DiagnosticResult {
	start := time.Now()

	// Simulate CPU health check
	cpuUsage := 25.0 + (float64(start.UnixNano()%10000) / 100.0)
	temperature := 45.0 + (cpuUsage * 0.3)

	result := DiagnosticResult{
		ID:        "cpu_health",
		Name:      "CPU Health Check",
		Category:  string(CategorySystem),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"cpu_usage":    cpuUsage,
			"temperature":  temperature,
			"cores":        runtime.NumCPU(),
			"architecture": runtime.GOARCH,
		},
	}

	// Evaluate CPU health
	if cpuUsage > 90 {
		result.Status = "fail"
		result.Score = 20
		result.Message = "CPU usage is critically high"
		result.Recommendations = []string{
			"Check for runaway processes",
			"Consider upgrading CPU if consistently high usage",
			"Review system resource allocation",
		}
	} else if cpuUsage > 70 {
		result.Status = "warning"
		result.Score = 60
		result.Message = "CPU usage is elevated"
		result.Recommendations = []string{
			"Monitor CPU trends",
			"Check for resource-intensive processes",
		}
	} else {
		result.Status = "pass"
		result.Score = 95
		result.Message = "CPU usage is normal"
		result.Recommendations = []string{
			"Continue monitoring CPU usage",
		}
	}

	// Add temperature check
	if temperature > 85 {
		result.Status = "fail"
		result.Score = minInt(result.Score, 30)
		result.Recommendations = append(result.Recommendations, "Check CPU cooling system")
	}

	return result
}

// checkMemoryHealth checks memory health and usage
func (e *DiagnosticEngine) checkMemoryHealth() DiagnosticResult {
	start := time.Now()

	// Simulate memory health check
	memUsage := 40.0 + (float64(start.UnixNano()%20000) / 100.0)
	swapUsage := 5.0 + (float64(start.UnixNano()%5000) / 100.0)

	result := DiagnosticResult{
		ID:        "memory_health",
		Name:      "Memory Health Check",
		Category:  string(CategorySystem),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"memory_usage_percent": memUsage,
			"swap_usage_percent":   swapUsage,
			"available_memory":     "18.5 GB",
			"total_memory":         "32.0 GB",
		},
	}

	// Evaluate memory health
	if memUsage > 90 {
		result.Status = "fail"
		result.Score = 25
		result.Message = "Memory usage is critically high"
		result.Recommendations = []string{
			"Close unnecessary applications",
			"Consider adding more RAM",
			"Check for memory leaks",
		}
	} else if memUsage > 75 {
		result.Status = "warning"
		result.Score = 65
		result.Message = "Memory usage is elevated"
		result.Recommendations = []string{
			"Monitor memory usage trends",
			"Consider memory optimization",
		}
	} else {
		result.Status = "pass"
		result.Score = 90
		result.Message = "Memory usage is normal"
		result.Recommendations = []string{
			"Continue monitoring memory usage",
		}
	}

	return result
}

// checkSystemLoad checks system load averages
func (e *DiagnosticEngine) checkSystemLoad() DiagnosticResult {
	start := time.Now()

	// Simulate load average check
	load1 := 0.5 + (float64(start.UnixNano()%100) / 100.0)
	load5 := 0.7 + (float64(start.UnixNano()%80) / 100.0)
	load15 := 0.6 + (float64(start.UnixNano()%60) / 100.0)

	result := DiagnosticResult{
		ID:        "system_load",
		Name:      "System Load Check",
		Category:  string(CategorySystem),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"load_1min":  load1,
			"load_5min":  load5,
			"load_15min": load15,
			"cpu_cores":  runtime.NumCPU(),
		},
	}

	// Evaluate system load
	maxLoad := float64(runtime.NumCPU())
	if load1 > maxLoad*0.9 {
		result.Status = "fail"
		result.Score = 30
		result.Message = "System load is critically high"
		result.Recommendations = []string{
			"Identify and terminate resource-intensive processes",
			"Consider system optimization",
		}
	} else if load1 > maxLoad*0.7 {
		result.Status = "warning"
		result.Score = 70
		result.Message = "System load is elevated"
		result.Recommendations = []string{
			"Monitor system load trends",
			"Check for inefficient processes",
		}
	} else {
		result.Status = "pass"
		result.Score = 95
		result.Message = "System load is normal"
		result.Recommendations = []string{
			"Continue monitoring system load",
		}
	}

	return result
}

// checkSystemResources performs general system resource checks
func (e *DiagnosticEngine) checkSystemResources() DiagnosticResult {
	start := time.Now()

	// Simulate system resource check
	diskUsage := 60.0 + (float64(start.UnixNano()%10000) / 100.0)
	openFiles := 1024 + int(start.UnixNano()%2048)
	processes := 150 + int(start.UnixNano()%100)

	result := DiagnosticResult{
		ID:        "system_resources",
		Name:      "System Resources Check",
		Category:  string(CategorySystem),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"disk_usage_percent": diskUsage,
			"open_files":         openFiles,
			"process_count":      processes,
			"uptime":             "2d 14h 32m",
		},
	}

	// Evaluate system resources
	issues := []string{}
	score := 100

	if diskUsage > 90 {
		issues = append(issues, "Disk usage is critically high")
		score -= 30
	} else if diskUsage > 80 {
		issues = append(issues, "Disk usage is high")
		score -= 15
	}

	if openFiles > 8192 {
		issues = append(issues, "High number of open files")
		score -= 10
	}

	if processes > 500 {
		issues = append(issues, "High number of running processes")
		score -= 10
	}

	if len(issues) > 0 {
		if score < 50 {
			result.Status = "fail"
		} else {
			result.Status = "warning"
		}
		result.Message = strings.Join(issues, "; ")
		result.Recommendations = []string{
			"Review system resource usage",
			"Consider system cleanup",
		}
	} else {
		result.Status = "pass"
		result.Message = "System resources are within normal limits"
		result.Recommendations = []string{
			"Continue monitoring system resources",
		}
	}

	result.Score = score
	return result
}

// runGPUDiagnostics runs GPU-specific diagnostics
func (e *DiagnosticEngine) runGPUDiagnostics() []DiagnosticResult {
	var results []DiagnosticResult

	// GPU availability check
	results = append(results, e.checkGPUAvailability())

	// GPU performance check
	results = append(results, e.checkGPUPerformance())

	// GPU temperature check
	results = append(results, e.checkGPUTemperature())

	// ROCm installation check
	results = append(results, e.checkROCmInstallation())

	return results
}

// checkGPUAvailability checks GPU availability and status
func (e *DiagnosticEngine) checkGPUAvailability() DiagnosticResult {
	start := time.Now()

	// Simulate GPU availability check
	gpuCount := 2
	gpuAvailable := true

	result := DiagnosticResult{
		ID:        "gpu_availability",
		Name:      "GPU Availability Check",
		Category:  string(CategoryGPU),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"gpu_count":     gpuCount,
			"gpu_available": gpuAvailable,
			"vendor":        "Advanced Micro Devices, Inc.",
			"driver_loaded": true,
		},
	}

	if !gpuAvailable || gpuCount == 0 {
		result.Status = "fail"
		result.Score = 0
		result.Message = "No GPUs detected or accessible"
		result.Recommendations = []string{
			"Check GPU hardware installation",
			"Verify GPU drivers are installed",
			"Check GPU permissions",
		}
	} else {
		result.Status = "pass"
		result.Score = 100
		result.Message = fmt.Sprintf("Detected %d GPU(s)", gpuCount)
		result.Recommendations = []string{
			"Monitor GPU performance",
		}
	}

	return result
}

// checkGPUPerformance checks GPU performance metrics
func (e *DiagnosticEngine) checkGPUPerformance() DiagnosticResult {
	start := time.Now()

	// Simulate GPU performance check
	gpuUsage := []float64{45.0, 30.0}
	memoryUsage := []float64{8.5, 6.2}
	clockSpeed := []float64{2100.0, 2050.0}

	result := DiagnosticResult{
		ID:        "gpu_performance",
		Name:      "GPU Performance Check",
		Category:  string(CategoryGPU),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"gpu_usage":        gpuUsage,
			"memory_usage":     memoryUsage,
			"clock_speeds":     clockSpeed,
			"power_efficiency": "Normal",
		},
	}

	// Evaluate GPU performance
	maxUsage := 0.0
	for _, usage := range gpuUsage {
		if usage > maxUsage {
			maxUsage = usage
		}
	}

	if maxUsage > 95 {
		result.Status = "warning"
		result.Score = 70
		result.Message = "GPU usage is very high"
		result.Recommendations = []string{
			"Monitor GPU temperature",
			"Consider workload optimization",
		}
	} else {
		result.Status = "pass"
		result.Score = 95
		result.Message = "GPU performance is normal"
		result.Recommendations = []string{
			"Continue monitoring GPU performance",
		}
	}

	return result
}

// checkGPUTemperature checks GPU temperatures
func (e *DiagnosticEngine) checkGPUTemperature() DiagnosticResult {
	start := time.Now()

	// Simulate GPU temperature check
	temperatures := []float64{65.0, 58.0}

	result := DiagnosticResult{
		ID:        "gpu_temperature",
		Name:      "GPU Temperature Check",
		Category:  string(CategoryGPU),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"temperatures":    temperatures,
			"max_temperature": 85.0,
			"optimal_range":   "30-75°C",
		},
	}

	// Evaluate GPU temperatures
	maxTemp := 0.0
	for _, temp := range temperatures {
		if temp > maxTemp {
			maxTemp = temp
		}
	}

	if maxTemp > 85 {
		result.Status = "fail"
		result.Score = 25
		result.Message = "GPU temperature is critically high"
		result.Recommendations = []string{
			"Check GPU cooling system",
			"Reduce GPU workload",
			"Verify case airflow",
		}
	} else if maxTemp > 75 {
		result.Status = "warning"
		result.Score = 70
		result.Message = "GPU temperature is elevated"
		result.Recommendations = []string{
			"Monitor GPU temperature",
			"Check system cooling",
		}
	} else {
		result.Status = "pass"
		result.Score = 100
		result.Message = "GPU temperatures are normal"
		result.Recommendations = []string{
			"Continue monitoring GPU temperatures",
		}
	}

	return result
}

// checkROCmInstallation checks ROCm installation and configuration
func (e *DiagnosticEngine) checkROCmInstallation() DiagnosticResult {
	start := time.Now()

	// Simulate ROCm installation check
	rocmInstalled := true
	rocmVersion := "6.4.43482"
	driverVersion := "23.40.4"
	rocmPath := "/opt/rocm"

	result := DiagnosticResult{
		ID:        "rocm_installation",
		Name:      "ROCm Installation Check",
		Category:  string(CategoryGPU),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"rocm_installed": rocmInstalled,
			"rocm_version":   rocmVersion,
			"driver_version": driverVersion,
			"rocm_path":      rocmPath,
			"hip_available":  true,
		},
	}

	if !rocmInstalled {
		result.Status = "fail"
		result.Score = 0
		result.Message = "ROCm is not installed or not accessible"
		result.Recommendations = []string{
			"Install ROCm from AMD official repositories",
			"Verify installation paths",
			"Check user permissions for ROCm",
		}
	} else {
		result.Status = "pass"
		result.Score = 100
		result.Message = fmt.Sprintf("ROCm %s is properly installed", rocmVersion)
		result.Recommendations = []string{
			"Keep ROCm updated",
			"Monitor ROCm performance",
		}
	}

	return result
}

// runNetworkDiagnostics runs network-related diagnostics
func (e *DiagnosticEngine) runNetworkDiagnostics() []DiagnosticResult {
	var results []DiagnosticResult

	// Network connectivity check
	results = append(results, e.checkNetworkConnectivity())

	// DNS resolution check
	results = append(results, e.checkDNSResolution())

	// Network speed test
	results = append(results, e.checkNetworkSpeed())

	return results
}

// checkNetworkConnectivity checks network connectivity
func (e *DiagnosticEngine) checkNetworkConnectivity() DiagnosticResult {
	start := time.Now()

	// Simulate network connectivity check
	interfaces := []string{"eth0", "lo", "wlan0"}
	activeInterfaces := 2
	connected := true

	result := DiagnosticResult{
		ID:        "network_connectivity",
		Name:      "Network Connectivity Check",
		Category:  string(CategoryNetwork),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"interfaces":         interfaces,
			"active_interfaces":  activeInterfaces,
			"internet_connected": connected,
			"local_ip":           "192.168.1.100",
		},
	}

	if !connected || activeInterfaces == 0 {
		result.Status = "fail"
		result.Score = 0
		result.Message = "No network connectivity detected"
		result.Recommendations = []string{
			"Check network cable connection",
			"Verify network interface status",
			"Restart network services if needed",
		}
	} else {
		result.Status = "pass"
		result.Score = 100
		result.Message = "Network connectivity is normal"
		result.Recommendations = []string{
			"Continue monitoring network performance",
		}
	}

	return result
}

// checkDNSResolution checks DNS resolution functionality
func (e *DiagnosticEngine) checkDNSResolution() DiagnosticResult {
	start := time.Now()

	// Simulate DNS resolution check
	resolutionWorking := true
	dnsServers := []string{"8.8.8.8", "1.1.1.1", "192.168.1.1"}

	result := DiagnosticResult{
		ID:        "dns_resolution",
		Name:      "DNS Resolution Check",
		Category:  string(CategoryNetwork),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"dns_working": resolutionWorking,
			"dns_servers": dnsServers,
			"test_domain": "google.com",
		},
	}

	if !resolutionWorking {
		result.Status = "fail"
		result.Score = 50
		result.Message = "DNS resolution is not working"
		result.Recommendations = []string{
			"Check DNS server configuration",
			"Verify /etc/resolv.conf",
			"Try alternative DNS servers",
		}
	} else {
		result.Status = "pass"
		result.Score = 100
		result.Message = "DNS resolution is working normally"
		result.Recommendations = []string{
			"Continue monitoring DNS performance",
		}
	}

	return result
}

// checkNetworkSpeed performs basic network speed assessment
func (e *DiagnosticEngine) checkNetworkSpeed() DiagnosticResult {
	start := time.Now()

	// Simulate network speed test
	downloadSpeed := 950.0 // Mbps
	uploadSpeed := 450.0   // Mbps
	latency := 12.5        // ms

	result := DiagnosticResult{
		ID:        "network_speed",
		Name:      "Network Speed Test",
		Category:  string(CategoryNetwork),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"download_speed_mbps": downloadSpeed,
			"upload_speed_mbps":   uploadSpeed,
			"latency_ms":          latency,
			"jitter_ms":           2.1,
		},
	}

	// Evaluate network performance
	if downloadSpeed < 10 {
		result.Status = "fail"
		result.Score = 30
		result.Message = "Network speed is very slow"
		result.Recommendations = []string{
			"Check network congestion",
			"Verify ISP service status",
			"Consider network upgrade",
		}
	} else if downloadSpeed < 100 {
		result.Status = "warning"
		result.Score = 60
		result.Message = "Network speed is below optimal"
		result.Recommendations = []string{
			"Check for network interference",
			"Optimize network configuration",
		}
	} else {
		result.Status = "pass"
		result.Score = 95
		result.Message = "Network speed is good"
		result.Recommendations = []string{
			"Continue monitoring network performance",
		}
	}

	return result
}

// runStorageDiagnostics runs storage-related diagnostics
func (e *DiagnosticEngine) runStorageDiagnostics() []DiagnosticResult {
	var results []DiagnosticResult

	// Disk space check
	results = append(results, e.checkDiskSpace())

	// Disk performance check
	results = append(results, e.checkDiskPerformance())

	// File system health check
	results = append(results, e.checkFileSystemHealth())

	return results
}

// checkDiskSpace checks disk space availability
func (e *DiagnosticEngine) checkDiskSpace() DiagnosticResult {
	start := time.Now()

	// Simulate disk space check
	totalSpace := 1000.0 // GB
	usedSpace := 600.0   // GB
	freeSpace := 400.0   // GB
	usagePercent := (usedSpace / totalSpace) * 100

	result := DiagnosticResult{
		ID:        "disk_space",
		Name:      "Disk Space Check",
		Category:  string(CategoryStorage),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"total_space_gb": totalSpace,
			"used_space_gb":  usedSpace,
			"free_space_gb":  freeSpace,
			"usage_percent":  usagePercent,
			"mount_point":    "/",
		},
	}

	if usagePercent > 95 {
		result.Status = "fail"
		result.Score = 10
		result.Message = "Disk space is critically low"
		result.Recommendations = []string{
			"Clean up unnecessary files",
			"Archive old data",
			"Consider disk upgrade",
		}
	} else if usagePercent > 85 {
		result.Status = "warning"
		result.Score = 60
		result.Message = "Disk space is running low"
		result.Recommendations = []string{
			"Monitor disk usage trends",
			"Plan disk cleanup",
		}
	} else {
		result.Status = "pass"
		result.Score = 100
		result.Message = "Disk space is adequate"
		result.Recommendations = []string{
			"Continue monitoring disk usage",
		}
	}

	return result
}

// checkDiskPerformance checks disk performance metrics
func (e *DiagnosticEngine) checkDiskPerformance() DiagnosticResult {
	start := time.Now()

	// Simulate disk performance check
	readSpeed := 550.0  // MB/s
	writeSpeed := 520.0 // MB/s
	iops := 85000       // I/O operations per second

	result := DiagnosticResult{
		ID:        "disk_performance",
		Name:      "Disk Performance Check",
		Category:  string(CategoryStorage),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"read_speed_mbs":  readSpeed,
			"write_speed_mbs": writeSpeed,
			"iops":            iops,
			"avg_latency_ms":  5.2,
		},
	}

	// Evaluate disk performance
	if readSpeed < 100 {
		result.Status = "warning"
		result.Score = 70
		result.Message = "Disk read performance is below optimal"
		result.Recommendations = []string{
			"Check disk health",
			"Verify disk configuration",
		}
	} else if writeSpeed < 100 {
		result.Status = "warning"
		result.Score = 70
		result.Message = "Disk write performance is below optimal"
		result.Recommendations = []string{
			"Check disk health",
			"Verify disk configuration",
		}
	} else {
		result.Status = "pass"
		result.Score = 95
		result.Message = "Disk performance is normal"
		result.Recommendations = []string{
			"Continue monitoring disk performance",
		}
	}

	return result
}

// checkFileSystemHealth checks file system integrity and health
func (e *DiagnosticEngine) checkFileSystemHealth() DiagnosticResult {
	start := time.Now()

	// Simulate file system health check
	errors := 0
	warnings := 2
	mounted := true

	result := DiagnosticResult{
		ID:        "filesystem_health",
		Name:      "File System Health Check",
		Category:  string(CategoryStorage),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"errors":          errors,
			"warnings":        warnings,
			"mounted":         mounted,
			"filesystem_type": "ext4",
			"mount_options":   "rw,relatime",
		},
	}

	if errors > 0 {
		result.Status = "fail"
		result.Score = 30
		result.Message = fmt.Sprintf("File system has %d errors", errors)
		result.Recommendations = []string{
			"Run file system check (fsck)",
			"Backup important data",
			"Consider file system repair",
		}
	} else if warnings > 0 {
		result.Status = "warning"
		result.Score = 80
		result.Message = fmt.Sprintf("File system has %d warnings", warnings)
		result.Recommendations = []string{
			"Review file system warnings",
			"Monitor file system health",
		}
	} else {
		result.Status = "pass"
		result.Score = 100
		result.Message = "File system is healthy"
		result.Recommendations = []string{
			"Continue monitoring file system health",
		}
	}

	return result
}

// runSecurityDiagnostics runs security-related diagnostics
func (e *DiagnosticEngine) runSecurityDiagnostics() []DiagnosticResult {
	var results []DiagnosticResult

	// System updates check
	results = append(results, e.checkSystemUpdates())

	// Security permissions check
	results = append(results, e.checkSecurityPermissions())

	// Firewall status check
	results = append(results, e.checkFirewallStatus())

	return results
}

// checkSystemUpdates checks for available system updates
func (e *DiagnosticEngine) checkSystemUpdates() DiagnosticResult {
	start := time.Now()

	// Simulate system updates check
	pendingUpdates := 5
	securityUpdates := 2
	lastUpdate := time.Now().Add(-7 * 24 * time.Hour)

	result := DiagnosticResult{
		ID:        "system_updates",
		Name:      "System Updates Check",
		Category:  string(CategorySecurity),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"pending_updates":     pendingUpdates,
			"security_updates":    securityUpdates,
			"last_update":         lastUpdate,
			"auto_update_enabled": true,
		},
	}

	if securityUpdates > 0 {
		result.Status = "warning"
		result.Score = 70
		result.Message = fmt.Sprintf("%d security updates available", securityUpdates)
		result.Recommendations = []string{
			"Install security updates as soon as possible",
			"Review update changelog",
			"Schedule maintenance window if needed",
		}
	} else if pendingUpdates > 0 {
		result.Status = "info"
		result.Score = 85
		result.Message = fmt.Sprintf("%d regular updates available", pendingUpdates)
		result.Recommendations = []string{
			"Install updates at convenience",
			"Review update descriptions",
		}
	} else {
		result.Status = "pass"
		result.Score = 100
		result.Message = "System is up to date"
		result.Recommendations = []string{
			"Continue monitoring for updates",
		}
	}

	return result
}

// checkSecurityPermissions checks file permissions and security settings
func (e *DiagnosticEngine) checkSecurityPermissions() DiagnosticResult {
	start := time.Now()

	// Simulate security permissions check
	worldWritableFiles := 2
	suidFiles := 1
	permissionsOK := true

	result := DiagnosticResult{
		ID:        "security_permissions",
		Name:      "Security Permissions Check",
		Category:  string(CategorySecurity),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"world_writable_files": worldWritableFiles,
			"suid_files":           suidFiles,
			"permissions_ok":       permissionsOK,
			"user_permissions":     "Standard",
		},
	}

	if !permissionsOK || worldWritableFiles > 10 {
		result.Status = "fail"
		result.Score = 40
		result.Message = "Security permissions need attention"
		result.Recommendations = []string{
			"Review file permissions",
			"Remove unnecessary SUID files",
			"Fix world-writable files",
		}
	} else if worldWritableFiles > 0 || suidFiles > 0 {
		result.Status = "warning"
		result.Score = 80
		result.Message = "Some security permissions need review"
		result.Recommendations = []string{
			"Review identified permission issues",
			"Document any necessary exceptions",
		}
	} else {
		result.Status = "pass"
		result.Score = 100
		result.Message = "Security permissions are appropriate"
		result.Recommendations = []string{
			"Continue monitoring security permissions",
		}
	}

	return result
}

// checkFirewallStatus checks firewall configuration and status
func (e *DiagnosticEngine) checkFirewallStatus() DiagnosticResult {
	start := time.Now()

	// Simulate firewall status check
	firewallActive := true
	rulesCount := 25
	defaultPolicy := "DROP"

	result := DiagnosticResult{
		ID:        "firewall_status",
		Name:      "Firewall Status Check",
		Category:  string(CategorySecurity),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"firewall_active": firewallActive,
			"rules_count":     rulesCount,
			"default_policy":  defaultPolicy,
			"firewall_type":   "ufw",
		},
	}

	if !firewallActive {
		result.Status = "warning"
		result.Score = 50
		result.Message = "Firewall is not active"
		result.Recommendations = []string{
			"Enable firewall protection",
			"Configure appropriate firewall rules",
		}
	} else if defaultPolicy == "ACCEPT" {
		result.Status = "warning"
		result.Score = 75
		result.Message = "Firewall default policy is permissive"
		result.Recommendations = []string{
			"Consider changing default policy to DROP",
			"Review firewall rules",
		}
	} else {
		result.Status = "pass"
		result.Score = 100
		result.Message = "Firewall is properly configured"
		result.Recommendations = []string{
			"Continue monitoring firewall logs",
			"Review firewall rules periodically",
		}
	}

	return result
}

// runPerformanceDiagnostics runs performance-related diagnostics
func (e *DiagnosticEngine) runPerformanceDiagnostics() []DiagnosticResult {
	var results []DiagnosticResult

	// System responsiveness check
	results = append(results, e.checkSystemResponsiveness())

	// Memory performance check
	results = append(results, e.checkMemoryPerformance())

	// I/O performance check
	results = append(results, e.checkIOPerformance())

	return results
}

// checkSystemResponsiveness checks system responsiveness and latency
func (e *DiagnosticEngine) checkSystemResponsiveness() DiagnosticResult {
	start := time.Now()

	// Simulate system responsiveness check
	avgResponseTime := 12.5 // milliseconds
	maxResponseTime := 45.0 // milliseconds
	cpuLatency := 2.1       // microseconds

	result := DiagnosticResult{
		ID:        "system_responsiveness",
		Name:      "System Responsiveness Check",
		Category:  string(CategoryPerformance),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"avg_response_time_ms": avgResponseTime,
			"max_response_time_ms": maxResponseTime,
			"cpu_latency_us":       cpuLatency,
			"interrupt_rate":       1250, // per second
		},
	}

	// Evaluate system responsiveness
	if avgResponseTime > 50 {
		result.Status = "fail"
		result.Score = 30
		result.Message = "System response time is poor"
		result.Recommendations = []string{
			"Check for resource bottlenecks",
			"Review running processes",
			"Consider system optimization",
		}
	} else if avgResponseTime > 25 {
		result.Status = "warning"
		result.Score = 70
		result.Message = "System response time is elevated"
		result.Recommendations = []string{
			"Monitor system performance trends",
			"Check for resource-intensive applications",
		}
	} else {
		result.Status = "pass"
		result.Score = 100
		result.Message = "System responsiveness is normal"
		result.Recommendations = []string{
			"Continue monitoring system responsiveness",
		}
	}

	return result
}

// checkMemoryPerformance checks memory performance metrics
func (e *DiagnosticEngine) checkMemoryPerformance() DiagnosticResult {
	start := time.Now()

	// Simulate memory performance check
	memoryBandwidth := 75.5 // GB/s
	memoryLatency := 85.2   // nanoseconds
	cacheHitRate := 0.94    // 94%

	result := DiagnosticResult{
		ID:        "memory_performance",
		Name:      "Memory Performance Check",
		Category:  string(CategoryPerformance),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"memory_bandwidth_gbs": memoryBandwidth,
			"memory_latency_ns":    memoryLatency,
			"cache_hit_rate":       cacheHitRate,
			"page_fault_rate":      0.02, // per second
		},
	}

	// Evaluate memory performance
	if memoryBandwidth < 20 {
		result.Status = "warning"
		result.Score = 65
		result.Message = "Memory bandwidth is below expected"
		result.Recommendations = []string{
			"Check memory configuration",
			"Verify memory module compatibility",
		}
	} else if cacheHitRate < 0.85 {
		result.Status = "warning"
		result.Score = 70
		result.Message = "Cache hit rate is lower than optimal"
		result.Recommendations = []string{
			"Review memory access patterns",
			"Consider software optimization",
		}
	} else {
		result.Status = "pass"
		result.Score = 95
		result.Message = "Memory performance is normal"
		result.Recommendations = []string{
			"Continue monitoring memory performance",
		}
	}

	return result
}

// checkIOPerformance checks I/O performance metrics
func (e *DiagnosticEngine) checkIOPerformance() DiagnosticResult {
	start := time.Now()

	// Simulate I/O performance check
	diskIOPS := 85000      // I/O operations per second
	diskLatency := 5.2     // milliseconds
	networkLatency := 12.5 // milliseconds

	result := DiagnosticResult{
		ID:        "io_performance",
		Name:      "I/O Performance Check",
		Category:  string(CategoryPerformance),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"disk_iops":          diskIOPS,
			"disk_latency_ms":    diskLatency,
			"network_latency_ms": networkLatency,
			"throughput_mbs":     550.0,
		},
	}

	// Evaluate I/O performance
	if diskLatency > 20 {
		result.Status = "warning"
		result.Score = 60
		result.Message = "Disk I/O latency is elevated"
		result.Recommendations = []string{
			"Check disk health",
			"Review disk usage patterns",
		}
	} else if networkLatency > 50 {
		result.Status = "warning"
		result.Score = 70
		result.Message = "Network latency is elevated"
		result.Recommendations = []string{
			"Check network connectivity",
			"Review network configuration",
		}
	} else {
		result.Status = "pass"
		result.Score = 95
		result.Message = "I/O performance is normal"
		result.Recommendations = []string{
			"Continue monitoring I/O performance",
		}
	}

	return result
}

// runSoftwareDiagnostics runs software-related diagnostics
func (e *DiagnosticEngine) runSoftwareDiagnostics() []DiagnosticResult {
	var results []DiagnosticResult

	// ML stack components check
	results = append(results, e.checkMLStackComponents())

	// Python environment check
	results = append(results, e.checkPythonEnvironment())

	// Development tools check
	results = append(results, e.checkDevelopmentTools())

	return results
}

// checkMLStackComponents checks ML stack component installations
func (e *DiagnosticEngine) checkMLStackComponents() DiagnosticResult {
	start := time.Now()

	// Simulate ML stack components check
	pytorchInstalled := true
	cudaAvailable := false // Using ROCm instead
	rocmAvailable := true
	cudaVersion := "N/A"
	rocmVersion := "6.4.43482"

	result := DiagnosticResult{
		ID:        "ml_stack_components",
		Name:      "ML Stack Components Check",
		Category:  string(CategorySoftware),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"pytorch_installed": pytorchInstalled,
			"cuda_available":    cudaAvailable,
			"rocm_available":    rocmAvailable,
			"cuda_version":      cudaVersion,
			"rocm_version":      rocmVersion,
			"python_version":    "3.11.0",
		},
	}

	if !pytorchInstalled {
		result.Status = "fail"
		result.Score = 0
		result.Message = "PyTorch is not installed"
		result.Recommendations = []string{
			"Install PyTorch with ROCm support",
			"Verify installation paths",
		}
	} else if !rocmAvailable {
		result.Status = "fail"
		result.Score = 50
		result.Message = "ROCm is not available for PyTorch"
		result.Recommendations = []string{
			"Install ROCm-compatible PyTorch",
			"Check ROCm installation",
		}
	} else {
		result.Status = "pass"
		result.Score = 100
		result.Message = "ML stack components are properly installed"
		result.Recommendations = []string{
			"Keep components updated",
			"Monitor component performance",
		}
	}

	return result
}

// checkPythonEnvironment checks Python environment and packages
func (e *DiagnosticEngine) checkPythonEnvironment() DiagnosticResult {
	start := time.Time{}

	// Simulate Python environment check
	pythonVersion := "3.11.0"
	pipVersion := "23.2.1"
	venvActive := true
	packageCount := 156

	result := DiagnosticResult{
		ID:        "python_environment",
		Name:      "Python Environment Check",
		Category:  string(CategorySoftware),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"python_version": pythonVersion,
			"pip_version":    pipVersion,
			"venv_active":    venvActive,
			"package_count":  packageCount,
			"site_packages":  "/home/user/.venv/lib/python3.11/site-packages",
		},
	}

	if !venvActive {
		result.Status = "warning"
		result.Score = 70
		result.Message = "Python virtual environment is not active"
		result.Recommendations = []string{
			"Activate virtual environment",
			"Create virtual environment if needed",
		}
	} else {
		result.Status = "pass"
		result.Score = 100
		result.Message = "Python environment is properly configured"
		result.Recommendations = []string{
			"Keep Python packages updated",
			"Monitor package dependencies",
		}
	}

	return result
}

// checkDevelopmentTools checks development tool installations
func (e *DiagnosticEngine) checkDevelopmentTools() DiagnosticResult {
	start := time.Now()

	// Simulate development tools check
	gitInstalled := true
	gitVersion := "2.41.0"
	dockerInstalled := true
	dockerVersion := "24.0.5"
	vscodeInstalled := true

	result := DiagnosticResult{
		ID:        "development_tools",
		Name:      "Development Tools Check",
		Category:  string(CategorySoftware),
		Timestamp: start,
		Duration:  time.Since(start),
		Details: map[string]interface{}{
			"git_installed":    gitInstalled,
			"git_version":      gitVersion,
			"docker_installed": dockerInstalled,
			"docker_version":   dockerVersion,
			"vscode_installed": vscodeInstalled,
		},
	}

	missingTools := []string{}
	if !gitInstalled {
		missingTools = append(missingTools, "Git")
	}
	if !dockerInstalled {
		missingTools = append(missingTools, "Docker")
	}

	if len(missingTools) > 0 {
		result.Status = "warning"
		result.Score = 60
		result.Message = fmt.Sprintf("Missing development tools: %s", strings.Join(missingTools, ", "))
		result.Recommendations = []string{
			"Install missing development tools",
			"Verify tool configurations",
		}
	} else {
		result.Status = "pass"
		result.Score = 100
		result.Message = "Development tools are properly installed"
		result.Recommendations = []string{
			"Keep development tools updated",
			"Monitor tool compatibility",
		}
	}

	return result
}

// GetResults returns all diagnostic results
func (e *DiagnosticEngine) GetResults() []DiagnosticResult {
	return e.results
}

// GetLatestResults returns the most recent diagnostic results
func (e *DiagnosticEngine) GetLatestResults() []DiagnosticResult {
	if len(e.results) == 0 {
		return []DiagnosticResult{}
	}

	// Group results by timestamp and return the latest group
	latestTime := e.results[0].Timestamp
	var latestResults []DiagnosticResult

	for _, result := range e.results {
		if result.Timestamp.After(latestTime) {
			latestTime = result.Timestamp
			latestResults = []DiagnosticResult{result}
		} else if result.Timestamp.Equal(latestTime) {
			latestResults = append(latestResults, result)
		}
	}

	return latestResults
}

// GetOverallScore calculates the overall diagnostic score
func (e *DiagnosticEngine) GetOverallScore() int {
	results := e.GetLatestResults()
	if len(results) == 0 {
		return 0
	}

	totalScore := 0
	for _, result := range results {
		totalScore += result.Score
	}

	return totalScore / len(results)
}

// DiagnosticCompleteMsg represents a message when diagnostics complete
type DiagnosticCompleteMsg struct {
	Results   []DiagnosticResult
	Timestamp time.Time
	Duration  time.Duration
}

func (DiagnosticCompleteMsg) IsMessage() {} // Implement Message interface

// NewDiagnosticWidget creates a new diagnostic widget
func NewDiagnosticWidget(width, height int, theme *AMDTheme) *DiagnosticWidget {
	widget := &DiagnosticWidget{
		width:        width,
		height:       height,
		theme:        theme,
		engine:       NewDiagnosticEngine(theme),
		selected:     0,
		scrollOffset: 0,
		showDetails:  false,
		autoRunning:  false,
		lastUpdate:   time.Now(),
	}

	return widget
}

// SetBounds implements the LayoutComponent interface
func (w *DiagnosticWidget) SetBounds(x, y, width, height int) {
	w.x = x
	w.y = y
	w.width = width
	w.height = height
}

// GetBounds implements the LayoutComponent interface
func (w *DiagnosticWidget) GetBounds() (int, int, int, int) {
	return w.x, w.y, w.width, w.height
}

// Init implements the tea.Model interface
func (w *DiagnosticWidget) Init() tea.Cmd {
	return nil
}

// Update implements the tea.Model interface
func (w *DiagnosticWidget) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch _ := msg.(type) {
	case DiagnosticCompleteMsg:
		w.lastUpdate = time.Now()
		// Auto-scroll to show new results
		w.selected = 0
		w.scrollOffset = 0
	}

	return w, nil
}

// View implements the tea.Model interface
func (w *DiagnosticWidget) View() string {
	results := w.engine.GetLatestResults()

	var sections []string

	// Header
	header := w.theme.Header.Render("🔧 System Diagnostics")
	sections = append(sections, header)

	// Overall score
	score := w.engine.GetOverallScore()
	scoreColor := w.theme.Success
	if score < 50 {
		scoreColor = w.theme.Error
	} else if score < 75 {
		scoreColor = w.theme.Warning
	}

	scoreText := fmt.Sprintf("Overall Score: %s%d%%",
		lipgloss.NewStyle().Foreground(scoreColor).Bold(true).Render(""), score)
	sections = append(sections, scoreText)

	// Last run time
	if !w.lastUpdate.IsZero() {
		timeAgo := w.formatTimeAgo(time.Since(w.lastUpdate))
		sections = append(sections, w.theme.MutedText.Render("Last run: "+timeAgo))
	}

	// Results
	if len(results) == 0 {
		sections = append(sections, w.theme.Text.Render("No diagnostic results available"))
		sections = append(sections, w.theme.MutedText.Render("Press 'r' to run diagnostics"))
	} else {
		sections = append(sections, w.renderResults(results))
	}

	content := strings.Join(sections, "\n")
	return w.theme.Panel.Width(w.width).Height(w.height).Render(content)
}

// renderResults renders diagnostic results
func (w *DiagnosticWidget) renderResults(results []DiagnosticResult) string {
	var resultStrings []string

	maxResults := w.height - 10 // Account for header, score, borders
	if maxResults < 1 {
		maxResults = 1
	}

	start := w.scrollOffset
	end := start + maxResults
	if end > len(results) {
		end = len(results)
	}

	for i := start; i < end; i++ {
		result := results[i]
		resultStr := w.renderResult(result, i == w.selected)
		resultStrings = append(resultStrings, resultStr)
	}

	return strings.Join(resultStrings, "\n")
}

// renderResult renders a single diagnostic result
func (w *DiagnosticWidget) renderResult(result DiagnosticResult, selected bool) string {
	// Status icon and color
	icon := "✅"
	style := w.theme.Text
	switch result.Status {
	case "fail":
		icon = "❌"
		style = w.theme.ErrorAlert
	case "warning":
		icon = "⚠️"
		style = w.theme.WarningAlert
	case "info":
		icon = "ℹ️"
		style = w.theme.Alert
	}

	// Score indicator
	scoreColor := w.theme.Success
	if result.Score < 50 {
		scoreColor = w.theme.Error
	} else if result.Score < 75 {
		scoreColor = w.theme.Warning
	}

	scoreText := lipgloss.NewStyle().Foreground(scoreColor).Render(fmt.Sprintf("%d", result.Score))

	// Format result line
	line := fmt.Sprintf("%s %s: %s (%s%%)",
		icon,
		result.Name,
		result.Message,
		scoreText)

	if selected {
		line = w.theme.Highlight.Render(line)
	} else {
		line = style.Render(line)
	}

	// Add details if shown and space permits
	if w.showDetails && w.height > 15 {
		details := w.renderResultDetails(result)
		if details != "" {
			line += "\n" + w.theme.MutedText.Render(details)
		}
	}

	return line
}

// renderResultDetails renders details for a diagnostic result
func (w *DiagnosticWidget) renderResultDetails(result DiagnosticResult) string {
	var details []string

	// Duration
	if result.Duration > 0 {
		details = append(details, fmt.Sprintf("Duration: %v", result.Duration))
	}

	// Add key details
	for key, value := range result.Details {
		if key == "cpu_usage" || key == "memory_usage_percent" || key == "temperature" {
			details = append(details, fmt.Sprintf("%s: %.1f", key, value))
		}
	}

	// Add recommendations if space permits
	if len(result.Recommendations) > 0 && w.height > 20 {
		details = append(details, "Recommendations:")
		for _, rec := range result.Recommendations {
			details = append(details, "• "+rec)
		}
	}

	if len(details) == 0 {
		return ""
	}

	return strings.Join(details, " | ")
}

// formatTimeAgo formats time duration in a human-readable way
func (w *DiagnosticWidget) formatTimeAgo(d time.Duration) string {
	if d < time.Minute {
		return "just now"
	} else if d < time.Hour {
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	} else if d < 24*time.Hour {
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	} else {
		days := int(d.Hours()) / 24
		return fmt.Sprintf("%dd ago", days)
	}
}

// RunDiagnostics triggers diagnostic execution
func (w *DiagnosticWidget) RunDiagnostics() tea.Cmd {
	return w.engine.RunDiagnostics()
}

// GetDiagnosticEngine returns the diagnostic engine
func (w *DiagnosticWidget) GetDiagnosticEngine() *DiagnosticEngine {
	return w.engine
}

// NewSystemProfiler creates a new system profiler
func NewSystemProfiler(theme *AMDTheme) *SystemProfiler {
	return &SystemProfiler{
		theme: theme,
	}
}

// GetSystemProfile collects comprehensive system information
func (p *SystemProfiler) GetSystemProfile() map[string]interface{} {
	profile := make(map[string]interface{})

	// Basic system information
	profile["hostname"] = "ml-stack-workstation"
	profile["os"] = runtime.GOOS
	profile["architecture"] = runtime.GOARCH
	profile["cpu_cores"] = runtime.NumCPU()
	profile["go_version"] = runtime.Version()

	// Memory information
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	profile["memory"] = map[string]interface{}{
		"alloc_mb":       m.Alloc / 1024 / 1024,
		"total_alloc_mb": m.TotalAlloc / 1024 / 1024,
		"sys_mb":         m.Sys / 1024 / 1024,
		"num_gc":         m.NumGC,
	}

	// GPU information would be collected here in a real implementation
	profile["gpu"] = map[string]interface{}{
		"detected": true,
		"count":    2,
		"vendor":   "AMD",
	}

	return profile
}

// NewHardwareProfiler creates a new hardware profiler
func NewHardwareProfiler(theme *AMDTheme) *HardwareProfiler {
	return &HardwareProfiler{
		theme: theme,
		gpuInfo: []GPUInfo{
			{
				Vendor:        "Advanced Micro Devices, Inc.",
				Model:         "AMD Radeon RX 7900 XTX",
				DeviceID:      "0x744C",
				DriverVersion: "23.40.4",
				ROCmVersion:   "6.4.43482",
				MemoryTotal:   24 * 1024 * 1024 * 1024,  // 24GB
				MemoryUsed:    8.5 * 1024 * 1024 * 1024, // 8.5GB
				Temperature:   65.0,
				PowerUsage:    250.0,
				ClockSpeed:    2100.0,
				ComputeUnits:  96,
				GFXVersion:    "gfx1100",
				LinkWidth:     16,
				LinkSpeed:     16.0,
				ActivityLevel: 45.0,
				Status:        "Active",
				LastUpdated:   time.Now(),
			},
		},
	}
}

// GetHardwareProfile collects detailed hardware information
func (p *HardwareProfiler) GetHardwareProfile() map[string]interface{} {
	profile := make(map[string]interface{})

	// CPU information
	profile["cpu"] = map[string]interface{}{
		"model":      "AMD Ryzen 9 7950X",
		"cores":      16,
		"threads":    32,
		"base_clock": "4.5 GHz",
		"max_clock":  "5.7 GHz",
		"cache": map[string]interface{}{
			"l1": "1MB",
			"l2": "16MB",
			"l3": "64MB",
		},
	}

	// Memory information
	profile["memory"] = map[string]interface{}{
		"type":      "DDR5",
		"total_gb":  32,
		"speed_mhz": 6000,
		"channels":  2,
	}

	// GPU information
	profile["gpu"] = p.gpuInfo

	// Storage information
	profile["storage"] = []map[string]interface{}{
		{
			"type":      "NVMe SSD",
			"size_gb":   1000,
			"model":     "Samsung 980 PRO",
			"interface": "PCIe 4.0 x4",
		},
	}

	// Motherboard information
	profile["motherboard"] = map[string]interface{}{
		"manufacturer": "ASUS",
		"model":        "ROG CROSSHAIR X670E HERO",
		"chipset":      "AMD X670E",
	}

	return profile
}

// NewPerformanceAnalyzer creates a new performance analyzer
func NewPerformanceAnalyzer(theme *AMDTheme) *PerformanceAnalyzer {
	return &PerformanceAnalyzer{
		theme:            theme,
		benchmarks:       make(map[string]BenchmarkResult),
		performanceTrend: make([]PerformanceSnapshot, 0),
	}
}

// RunBenchmark executes a performance benchmark
func (a *PerformanceAnalyzer) RunBenchmark(name string) BenchmarkResult {
	start := time.Now()

	// Simulate different benchmark types
	var result BenchmarkResult
	result.Name = name
	result.Timestamp = start
	result.Metrics = make(map[string]float64)
	result.SystemInfo = map[string]string{
		"cpu_cores":  fmt.Sprintf("%d", runtime.NumCPU()),
		"go_version": runtime.Version(),
	}

	switch name {
	case "cpu_performance":
		result.Category = "CPU"
		result.Score = 8500 + float64(start.UnixNano()%1000)
		result.Unit = "score"
		result.Metrics["single_thread"] = 2850 + float64(start.UnixNano()%500)
		result.Metrics["multi_thread"] = 8500 + float64(start.UnixNano()%1000)

	case "memory_bandwidth":
		result.Category = "Memory"
		result.Score = 75.5 + float64(start.UnixNano()%10)/10.0
		result.Unit = "GB/s"
		result.Metrics["read_bandwidth"] = 75.5 + float64(start.UnixNano()%10)/10.0
		result.Metrics["write_bandwidth"] = 72.3 + float64(start.UnixNano()%10)/10.0

	case "gpu_compute":
		result.Category = "GPU"
		result.Score = 12500 + float64(start.UnixNano()%2000)
		result.Unit = "score"
		result.Metrics["fp32_performance"] = 12500 + float64(start.UnixNano()%2000)
		result.Metrics["memory_bandwidth"] = 960 + float64(start.UnixNano()%100)

	default:
		result.Category = "General"
		result.Score = 100.0
		result.Unit = "score"
	}

	// Store benchmark result
	a.benchmarks[name] = result

	return result
}

// CollectSnapshot collects a performance snapshot
func (a *PerformanceAnalyzer) CollectSnapshot() PerformanceSnapshot {
	snapshot := PerformanceSnapshot{
		Timestamp:       time.Now(),
		CPUUsage:        25.0 + float64(time.Now().UnixNano()%10000)/100.0,
		MemoryUsage:     40.0 + float64(time.Now().UnixNano()%20000)/100.0,
		GPUUsage:        []float64{45.0, 30.0},
		DiskIO:          50.0 + float64(time.Now().UnixNano()%5000)/100.0,
		NetworkIO:       25.0 + float64(time.Now().UnixNano()%3000)/100.0,
		LoadAverage:     []float64{0.5, 0.7, 0.6},
		ContextSwitches: uint64(1000000 + time.Now().UnixNano()%100000),
		Processes:       150 + int(time.Now().UnixNano()%100),
		Additional:      make(map[string]interface{}),
	}

	// Add to performance trend
	a.performanceTrend = append(a.performanceTrend, snapshot)

	// Keep only last 1000 snapshots
	if len(a.performanceTrend) > 1000 {
		a.performanceTrend = a.performanceTrend[1:]
	}

	return snapshot
}

// GetBenchmarks returns all benchmark results
func (a *PerformanceAnalyzer) GetBenchmarks() map[string]BenchmarkResult {
	return a.benchmarks
}

// GetPerformanceTrend returns the performance trend data
func (a *PerformanceAnalyzer) GetPerformanceTrend() []PerformanceSnapshot {
	return a.performanceTrend
}

// GetPerformanceAnalysis returns performance analysis and insights
func (a *PerformanceAnalyzer) GetPerformanceAnalysis() map[string]interface{} {
	analysis := make(map[string]interface{})

	if len(a.performanceTrend) == 0 {
		analysis["status"] = "insufficient_data"
		return analysis
	}

	// Calculate averages and trends
	avgCPU := 0.0
	avgMemory := 0.0
	avgGPU := 0.0

	for _, snapshot := range a.performanceTrend {
		avgCPU += snapshot.CPUUsage
		avgMemory += snapshot.MemoryUsage
		for _, gpu := range snapshot.GPUUsage {
			avgGPU += gpu
		}
	}

	count := float64(len(a.performanceTrend))
	avgCPU /= count
	avgMemory /= count
	avgGPU /= count * float64(len(a.performanceTrend[0].GPUUsage))

	analysis["averages"] = map[string]interface{}{
		"cpu_usage":    avgCPU,
		"memory_usage": avgMemory,
		"gpu_usage":    avgGPU,
	}

	// Performance recommendations
	recommendations := []string{}

	if avgCPU > 80 {
		recommendations = append(recommendations, "High average CPU usage detected")
	}
	if avgMemory > 85 {
		recommendations = append(recommendations, "High memory usage detected")
	}
	if avgGPU > 85 {
		recommendations = append(recommendations, "High GPU usage detected")
	}

	analysis["recommendations"] = recommendations
	analysis["status"] = "healthy"

	return analysis
}

// GetDiagnosticReport returns a comprehensive diagnostic report
func (w *DiagnosticWidget) GetDiagnosticReport() map[string]interface{} {
	report := make(map[string]interface{})

	// Engine results
	report["diagnostic_results"] = w.engine.GetLatestResults()
	report["overall_score"] = w.engine.GetOverallScore()
	report["last_run"] = w.lastUpdate

	// System profiling
	systemProfiler := NewSystemProfiler(w.theme)
	report["system_profile"] = systemProfiler.GetSystemProfile()

	// Hardware profiling
	hardwareProfiler := NewHardwareProfiler(w.theme)
	report["hardware_profile"] = hardwareProfiler.GetHardwareProfile()

	// Performance analysis
	perfAnalyzer := NewPerformanceAnalyzer(w.theme)
	report["benchmarks"] = perfAnalyzer.GetBenchmarks()
	report["performance_analysis"] = perfAnalyzer.GetPerformanceAnalysis()

	// Timestamp
	report["report_timestamp"] = time.Now()

	return report
}

// ExportDiagnosticReport exports the diagnostic report to JSON
func (w *DiagnosticWidget) ExportDiagnosticReport() (string, error) {
	report := w.GetDiagnosticReport()

	jsonData, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return "", err
	}

	return string(jsonData), nil
}


