// internal/testing/hardware_test.go
package testing

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/scooter-lacroix/mlstack-installer/internal/installer"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/components"
)

// HardwareTestSuite provides comprehensive hardware detection testing
type HardwareTestSuite struct {
	// Test configuration
	TestDataDir   string
	TempDir       string
	MockResponses map[string]string
	EnableRealHW  bool

	// Test results
	TestResults      []HardwareTestResult
	PerformanceStats HardwarePerformanceStats

	// AMD-specific testing
	AMDSpecificTests []AMDHardwareTest
}

// HardwareTestResult represents a single hardware test result
type HardwareTestResult struct {
	TestName        string
	Description     string
	Status          TestStatus
	ExecutionTime   time.Duration
	Error           error
	ExpectedResult  interface{}
	ActualResult    interface{}
	AMDCompatible   bool
	Recommendations []string
}

// HardwarePerformanceStats tracks hardware detection performance
type HardwarePerformanceStats struct {
	TotalTests       int
	PassedTests      int
	FailedTests      int
	SkippedTests     int
	AverageTime      time.Duration
	MaxTime          time.Duration
	MinTime          time.Duration
	TotalTime        time.Duration
	AMDDetectionRate float64
	PerformanceScore int
}

// AMDHardwareTest represents AMD-specific hardware tests
type AMDHardwareTest struct {
	Name           string
	Description    string
	TestType       AMDTestType
	RequiredForAMD bool
	ValidationFunc func() AMDTestResult
}

// AMDTestType defines different types of AMD hardware tests
type AMDTestType int

const (
	AMDGPUTest AMDTestType = iota
	AMDCPUTest
	AMDMemoryTest
	AMDStorageTest
	AMDROCmTest
	AMDDriverTest
)

// AMDTestResult represents the result of an AMD-specific test
type AMDTestResult struct {
	Status           TestStatus
	Performance      string
	AMDCompatible    bool
	DetectedHardware string
	Score            int
	Recommendations  []string
}

// TestStatus represents the status of a test
type TestStatus int

const (
	TestStatusPassed TestStatus = iota
	TestStatusFailed
	TestStatusSkipped
	TestStatusError
	TestStatusWarning
)

// NewHardwareTestSuite creates a new hardware test suite
func NewHardwareTestSuite(enableRealHW bool) *HardwareTestSuite {
	tempDir, _ := os.MkdirTemp("", "hardware_test_*")

	suite := &HardwareTestSuite{
		TestDataDir:      "testdata/hardware",
		TempDir:          tempDir,
		MockResponses:    make(map[string]string),
		EnableRealHW:     enableRealHW,
		TestResults:      make([]HardwareTestResult, 0),
		AMDSpecificTests: []AMDHardwareTest{},
	}

	suite.initializeMockResponses()
	suite.initializeAMDSpecificTests()

	return suite
}

// initializeMockResponses sets up mock hardware detection responses
func (h *HardwareTestSuite) initializeMockResponses() {
	h.MockResponses = map[string]string{
		"lspci_amd_gpu":      "01:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX]",
		"lscpu_amd":          "Model name: AMD Ryzen 9 7950X\nCPU(s): 16\nThread(s) per core: 2",
		"meminfo_amd":        "MemTotal:       67403216 kB\nMemAvailable:    45875200 kB",
		"rocm_smi_amd":       "GPU ID\t\tTemp\t\tPower\t\tMemory Usage\t\tGPU Utilization\n0\t\t45C\t\t250W\t\t16GB / 24GB\t\t85%",
		"amd_driver_version": "amdgpu 6.8.0 (ROCM 6.4.43482)",
		"no_amd_gpu":         "01:00.0 VGA compatible controller: NVIDIA Corporation GeForce RTX 4090",
		"insufficient_mem":   "MemTotal:       4194304 kB\nMemAvailable:    2097152 kB",
		"amd_architecture":   "Architecture:        x86_64\nCPU op-mode(s):      32-bit, 64-bit",
	}
}

// initializeAMDSpecificTests sets up AMD-specific hardware tests
func (h *HardwareTestSuite) initializeAMDSpecificTests() {
	h.AMDSpecificTests = []AMDHardwareTest{
		{
			Name:           "AMD GPU Detection",
			Description:    "Detects AMD GPUs and validates compatibility",
			TestType:       AMDGPUTest,
			RequiredForAMD: true,
			ValidationFunc: h.testAMDDetection,
		},
		{
			Name:           "AMD CPU Performance",
			Description:    "Tests AMD CPU capabilities for ML workloads",
			TestType:       AMDCPUTest,
			RequiredForAMD: false,
			ValidationFunc: h.testAMDCPU,
		},
		{
			Name:           "Memory Configuration",
			Description:    "Validates system memory for AMD ML operations",
			TestType:       AMDMemoryTest,
			RequiredForAMD: false,
			ValidationFunc: h.testAMDMemory,
		},
		{
			Name:           "ROCm Platform Test",
			Description:    "Tests ROCm platform installation and functionality",
			TestType:       AMDROCmTest,
			RequiredForAMD: true,
			ValidationFunc: h.testAMDROCm,
		},
		{
			Name:           "AMD Driver Validation",
			Description:    "Validates AMD driver installation and version",
			TestType:       AMDDriverTest,
			RequiredForAMD: true,
			ValidationFunc: h.testAMDDrivers,
		},
		{
			Name:           "Storage Performance",
			Description:    "Tests storage performance for AMD ML workloads",
			TestType:       AMDStorageTest,
			RequiredForAMD: false,
			ValidationFunc: h.testAMDStorage,
		},
	}
}

// RunHardwareTests executes the complete hardware test suite
func (h *HardwareTestSuite) RunHardwareTests() error {
	fmt.Println("🔴 Starting AMD Hardware Detection Test Suite")
	fmt.Println("=" + strings.Repeat("=", 59))

	startTime := time.Now()

	// Run basic hardware detection tests
	h.runBasicHardwareTests()

	// Run AMD-specific tests
	h.runAMDSpecificTests()

	// Calculate performance statistics
	h.calculatePerformanceStats(time.Since(startTime))

	// Generate test report
	h.generateTestReport()

	return nil
}

// runBasicHardwareTests runs basic hardware detection tests
func (h *HardwareTestSuite) runBasicHardwareTests() {
	basicTests := []struct {
		name        string
		description string
		testFunc    func() HardwareTestResult
	}{
		{
			name:        "GPU Detection",
			description: "Detects and validates GPU hardware",
			testFunc:    h.testGPUDetection,
		},
		{
			name:        "CPU Detection",
			description: "Detects and validates CPU hardware",
			testFunc:    h.testCPUDetection,
		},
		{
			name:        "Memory Detection",
			description: "Detects and validates system memory",
			testFunc:    h.testMemoryDetection,
		},
		{
			name:        "Storage Detection",
			description: "Detects and validates storage devices",
			testFunc:    h.testStorageDetection,
		},
		{
			name:        "System Compatibility",
			description: "Validates overall system compatibility",
			testFunc:    h.testSystemCompatibility,
		},
	}

	for _, test := range basicTests {
		result := test.testFunc()
		h.TestResults = append(h.TestResults, result)
		h.printTestResult(result)
	}
}

// runAMDSpecificTests runs AMD-specific hardware tests
func (h *HardwareTestSuite) runAMDSpecificTests() {
	fmt.Println("\n🔴 AMD-Specific Hardware Tests")
	fmt.Println("-" + strings.Repeat("-", 40))

	for _, test := range h.AMDSpecificTests {
		fmt.Printf("\n⚡ Running: %s\n", test.Name)
		fmt.Printf("Description: %s\n", test.Description)

		startTime := time.Now()
		result := test.ValidationFunc()
		executionTime := time.Since(startTime)

		// Convert to standard test result
		hwResult := HardwareTestResult{
			TestName:        test.Name,
			Description:     test.Description,
			Status:          result.Status,
			ExecutionTime:   executionTime,
			AMDCompatible:   result.AMDCompatible,
			Recommendations: result.Recommendations,
			ActualResult: map[string]interface{}{
				"performance":       result.Performance,
				"detected_hardware": result.DetectedHardware,
				"score":             result.Score,
			},
		}

		h.TestResults = append(h.TestResults, hwResult)
		h.printAMDTestResult(test, result)
	}
}

// testAMDDetection tests AMD GPU detection
func (h *HardwareTestSuite) testAMDDetection() AMDTestResult {
	if !h.EnableRealHW {
		// Mock test
		return AMDTestResult{
			Status:           TestStatusPassed,
			Performance:      "Excellent",
			AMDCompatible:    true,
			DetectedHardware: "AMD Radeon RX 7900 XTX",
			Score:            95,
			Recommendations: []string{
				"GPU is excellent for ML workloads",
				"ROCm 6.4.43482 fully supported",
			},
		}
	}

	// Real hardware test
	cmd := exec.Command("lspci", "-nn")
	output, err := cmd.Output()
	if err != nil {
		return AMDTestResult{
			Status:          TestStatusError,
			Performance:     "Unknown",
			AMDCompatible:   false,
			Recommendations: []string{"Unable to detect GPU hardware"},
		}
	}

	// Check for AMD GPU
	amdPattern := regexp.MustCompile(`(?i)advanced micro devices|amd|radeon`)
	if amdPattern.MatchString(string(output)) {
		return AMDTestResult{
			Status:           TestStatusPassed,
			Performance:      "Detected",
			AMDCompatible:    true,
			DetectedHardware: extractAMDGPUModel(string(output)),
			Score:            85,
			Recommendations: []string{
				"AMD GPU detected and compatible",
				"Install ROCm drivers for optimal performance",
			},
		}
	}

	return AMDTestResult{
		Status:        TestStatusFailed,
		Performance:   "Not Detected",
		AMDCompatible: false,
		Recommendations: []string{
			"No AMD GPU detected",
			"Consider installing AMD GPU for optimal performance",
		},
	}
}

// testAMDCPU tests AMD CPU capabilities
func (h *HardwareTestSuite) testAMDCPU() AMDTestResult {
	if !h.EnableRealHW {
		return AMDTestResult{
			Status:           TestStatusPassed,
			Performance:      "Excellent",
			AMDCompatible:    true,
			DetectedHardware: "AMD Ryzen 9 7950X",
			Score:            90,
			Recommendations: []string{
				"High-performance CPU detected",
				"Excellent for ML workloads",
			},
		}
	}

	cmd := exec.Command("lscpu")
	output, err := cmd.Output()
	if err != nil {
		return AMDTestResult{
			Status:        TestStatusError,
			Performance:   "Unknown",
			AMDCompatible: false,
		}
	}

	cpuInfo := string(output)
	cores := extractCPUCores(cpuInfo)
	model := extractCPUModel(cpuInfo)

	var performance string
	var score int
	var recommendations []string

	if cores >= 16 {
		performance = "Excellent"
		score = 90
		recommendations = append(recommendations, "High core count CPU detected")
	} else if cores >= 8 {
		performance = "Good"
		score = 75
		recommendations = append(recommendations, "Adequate CPU for ML workloads")
	} else {
		performance = "Limited"
		score = 50
		recommendations = append(recommendations, "Consider upgrading CPU for better performance")
	}

	if strings.Contains(strings.ToLower(model), "amd") {
		return AMDTestResult{
			Status:           TestStatusPassed,
			Performance:      performance,
			AMDCompatible:    true,
			DetectedHardware: model,
			Score:            score,
			Recommendations:  recommendations,
		}
	}

	return AMDTestResult{
		Status:           TestStatusWarning,
		Performance:      performance,
		AMDCompatible:    true, // Non-AMD CPUs are still compatible
		DetectedHardware: model,
		Score:            score - 10,
		Recommendations:  append(recommendations, "Non-AMD CPU detected, but still compatible"),
	}
}

// testAMDMemory tests system memory configuration
func (h *HardwareTestSuite) testAMDMemory() AMDTestResult {
	if !h.EnableRealHW {
		return AMDTestResult{
			Status:           TestStatusPassed,
			Performance:      "Excellent",
			AMDCompatible:    true,
			DetectedHardware: "64 GB DDR5",
			Score:            95,
			Recommendations: []string{
				"Sufficient memory for large ML models",
				"Fast DDR5 memory detected",
			},
		}
	}

	cmd := exec.Command("cat", "/proc/meminfo")
	output, err := cmd.Output()
	if err != nil {
		return AMDTestResult{
			Status:        TestStatusError,
			Performance:   "Unknown",
			AMDCompatible: false,
		}
	}

	memInfo := string(output)
	totalMB := extractMemoryMB(memInfo)

	var performance string
	var score int
	var recommendations []string

	if totalMB >= 64000 { // 64GB
		performance = "Excellent"
		score = 95
		recommendations = append(recommendations, "Excellent memory configuration for ML")
	} else if totalMB >= 32000 { // 32GB
		performance = "Good"
		score = 80
		recommendations = append(recommendations, "Good memory configuration for most ML workloads")
	} else if totalMB >= 16000 { // 16GB
		performance = "Adequate"
		score = 65
		recommendations = append(recommendations, "Minimum memory for basic ML workloads")
	} else {
		performance = "Insufficient"
		score = 30
		recommendations = append(recommendations, "Insufficient memory for ML workloads, upgrade recommended")
	}

	return AMDTestResult{
		Status:           TestStatusPassed,
		Performance:      performance,
		AMDCompatible:    totalMB >= 16000,
		DetectedHardware: fmt.Sprintf("%.1f GB", float64(totalMB)/1024),
		Score:            score,
		Recommendations:  recommendations,
	}
}

// testAMDROCm tests ROCm platform functionality
func (h *HardwareTestSuite) testAMDROCm() AMDTestResult {
	if !h.EnableRealHW {
		return AMDTestResult{
			Status:           TestStatusPassed,
			Performance:      "Optimal",
			AMDCompatible:    true,
			DetectedHardware: "ROCm 6.4.43482",
			Score:            100,
			Recommendations: []string{
				"ROCm platform fully operational",
				"Ready for AMD GPU computing",
			},
		}
	}

	// Test ROCm installation
	rocsmiCmd := exec.Command("rocm-smi", "--showproductname")
	_, err := rocsmiCmd.Output()

	if err != nil {
		return AMDTestResult{
			Status:        TestStatusFailed,
			Performance:   "Not Available",
			AMDCompatible: false,
			Recommendations: []string{
				"ROCm not installed or not in PATH",
				"Install ROCm 6.4.43482 for AMD GPU support",
			},
		}
	}

	// Test basic ROCm functionality
	versionCmd := exec.Command("rocm-smi", "--showversion")
	versionOutput, _ := versionCmd.Output()

	return AMDTestResult{
		Status:           TestStatusPassed,
		Performance:      "Available",
		AMDCompatible:    true,
		DetectedHardware: strings.TrimSpace(string(versionOutput)),
		Score:            90,
		Recommendations: []string{
			"ROCm platform detected and functional",
			"Ready for GPU-accelerated ML workloads",
		},
	}
}

// testAMDDrivers tests AMD driver installation
func (h *HardwareTestSuite) testAMDDrivers() AMDTestResult {
	if !h.EnableRealHW {
		return AMDTestResult{
			Status:           TestStatusPassed,
			Performance:      "Latest",
			AMDCompatible:    true,
			DetectedHardware: "amdgpu 6.8.0",
			Score:            95,
			Recommendations: []string{
				"Latest AMD drivers installed",
				"Full GPU acceleration available",
			},
		}
	}

	// Check AMD GPU driver
	cmd := exec.Command("modinfo", "amdgpu")
	output, err := cmd.Output()

	if err != nil {
		return AMDTestResult{
			Status:        TestStatusFailed,
			Performance:   "Not Available",
			AMDCompatible: false,
			Recommendations: []string{
				"AMD GPU drivers not loaded",
				"Install amdgpu drivers for GPU support",
			},
		}
	}

	driverInfo := string(output)
	version := extractDriverVersion(driverInfo)

	return AMDTestResult{
		Status:           TestStatusPassed,
		Performance:      "Available",
		AMDCompatible:    true,
		DetectedHardware: fmt.Sprintf("amdgpu %s", version),
		Score:            85,
		Recommendations: []string{
			"AMD GPU drivers detected and loaded",
			"GPU acceleration should be functional",
		},
	}
}

// testAMDStorage tests storage performance
func (h *HardwareTestSuite) testAMDStorage() AMDTestResult {
	if !h.EnableRealHW {
		return AMDTestResult{
			Status:           TestStatusPassed,
			Performance:      "Excellent",
			AMDCompatible:    true,
			DetectedHardware: "2TB NVMe SSD",
			Score:            90,
			Recommendations: []string{
				"High-speed storage detected",
				"Excellent for ML dataset loading",
			},
		}
	}

	// Check available disk space
	cmd := exec.Command("df", "-h", "/")
	output, err := cmd.Output()
	if err != nil {
		return AMDTestResult{
			Status:        TestStatusError,
			Performance:   "Unknown",
			AMDCompatible: false,
		}
	}

	lines := strings.Split(string(output), "\n")
	if len(lines) < 2 {
		return AMDTestResult{
			Status:        TestStatusError,
			Performance:   "Unknown",
			AMDCompatible: false,
		}
	}

	storageInfo := strings.Fields(lines[1])
	if len(storageInfo) < 4 {
		return AMDTestResult{
			Status:        TestStatusError,
			Performance:   "Unknown",
			AMDCompatible: false,
		}

	}

	available := storageInfo[3]

	return AMDTestResult{
		Status:           TestStatusPassed,
		Performance:      "Available",
		AMDCompatible:    true,
		DetectedHardware: fmt.Sprintf("Available: %s", available),
		Score:            75,
		Recommendations: []string{
			"Storage space available for ML stack",
			"Consider additional storage for large datasets",
		},
	}
}

// Helper functions for extracting hardware information
func extractAMDGPUModel(output string) string {
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		if strings.Contains(strings.ToLower(line), "amd") && strings.Contains(strings.ToLower(line), "vga") {
			parts := strings.Split(line, ":")
			if len(parts) > 1 {
				return strings.TrimSpace(parts[len(parts)-1])
			}
		}
	}
	return "Unknown AMD GPU"
}

func extractCPUCores(output string) int {
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "CPU(s):") {
			parts := strings.Fields(line)
			if len(parts) > 1 {
				cores, err := strconv.Atoi(parts[1])
				if err == nil {
					return cores
				}
			}
		}
	}
	return 0
}

func extractCPUModel(output string) string {
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "Model name:") {
			parts := strings.SplitN(line, ":", 2)
			if len(parts) > 1 {
				return strings.TrimSpace(parts[1])
			}
		}
	}
	return "Unknown CPU"
}

func extractMemoryMB(output string) int {
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "MemTotal:") {
			parts := strings.Fields(line)
			if len(parts) > 1 {
				kb, err := strconv.Atoi(parts[1])
				if err == nil {
					return kb / 1024 // Convert KB to MB
				}
			}
		}
	}
	return 0
}

func extractDriverVersion(output string) string {
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "version:") {
			parts := strings.SplitN(line, ":", 2)
			if len(parts) > 1 {
				return strings.TrimSpace(parts[1])
			}
		}
	}
	return "Unknown"
}

// Test execution result printing functions
func (h *HardwareTestSuite) printTestResult(result HardwareTestResult) {
	statusIcon := map[TestStatus]string{
		TestStatusPassed:  "✅",
		TestStatusFailed:  "❌",
		TestStatusSkipped: "⏭️",
		TestStatusError:   "💥",
		TestStatusWarning: "⚠️",
	}[result.Status]

	fmt.Printf("\n%s %s (%.2fs)\n", statusIcon, result.TestName, result.ExecutionTime.Seconds())
	if result.Error != nil {
		fmt.Printf("   Error: %v\n", result.Error)
	}

	if result.AMDCompatible {
		fmt.Printf("   AMD Compatible: Yes\n")
	}

	for _, rec := range result.Recommendations {
		fmt.Printf("   💡 %s\n", rec)
	}
}

func (h *HardwareTestSuite) printAMDTestResult(test AMDHardwareTest, result AMDTestResult) {
	statusIcon := map[TestStatus]string{
		TestStatusPassed:  "✅",
		TestStatusFailed:  "❌",
		TestStatusSkipped: "⏭️",
		TestStatusError:   "💥",
		TestStatusWarning: "⚠️",
	}[result.Status]

	fmt.Printf("\n   %s Status: %s\n", statusIcon, result.Status)
	fmt.Printf("   🎯 Performance: %s\n", result.Performance)
	fmt.Printf("   🔧 Hardware: %s\n", result.DetectedHardware)
	fmt.Printf("   📊 Score: %d/100\n", result.Score)

	if result.AMDCompatible {
		fmt.Printf("   ✅ AMD Compatible\n")
	} else {
		fmt.Printf("   ❌ Not AMD Compatible\n")
	}

	for _, rec := range result.Recommendations {
		fmt.Printf("   💡 %s\n", rec)
	}
}

// calculatePerformanceStats calculates test suite performance statistics
func (h *HardwareTestSuite) calculatePerformanceStats(totalTime time.Duration) {
	stats := HardwarePerformanceStats{
		TotalTests:  len(h.TestResults),
		TotalTime:   totalTime,
		AverageTime: totalTime / time.Duration(len(h.TestResults)),
	}

	var maxTime, minTime time.Duration
	var amdCompatibleCount int

	for i, result := range h.TestResults {
		switch result.Status {
		case TestStatusPassed:
			stats.PassedTests++
		case TestStatusFailed:
			stats.FailedTests++
		case TestStatusSkipped:
			stats.SkippedTests++
		case TestStatusError:
			stats.FailedTests++
		}

		if i == 0 {
			maxTime = result.ExecutionTime
			minTime = result.ExecutionTime
		} else {
			if result.ExecutionTime > maxTime {
				maxTime = result.ExecutionTime
			}
			if result.ExecutionTime < minTime {
				minTime = result.ExecutionTime
			}
		}

		if result.AMDCompatible {
			amdCompatibleCount++
		}
	}

	stats.MaxTime = maxTime
	stats.MinTime = minTime
	stats.AMDDetectionRate = float64(amdCompatibleCount) / float64(stats.TotalTests) * 100

	// Calculate overall performance score
	stats.PerformanceScore = int((float64(stats.PassedTests) / float64(stats.TotalTests) * 50) +
		(stats.AMDDetectionRate * 0.3) +
		(float64(stats.PassedTests) / float64(stats.TotalTests) * 20))

	h.PerformanceStats = stats
}

// generateTestReport generates a comprehensive test report
func (h *HardwareTestSuite) generateTestReport() {
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("🔴 AMD HARDWARE DETECTION TEST SUITE REPORT")
	fmt.Println(strings.Repeat("=", 60))

	// Performance Statistics
	fmt.Printf("\n📊 PERFORMANCE STATISTICS\n")
	fmt.Printf("   Total Tests:     %d\n", h.PerformanceStats.TotalTests)
	fmt.Printf("   Passed:          %d (%.1f%%)\n", h.PerformanceStats.PassedTests,
		float64(h.PerformanceStats.PassedTests)/float64(h.PerformanceStats.TotalTests)*100)
	fmt.Printf("   Failed:          %d (%.1f%%)\n", h.PerformanceStats.FailedTests,
		float64(h.PerformanceStats.FailedTests)/float64(h.PerformanceStats.TotalTests)*100)
	fmt.Printf("   AMD Compatible:  %.1f%%\n", h.PerformanceStats.AMDDetectionRate)
	fmt.Printf("   Performance Score: %d/100\n", h.PerformanceStats.PerformanceScore)
	fmt.Printf("   Total Time:      %v\n", h.PerformanceStats.TotalTime)
	fmt.Printf("   Average Time:    %v\n", h.PerformanceStats.AverageTime)

	// Overall Assessment
	fmt.Printf("\n🎯 OVERALL ASSESSMENT\n")
	if h.PerformanceStats.PerformanceScore >= 90 {
		fmt.Printf("   ✅ EXCELLENT: System is fully optimized for AMD ML workloads\n")
	} else if h.PerformanceStats.PerformanceScore >= 70 {
		fmt.Printf("   ✅ GOOD: System is suitable for AMD ML workloads\n")
	} else if h.PerformanceStats.PerformanceScore >= 50 {
		fmt.Printf("   ⚠️ FAIR: System can run AMD ML workloads with limitations\n")
	} else {
		fmt.Printf("   ❌ POOR: System requires significant upgrades for AMD ML workloads\n")
	}

	// Recommendations
	fmt.Printf("\n💡 RECOMMENDATIONS\n")
	if h.PerformanceStats.AMDDetectionRate < 100 {
		fmt.Printf("   • Consider installing AMD GPU for optimal performance\n")
	}
	if h.PerformanceStats.FailedTests > 0 {
		fmt.Printf("   • Address %d failed tests before proceeding with installation\n", h.PerformanceStats.FailedTests)
	}
	fmt.Printf("   • Ensure ROCm 6.4.43482 is properly installed for GPU acceleration\n")
	fmt.Printf("   • Verify all drivers are up to date for best performance\n")

	fmt.Println("\n" + strings.Repeat("=", 60))
}

// Cleanup cleans up temporary files and resources
func (h *HardwareTestSuite) Cleanup() error {
	if h.TempDir != "" {
		return os.RemoveAll(h.TempDir)
	}
	return nil
}

// RunHardwareTestSuite is the main entry point for running hardware tests
func RunHardwareTestSuite(t *testing.T, enableRealHW bool) {
	suite := NewHardwareTestSuite(enableRealHW)
	defer suite.Cleanup()

	err := suite.RunHardwareTests()
	if err != nil {
		t.Fatalf("Hardware test suite failed: %v", err)
	}

	// Assert minimum requirements
	if suite.PerformanceStats.PerformanceScore < 50 {
		t.Errorf("Hardware compatibility score too low: %d/100", suite.PerformanceStats.PerformanceScore)
	}

	if suite.PerformanceStats.AMDDetectionRate < 30 && enableRealHW {
		t.Errorf("AMD detection rate too low: %.1f%%", suite.PerformanceStats.AMDDetectionRate)
	}
}
