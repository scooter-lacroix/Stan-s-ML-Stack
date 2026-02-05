// internal/utils/system.go
package utils

import (
	"os/exec"
	"runtime"
	"strings"
)

// SystemInfo holds system information
type SystemInfo struct {
	OS           string
	Architecture string
	GPUInfo      string
	Memory       string
}

// DetectSystemInfo detects system information
func DetectSystemInfo() (SystemInfo, error) {
	var info SystemInfo

	// Get OS and architecture
	info.OS = runtime.GOOS
	info.Architecture = runtime.GOARCH

	// Get GPU info (ROCm specific)
	gpuCmd := exec.Command("rocminfo")
	gpuOutput, err := gpuCmd.CombinedOutput()
	if err == nil {
		info.GPUInfo = string(gpuOutput)
	} else {
		// Fallback to lspci for GPU detection
		lspciCmd := exec.Command("lspci", "|", "grep", "-i", "vga")
		lspciOutput, _ := lspciCmd.CombinedOutput()
		info.GPUInfo = string(lspciOutput)
	}

	// Get memory info
	memCmd := exec.Command("free", "-h")
	memOutput, err := memCmd.CombinedOutput()
	if err == nil {
		info.Memory = strings.TrimSpace(string(memOutput))
	}

	return info, nil
}

// IsRoot checks if running with root privileges
func IsRoot() bool {
	return false // We'll check this in main.go since os.Geteuid() is used there
}
