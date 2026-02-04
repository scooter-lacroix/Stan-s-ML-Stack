// internal/ui/models.go
package ui

import (
	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
)

// UI stages (different from installer stages)
type UIStage int

const (
	UIStageWelcome UIStage = iota
	UIStageHardwareDetect
	UIStageComponentSelect
	UIStageConfirm
	UIStageInstalling
	UIStageComplete
)

// Component represents an installable ML stack component
type Component struct {
	ID          string
	Name        string
	Description string
	Script      string
	Category    string // "foundation", "core", "extension", "environment"
	Required    bool
	Selected    bool
	Installed   bool
	Progress    float64
	Size        int64  // Estimated installation size in MB
	Estimate    string // Time estimate
}

// Foundation Components (Required Core ML Stack)
var FoundationComponents = []Component{
	{
		ID:          "rocm",
		Name:        "ROCm Platform",
		Description: "AMD's open software platform for GPU computing (6.4.43482)",
		Script:      "install_rocm.sh",
		Category:    "foundation",
		Required:    true,
		Selected:    true,
		Size:        8000, // ~8GB
		Estimate:    "30-45 min",
	},
	{
		ID:          "pytorch",
		Name:        "PyTorch with ROCm",
		Description: "Deep learning framework with AMD GPU support",
		Script:      "install_pytorch_rocm.sh",
		Category:    "foundation",
		Required:    true,
		Selected:    true,
		Size:        2000, // ~2GB
		Estimate:    "10-15 min",
	},
	{
		ID:          "triton",
		Name:        "Triton",
		Description: "Compiler for parallel programming",
		Script:      "install_triton.sh",
		Category:    "foundation",
		Required:    true,
		Selected:    true,
		Size:        500, // ~500MB
		Estimate:    "5-10 min",
	},
	{
		ID:          "mpi4py",
		Name:        "MPI4Py",
		Description: "Message Passing Interface for distributed computing",
		Script:      "install_mpi4py.sh",
		Category:    "foundation",
		Required:    true,
		Selected:    true,
		Size:        300, // ~300MB
		Estimate:    "3-5 min",
	},
	{
		ID:          "deepspeed",
		Name:        "DeepSpeed",
		Description: "Deep learning optimization library for large models",
		Script:      "install_deepspeed.sh",
		Category:    "foundation",
		Required:    true,
		Selected:    true,
		Size:        800, // ~800MB
		Estimate:    "8-12 min",
	},
}

// Core Components (Optional Core ML Stack)
var CoreComponents = []Component{
	{
		ID:          "ml-stack-core",
		Name:        "ML Stack Core",
		Description: "Core ML Stack components installation",
		Script:      "install_ml_stack.sh",
		Category:    "core",
		Required:    false,
		Selected:    true,
		Size:        1000, // ~1GB
		Estimate:    "15-20 min",
	},
	{
		ID:          "flash-attn",
		Name:        "Flash Attention",
		Description: "Efficient attention computation",
		Script:      "install_flash_attention_ck.sh",
		Category:    "core",
		Required:    false,
		Selected:    true,
		Size:        1200, // ~1.2GB
		Estimate:    "20-30 min",
	},
	{
		ID:          "repair-stack",
		Name:        "Repair ML Stack",
		Description: "Fix and repair ML Stack installation",
		Script:      "repair_ml_stack.sh",
		Category:    "core",
		Required:    false,
		Selected:    false,
		Size:        0, // No size estimate for repair
		Estimate:    "5-10 min",
	},
}

// Extension Components (Advanced ML Features)
var ExtensionComponents = []Component{
	{
		ID:          "megatron",
		Name:        "Megatron-LM",
		Description: "Large-scale training framework for transformer models",
		Script:      "install_megatron.sh",
		Category:    "extension",
		Required:    false,
		Selected:    false,
		Size:        3000, // ~3GB
		Estimate:    "25-40 min",
	},
	{
		ID:          "vllm",
		Name:        "vLLM",
		Description: "High-throughput inference engine for LLMs",
		Script:      "install_vllm.sh",
		Category:    "extension",
		Required:    false,
		Selected:    false,
		Size:        1500, // ~1.5GB
		Estimate:    "15-25 min",
	},
	{
		ID:          "onnx",
		Name:        "ONNX Runtime",
		Description: "Cross-platform inference accelerator (BUILDS TAKE HOURS)",
		Script:      "build_onnxruntime.sh",
		Category:    "extension",
		Required:    false,
		Selected:    false,
		Size:        2500, // ~2.5GB
		Estimate:    "45-90 min",
	},
	{
		ID:          "bitsandbytes",
		Name:        "BITSANDBYTES",
		Description: "Efficient quantization for deep learning models",
		Script:      "install_bitsandbytes.sh",
		Category:    "extension",
		Required:    false,
		Selected:    false,
		Size:        200, // ~200MB
		Estimate:    "3-5 min",
	},
	{
		ID:          "rocm-smi",
		Name:        "ROCm SMI",
		Description: "System monitoring and management for AMD GPUs",
		Script:      "install_rocm_smi.sh",
		Category:    "extension",
		Required:    false,
		Selected:    true,
		Size:        100, // ~100MB
		Estimate:    "2-3 min",
	},
	{
		ID:          "migraphx",
		Name:        "MIGraphX Python Wrapper",
		Description: "Python wrapper for MIGraphX library",
		Script:      "install_migraphx_python.sh",
		Category:    "extension",
		Required:    false,
		Selected:    false,
		Size:        800, // ~800MB
		Estimate:    "10-15 min",
	},
	{
		ID:          "pytorch-profiler",
		Name:        "PyTorch Profiler",
		Description: "Performance analysis for PyTorch models",
		Script:      "install_pytorch_profiler.sh",
		Category:    "extension",
		Required:    false,
		Selected:    false,
		Size:        150, // ~150MB
		Estimate:    "3-5 min",
	},
	{
		ID:          "wandb",
		Name:        "Weights & Biases",
		Description: "Experiment tracking and visualization",
		Script:      "install_wandb.sh",
		Category:    "extension",
		Required:    false,
		Selected:    false,
		Size:        500, // ~500MB
		Estimate:    "5-8 min",
	},
}

// Environment Setup Components
var EnvironmentComponents = []Component{
	{
		ID:          "basic-env",
		Name:        "Environment Setup",
		Description: "Basic environment setup for simple GPU configurations",
		Script:      "setup_environment.sh",
		Category:    "environment",
		Required:    true,
		Selected:    true,
		Size:        0, // No size for environment setup
		Estimate:    "1-2 min",
	},
	{
		ID:          "enhanced-env",
		Name:        "Enhanced Environment Setup",
		Description: "Advanced setup for complex GPU configs with iGPU filtering",
		Script:      "enhanced_setup_environment.sh",
		Category:    "environment",
		Required:    false,
		Selected:    false,
		Size:        0, // No size for environment setup
		Estimate:    "2-3 min",
	},
}

// Verification Components
var VerificationComponents = []Component{
	{
		ID:          "verify-basic",
		Name:        "Verify Installation",
		Description: "Basic verification of ML Stack installation",
		Script:      "verify_installation.sh",
		Category:    "verification",
		Required:    false,
		Selected:    false,
		Size:        0, // No size for verification
		Estimate:    "2-5 min",
	},
	{
		ID:          "verify-enhanced",
		Name:        "Enhanced Verify Installation",
		Description: "Advanced verification with detailed diagnostics",
		Script:      "enhanced_verify_installation.sh",
		Category:    "verification",
		Required:    false,
		Selected:    false,
		Size:        0, // No size for verification
		Estimate:    "5-10 min",
	},
	{
		ID:          "verify-build",
		Name:        "Verify and Build",
		Description: "Verify installation and build components",
		Script:      "verify_and_build.sh",
		Category:    "verification",
		Required:    false,
		Selected:    false,
		Size:        0, // No size for verification
		Estimate:    "10-15 min",
	},
}

// Model represents the application state
type UIModel struct {
	stage      UIStage
	components []Component
	list       list.Model
	progress   progress.Model
	spinner    spinner.Model

	// Hardware detection
	gpuInfo    string
	systemInfo string

	// Installation tracking
	currentComponent int
	installLog       []string
	errorLog         []string

	// Component selection tracking
	selectedCategories map[string]bool

	// UI state
	width    int
	height   int
	ready    bool
	quitting bool
}

// AllComponents returns all available components organized by category
func AllComponents() []Component {
	all := append([]Component{}, FoundationComponents...)
	all = append(all, CoreComponents...)
	all = append(all, ExtensionComponents...)
	all = append(all, EnvironmentComponents...)
	all = append(all, VerificationComponents...)
	return all
}

// GetComponentsByCategory returns components filtered by category
func GetComponentsByCategory(category string) []Component {
	var components []Component
	switch category {
	case "foundation":
		components = FoundationComponents
	case "core":
		components = CoreComponents
	case "extension":
		components = ExtensionComponents
	case "environment":
		components = EnvironmentComponents
	case "verification":
		components = VerificationComponents
	}
	return components
}

// GetSelectedComponents returns all selected components
func (m *UIModel) GetSelectedComponents() []Component {
	var selected []Component
	for _, c := range m.components {
		if c.Selected {
			selected = append(selected, c)
		}
	}
	return selected
}

// GetRequiredComponents returns all required components
func (m *UIModel) GetRequiredComponents() []Component {
	var required []Component
	for _, c := range m.components {
		if c.Required {
			required = append(required, c)
		}
	}
	return required
}

// GetTotalSize returns the total size of selected components
func (m *UIModel) GetTotalSize() int64 {
	var total int64
	for _, c := range m.GetSelectedComponents() {
		total += c.Size
	}
	return total
}

// GetTotalTime returns the estimated total installation time
func (m *UIModel) GetTotalTime() string {
	var maxTime string
	for _, c := range m.GetSelectedComponents() {
		if c.Estimate > maxTime {
			maxTime = c.Estimate
		}
	}
	return maxTime
}

func NewInstallerModel() Model {
	// Initialize progress bar
	prog := progress.New(
		progress.WithDefaultGradient(),
		progress.WithWidth(60),
	)

	// Initialize spinner for loading states
	s := spinner.New()
	s.Spinner = spinner.Dot

	// Initialize components
	allComponents := AllComponents()

	// Set default selections (all required components and some common ones)
	for i := range allComponents {
		switch allComponents[i].ID {
		case "rocm", "pytorch", "triton", "mpi4py", "deepspeed":
			allComponents[i].Selected = true
		case "ml-stack-core", "flash-attn":
			allComponents[i].Selected = true
		case "rocm-smi":
			allComponents[i].Selected = true
		case "basic-env":
			allComponents[i].Selected = true
		}
	}

	return Model{
		stage:              StageWelcome,
		components:         allComponents,
		progress:           prog,
		spinner:            s,
		selectedCategories: make(map[string]bool),
	}
}
