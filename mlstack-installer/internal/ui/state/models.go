// internal/ui/state/models.go
package state

import (
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// Foundation Components (Required Core ML Stack)
var FoundationComponents = []types.Component{
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
var CoreComponents = []types.Component{
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
var ExtensionComponents = []types.Component{
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
		Script:      "install_vllm_multi.sh",
		Category:    "extension",
		Required:    false,
		Selected:    false,
		Size:        1500, // ~1.5GB
		Estimate:    "15-25 min",
	},
	{
		ID:          "aiter",
		Name:        "AITER",
		Description: "AMD AITER optimization tooling",
		Script:      "install_aiter.sh",
		Category:    "extension",
		Required:    false,
		Selected:    false,
		Size:        600, // ~600MB
		Estimate:    "10-20 min",
	},
	{
		ID:          "vllm-studio",
		Name:        "vLLM Studio",
		Description: "Model lifecycle manager for vLLM/SGLang",
		Script:      "install_vllm_studio.sh",
		Category:    "extension",
		Required:    false,
		Selected:    false,
		Size:        1200, // ~1.2GB
		Estimate:    "10-20 min",
	},
	{
		ID:          "onnx",
		Name:        "ONNX Runtime",
		Description: "Cross-platform inference accelerator (BUILDS TAKE HOURS)",
		Script:      "build_onnxruntime_multi.sh",
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
		Script:      "install_bitsandbytes_multi.sh",
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
		Script:      "install_migraphx_multi.sh",
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
var EnvironmentComponents = []types.Component{
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
var VerificationComponents = []types.Component{
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

// AllComponents returns all available components organized by category
func AllComponents() []types.Component {
	all := append([]types.Component{}, FoundationComponents...)
	all = append(all, CoreComponents...)
	all = append(all, ExtensionComponents...)
	all = append(all, EnvironmentComponents...)
	all = append(all, VerificationComponents...)
	return all
}

// GetComponentsByCategory returns components filtered by category
func GetComponentsByCategory(category string) []types.Component {
	switch category {
	case "foundation":
		return FoundationComponents
	case "core":
		return CoreComponents
	case "extension":
		return ExtensionComponents
	case "environment":
		return EnvironmentComponents
	case "verification":
		return VerificationComponents
	default:
		return []types.Component{}
	}
}

// CategoryInfo provides information about component categories
type CategoryInfo struct {
	Name        string
	Description string
	Icon        string
	Priority    int
}

// GetCategoryInfo returns information about all categories
func GetCategoryInfo() map[string]CategoryInfo {
	return map[string]CategoryInfo{
		"foundation": {
			Name:        "Foundation",
			Description: "Core ML stack components required for basic functionality",
			Icon:        "🏗️",
			Priority:    1,
		},
		"core": {
			Name:        "Core ML Stack",
			Description: "Essential ML tools and libraries",
			Icon:        "⚡",
			Priority:    2,
		},
		"extension": {
			Name:        "Advanced Features",
			Description: "Optional components for advanced ML workflows",
			Icon:        "🚀",
			Priority:    3,
		},
		"environment": {
			Name:        "Environment Setup",
			Description: "System environment configuration utilities",
			Icon:        "⚙️",
			Priority:    4,
		},
		"verification": {
			Name:        "Verification Tools",
			Description: "Installation verification and diagnostic tools",
			Icon:        "✅",
			Priority:    5,
		},
	}
}

// GetCategoriesByPriority returns categories ordered by priority
func GetCategoriesByPriority() []string {
	info := GetCategoryInfo()
	categories := make([]string, 0, len(info))

	// Create slice of categories sorted by priority
	for category, categoryInfo := range info {
		inserted := false
		for i, existing := range categories {
			if info[existing].Priority > categoryInfo.Priority {
				categories = append(categories[:i], append([]string{category}, categories[i:]...)...)
				inserted = true
				break
			}
		}
		if !inserted {
			categories = append(categories, category)
		}
	}

	return categories
}
