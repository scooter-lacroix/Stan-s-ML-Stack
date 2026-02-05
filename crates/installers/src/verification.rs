//! ML Stack Verification Logic
//!
//! Provides comprehensive verification of the ML stack, including ROCm,
//! PyTorch, ML components, and environment configuration.

use std::process::Command;
use std::path::Path;
use serde::{Deserialize, Serialize};

/// Status of a verification item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStatus {
    /// Verification passed
    Success,
    /// Verification passed with warnings
    Warning,
    /// Verification failed
    Failure,
    /// Component not detected
    NotDetected,
}

/// A single verification item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationItem {
    /// Name of the check
    pub name: String,
    /// Description of the check
    pub description: String,
    /// Status of the check
    pub status: VerificationStatus,
    /// Optional message or error details
    pub message: Option<String>,
}

/// Module for running all verifications.
pub struct VerificationModule;

impl VerificationModule {
    /// Verifies ROCm installation.
    pub async fn verify_rocm() -> VerificationItem {
        let name = "ROCm Installation".to_string();
        let description = "Checks if ROCm is installed and functional via rocminfo".to_string();
        
        if !Path::new("/opt/rocm/bin/rocminfo").exists() {
            return VerificationItem {
                name,
                description,
                status: VerificationStatus::NotDetected,
                message: Some("ROCm not found at /opt/rocm".to_string()),
            };
        }

        let output = Command::new("/opt/rocm/bin/rocminfo").output();
        match output {
            Ok(o) if o.status.success() => {
                VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::Success,
                    message: Some("rocminfo executed successfully".to_string()),
                }
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::Failure,
                    message: Some(format!("rocminfo failed: {}", stderr)),
                }
            }
            Err(e) => {
                VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::Failure,
                    message: Some(format!("Failed to execute rocminfo: {}", e)),
                }
            }
        }
    }

    /// Verifies PyTorch ROCm support.
    pub async fn verify_pytorch() -> VerificationItem {
        let name = "PyTorch ROCm Support".to_string();
        let description = "Verifies if PyTorch is installed and has ROCm/HIP support enabled".to_string();

        let output = Command::new("python3")
            .args(["-c", "import torch; print(f'ROCm: {torch.version.hip}, Available: {torch.cuda.is_available()}')"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                if stdout.contains("ROCm: None") {
                    VerificationItem {
                        name,
                        description,
                        status: VerificationStatus::Failure,
                        message: Some("PyTorch installed but ROCm/HIP is NOT enabled (None detected)".to_string()),
                    }
                } else if stdout.contains("Available: False") {
                    VerificationItem {
                        name,
                        description,
                        status: VerificationStatus::Warning,
                        message: Some(format!("ROCm PyTorch detected but GPU is not available to it: {}", stdout.trim())),
                    }
                } else {
                    VerificationItem {
                        name,
                        description,
                        status: VerificationStatus::Success,
                        message: Some(stdout.trim().to_string()),
                    }
                }
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::Failure,
                    message: Some(format!("Failed to verify PyTorch: {}", stderr)),
                }
            }
            Err(_) => {
                VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::NotDetected,
                    message: Some("PyTorch or Python3 not found".to_string()),
                }
            }
        }
    }

    /// Verifies vLLM ROCm support.
    pub async fn verify_vllm() -> VerificationItem {
        let name = "vLLM ROCm Support".to_string();
        let description = "Checks if vLLM is installed and recognizes ROCm/HIP".to_string();

        let output = Command::new("python3")
            .args(["-c", "import vllm; from vllm.utils import is_hip; print(is_hip())"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                if stdout.trim() == "True" {
                    VerificationItem {
                        name,
                        description,
                        status: VerificationStatus::Success,
                        message: Some("vLLM ROCm support detected".to_string()),
                    }
                } else {
                    VerificationItem {
                        name,
                        description,
                        status: VerificationStatus::Failure,
                        message: Some("vLLM installed but ROCm support not active".to_string()),
                    }
                }
            }
            _ => VerificationItem {
                name,
                description,
                status: VerificationStatus::NotDetected,
                message: Some("vLLM not installed".to_string()),
            },
        }
    }

    /// Verifies Triton ROCm support.
    pub async fn verify_triton() -> VerificationItem {
        let name = "Triton ROCm Support".to_string();
        let description = "Checks if Triton is installed and has ROCm backends".to_string();

        let output = Command::new("python3")
            .args(["-c", "import triton; print(triton.backends.backends)"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout).to_lowercase();
                if stdout.contains("rocm") || stdout.contains("hip") {
                    VerificationItem {
                        name,
                        description,
                        status: VerificationStatus::Success,
                        message: Some(format!("Triton ROCm backends: {}", stdout.trim())),
                    }
                } else {
                    VerificationItem {
                        name,
                        description,
                        status: VerificationStatus::Failure,
                        message: Some("Triton installed but ROCm backend not found".to_string()),
                    }
                }
            }
            _ => VerificationItem {
                name,
                description,
                status: VerificationStatus::NotDetected,
                message: Some("Triton not installed".to_string()),
            },
        }
    }

    /// Verifies Flash Attention installation.
    pub async fn verify_flash_attn() -> VerificationItem {
        let name = "Flash Attention ROCm".to_string();
        let description = "Checks if Flash Attention is installed and functional".to_string();

        let output = Command::new("python3")
            .args(["-c", "import flash_attn; print(flash_attn.__version__)"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::Success,
                    message: Some(format!("Flash Attention version: {}", stdout.trim())),
                }
            }
            _ => VerificationItem {
                name,
                description,
                status: VerificationStatus::NotDetected,
                message: Some("Flash Attention not installed".to_string()),
            },
        }
    }

    /// Verifies ONNX Runtime ROCm support.
    pub async fn verify_onnx() -> VerificationItem {
        let name = "ONNX Runtime ROCm".to_string();
        let description = "Checks if ONNX Runtime is installed and has ROCm provider".to_string();

        let output = Command::new("python3")
            .args(["-c", "import onnxruntime as ort; print(ort.get_available_providers())"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                if stdout.contains("ROCMExecutionProvider") {
                    VerificationItem {
                        name,
                        description,
                        status: VerificationStatus::Success,
                        message: Some(format!("Available providers: {}", stdout.trim())),
                    }
                } else {
                    VerificationItem {
                        name,
                        description,
                        status: VerificationStatus::Failure,
                        message: Some("ONNX Runtime installed but ROCMExecutionProvider not found".to_string()),
                    }
                }
            }
            _ => VerificationItem {
                name,
                description,
                status: VerificationStatus::NotDetected,
                message: Some("ONNX Runtime not installed".to_string()),
            },
        }
    }

    /// Verifies bitsandbytes ROCm support.
    pub async fn verify_bitsandbytes() -> VerificationItem {
        let name = "bitsandbytes ROCm".to_string();
        let description = "Checks if bitsandbytes is installed and functional (expected 0.48.2)".to_string();

        let output = Command::new("python3")
            .args(["-c", "import bitsandbytes; print(bitsandbytes.__version__)"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::Success,
                    message: Some(format!("bitsandbytes version: {}", stdout.trim())),
                }
            }
            _ => VerificationItem {
                name,
                description,
                status: VerificationStatus::Failure,
                message: Some("bitsandbytes not installed or failed to import".to_string()),
            },
        }
    }

    /// Verifies MIGraphX ROCm support.
    pub async fn verify_migraphx() -> VerificationItem {
        let name = "MIGraphX ROCm".to_string();
        let description = "Checks if MIGraphX is installed and functional".to_string();

        let output = Command::new("python3")
            .args(["-c", "import migraphx; print('MIGraphX import successful')"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::Success,
                    message: Some("MIGraphX detected and importable".to_string()),
                }
            }
            _ => VerificationItem {
                name,
                description,
                status: VerificationStatus::Failure,
                message: Some("MIGraphX not installed or failed to import".to_string()),
            },
        }
    }

    /// Verifies environment variables.
    pub fn verify_env_vars() -> Vec<VerificationItem> {
        let mut items = Vec::new();
        let critical_vars = [
            ("ROCM_PATH", "Base path for ROCm installation"),
            ("HIP_PATH", "Path to HIP toolkit"),
            ("HSA_OVERRIDE_GFX_VERSION", "GFX architecture override for unsupported GPUs"),
            ("LD_LIBRARY_PATH", "System library path"),
        ];

        for (var, desc) in critical_vars {
            let val = std::env::var(var);
            let status = match (var, &val) {
                (_, Ok(_)) => VerificationStatus::Success,
                ("HSA_OVERRIDE_GFX_VERSION", Err(_)) => VerificationStatus::Warning, // Optional
                _ => VerificationStatus::Warning,
            };

            items.push(VerificationItem {
                name: format!("Env Var: {}", var),
                description: desc.to_string(),
                status,
                message: Some(val.unwrap_or_else(|_| "Not set".to_string())),
            });
        }

        items
    }

    /// Runs all verifications.
    pub async fn run_all() -> Vec<VerificationItem> {
        let mut results = Vec::new();

        // Core components
        results.push(Self::verify_rocm().await);
        results.push(Self::verify_pytorch().await);
        results.push(Self::verify_vllm().await);
        results.push(Self::verify_triton().await);
        results.push(Self::verify_flash_attn().await);
        results.push(Self::verify_onnx().await);
        results.push(Self::verify_bitsandbytes().await);
        results.push(Self::verify_migraphx().await);

        // New components
        results.push(Self::verify_aiter().await);
        results.push(Self::verify_deepspeed().await);
        results.push(Self::verify_rocm_smi().await);
        results.push(Self::verify_wandb().await);
        results.push(Self::verify_pytorch_profiler().await);
        results.push(Self::verify_megatron().await);

        results.extend(Self::verify_env_vars());

        results
    }

    /// Verifies AITER installation.
    pub async fn verify_aiter() -> VerificationItem {
        let name = "AITER".to_string();
        let description = "Checks if AITER is installed for RDNA 3 GPU support".to_string();

        let output = Command::new("python3")
            .args(["-c", "import aiter; print(f'AITER {aiter.__version__}')"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::Success,
                    message: Some(format!("AITER detected: {}", stdout.trim())),
                }
            }
            _ => VerificationItem {
                name,
                description,
                status: VerificationStatus::NotDetected,
                message: Some("AITER not installed".to_string()),
            },
        }
    }

    /// Verifies DeepSpeed installation.
    pub async fn verify_deepspeed() -> VerificationItem {
        let name = "DeepSpeed".to_string();
        let description = "Checks if DeepSpeed is installed with ZeRO optimization".to_string();

        let output = Command::new("python3")
            .args(["-c", "import deepspeed; print(f'DeepSpeed {deepspeed.__version__}')"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::Success,
                    message: Some(format!("DeepSpeed detected: {}", stdout.trim())),
                }
            }
            _ => VerificationItem {
                name,
                description,
                status: VerificationStatus::NotDetected,
                message: Some("DeepSpeed not installed".to_string()),
            },
        }
    }

    /// Verifies ROCm SMI (system management interface).
    pub async fn verify_rocm_smi() -> VerificationItem {
        let name = "ROCm SMI".to_string();
        let description = "Checks if rocm-smi is installed and functional".to_string();

        // First check if the binary exists
        let smi_path = Path::new("/opt/rocm/bin/rocm-smi");
        if !smi_path.exists() {
            // Also check if it's in PATH
            let which_output = Command::new("which").arg("rocm-smi").output();
            if which_output.is_err() || !which_output.unwrap().status.success() {
                return VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::NotDetected,
                    message: Some("rocm-smi not found".to_string()),
                };
            }
        }

        // Try to run rocm-smi
        let output = Command::new("rocm-smi")
            .args(["--showproductname", "--csv"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                let gpu_count = stdout.lines().skip(1).count();
                VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::Success,
                    message: Some(format!("rocm-smi functional, {} GPU(s) detected", gpu_count)),
                }
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::Warning,
                    message: Some(format!("rocm-smi found but error running: {}", stderr)),
                }
            }
            Err(e) => {
                VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::Failure,
                    message: Some(format!("Failed to run rocm-smi: {}", e)),
                }
            }
        }
    }

    /// Verifies Weights & Biases installation.
    pub async fn verify_wandb() -> VerificationItem {
        let name = "Weights & Biases".to_string();
        let description = "Checks if W&B is installed and configured".to_string();

        let output = Command::new("python3")
            .args(["-c", "import wandb; print(f'wandb {wandb.__version__}')"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                // Also check if logged in
                let login_check = Command::new("wandb")
                    .args(["whoami"])
                    .output();

                let login_status = if let Ok(lo) = login_check {
                    if lo.status.success() {
                        format!("Logged in: {}", String::from_utf8_lossy(&lo.stdout).trim())
                    } else {
                        "Not logged in".to_string()
                    }
                } else {
                    "Login check failed".to_string()
                };

                VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::Success,
                    message: Some(format!("W&B {} - {}", stdout.trim(), login_status)),
                }
            }
            _ => VerificationItem {
                name,
                description,
                status: VerificationStatus::NotDetected,
                message: Some("W&B not installed".to_string()),
            },
        }
    }

    /// Verifies PyTorch Profiler availability.
    pub async fn verify_pytorch_profiler() -> VerificationItem {
        let name = "PyTorch Profiler".to_string();
        let description = "Checks if PyTorch profiler is available".to_string();

        let output = Command::new("python3")
            .args(["-c", "from torch.profiler import profile; print('Profiler available')"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::Success,
                    message: Some("PyTorch Profiler is available".to_string()),
                }
            }
            _ => VerificationItem {
                name,
                description,
                status: VerificationStatus::Failure,
                message: Some("PyTorch Profiler not available - check PyTorch installation".to_string()),
            },
        }
    }

    /// Verifies Megatron-LM installation.
    pub async fn verify_megatron() -> VerificationItem {
        let name = "Megatron-LM".to_string();
        let description = "Checks if Megatron-LM is installed".to_string();

        let output = Command::new("python3")
            .args(["-c", "import megatron; print('Megatron-LM available')"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                VerificationItem {
                    name,
                    description,
                    status: VerificationStatus::Success,
                    message: Some("Megatron-LM is available".to_string()),
                }
            }
            _ => VerificationItem {
                name,
                description,
                status: VerificationStatus::NotDetected,
                message: Some("Megatron-LM not installed".to_string()),
            },
        }
    }
}
