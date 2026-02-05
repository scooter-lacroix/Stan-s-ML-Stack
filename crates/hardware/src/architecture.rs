//! GPU Architecture Definitions
//!
//! This module defines all supported GPU architectures for Stan's ML Stack,
//! including RDNA 2/3/4 and CDNA 2/3 series GPUs.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Represents the family of GPU architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchitectureFamily {
    /// RDNA 4 architecture (latest consumer GPUs)
    RDNA4,
    /// RDNA 3 architecture (current consumer GPUs)
    RDNA3,
    /// RDNA 2 architecture (previous generation)
    RDNA2,
    /// CDNA 3 architecture (data center MI300 series)
    CDNA3,
    /// CDNA 2 architecture (data center MI200 series)
    CDNA2,
    /// Legacy architectures (Vega, MI100)
    Legacy,
}

impl fmt::Display for ArchitectureFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArchitectureFamily::RDNA4 => write!(f, "RDNA 4"),
            ArchitectureFamily::RDNA3 => write!(f, "RDNA 3"),
            ArchitectureFamily::RDNA2 => write!(f, "RDNA 2"),
            ArchitectureFamily::CDNA3 => write!(f, "CDNA 3"),
            ArchitectureFamily::CDNA2 => write!(f, "CDNA 2"),
            ArchitectureFamily::Legacy => write!(f, "Legacy"),
        }
    }
}

/// Represents specific GPU architecture variants.
///
/// This enum covers all GPU architectures supported by Stan's ML Stack,
/// from legacy Vega GPUs to the latest RDNA 4 and CDNA 3 architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GPUArchitecture {
    /// gfx906 - Vega 20 (MI100, Radeon VII)
    Gfx906,
    /// gfx908 - Vega 10 (MI50, MI60)
    Gfx908,
    /// gfx90a - Aldebaran (MI200 series: MI210, MI250, MI250X)
    Gfx90a,
    /// gfx942 - MI300 series (MI300X, MI300A)
    Gfx942,
    /// gfx1030 - Navi 21 (RX 6900 XT, RX 6800 XT)
    Gfx1030,
    /// gfx1100 - Navi 31 (RX 7900 XTX, RX 7900 XT)
    Gfx1100,
    /// gfx1101 - Navi 32 (RX 7800 XT, RX 7700 XT)
    Gfx1101,
    /// gfx1151 - Navi 33 (Strix Halo, Ryzen AI)
    Gfx1151,
    /// gfx1200 - Navi 44 (RX 9060 XT) - RDNA 4
    Gfx1200,
    /// gfx1201 - Navi 48 (RX 9070 XT) - RDNA 4
    Gfx1201,
}

impl GPUArchitecture {
    /// Returns the architecture family for this GPU.
    ///
    /// # Examples
    ///
    /// ```
    /// use mlstack_hardware::GPUArchitecture;
    ///
    /// let arch = GPUArchitecture::Gfx1100;
    /// assert!(arch.family().to_string().contains("RDNA 3"));
    /// ```
    pub fn family(&self) -> ArchitectureFamily {
        match self {
            GPUArchitecture::Gfx1200 | GPUArchitecture::Gfx1201 => ArchitectureFamily::RDNA4,
            GPUArchitecture::Gfx1100 | GPUArchitecture::Gfx1101 | GPUArchitecture::Gfx1151 => {
                ArchitectureFamily::RDNA3
            }
            GPUArchitecture::Gfx1030 => ArchitectureFamily::RDNA2,
            GPUArchitecture::Gfx942 => ArchitectureFamily::CDNA3,
            GPUArchitecture::Gfx90a => ArchitectureFamily::CDNA2,
            GPUArchitecture::Gfx906 | GPUArchitecture::Gfx908 => ArchitectureFamily::Legacy,
        }
    }

    /// Returns true if this is an RDNA 4 architecture.
    pub fn is_rdna4(&self) -> bool {
        matches!(self.family(), ArchitectureFamily::RDNA4)
    }

    /// Returns true if this is an RDNA 3 architecture.
    pub fn is_rdna3(&self) -> bool {
        matches!(self.family(), ArchitectureFamily::RDNA3)
    }

    /// Returns true if this is an RDNA 2 architecture.
    pub fn is_rdna2(&self) -> bool {
        matches!(self.family(), ArchitectureFamily::RDNA2)
    }

    /// Returns true if this is a CDNA 3 architecture.
    pub fn is_cdna3(&self) -> bool {
        matches!(self.family(), ArchitectureFamily::CDNA3)
    }

    /// Returns true if this is a CDNA 2 architecture.
    pub fn is_cdna2(&self) -> bool {
        matches!(self.family(), ArchitectureFamily::CDNA2)
    }

    /// Returns true if this is a legacy architecture.
    pub fn is_legacy(&self) -> bool {
        matches!(self.family(), ArchitectureFamily::Legacy)
    }

    /// Returns the gfx string representation (e.g., "gfx1100").
    pub fn as_gfx_string(&self) -> &'static str {
        match self {
            GPUArchitecture::Gfx906 => "gfx906",
            GPUArchitecture::Gfx908 => "gfx908",
            GPUArchitecture::Gfx90a => "gfx90a",
            GPUArchitecture::Gfx942 => "gfx942",
            GPUArchitecture::Gfx1030 => "gfx1030",
            GPUArchitecture::Gfx1100 => "gfx1100",
            GPUArchitecture::Gfx1101 => "gfx1101",
            GPUArchitecture::Gfx1151 => "gfx1151",
            GPUArchitecture::Gfx1200 => "gfx1200",
            GPUArchitecture::Gfx1201 => "gfx1201",
        }
    }

    /// Returns a human-readable GPU name for this architecture.
    pub fn gpu_name(&self) -> &'static str {
        match self {
            GPUArchitecture::Gfx906 => "Vega 20 / MI100",
            GPUArchitecture::Gfx908 => "Vega 10 / MI50/MI60",
            GPUArchitecture::Gfx90a => "Aldebaran / MI200 Series",
            GPUArchitecture::Gfx942 => "MI300 Series",
            GPUArchitecture::Gfx1030 => "Navi 21 / RX 6900/6800 Series",
            GPUArchitecture::Gfx1100 => "Navi 31 / RX 7900 XTX/XT",
            GPUArchitecture::Gfx1101 => "Navi 32 / RX 7800/7700 XT",
            GPUArchitecture::Gfx1151 => "Navi 33 / Strix Halo",
            GPUArchitecture::Gfx1200 => "Navi 44 / RX 9060 XT",
            GPUArchitecture::Gfx1201 => "Navi 48 / RX 9070 XT",
        }
    }

    /// Returns the HSA_OVERRIDE_GFX_VERSION value for this architecture.
    ///
    /// This is used for PyTorch compatibility on RDNA 3/4 GPUs.
    pub fn hsa_override_gfx_version(&self) -> Option<&'static str> {
        match self {
            GPUArchitecture::Gfx1100 | GPUArchitecture::Gfx1101 | GPUArchitecture::Gfx1151 => {
                Some("11.0.0")
            }
            GPUArchitecture::Gfx1200 | GPUArchitecture::Gfx1201 => Some("12.0.0"),
            _ => None,
        }
    }
}

impl fmt::Display for GPUArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_gfx_string())
    }
}

impl FromStr for GPUArchitecture {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalized = s.to_lowercase().trim().to_string();
        match normalized.as_str() {
            "gfx906" => Ok(GPUArchitecture::Gfx906),
            "gfx908" => Ok(GPUArchitecture::Gfx908),
            "gfx90a" => Ok(GPUArchitecture::Gfx90a),
            "gfx942" => Ok(GPUArchitecture::Gfx942),
            "gfx1030" => Ok(GPUArchitecture::Gfx1030),
            "gfx1100" => Ok(GPUArchitecture::Gfx1100),
            "gfx1101" => Ok(GPUArchitecture::Gfx1101),
            "gfx1151" => Ok(GPUArchitecture::Gfx1151),
            "gfx1200" => Ok(GPUArchitecture::Gfx1200),
            "gfx1201" => Ok(GPUArchitecture::Gfx1201),
            _ => Err(format!("Unknown GPU architecture: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_family() {
        assert_eq!(GPUArchitecture::Gfx1200.family(), ArchitectureFamily::RDNA4);
        assert_eq!(GPUArchitecture::Gfx1201.family(), ArchitectureFamily::RDNA4);
        assert_eq!(GPUArchitecture::Gfx1100.family(), ArchitectureFamily::RDNA3);
        assert_eq!(GPUArchitecture::Gfx1101.family(), ArchitectureFamily::RDNA3);
        assert_eq!(GPUArchitecture::Gfx1151.family(), ArchitectureFamily::RDNA3);
        assert_eq!(GPUArchitecture::Gfx1030.family(), ArchitectureFamily::RDNA2);
        assert_eq!(GPUArchitecture::Gfx942.family(), ArchitectureFamily::CDNA3);
        assert_eq!(GPUArchitecture::Gfx90a.family(), ArchitectureFamily::CDNA2);
        assert_eq!(GPUArchitecture::Gfx906.family(), ArchitectureFamily::Legacy);
        assert_eq!(GPUArchitecture::Gfx908.family(), ArchitectureFamily::Legacy);
    }

    #[test]
    fn test_architecture_checks() {
        let rdna4 = GPUArchitecture::Gfx1200;
        assert!(rdna4.is_rdna4());
        assert!(!rdna4.is_rdna3());
        assert!(!rdna4.is_legacy());

        let rdna3 = GPUArchitecture::Gfx1100;
        assert!(rdna3.is_rdna3());
        assert!(!rdna3.is_rdna4());

        let legacy = GPUArchitecture::Gfx906;
        assert!(legacy.is_legacy());
        assert!(!legacy.is_rdna3());
    }

    #[test]
    fn test_from_str() {
        assert_eq!(
            "gfx1100".parse::<GPUArchitecture>().unwrap(),
            GPUArchitecture::Gfx1100
        );
        assert_eq!(
            "gfx1201".parse::<GPUArchitecture>().unwrap(),
            GPUArchitecture::Gfx1201
        );
        assert_eq!(
            "GFX1100".parse::<GPUArchitecture>().unwrap(),
            GPUArchitecture::Gfx1100
        );
        assert!("unknown".parse::<GPUArchitecture>().is_err());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", GPUArchitecture::Gfx1100), "gfx1100");
        assert_eq!(format!("{}", GPUArchitecture::Gfx942), "gfx942");
    }

    #[test]
    fn test_hsa_override() {
        assert_eq!(
            GPUArchitecture::Gfx1100.hsa_override_gfx_version(),
            Some("11.0.0")
        );
        assert_eq!(
            GPUArchitecture::Gfx1200.hsa_override_gfx_version(),
            Some("12.0.0")
        );
        assert_eq!(GPUArchitecture::Gfx906.hsa_override_gfx_version(), None);
    }
}
