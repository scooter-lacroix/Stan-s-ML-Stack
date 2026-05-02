//! Per-distro package name mappings.
//!
//! Maps generic (Debian-style) package names to the native names for each
//! package manager (apt, pacman, dnf, yum, zypper). Covers the same packages
//! as `scripts/lib/package_mappings.sh`.
//!
//! # Validation Assertions
//!
//! - **VAL-INFRA-004**: Package name mapping correct

use crate::platform::detection::PackageManager;

/// Lookup a mapped package name for a given package manager.
///
/// Returns the native package name if a mapping exists, otherwise returns
/// the original name unchanged.
pub fn map_package_name(generic_name: &str, pkg_mgr: PackageManager) -> String {
    match pkg_mgr {
        PackageManager::Apt => map_apt(generic_name),
        PackageManager::Pacman => map_pacman(generic_name),
        PackageManager::Dnf => map_dnf(generic_name),
        PackageManager::Yum => map_yum(generic_name),
        PackageManager::Zypper => map_zypper(generic_name),
        PackageManager::Unknown => generic_name.to_string(),
    }
}

/// Map multiple package names at once.
pub fn map_package_names(generic_names: &[&str], pkg_mgr: PackageManager) -> Vec<String> {
    generic_names
        .iter()
        .map(|name| map_package_name(name, pkg_mgr))
        .collect()
}

/// Check if a mapping exists for a given package name and package manager.
pub fn has_mapping(generic_name: &str, pkg_mgr: PackageManager) -> bool {
    let mapped = map_package_name(generic_name, pkg_mgr);
    // If the mapped name is different from the input, a mapping exists
    mapped != generic_name || get_known_packages().contains(&generic_name)
}

/// Get all known generic package names.
pub fn get_known_packages() -> &'static [&'static str] {
    &[
        // Build essentials
        "build-essential",
        "gcc",
        "g++",
        "make",
        "cmake",
        "ninja-build",
        "pkg-config",
        "autoconf",
        "automake",
        "libtool",
        // Python development
        "python3",
        "python3-dev",
        "python3-pip",
        "python3-venv",
        "python3-setuptools",
        "python3-wheel",
        "python3-numpy",
        // ROCm tools
        "rocminfo",
        "rocprofiler",
        "rocm-smi-lib",
        "rocm-dev",
        "rocm-libs",
        "hip-runtime",
        // ROCm libraries
        "librccl-dev",
        "librccl1",
        "rccl",
        "rccl-dev",
        "migraphx",
        "migraphx-dev",
        "half",
        "miopen-hip",
        "rocblas",
        // MPI
        "libopenmpi-dev",
        "openmpi-bin",
        "openmpi",
        "openmpi-devel",
        "environment-modules",
        // Build tools
        "llvm-dev",
        "clang",
        "lld",
        "lldb",
        "libssl-dev",
        "libffi-dev",
        "zlib1g-dev",
        // System utilities
        "pciutils",
        "mesa-utils",
        "clinfo",
        "libnuma-dev",
        "numactl",
        "hwloc",
        "lshw",
        "dmidecode",
        // Version control
        "git",
        "git-lfs",
        "gh",
        // Network
        "wget",
        "curl",
        "gnupg",
        "ca-certificates",
        "rsync",
        "ssh",
        // Libraries
        "libbz2-dev",
        "liblzma-dev",
        "libsqlite3-dev",
        "libncurses-dev",
        "libreadline-dev",
        "uuid-dev",
        "libgdbm-dev",
    ]
}

// =========================================================================
// Helper macro to avoid repeating .to_string() on every match arm
// =========================================================================

/// Map a package name using a match expression where all arms return `&str`.
/// The `_ => name` fallback returns the original name.
macro_rules! pkg_map {
    ($name:expr, { $($pat:pat => $val:expr),* $(,)? }) => {
        match $name {
            $($pat => $val.to_string(),)*
            _ => $name.to_string(),
        }
    };
}

// =========================================================================
// Apt mappings (Debian/Ubuntu) — mostly identity
// =========================================================================

fn map_apt(name: &str) -> String {
    // Apt packages are mostly the canonical names, but a few need mapping
    match name {
        "openmpi" => "openmpi-bin libopenmpi-dev".to_string(),
        "hip-runtime" => "hip-runtime-amd".to_string(),
        _ => name.to_string(),
    }
}

// =========================================================================
// Pacman mappings (Arch Linux)
// =========================================================================

fn map_pacman(name: &str) -> String {
    pkg_map!(name, {
        // Build essentials
        "build-essential" => "base-devel",
        "g++" => "gcc",
        "ninja-build" => "ninja",
        "pkg-config" => "pkgconf",

        // Python development
        "python3" => "python",
        "python3-dev" => "python",
        "python3-pip" => "python-pip",
        "python3-venv" => "python",
        "python3-setuptools" => "python-setuptools",
        "python3-wheel" => "python-wheel",
        "python3-numpy" => "python-numpy",

        // ROCm tools
        "hip-runtime" => "hip-runtime-amd",

        // ROCm libraries
        "librccl-dev" => "rccl",
        "librccl1" => "rccl",
        "rccl-dev" => "rccl",
        "migraphx-dev" => "migraphx",

        // MPI
        "libopenmpi-dev" => "openmpi",
        "openmpi-bin" => "openmpi",
        "openmpi" => "openmpi",
        "openmpi-devel" => "openmpi",

        // Build tools
        "llvm-dev" => "llvm",
        "libssl-dev" => "openssl",
        "libffi-dev" => "libffi",
        "zlib1g-dev" => "zlib",

        // System utilities
        "mesa-utils" => "mesa-demos",
        "libnuma-dev" => "numactl",

        // Version control
        "gh" => "github-cli",

        // Network
        "ssh" => "openssh",

        // Libraries
        "libbz2-dev" => "bzip2",
        "liblzma-dev" => "xz",
        "libsqlite3-dev" => "sqlite",
        "libncurses-dev" => "ncurses",
        "libreadline-dev" => "readline",
        "uuid-dev" => "util-linux",
        "libgdbm-dev" => "gdbm",
    })
}

// =========================================================================
// DNF mappings (Fedora/RHEL)
// =========================================================================

fn map_dnf(name: &str) -> String {
    pkg_map!(name, {
        // Build essentials
        "build-essential" => "@development-tools",
        "g++" => "gcc-c++",
        "pkg-config" => "pkgconf",

        // Python development
        "python3-dev" => "python3-devel",
        "python3-venv" => "python3",

        // ROCm tools
        "rocm-smi-lib" => "rocm-smi",

        // ROCm libraries
        "librccl-dev" => "rccl-devel",
        "rccl-dev" => "rccl-devel",
        "migraphx-dev" => "migraphx-devel",

        // MPI
        "libopenmpi-dev" => "openmpi-devel",
        "openmpi" => "openmpi openmpi-devel",
        "openmpi-devel" => "openmpi-devel",

        // Build tools
        "llvm-dev" => "llvm-devel",
        "libssl-dev" => "openssl-devel",
        "libffi-dev" => "libffi-devel",
        "zlib1g-dev" => "zlib-devel",

        // System utilities
        "mesa-utils" => "mesa-demos",
        "libnuma-dev" => "numactl-devel",

        // Network
        "gnupg" => "gnupg2",
        "ssh" => "openssh-clients",

        // Libraries
        "libbz2-dev" => "bzip2-devel",
        "liblzma-dev" => "xz-devel",
        "libsqlite3-dev" => "sqlite-devel",
        "libncurses-dev" => "ncurses-devel",
        "libreadline-dev" => "readline-devel",
        "uuid-dev" => "libuuid-devel",
        "libgdbm-dev" => "gdbm-devel",
    })
}

// =========================================================================
// YUM mappings (older RHEL/CentOS)
// =========================================================================

fn map_yum(name: &str) -> String {
    pkg_map!(name, {
        // Build essentials
        "build-essential" => "@development-tools",
        "g++" => "gcc-c++",
        "pkg-config" => "pkgconf",

        // Python development
        "python3-dev" => "python3-devel",
        "python3-venv" => "python3",

        // ROCm tools
        "rocm-smi-lib" => "rocm-smi",

        // ROCm libraries
        "librccl-dev" => "rccl-devel",
        "rccl-dev" => "rccl-devel",
        "migraphx-dev" => "migraphx-devel",

        // MPI
        "libopenmpi-dev" => "openmpi-devel",
        "openmpi" => "openmpi openmpi-devel",
        "openmpi-devel" => "openmpi-devel",

        // Build tools
        "llvm-dev" => "llvm-devel",
        "libssl-dev" => "openssl-devel",
        "libffi-dev" => "libffi-devel",
        "zlib1g-dev" => "zlib-devel",

        // System utilities
        "mesa-utils" => "mesa-demos",
        "libnuma-dev" => "numactl-devel",

        // Network
        "gnupg" => "gnupg2",
        "ssh" => "openssh-clients",

        // Libraries
        "libbz2-dev" => "bzip2-devel",
        "liblzma-dev" => "xz-devel",
        "libsqlite3-dev" => "sqlite-devel",
        "libncurses-dev" => "ncurses-devel",
        "libreadline-dev" => "readline-devel",
        "uuid-dev" => "libuuid-devel",
        "libgdbm-dev" => "gdbm-devel",
    })
}

// =========================================================================
// Zypper mappings (openSUSE/SLES)
// =========================================================================

fn map_zypper(name: &str) -> String {
    pkg_map!(name, {
        // Build essentials
        "build-essential" => "patterns-devel-base-devel_basis",
        "g++" => "gcc-c++",
        "ninja-build" => "ninja",
        "pkg-config" => "pkgconf",

        // Python development
        "python3-dev" => "python3-devel",
        "python3-venv" => "python3",

        // ROCm tools
        "rocm-smi-lib" => "rocm-smi",

        // ROCm libraries
        "librccl-dev" => "rccl-devel",
        "rccl-dev" => "rccl-devel",
        "migraphx-dev" => "migraphx-devel",

        // MPI
        "libopenmpi-dev" => "openmpi-devel",
        "openmpi" => "openmpi openmpi-devel",
        "openmpi-devel" => "openmpi-devel",

        // Build tools
        "llvm-dev" => "llvm-devel",
        "libssl-dev" => "libopenssl-devel",
        "libffi-dev" => "libffi-devel",
        "zlib1g-dev" => "zlib-devel",

        // System utilities
        "mesa-utils" => "Mesa-demo",
        "libnuma-dev" => "libnuma-devel",

        // Network
        "gnupg" => "gpg2",
        "ssh" => "openssh",

        // Libraries
        "libbz2-dev" => "libbz2-devel",
        "liblzma-dev" => "xz-devel",
        "libsqlite3-dev" => "sqlite3-devel",
        "libncurses-dev" => "ncurses-devel",
        "libreadline-dev" => "readline-devel",
        "uuid-dev" => "libuuid-devel",
        "libgdbm-dev" => "gdbm-devel",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // =====================================================================
    // VAL-INFRA-004: Package name mapping correct
    // =====================================================================

    // --- Build Essentials ---

    #[test]
    fn test_build_essential_apt() {
        assert_eq!(
            map_package_name("build-essential", PackageManager::Apt),
            "build-essential"
        );
    }
    #[test]
    fn test_build_essential_pacman() {
        assert_eq!(
            map_package_name("build-essential", PackageManager::Pacman),
            "base-devel"
        );
    }
    #[test]
    fn test_build_essential_dnf() {
        assert_eq!(
            map_package_name("build-essential", PackageManager::Dnf),
            "@development-tools"
        );
    }
    #[test]
    fn test_build_essential_yum() {
        assert_eq!(
            map_package_name("build-essential", PackageManager::Yum),
            "@development-tools"
        );
    }
    #[test]
    fn test_build_essential_zypper() {
        assert_eq!(
            map_package_name("build-essential", PackageManager::Zypper),
            "patterns-devel-base-devel_basis"
        );
    }

    // --- g++ ---
    #[test]
    fn test_gplusplus_pacman() {
        assert_eq!(map_package_name("g++", PackageManager::Pacman), "gcc");
    }
    #[test]
    fn test_gplusplus_dnf() {
        assert_eq!(map_package_name("g++", PackageManager::Dnf), "gcc-c++");
    }

    // --- pkg-config ---
    #[test]
    fn test_pkg_config_pacman() {
        assert_eq!(
            map_package_name("pkg-config", PackageManager::Pacman),
            "pkgconf"
        );
    }
    #[test]
    fn test_pkg_config_dnf() {
        assert_eq!(
            map_package_name("pkg-config", PackageManager::Dnf),
            "pkgconf"
        );
    }

    // --- ninja-build ---
    #[test]
    fn test_ninja_build_pacman() {
        assert_eq!(
            map_package_name("ninja-build", PackageManager::Pacman),
            "ninja"
        );
    }

    // --- Python Development ---

    #[test]
    fn test_python3_pacman() {
        assert_eq!(
            map_package_name("python3", PackageManager::Pacman),
            "python"
        );
    }
    #[test]
    fn test_python3_dev_pacman() {
        assert_eq!(
            map_package_name("python3-dev", PackageManager::Pacman),
            "python"
        );
    }
    #[test]
    fn test_python3_dev_dnf() {
        assert_eq!(
            map_package_name("python3-dev", PackageManager::Dnf),
            "python3-devel"
        );
    }
    #[test]
    fn test_python3_dev_zypper() {
        assert_eq!(
            map_package_name("python3-dev", PackageManager::Zypper),
            "python3-devel"
        );
    }
    #[test]
    fn test_python3_pip_pacman() {
        assert_eq!(
            map_package_name("python3-pip", PackageManager::Pacman),
            "python-pip"
        );
    }
    #[test]
    fn test_python3_venv_pacman() {
        assert_eq!(
            map_package_name("python3-venv", PackageManager::Pacman),
            "python"
        );
    }

    // --- ROCm Tools ---

    #[test]
    fn test_hip_runtime_apt() {
        assert_eq!(
            map_package_name("hip-runtime", PackageManager::Apt),
            "hip-runtime-amd"
        );
    }
    #[test]
    fn test_hip_runtime_pacman() {
        assert_eq!(
            map_package_name("hip-runtime", PackageManager::Pacman),
            "hip-runtime-amd"
        );
    }
    #[test]
    fn test_rocm_smi_lib_dnf() {
        assert_eq!(
            map_package_name("rocm-smi-lib", PackageManager::Dnf),
            "rocm-smi"
        );
    }

    // --- ROCm Libraries ---

    #[test]
    fn test_librccl_dev_pacman() {
        assert_eq!(
            map_package_name("librccl-dev", PackageManager::Pacman),
            "rccl"
        );
    }
    #[test]
    fn test_librccl_dev_dnf() {
        assert_eq!(
            map_package_name("librccl-dev", PackageManager::Dnf),
            "rccl-devel"
        );
    }
    #[test]
    fn test_migraphx_dev_dnf() {
        assert_eq!(
            map_package_name("migraphx-dev", PackageManager::Dnf),
            "migraphx-devel"
        );
    }
    #[test]
    fn test_migraphx_dev_pacman() {
        assert_eq!(
            map_package_name("migraphx-dev", PackageManager::Pacman),
            "migraphx"
        );
    }

    // --- MPI ---

    #[test]
    fn test_libopenmpi_dev_pacman() {
        assert_eq!(
            map_package_name("libopenmpi-dev", PackageManager::Pacman),
            "openmpi"
        );
    }
    #[test]
    fn test_libopenmpi_dev_dnf() {
        assert_eq!(
            map_package_name("libopenmpi-dev", PackageManager::Dnf),
            "openmpi-devel"
        );
    }
    #[test]
    fn test_openmpi_apt() {
        assert_eq!(
            map_package_name("openmpi", PackageManager::Apt),
            "openmpi-bin libopenmpi-dev"
        );
    }
    #[test]
    fn test_openmpi_dnf() {
        assert_eq!(
            map_package_name("openmpi", PackageManager::Dnf),
            "openmpi openmpi-devel"
        );
    }

    // --- Build Tools ---

    #[test]
    fn test_llvm_dev_pacman() {
        assert_eq!(map_package_name("llvm-dev", PackageManager::Pacman), "llvm");
    }
    #[test]
    fn test_llvm_dev_dnf() {
        assert_eq!(
            map_package_name("llvm-dev", PackageManager::Dnf),
            "llvm-devel"
        );
    }
    #[test]
    fn test_libssl_dev_pacman() {
        assert_eq!(
            map_package_name("libssl-dev", PackageManager::Pacman),
            "openssl"
        );
    }
    #[test]
    fn test_libssl_dev_dnf() {
        assert_eq!(
            map_package_name("libssl-dev", PackageManager::Dnf),
            "openssl-devel"
        );
    }
    #[test]
    fn test_libssl_dev_zypper() {
        assert_eq!(
            map_package_name("libssl-dev", PackageManager::Zypper),
            "libopenssl-devel"
        );
    }
    #[test]
    fn test_libffi_dev_dnf() {
        assert_eq!(
            map_package_name("libffi-dev", PackageManager::Dnf),
            "libffi-devel"
        );
    }
    #[test]
    fn test_zlib1g_dev_pacman() {
        assert_eq!(
            map_package_name("zlib1g-dev", PackageManager::Pacman),
            "zlib"
        );
    }
    #[test]
    fn test_zlib1g_dev_dnf() {
        assert_eq!(
            map_package_name("zlib1g-dev", PackageManager::Dnf),
            "zlib-devel"
        );
    }

    // --- System Utilities ---

    #[test]
    fn test_mesa_utils_pacman() {
        assert_eq!(
            map_package_name("mesa-utils", PackageManager::Pacman),
            "mesa-demos"
        );
    }
    #[test]
    fn test_mesa_utils_dnf() {
        assert_eq!(
            map_package_name("mesa-utils", PackageManager::Dnf),
            "mesa-demos"
        );
    }
    #[test]
    fn test_mesa_utils_zypper() {
        assert_eq!(
            map_package_name("mesa-utils", PackageManager::Zypper),
            "Mesa-demo"
        );
    }
    #[test]
    fn test_libnuma_dev_pacman() {
        assert_eq!(
            map_package_name("libnuma-dev", PackageManager::Pacman),
            "numactl"
        );
    }
    #[test]
    fn test_libnuma_dev_dnf() {
        assert_eq!(
            map_package_name("libnuma-dev", PackageManager::Dnf),
            "numactl-devel"
        );
    }

    // --- Version Control ---

    #[test]
    fn test_gh_pacman() {
        assert_eq!(map_package_name("gh", PackageManager::Pacman), "github-cli");
    }

    // --- Network ---

    #[test]
    fn test_gnupg_dnf() {
        assert_eq!(map_package_name("gnupg", PackageManager::Dnf), "gnupg2");
    }
    #[test]
    fn test_gnupg_zypper() {
        assert_eq!(map_package_name("gnupg", PackageManager::Zypper), "gpg2");
    }
    #[test]
    fn test_ssh_pacman() {
        assert_eq!(map_package_name("ssh", PackageManager::Pacman), "openssh");
    }
    #[test]
    fn test_ssh_dnf() {
        assert_eq!(
            map_package_name("ssh", PackageManager::Dnf),
            "openssh-clients"
        );
    }

    // --- Libraries ---

    #[test]
    fn test_libbz2_dev_pacman() {
        assert_eq!(
            map_package_name("libbz2-dev", PackageManager::Pacman),
            "bzip2"
        );
    }
    #[test]
    fn test_libbz2_dev_dnf() {
        assert_eq!(
            map_package_name("libbz2-dev", PackageManager::Dnf),
            "bzip2-devel"
        );
    }
    #[test]
    fn test_libsqlite3_dev_pacman() {
        assert_eq!(
            map_package_name("libsqlite3-dev", PackageManager::Pacman),
            "sqlite"
        );
    }
    #[test]
    fn test_libsqlite3_dev_zypper() {
        assert_eq!(
            map_package_name("libsqlite3-dev", PackageManager::Zypper),
            "sqlite3-devel"
        );
    }
    #[test]
    fn test_uuid_dev_pacman() {
        assert_eq!(
            map_package_name("uuid-dev", PackageManager::Pacman),
            "util-linux"
        );
    }
    #[test]
    fn test_uuid_dev_dnf() {
        assert_eq!(
            map_package_name("uuid-dev", PackageManager::Dnf),
            "libuuid-devel"
        );
    }

    // --- Unknown package returns original name ---

    #[test]
    fn test_unknown_package_returns_original() {
        assert_eq!(
            map_package_name("some-unknown-package", PackageManager::Apt),
            "some-unknown-package"
        );
        assert_eq!(
            map_package_name("some-unknown-package", PackageManager::Pacman),
            "some-unknown-package"
        );
    }

    #[test]
    fn test_unknown_package_manager_returns_original() {
        assert_eq!(
            map_package_name("build-essential", PackageManager::Unknown),
            "build-essential"
        );
    }

    // --- map_package_names batch ---

    #[test]
    fn test_map_package_names_batch() {
        let names = vec!["build-essential", "cmake", "python3-dev"];
        let mapped = map_package_names(&names, PackageManager::Pacman);
        assert_eq!(mapped, vec!["base-devel", "cmake", "python"]);
    }

    // --- Identity mappings (same on all platforms) ---

    #[test]
    fn test_identity_mappings() {
        // These should be the same on all platforms
        for pkg in &[
            "gcc", "make", "cmake", "git", "curl", "wget", "clang", "lld",
        ] {
            assert_eq!(map_package_name(pkg, PackageManager::Apt), *pkg);
            assert_eq!(map_package_name(pkg, PackageManager::Pacman), *pkg);
        }
    }

    // --- has_mapping ---

    #[test]
    fn test_has_mapping_known() {
        assert!(has_mapping("build-essential", PackageManager::Pacman));
    }

    #[test]
    fn test_has_mapping_identity() {
        // gcc maps to itself on all platforms, but it's still a known package
        assert!(has_mapping("gcc", PackageManager::Apt));
    }

    #[test]
    fn test_has_mapping_unknown() {
        assert!(!has_mapping("totally-fake-package", PackageManager::Apt));
    }

    // --- get_known_packages coverage ---

    #[test]
    fn test_known_packages_contains_key_packages() {
        let known = get_known_packages();
        assert!(known.contains(&"build-essential"));
        assert!(known.contains(&"python3-dev"));
        assert!(known.contains(&"libopenmpi-dev"));
        assert!(known.contains(&"git"));
        assert!(known.contains(&"libssl-dev"));
        assert!(known.contains(&"rocminfo"));
    }
}
