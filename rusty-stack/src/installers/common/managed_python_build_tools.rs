//! Rusty-owned Python tooling and import-time dependencies.
//!
//! Component installers consume these commands but never resolve or mutate their
//! own dependency graph. Every package is exact-pinned and installed without
//! dependency resolution so the managed ROCm environment remains authoritative.

/// Exact Python distributions owned by the Rusty environment.
///
/// Keep this order stable: command construction and audit logs intentionally
/// preserve it.
pub const MANAGED_PYTHON_PACKAGES: [(&str, &str); 11] = [
    ("setuptools", "81.0.0"),
    ("cmake", "4.3.4"),
    ("scikit-build-core", "1.0.3"),
    ("pathspec", "1.1.1"),
    ("wheel", "0.47.0"),
    ("ninja", "1.13.0"),
    ("imageio", "2.36.0"),
    ("diffusers", "0.33.1"),
    ("remote-pdb", "2.1.0"),
    ("ftfy", "6.3.1"),
    ("wcwidth", "0.2.13"),
];

/// A command targeting the Python interpreter selected by Rusty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedPythonCommand {
    /// Selected Python executable.
    pub program: String,
    /// Arguments passed directly to the executable.
    pub args: Vec<String>,
}

/// Build the exact, resolver-free install command for Rusty-owned Python packages.
///
/// The selected managed prefix is PEP 668 protected, so its existing pip call
/// pattern requires `--break-system-packages`. `--no-deps` is mandatory:
/// these pins are the complete centrally owned set and component dependency
/// resolution must never replace ROCm stack packages.
pub fn managed_python_install_command(python_bin: impl Into<String>) -> ManagedPythonCommand {
    let mut args = [
        "-m",
        "pip",
        "install",
        "--break-system-packages",
        "--no-cache-dir",
        "--no-deps",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    args.extend(
        MANAGED_PYTHON_PACKAGES
            .iter()
            .map(|(name, version)| format!("{name}=={version}")),
    );

    ManagedPythonCommand {
        program: python_bin.into(),
        args,
    }
}

/// Build a read-only command that verifies every Rusty-owned package version.
pub fn managed_python_verify_command(python_bin: impl Into<String>) -> ManagedPythonCommand {
    let expected = MANAGED_PYTHON_PACKAGES
        .iter()
        .map(|(name, version)| format!("    {name:?}: {version:?},"))
        .collect::<Vec<_>>()
        .join("\n");
    let script = format!(
        r#"import importlib.metadata

expected_versions = {{
{expected}
}}
problems = []
for package, expected_version in expected_versions.items():
    try:
        actual = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        problems.append(f"{{package}}: missing (expected {{expected_version}})")
        continue
    if actual != expected_version:
        problems.append(f"{{package}}: {{actual}} (expected {{expected_version}})")

if problems:
    raise SystemExit("Rusty managed Python package mismatch:\\n" + "\\n".join(problems))
print("Rusty managed Python package versions verified")
"#
    );

    ManagedPythonCommand {
        program: python_bin.into(),
        args: vec!["-c".to_string(), script],
    }
}

#[cfg(test)]
mod tests {
    use super::{
        managed_python_install_command, managed_python_verify_command, MANAGED_PYTHON_PACKAGES,
    };

    const EXPECTED_PACKAGES: [(&str, &str); 11] = [
        ("setuptools", "81.0.0"),
        ("cmake", "4.3.4"),
        ("scikit-build-core", "1.0.3"),
        ("pathspec", "1.1.1"),
        ("wheel", "0.47.0"),
        ("ninja", "1.13.0"),
        ("imageio", "2.36.0"),
        ("diffusers", "0.33.1"),
        ("remote-pdb", "2.1.0"),
        ("ftfy", "6.3.1"),
        ("wcwidth", "0.2.13"),
    ];

    #[test]
    fn install_uses_selected_python_and_exact_argument_order() {
        let command = managed_python_install_command("/managed/python3");
        let mut expected_args = vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
            "--break-system-packages".to_string(),
            "--no-cache-dir".to_string(),
            "--no-deps".to_string(),
        ];
        expected_args.extend(
            EXPECTED_PACKAGES
                .iter()
                .map(|(name, version)| format!("{name}=={version}")),
        );

        assert_eq!(command.program, "/managed/python3");
        assert_eq!(command.args, expected_args);
    }

    #[test]
    fn managed_packages_are_exact_and_resolver_safe() {
        assert_eq!(MANAGED_PYTHON_PACKAGES, EXPECTED_PACKAGES);

        let command = managed_python_install_command("selected-python");
        let package_args = &command.args[6..];
        assert_eq!(package_args.len(), EXPECTED_PACKAGES.len());
        assert!(package_args.iter().all(|package| {
            let mut pieces = package.split("==");
            matches!(
                (pieces.next(), pieces.next(), pieces.next()),
                (Some(name), Some(version), None) if !name.is_empty() && !version.is_empty()
            )
        }));

        let rendered = command.args.join(" ").to_ascii_lowercase();
        for forbidden in [
            "--upgrade",
            "--index-url",
            "--extra-index-url",
            "nvidia",
            "cuda",
            "cu11",
            "cu12",
        ] {
            assert!(
                !rendered.contains(forbidden),
                "managed install command contains forbidden token {forbidden}: {rendered}"
            );
        }
        assert!(command.args.iter().any(|arg| arg == "--no-deps"));
    }

    #[test]
    fn verification_is_read_only_and_checks_every_exact_version() {
        let command = managed_python_verify_command("/managed/python3");
        assert_eq!(command.program, "/managed/python3");
        assert_eq!(command.args.first().map(String::as_str), Some("-c"));
        assert_eq!(command.args.len(), 2);

        let script = &command.args[1];
        for (name, version) in EXPECTED_PACKAGES {
            assert!(script.contains(&format!("{name:?}: {version:?}")));
        }
        for forbidden in ["subprocess", "pip install", "--upgrade", "--index-url"] {
            assert!(
                !script.to_ascii_lowercase().contains(forbidden),
                "verification command must be read-only: found {forbidden}"
            );
        }
        assert!(script.contains("importlib.metadata"));
        assert!(script.contains("actual != expected_version"));
    }
}
