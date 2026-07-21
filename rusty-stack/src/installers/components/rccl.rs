//! Targeted workaround for the ROCm 7.2.1+ RCCL compiler regression.
//!
//! The distro RCCL package remains authoritative. This module only builds and
//! activates a sealed overlay when a two-GPU collective reproduces a known
//! ROCm/RCCL multi-GPU failure fingerprint.

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub const RCCL_COMMIT: &str = "96a25b5fd6f73fba58c7d83eb57cf19a50230aa4";
const RCCL_REPO: &str = "https://github.com/ROCm/rccl.git";
const LIBXML_URL: &str = "https://archive.ubuntu.com/ubuntu/pool/main/libx/libxml2/libxml2_2.9.13+dfsg-1ubuntu0.12_amd64.deb";
const LIBXML_SHA256: &str = "b3678e6e4b166bc0e4226fb118d489ab51802c772914421397c9dcb2dd0e0d2b";
const ICU_URL: &str = "https://archive.ubuntu.com/ubuntu/pool/main/i/icu/libicu70_70.1-2_amd64.deb";
const ICU_SHA256: &str = "58a154f6307289813da2276f900498ef536ae7c0522d2cf31a3c3c5cf62dfd9a";
const ISSUE_URL: &str = "https://github.com/ROCm/ROCm/issues/6074";

#[derive(Debug, Clone, Copy)]
struct RcclRepairProfile {
    id: &'static str,
    compiler_id: &'static str,
    compiler_url: &'static str,
    compiler_sha256: &'static str,
    compiler_size: u64,
    hipify_id: &'static str,
    hipify_url: &'static str,
    hipify_sha256: &'static str,
    hipify_size: u64,
}

const ROCM_72_REPAIR_PROFILE: RcclRepairProfile = RcclRepairProfile {
    id: "rocm-7.2.x",
    compiler_id: "rocm-llvm-7.2.0",
    compiler_url: "https://repo.radeon.com/rocm/apt/7.2/pool/main/r/rocm-llvm/rocm-llvm_22.0.0.26014.70200-43~22.04_amd64.deb",
    compiler_sha256: "88d604fe0ab4a2502f0c7e0f8d3b7b49e829ae3057971de681a68b2fdb2505d2",
    compiler_size: 549_321_042,
    hipify_id: "hipify-rocm-7.2.0",
    hipify_url: "https://raw.githubusercontent.com/ROCm/HIPIFY/refs/tags/rocm-7.2.0/bin/hipify-perl",
    hipify_sha256: "036b3b07a4fe9a921d220f21a1fa5a3cfc22343bb0417d729015b87ae0463291",
    hipify_size: 948_312,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellCommand {
    pub program: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct RcclInstaller {
    home: PathBuf,
    python_bin: PathBuf,
    rocm_path: PathBuf,
    rocm_version: String,
    rocm_channel: String,
    gpu_archs: Vec<String>,
    gpu_count: usize,
    rocr_tokens: Vec<String>,
}

impl RcclInstaller {
    pub fn new(
        home: impl Into<PathBuf>,
        python_bin: impl Into<PathBuf>,
        rocm_path: impl Into<PathBuf>,
        rocm_version: impl Into<String>,
        rocm_channel: impl Into<String>,
        rocr_visible_devices: Option<String>,
    ) -> Self {
        let gpus = crate::gpu::detect_discrete_amd_gpus();
        let gpu_archs = gpus
            .iter()
            .filter_map(|gpu| gpu.gfx_arch.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let rocr_tokens = rocr_visible_devices
            .as_deref()
            .map(parse_visibility_tokens)
            .filter(|tokens| !tokens.is_empty())
            .unwrap_or_else(|| (0..gpus.len()).map(|idx| idx.to_string()).collect());
        Self {
            home: home.into(),
            python_bin: python_bin.into(),
            rocm_path: rocm_path.into(),
            rocm_version: rocm_version.into(),
            rocm_channel: rocm_channel.into(),
            gpu_count: gpus.len(),
            gpu_archs,
            rocr_tokens,
        }
    }

    /// Probe the installed RCCL and return a build command only for ROCm#6074.
    pub fn prepare_remediation(&self) -> Result<Option<ShellCommand>> {
        if self.gpu_count < 2 {
            return Ok(None);
        }
        if self.gpu_archs.is_empty() {
            bail!(
                "RCCL remediation requires detected gfx architectures; refusing a guessed target"
            );
        }
        if self.rocr_tokens.len() < 2 {
            bail!("RCCL remediation requires at least two ROCr visibility tokens");
        }

        match self.active_overlay_state()? {
            OverlayState::Valid { version_dir, sha } => {
                let probe = self.run_probe(Some((&version_dir, &sha)))?;
                if overlay_probe_succeeded(&probe) {
                    return Ok(None);
                }
                if !unseal_requested() {
                    bail!(
                        "sealed RCCL overlay failed functional revalidation; set MLSTACK_UNSEAL_CORE=1 to rebuild it:\n{}",
                        output_text(&probe)
                    );
                }
            }
            OverlayState::Invalid(reason) if !unseal_requested() => {
                bail!(
                    "sealed RCCL overlay is invalid ({reason}); set MLSTACK_UNSEAL_CORE=1 to rebuild it"
                )
            }
            OverlayState::Invalid(_) | OverlayState::Missing => {}
        }

        let probe = self.run_probe(None)?;
        if probe.status.success() {
            return Ok(None);
        }
        let output = output_text(&probe);
        if !is_compiler_regression(&output) {
            bail!("RCCL multi-GPU probe failed outside the known ROCm/RCCL collective failure fingerprints:\n{output}");
        }

        let profile = self.repair_profile()?;
        let compiler = self.ensure_toolchain(profile)?;
        let hipify_perl = self.ensure_hipify_perl(profile)?;
        Ok(Some(self.build_command(&compiler, &hipify_perl)?))
    }

    /// Publish the completed build as a read-only overlay and prove both ranks use it.
    pub fn activate_and_verify(&self) -> Result<()> {
        let profile = self.repair_profile()?;
        let source = self.source_dir();
        let built = find_named(&source.join("build"), "librccl.so.1.0")
            .or_else(|| find_named(&source.join("build"), "librccl.so.1"))
            .context("RCCL build completed without librccl.so.1.0")?;
        let sha = sha256_file(&built)?;
        let version_name = format!("{}-{}", &RCCL_COMMIT[..12], profile.compiler_id);
        let root = self.overlay_root();
        let version_dir = root.join(&version_name);
        let staging = root.join(format!(".{version_name}.tmp-{}", std::process::id()));

        fs::create_dir_all(&root)?;
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        if version_dir.exists() {
            if !unseal_requested() {
                bail!(
                    "RCCL overlay version already exists but was not reusable; refusing overwrite"
                );
            }
            make_tree_writable(&version_dir)?;
            fs::remove_dir_all(&version_dir)?;
        }

        let lib_dir = staging.join("lib");
        let py_dir = staging.join("python");
        fs::create_dir_all(&lib_dir)?;
        fs::create_dir_all(&py_dir)?;
        fs::copy(&built, lib_dir.join("librccl.so.1.0"))?;
        make_symlink("librccl.so.1.0", &lib_dir.join("librccl.so.1"))?;
        make_symlink("librccl.so.1", &lib_dir.join("librccl.so"))?;
        fs::write(py_dir.join("sitecustomize.py"), SITECUSTOMIZE_SOURCE)?;

        let archs = self.gpu_archs.join(";");
        fs::write(
            staging.join("manifest"),
            format!(
                "source={RCCL_REPO}\ncommit={RCCL_COMMIT}\nprofile={}\ncompiler={}\nhipify={}\narchitectures={archs}\nsha256={sha}\nissue={ISSUE_URL}\n",
                profile.id, profile.compiler_id, profile.hipify_id
            ),
        )?;
        let suffix = concat!("$", "{PYTHONPATH:+:$PYTHONPATH}");
        fs::write(
            staging.join("env.sh"),
            format!(
                "export MLSTACK_RCCL_OVERLAY_LIB={}\nexport MLSTACK_RCCL_OVERLAY_SHA256={sha}\nexport NCCL_P2P_DISABLE=1\nexport RCCL_P2P_DISABLE=1\nexport PYTHONPATH={}{suffix}\n",
                shell_quote_path(&version_dir.join("lib/librccl.so.1.0")),
                shell_quote_path(&version_dir.join("python"))
            ),
        )?;
        fs::write(
            staging.join("env.fish"),
            format!(
                "set -gx MLSTACK_RCCL_OVERLAY_LIB {}\nset -gx MLSTACK_RCCL_OVERLAY_SHA256 {sha}\nset -gx NCCL_P2P_DISABLE 1\nset -gx RCCL_P2P_DISABLE 1\nset -gx PYTHONPATH {} $PYTHONPATH\n",
                fish_quote_path(&version_dir.join("lib/librccl.so.1.0")),
                fish_quote_path(&version_dir.join("python"))
            ),
        )?;

        set_tree_read_only(&staging)?;
        fs::rename(&staging, &version_dir)?;

        let probe = self.run_probe(Some((&version_dir, &sha)))?;
        let output = output_text(&probe);
        if !overlay_probe_succeeded(&probe) {
            make_tree_writable(&version_dir)?;
            fs::remove_dir_all(&version_dir)?;
            bail!("corrected RCCL overlay failed final two-GPU validation:\n{output}");
        }

        let active_tmp = root.join(".active.tmp");
        if active_tmp.exists() {
            fs::remove_file(&active_tmp)?;
        }
        make_symlink(&version_name, &active_tmp)?;
        fs::rename(&active_tmp, root.join("active"))?;
        Ok(())
    }

    fn overlay_root(&self) -> PathBuf {
        self.home.join(".mlstack/components/rccl")
    }

    fn source_dir(&self) -> PathBuf {
        self.home
            .join(".mlstack/src")
            .join(format!("rccl-{}", &RCCL_COMMIT[..12]))
    }

    fn active_overlay_state(&self) -> Result<OverlayState> {
        let active = self.overlay_root().join("active");
        let metadata = match fs::symlink_metadata(&active) {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Ok(OverlayState::Missing);
            }
            Err(err) => return Err(err.into()),
        };
        if !metadata.file_type().is_symlink() {
            return Ok(OverlayState::Invalid(
                "active marker is not a symlink".into(),
            ));
        }
        if !active.exists() {
            return Ok(OverlayState::Invalid(
                "active symlink target is missing".into(),
            ));
        }
        let manifest = active.join("manifest");
        let library = active.join("lib/librccl.so.1.0");
        if !manifest.is_file() || !library.is_file() {
            return Ok(OverlayState::Invalid("missing manifest or library".into()));
        }
        let text = fs::read_to_string(manifest)?;
        let expected = text
            .lines()
            .find_map(|line| line.strip_prefix("sha256="))
            .unwrap_or_default();
        if expected.is_empty() || sha256_file(&library)? != expected {
            return Ok(OverlayState::Invalid("SHA-256 mismatch".into()));
        }
        if !active.join("python/sitecustomize.py").is_file()
            || !active.join("env.sh").is_file()
            || !active.join("env.fish").is_file()
        {
            return Ok(OverlayState::Invalid("activation files missing".into()));
        }
        Ok(OverlayState::Valid {
            version_dir: active,
            sha: expected.to_string(),
        })
    }

    fn probe_path(&self) -> PathBuf {
        self.home.join(".mlstack/cache/rccl-two-rank-probe.py")
    }

    fn run_probe(&self, overlay: Option<(&Path, &str)>) -> Result<Output> {
        let path = self.probe_path();
        fs::create_dir_all(path.parent().context("probe cache has no parent")?)?;
        fs::write(&path, PROBE_SOURCE)?;

        let mut command = Command::new(&self.python_bin);
        command.arg(&path);
        command.env_remove("HIP_VISIBLE_DEVICES");
        command.env_remove("CUDA_VISIBLE_DEVICES");
        command.env_remove("GPU_DEVICE_ORDINAL");
        command.env_remove("MLSTACK_RCCL_OVERLAY_LIB");
        command.env_remove("MLSTACK_RCCL_OVERLAY_SHA256");
        command.env_remove("MLSTACK_RCCL_LOADER_PID");
        command.env("ROCR_VISIBLE_DEVICES", self.rocr_tokens.join(","));
        if let Some((version_dir, sha)) = overlay {
            command.env(
                "MLSTACK_RCCL_OVERLAY_LIB",
                version_dir.join("lib/librccl.so.1.0"),
            );
            command.env("MLSTACK_RCCL_OVERLAY_SHA256", sha);
            command.env("NCCL_P2P_DISABLE", "1");
            command.env("RCCL_P2P_DISABLE", "1");
            let existing = std::env::var_os("PYTHONPATH").unwrap_or_default();
            let mut paths = vec![version_dir.join("python")];
            paths.extend(std::env::split_paths(&existing));
            command.env("PYTHONPATH", std::env::join_paths(paths)?);
        }
        command.output().with_context(|| {
            format!(
                "failed to launch RCCL probe with {}",
                self.python_bin.display()
            )
        })
    }

    fn repair_profile(&self) -> Result<&'static RcclRepairProfile> {
        if self.rocm_channel.eq_ignore_ascii_case("latest") && self.rocm_version.starts_with("7.2")
        {
            return Ok(&ROCM_72_REPAIR_PROFILE);
        }
        bail!(
            "RCCL repair has no pinned profile for ROCm channel '{}' version '{}'",
            self.rocm_channel,
            self.rocm_version
        )
    }

    fn ensure_toolchain(&self, profile: &RcclRepairProfile) -> Result<PathBuf> {
        let root = self
            .home
            .join(format!(".mlstack/toolchains/{}", profile.compiler_id));
        if let Some(clang) = find_named(&root, "clang++") {
            if compiler_works(&clang, &compat_library_path(&root)) {
                return Ok(clang);
            }
        }

        let cache = self.home.join(".mlstack/cache/downloads");
        fs::create_dir_all(&cache)?;
        let compiler_deb = cache.join(format!("{}.deb", profile.compiler_id));
        download_pinned(
            profile.compiler_url,
            profile.compiler_sha256,
            Some(profile.compiler_size),
            &compiler_deb,
        )?;
        fs::create_dir_all(&root)?;
        extract_deb(&compiler_deb, &root)?;

        let clang = find_named(&root, "clang++").context("pinned ROCm compiler lacks clang++")?;
        if !compiler_works(&clang, "") {
            let compat = root.join("compat");
            fs::create_dir_all(&compat)?;
            let xml = cache.join("libxml2-2.9.13.deb");
            let icu = cache.join("libicu70.deb");
            download_pinned(LIBXML_URL, LIBXML_SHA256, None, &xml)?;
            download_pinned(ICU_URL, ICU_SHA256, None, &icu)?;
            extract_deb(&xml, &compat)?;
            extract_deb(&icu, &compat)?;
        }
        if !compiler_works(&clang, &compat_library_path(&root)) {
            bail!(
                "pinned ROCm compiler profile {} cannot run, even with isolated compatibility DSOs",
                profile.id
            );
        }
        Ok(clang)
    }

    fn ensure_hipify_perl(&self, profile: &RcclRepairProfile) -> Result<PathBuf> {
        let tool = self.home.join(format!(
            ".mlstack/toolchains/{}/bin/hipify-perl",
            profile.hipify_id
        ));
        if tool.is_file() && sha256_file(&tool)?.eq_ignore_ascii_case(profile.hipify_sha256) {
            make_executable(&tool)?;
            return Ok(tool);
        }

        let cache = self.home.join(".mlstack/cache/downloads");
        fs::create_dir_all(&cache)?;
        let cached = cache.join(profile.hipify_id);
        download_pinned(
            profile.hipify_url,
            profile.hipify_sha256,
            Some(profile.hipify_size),
            &cached,
        )?;
        fs::create_dir_all(tool.parent().context("hipify tool path has no parent")?)?;
        fs::copy(&cached, &tool)?;
        make_executable(&tool)?;
        if !sha256_file(&tool)?.eq_ignore_ascii_case(profile.hipify_sha256) {
            bail!("cached hipify-perl SHA-256 changed during installation");
        }
        Ok(tool)
    }

    fn build_command(&self, compiler: &Path, hipify_perl: &Path) -> Result<ShellCommand> {
        let source = self.source_dir();
        fs::create_dir_all(source.parent().context("RCCL source has no parent")?)?;
        let archs = self.gpu_archs.join(";");
        let profile = self.repair_profile()?;
        let compat = compat_library_path(
            &self
                .home
                .join(format!(".mlstack/toolchains/{}", profile.compiler_id)),
        );
        let script = format!(
            r#"set -eu
if [ ! -d {src}/.git ]; then git clone --no-checkout {repo} {src}; fi
git -C {src} fetch --depth 1 origin {commit}
git -C {src} checkout --detach --force {commit}
test "$(git -C {src} rev-parse HEAD)" = {commit}
rm -rf {src}/build
git -C {src} diff --quiet
test -z "$(git -C {src} status --porcelain --untracked-files=all)"
CC={cc} CXX={cxx} CFLAGS=--rocm-path={rocm} CXXFLAGS=--rocm-path={rocm} \
cmake -S {src} -B {src}/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF \
-DROCM_PATH={rocm} -DCMAKE_PREFIX_PATH={rocm} -DCMAKE_C_COMPILER={cc} \
-DCMAKE_CXX_COMPILER={cxx} -DCMAKE_HIP_COMPILER={cxx} \
-Dhipify-perl_executable={hipify} \
-DGPU_TARGETS={archs_q} -DAMDGPU_TARGETS={archs_q}
cmake --build {src}/build --parallel 2
"#,
            src = shell_quote_path(&source),
            repo = shell_quote(RCCL_REPO),
            commit = shell_quote(RCCL_COMMIT),
            cc = shell_quote_path(&compiler.with_file_name("clang")),
            cxx = shell_quote_path(compiler),
            hipify = shell_quote_path(hipify_perl),
            rocm = shell_quote_path(&self.rocm_path),
            archs_q = shell_quote(&archs),
        );
        let mut env = Vec::new();
        if !compat.is_empty() {
            let inherited = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
            env.push((
                "LD_LIBRARY_PATH".into(),
                if inherited.is_empty() {
                    compat
                } else {
                    format!("{compat}:{inherited}")
                },
            ));
        }
        Ok(ShellCommand {
            program: "sh".into(),
            args: vec!["-c".into(), script],
            env,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum OverlayState {
    Missing,
    Valid { version_dir: PathBuf, sha: String },
    Invalid(String),
}

pub fn is_compiler_regression(output: &str) -> bool {
    let lower = output.to_ascii_lowercase();
    let collective_context =
        (lower.contains("rccl") || lower.contains("nccl") || lower.contains("processgroup"))
            && (lower.contains("distributed")
                || lower.contains("all_reduce")
                || lower.contains("allreduce"));
    let present_state_error = lower.contains("operation cannot be performed in present state")
        || lower.contains("the operation cannot be performed in the present state");
    let invalid_pointer_error = lower.contains("invalid device pointer")
        && (lower.contains("ncclunhandledcudaerror")
            || lower.contains("hip failure")
            || lower.contains("unhandled cuda error"));
    collective_context && (present_state_error || invalid_pointer_error)
}

fn output_text(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn parse_visibility_tokens(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect()
}

fn overlay_probe_succeeded(output: &Output) -> bool {
    if !output.status.success() {
        return false;
    }
    let text = output_text(output);
    text.matches("RCCL_OVERLAY_OK").count() >= 2 && text.matches("sum=3.0").count() >= 2
}

fn unseal_requested() -> bool {
    std::env::var("MLSTACK_UNSEAL_CORE").as_deref() == Ok("1")
}

fn download_pinned(url: &str, expected: &str, size: Option<u64>, dest: &Path) -> Result<()> {
    if dest.is_file()
        && sha256_file(dest)?.eq_ignore_ascii_case(expected)
        && size.is_none_or(|n| dest.metadata().map(|m| m.len() == n).unwrap_or(false))
    {
        return Ok(());
    }
    let part = dest.with_extension("part");
    let _ = fs::remove_file(&part);
    let response = ureq::get(url)
        .header("User-Agent", "rusty-stack-rccl-remediation")
        .call()
        .with_context(|| format!("failed to download pinned artifact {url}"))?;
    let mut reader = response.into_body().into_reader();
    let mut file = fs::File::create(&part)?;
    std::io::copy(&mut reader, &mut file)?;
    file.flush()?;
    file.sync_all()?;
    if size.is_some_and(|n| part.metadata().map(|m| m.len() != n).unwrap_or(true)) {
        let _ = fs::remove_file(&part);
        bail!("pinned artifact size mismatch for {url}");
    }
    let actual = sha256_file(&part)?;
    if !actual.eq_ignore_ascii_case(expected) {
        let _ = fs::remove_file(&part);
        bail!("pinned artifact SHA-256 mismatch for {url}: expected {expected}, got {actual}");
    }
    fs::rename(part, dest)?;
    Ok(())
}

fn extract_deb(deb: &Path, dest: &Path) -> Result<()> {
    let temp = tempfile::tempdir()?;
    let status = Command::new("ar")
        .arg("x")
        .arg(deb)
        .current_dir(temp.path())
        .status()
        .context("ar is required to extract the pinned ROCm toolchain")?;
    if !status.success() {
        bail!("failed to extract {}", deb.display());
    }
    let data = fs::read_dir(temp.path())?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .find(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("data.tar"))
        })
        .context("Debian artifact has no data archive")?;
    let status = Command::new("tar")
        .args(["-xf"])
        .arg(data)
        .arg("-C")
        .arg(dest)
        .status()
        .context("tar is required to extract the pinned ROCm toolchain")?;
    if !status.success() {
        bail!("failed to unpack {}", deb.display());
    }
    Ok(())
}

fn compiler_works(clang: &Path, library_path: &str) -> bool {
    let lld = clang.with_file_name("ld.lld");
    let mut cmd = Command::new(if lld.is_file() { &lld } else { clang });
    cmd.arg("--version");
    if !library_path.is_empty() {
        cmd.env("LD_LIBRARY_PATH", library_path);
    }
    cmd.output().is_ok_and(|output| output.status.success())
}

fn compat_library_path(root: &Path) -> String {
    let compat = root.join("compat");
    let mut dirs = BTreeSet::new();
    collect_library_dirs(&compat, &mut dirs);
    std::env::join_paths(dirs)
        .ok()
        .and_then(|p| p.into_string().ok())
        .unwrap_or_default()
}

fn collect_library_dirs(path: &Path, dirs: &mut BTreeSet<PathBuf>) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let child = entry.path();
        if child.is_dir() {
            collect_library_dirs(&child, dirs);
        } else if child
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("libxml2.so") || n.starts_with("libicu"))
        {
            if let Some(parent) = child.parent() {
                dirs.insert(parent.to_path_buf());
            }
        }
    }
}

fn find_named(root: &Path, name: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_named(&path, name) {
                return Some(found);
            }
        } else if path.file_name().and_then(|n| n.to_str()) == Some(name) {
            return Some(path);
        }
    }
    None
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[cfg(unix)]
fn make_symlink(target: impl AsRef<Path>, link: &Path) -> Result<()> {
    std::os::unix::fs::symlink(target, link)?;
    Ok(())
}

#[cfg(not(unix))]
fn make_symlink(_target: impl AsRef<Path>, _link: &Path) -> Result<()> {
    bail!("RCCL overlay activation requires Unix symbolic links")
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<()> {
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
    Ok(())
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<()> {
    bail!("RCCL hipify tool requires Unix executable permissions")
}

#[cfg(unix)]
fn set_tree_read_only(root: &Path) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            set_tree_read_only(&path)?;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o555))?;
        } else if !path.is_symlink() {
            fs::set_permissions(&path, fs::Permissions::from_mode(0o444))?;
        }
    }
    fs::set_permissions(root, fs::Permissions::from_mode(0o555))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_tree_read_only(_root: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn make_tree_writable(root: &Path) -> Result<()> {
    fs::set_permissions(root, fs::Permissions::from_mode(0o755))?;
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            make_tree_writable(&path)?;
        } else if !path.is_symlink() {
            fs::set_permissions(&path, fs::Permissions::from_mode(0o644))?;
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn make_tree_writable(_root: &Path) -> Result<()> {
    Ok(())
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn shell_quote_path(path: &Path) -> String {
    shell_quote(&path.to_string_lossy())
}

fn fish_quote_path(path: &Path) -> String {
    shell_quote_path(path)
}

const SITECUSTOMIZE_SOURCE: &str = r#"import hashlib
import importlib.machinery
import os
from pathlib import Path
import sys
import sysconfig

def _activate_rccl_overlay():
    lib = os.environ.get("MLSTACK_RCCL_OVERLAY_LIB")
    expected = os.environ.get("MLSTACK_RCCL_OVERLAY_SHA256")
    if not (lib and expected):
        return
    path = Path(lib).resolve()
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
    except OSError as exc:
        sys.stderr.write(f"rusty-stack: sealed RCCL overlay unreadable: {exc}\n")
        os._exit(126)
    if actual != expected:
        sys.stderr.write("rusty-stack: sealed RCCL overlay SHA-256 mismatch\n")
        os._exit(126)

    spec = importlib.machinery.PathFinder.find_spec("torch", sys.path)
    if spec is not None and spec.origin:
        torch_dir = Path(spec.origin).parent
        torch_lib = torch_dir / "lib"
        targets = list(torch_dir.glob("_C*.so"))
        targets += [torch_lib / name for name in (
            "libtorch_python.so", "libtorch.so", "libtorch_hip.so"
        ) if (torch_lib / name).exists()]
        loaders = (
            Path("/lib64/ld-linux-x86-64.so.2"),
            Path("/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"),
        )
        loader = next((candidate for candidate in loaders if candidate.exists()), None)
        if loader is None or not targets:
            sys.stderr.write("rusty-stack: cannot activate sealed RCCL overlay safely\n")
            os._exit(126)
        library_path = [str(path.parent), str(torch_lib)]
        libdir = sysconfig.get_config_var("LIBDIR")
        if libdir:
            library_path.append(libdir)
        if os.environ.get("LD_LIBRARY_PATH"):
            library_path.append(os.environ["LD_LIBRARY_PATH"])
        env = os.environ.copy()
        env["MLSTACK_RCCL_LOADER_PID"] = str(os.getpid())
        argv = list(getattr(sys, "orig_argv", [sys.executable, *sys.argv]))
        os.execve(str(loader), [
            str(loader), "--inhibit-rpath", ":".join(map(str, targets)),
            "--library-path", ":".join(library_path),
            sys.executable, *argv[1:]
        ], env)

if os.environ.get("MLSTACK_RCCL_LOADER_PID") == str(os.getpid()):
    os.environ.pop("MLSTACK_RCCL_LOADER_PID", None)
elif os.environ.get("MLSTACK_RCCL_OVERLAY_LIB") and os.environ.get("MLSTACK_RCCL_OVERLAY_SHA256"):
    class _RustyRcclImportHook:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch" or fullname.startswith("torch."):
                _activate_rccl_overlay()
            return None
    sys.meta_path.insert(0, _RustyRcclImportHook())
"#;

const PROBE_SOURCE: &str = r#"import os
from pathlib import Path
import subprocess
import sys
import tempfile

rank_source = r'''
import os
from pathlib import Path
import sys

rank = int(sys.argv[1])
rendezvous = sys.argv[2]
token = sys.argv[3]
os.environ["ROCR_VISIBLE_DEVICES"] = token
os.environ.pop("HIP_VISIBLE_DEVICES", None)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ.pop("GPU_DEVICE_ORDINAL", None)

import torch
import torch.distributed as dist

device_count = torch.cuda.device_count()
if device_count != 1:
    raise RuntimeError(f"rank {rank} expected one isolated GPU for ROCR token {token}, found {device_count}")
name = torch.cuda.get_device_name(0)

initialized = False
try:
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    initialized = True
    value = torch.tensor([float(rank + 1)], device="cuda:0")
    dist.all_reduce(value)
    torch.cuda.synchronize()
    loaded = [line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
              if "librccl.so" in line and "/" in line]
    expected = os.environ.get("MLSTACK_RCCL_OVERLAY_LIB")
    if expected and not any(Path(item).resolve() == Path(expected).resolve() for item in loaded):
        raise RuntimeError(f"RCCL overlay not loaded: {loaded}")
    if value.item() != 3.0:
        raise RuntimeError(f"wrong collective sum {value.item()}")
    print(f"RCCL_OVERLAY_OK rank={rank} token={token} name={name} sum={value.item():.1f} lib={loaded}", flush=True)
finally:
    if initialized:
        dist.destroy_process_group()
'''

def rank_env(token):
    env = os.environ.copy()
    env["ROCR_VISIBLE_DEVICES"] = token
    env.pop("HIP_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("GPU_DEVICE_ORDINAL", None)
    env.pop("MLSTACK_RCCL_LOADER_PID", None)
    return env

if __name__ == "__main__":
    devices = [item.strip() for item in os.environ.get("ROCR_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(devices) < 2:
        raise SystemExit("RCCL probe requires at least two ROCr visibility tokens")
    fd, rendezvous = tempfile.mkstemp(prefix="rusty-rccl-")
    os.close(fd)
    os.unlink(rendezvous)
    procs = []
    try:
        for rank, token in enumerate(devices[:2]):
            procs.append((
                rank,
                token,
                subprocess.Popen(
                    [sys.executable, "-c", rank_source, str(rank), rendezvous, token],
                    env=rank_env(token),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ),
            ))
        output = []
        failed = False
        for rank, token, proc in procs:
            try:
                stdout, stderr = proc.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                failed = True
                stderr += f"\nrank {rank} token {token} timed out\n"
            output.append(stdout)
            output.append(stderr)
            if proc.returncode != 0:
                failed = True
        text = "".join(output)
        print(text, end="")
        if failed:
            raise SystemExit(1)
        for rank, token in enumerate(devices[:2]):
            if f"RCCL_OVERLAY_OK rank={rank} token={token} " not in text:
                raise SystemExit(f"rank {rank} token {token} did not report success")
    finally:
        for _, _, proc in procs:
            if proc.poll() is None:
                proc.kill()
        Path(rendezvous).unlink(missing_ok=True)
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_only_known_rocm_rccl_collective_failures() {
        assert!(is_compiler_regression(
            "torch.distributed ProcessGroupNCCL all_reduce: operation cannot be performed in present state"
        ));
        assert!(is_compiler_regression(
            "torch.distributed all_reduce NCCL error ncclUnhandledCudaError HIP failure: invalid device pointer"
        ));
        assert!(!is_compiler_regression(
            "operation cannot be performed in present state"
        ));
        assert!(!is_compiler_regression(
            "HIP failure: invalid device pointer"
        ));
        assert!(!is_compiler_regression("distributed nccl timeout"));
    }

    #[test]
    fn remediation_is_pinned_amd_only_and_fail_closed() {
        assert!(ROCM_72_REPAIR_PROFILE
            .compiler_url
            .starts_with("https://repo.radeon.com/rocm/"));
        assert!(ROCM_72_REPAIR_PROFILE
            .hipify_url
            .starts_with("https://raw.githubusercontent.com/ROCm/HIPIFY/"));
        assert!(LIBXML_URL.starts_with("https://archive.ubuntu.com/ubuntu/"));
        assert!(ICU_URL.starts_with("https://archive.ubuntu.com/ubuntu/"));
        assert!(!SITECUSTOMIZE_SOURCE.contains("LD_PRELOAD"));
        assert!(SITECUSTOMIZE_SOURCE.contains("--inhibit-rpath"));
        assert!(SITECUSTOMIZE_SOURCE.contains("sha256"));
        assert!(SITECUSTOMIZE_SOURCE.contains("MLSTACK_RCCL_LOADER_PID"));
        assert!(SITECUSTOMIZE_SOURCE.contains("os.getpid()"));
        assert!(SITECUSTOMIZE_SOURCE.contains("sys.meta_path.insert"));
        assert!(PROBE_SOURCE.contains("ROCR_VISIBLE_DEVICES"));
        assert!(PROBE_SOURCE.contains("RCCL_OVERLAY_OK"));
        assert!(PROBE_SOURCE.contains("pop(\"HIP_VISIBLE_DEVICES\""));
        assert!(PROBE_SOURCE.contains("pop(\"CUDA_VISIBLE_DEVICES\""));
        assert!(PROBE_SOURCE.contains("pop(\"GPU_DEVICE_ORDINAL\""));
        assert!(PROBE_SOURCE.contains("subprocess.Popen"));
        assert!(!PROBE_SOURCE.contains("torch.multiprocessing"));
    }

    #[test]
    fn overlay_lives_outside_python_environment() {
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path().to_path_buf();
        let installer = RcclInstaller {
            python_bin: home.join(".mlstack/global/bin/python"),
            home: home.clone(),
            rocm_path: PathBuf::from("/opt/rocm"),
            rocm_version: "7.2.4".into(),
            rocm_channel: "latest".into(),
            gpu_archs: vec!["gfx1100".into(), "gfx1101".into()],
            gpu_count: 2,
            rocr_tokens: vec!["2".into(), "5".into()],
        };
        assert_eq!(
            installer.overlay_root(),
            home.join(".mlstack/components/rccl")
        );
        let command = installer
            .build_command(
                Path::new("/toolchain/bin/clang++"),
                Path::new("/toolchain/bin/hipify-perl"),
            )
            .unwrap();
        let script = command.args.last().unwrap();
        assert!(script.contains("BUILD_TESTS=OFF"));
        assert!(script.contains("--parallel 2"));
        assert!(script.contains("gfx1100;gfx1101"));
        assert!(script.contains("--rocm-path="));
        assert!(script.contains("-Dhipify-perl_executable="));
    }

    #[test]
    fn overlay_exports_safe_consumer_rdna_transport() {
        assert!(SITECUSTOMIZE_SOURCE.contains("MLSTACK_RCCL_OVERLAY_LIB"));
        let env_text = format!(
            "export MLSTACK_RCCL_OVERLAY_LIB={}\nexport MLSTACK_RCCL_OVERLAY_SHA256=sha\nexport NCCL_P2P_DISABLE=1\nexport RCCL_P2P_DISABLE=1\n",
            shell_quote_path(Path::new("/overlay/lib/librccl.so.1.0"))
        );
        assert!(env_text.contains("NCCL_P2P_DISABLE=1"));
        assert!(env_text.contains("RCCL_P2P_DISABLE=1"));
    }

    #[test]
    fn rccl_repair_profile_is_version_aware() {
        let installer = RcclInstaller {
            home: PathBuf::from("/home/test"),
            python_bin: PathBuf::from("python3"),
            rocm_path: PathBuf::from("/opt/rocm"),
            rocm_version: "7.2.4".into(),
            rocm_channel: "latest".into(),
            gpu_archs: vec!["gfx1100".into(), "gfx1101".into()],
            gpu_count: 2,
            rocr_tokens: vec!["0".into(), "1".into()],
        };
        assert_eq!(installer.repair_profile().unwrap().id, "rocm-7.2.x");

        let legacy = RcclInstaller {
            rocm_version: "6.4.3".into(),
            rocm_channel: "legacy".into(),
            ..installer
        };
        assert!(legacy.repair_profile().is_err());
    }

    #[test]
    fn visibility_tokens_are_trimmed_without_guessing() {
        assert_eq!(
            parse_visibility_tokens(" 2, 5 ,,"),
            vec!["2".to_string(), "5".to_string()]
        );
    }

    #[cfg(unix)]
    #[test]
    fn dangling_active_symlink_fails_closed() {
        let temp = tempfile::tempdir().unwrap();
        let installer = RcclInstaller {
            home: temp.path().to_path_buf(),
            python_bin: PathBuf::from("python3"),
            rocm_path: PathBuf::from("/opt/rocm"),
            rocm_version: "7.2.4".into(),
            rocm_channel: "latest".into(),
            gpu_archs: vec!["gfx1100".into(), "gfx1101".into()],
            gpu_count: 2,
            rocr_tokens: vec!["0".into(), "1".into()],
        };
        fs::create_dir_all(installer.overlay_root()).unwrap();
        make_symlink("missing-version", &installer.overlay_root().join("active")).unwrap();

        assert!(matches!(
            installer.active_overlay_state().unwrap(),
            OverlayState::Invalid(reason) if reason.contains("missing")
        ));
    }
}
