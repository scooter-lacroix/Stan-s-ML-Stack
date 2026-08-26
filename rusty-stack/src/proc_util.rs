//! Bounded subprocess execution for hardware/component detection.
//!
//! Detection probes used to call `.output()` directly: no timeout, inheriting
//! the TUI's stdio, serially queued behind whichever process currently holds
//! `/dev/kfd`. Under load that turned a 1–2 s hardware phase into ~60 s.
//! Every detection-path subprocess goes through [`run_with_detection_timeout`]
//! instead: piped stdio, hard cap, killed-and-reaped children, and optional
//! `[hw-timing]` traces when `MLSTACK_HW_TIMING=1`.

use std::process::{Command, Output};
use std::time::{Duration, Instant};

/// Upper bound for any single detection-phase subprocess. Chosen above the
/// worst observed idle time (~1 s for a full `rocminfo`) but far below the
/// multi-second stalls seen when GPUs are contended.
pub const DETECTION_EXEC_TIMEOUT: Duration = Duration::from_secs(3);

/// Upper bound for heavyweight runtime health probes (FastVideo / vLLM
/// detection): they legitimately import torch + native extensions, which can
/// take tens of seconds cold — but must never hang indefinitely behind a busy
/// `/dev/kfd`.
pub const HEAVY_PROBE_EXEC_TIMEOUT: Duration = Duration::from_secs(45);

fn hw_timing_enabled() -> bool {
    std::env::var("MLSTACK_HW_TIMING").ok().as_deref() == Some("1")
}

/// Cheap-path wrapper: [`DETECTION_EXEC_TIMEOUT`] cap.
pub fn run_with_detection_timeout(cmd: &mut Command, label: &str) -> Option<Output> {
    run_command_timeout(cmd, label, DETECTION_EXEC_TIMEOUT)
}

/// Run `cmd`, hard-capped at `limit`; children run in their own process group
/// so a timeout kills the whole probe tree (e.g. FastVideo's per-GPU children).
///
/// Returns `None` on timeout, spawn failure, or wait error — callers treat it
/// exactly like the previous non-success `.output()` paths (skip enrichment).
pub fn run_command_timeout(cmd: &mut Command, label: &str, limit: Duration) -> Option<Output> {
    let timing = hw_timing_enabled();
    let start = Instant::now();
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(_) => {
            if timing {
                eprintln!("[hw-timing] {label} spawn failed");
            }
            return None;
        }
    };

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // own process group ⇒ timeout can TERM the whole probe tree
        cmd.process_group(0);
    }
    let deadline = start + limit;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if Instant::now() >= deadline {
                    // signal the whole probe group first (FastVideo spawns
                    // per-GPU children), then hard-reap the leader
                    #[cfg(unix)]
                    {
                        let _ = std::process::Command::new("kill")
                            .arg("-TERM")
                            .arg(format!("-{}", child.id()))
                            .status();
                    }

                    let _ = child.kill();
                    let _ = child.wait();
                    if timing {
                        eprintln!("[hw-timing] {label} TIMEOUT after {:?}", start.elapsed());
                    }
                    return None;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(_) => {
                if timing {
                    eprintln!("[hw-timing] {label} wait error");
                }
                return None;
            }
        }
    }

    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(_) => return None,
    };
    if timing {
        eprintln!(
            "[hw-timing] {label} ok={} {:?}",
            output.status.success(),
            start.elapsed()
        );
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_output_within_limit() {
        let mut cmd = Command::new("echo");
        cmd.arg("hello");
        let out = run_with_detection_timeout(&mut cmd, "test-echo").expect("echo must run");
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "hello");
    }

    #[test]
    fn times_out_and_kills_runaway() {
        let mut cmd = Command::new("sleep");
        cmd.arg("30");
        let started = Instant::now();
        let out = run_with_detection_timeout(&mut cmd, "test-sleep");
        assert!(out.is_none(), "long sleep must be killed");
        assert!(started.elapsed() < DETECTION_EXEC_TIMEOUT + Duration::from_secs(2));
    }

    #[test]
    fn missing_binary_is_none_not_panic() {
        let mut cmd = Command::new("definitely-not-a-binary-rusty-stack");
        assert!(run_with_detection_timeout(&mut cmd, "test-missing").is_none());
    }
}
