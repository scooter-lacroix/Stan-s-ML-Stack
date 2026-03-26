//! rusty-stack-update - Binary wrapper for scripts/update_stack.sh
//!
//! Detects the scripts directory and forwards all arguments to the shell script,
//! inheriting stdin/stdout/stderr for interactive menu support.

use std::env;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn find_scripts_dir() -> Option<PathBuf> {
    let exe_dir = env::current_exe().ok()?.parent()?.to_path_buf();

    // Check relative to binary: target/release/../../scripts
    let candidates = [
        exe_dir.join("../../scripts"),
        exe_dir.join("../../../scripts"),
        PathBuf::from("./scripts"),
        PathBuf::from("../scripts"),
    ];

    for dir in &candidates {
        if dir.join("update_stack.sh").exists() {
            if let Ok(canonical) = dir.canonicalize() {
                return Some(canonical);
            }
        }
    }

    None
}

fn main() {
    let scripts_dir = match find_scripts_dir() {
        Some(dir) => dir,
        None => {
            eprintln!("Error: Could not find scripts directory containing update_stack.sh");
            eprintln!("Searched: ./scripts, ../scripts, and relative to binary location");
            std::process::exit(1);
        }
    };

    let update_script = scripts_dir.join("update_stack.sh");

    let args: Vec<String> = env::args().skip(1).collect();

    let status = Command::new("bash")
        .arg(&update_script)
        .args(&args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status();

    match status {
        Ok(s) => std::process::exit(s.code().unwrap_or(1)),
        Err(e) => {
            eprintln!("Error executing update_stack.sh: {}", e);
            std::process::exit(1);
        }
    }
}
