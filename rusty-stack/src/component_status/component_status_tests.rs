//! Tests for component_status.rs, focused on llama-cpp detection and verification.

use crate::component_status::is_component_installed_by_id;
use crate::installers::components::llama_cpp::{
    has_partial_artifacts, is_llama_cli_functional, is_llama_cli_on_path,
};
use std::fs;
use std::process::Command;

#[test]
fn test_llama_cpp_detection_absent() {
    let _env = crate::test_support::lock_env();
    let saved_home = std::env::var("HOME").ok();
    let saved_mlstack_home = std::env::var("MLSTACK_USER_HOME").ok();
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().to_string_lossy().to_string();
    std::env::set_var("HOME", &home);
    std::env::set_var("MLSTACK_USER_HOME", &home);
    fs::create_dir_all(dir.path().join(".mlstack")).unwrap();

    // Mock a home dir with no llama-cpp artifacts.
    assert!(!is_llama_cli_functional(&home));
    assert!(!is_llama_cli_on_path());
    assert!(!has_partial_artifacts(&home));
    assert!(!is_component_installed_by_id("llama-cpp", &[]));

    match saved_home {
        Some(v) => std::env::set_var("HOME", v),
        None => std::env::remove_var("HOME"),
    }
    match saved_mlstack_home {
        Some(v) => std::env::set_var("MLSTACK_USER_HOME", v),
        None => std::env::remove_var("MLSTACK_USER_HOME"),
    }
}

#[test]
fn test_llama_cpp_detection_partial_artifacts() {
    let _env = crate::test_support::lock_env();
    let saved_home = std::env::var("HOME").ok();
    let saved_mlstack_home = std::env::var("MLSTACK_USER_HOME").ok();
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().to_string_lossy().to_string();
    std::env::set_var("HOME", &home);
    std::env::set_var("MLSTACK_USER_HOME", &home);

    // Mock a home dir with partial artifacts but no functional binary.
    fs::create_dir_all(dir.path().join(".mlstack/components/llama-cpp")).unwrap();

    assert!(!is_llama_cli_functional(&home));
    assert!(has_partial_artifacts(&home));
    assert!(!is_component_installed_by_id("llama-cpp", &[]));

    match saved_home {
        Some(v) => std::env::set_var("HOME", v),
        None => std::env::remove_var("HOME"),
    }
    match saved_mlstack_home {
        Some(v) => std::env::set_var("MLSTACK_USER_HOME", v),
        None => std::env::remove_var("MLSTACK_USER_HOME"),
    }
}

#[test]
fn test_llama_cpp_detection_functional_binary() {
    let _env = crate::test_support::lock_env();
    let saved_home = std::env::var("HOME").ok();
    let saved_mlstack_home = std::env::var("MLSTACK_USER_HOME").ok();
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().to_string_lossy().to_string();
    std::env::set_var("HOME", &home);
    std::env::set_var("MLSTACK_USER_HOME", &home);

    // Mock a home dir with a non-functional binary.
    let bin_path = dir
        .path()
        .join(".mlstack/components/llama-cpp/bin/llama-cli");
    fs::create_dir_all(bin_path.parent().unwrap()).unwrap();

    // Create a dummy binary that will fail the --help check
    let _ = fs::write(&bin_path, b"#!/bin/bash\nexit 1");
    let _ = Command::new("chmod").arg("+x").arg(&bin_path).output();

    assert!(!is_llama_cli_functional(&home));
    assert!(!is_component_installed_by_id("llama-cpp", &[]));

    match saved_home {
        Some(v) => std::env::set_var("HOME", v),
        None => std::env::remove_var("HOME"),
    }
    match saved_mlstack_home {
        Some(v) => std::env::set_var("MLSTACK_USER_HOME", v),
        None => std::env::remove_var("MLSTACK_USER_HOME"),
    }
}
