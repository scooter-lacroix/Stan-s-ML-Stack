//! Tests for component_status.rs, focused on llama-cpp detection and verification.

use crate::component_status::is_component_installed_by_id;
use crate::installers::components::llama_cpp::{
    has_partial_artifacts, is_llama_cli_functional, is_llama_cli_on_path,
};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn test_llama_cpp_detection_absent() {
    // Mock a home dir with no llama-cpp artifacts
    let home = "/nonexistent/path/that/does/not/exist";
    assert!(!is_llama_cli_functional(home));
    assert!(!is_llama_cli_on_path());
    assert!(!has_partial_artifacts(home));
    assert!(!is_component_installed_by_id("llama-cpp", &[]));
}

#[test]
fn test_llama_cpp_detection_partial_artifacts() {
    // Mock a home dir with partial artifacts but no functional binary
    let home = "/tmp/llama_cpp_test_partial";
    let _ = fs::create_dir_all(home);
    let _ = fs::create_dir_all(format!("{}/.mlstack/components/llama-cpp", home));

    assert!(!is_llama_cli_functional(home));
    assert!(has_partial_artifacts(home));
    assert!(!is_component_installed_by_id("llama-cpp", &[]));

    let _ = fs::remove_dir_all(home);
}

#[test]
fn test_llama_cpp_detection_functional_binary() {
    // Mock a home dir with a functional binary (not implemented here, but the test structure is ready)
    // For now, just verify the detection logic for a missing binary
    let home = "/tmp/llama_cpp_test_functional";
    let _ = fs::create_dir_all(home);
    let bin_path = format!("{}/.mlstack/components/llama-cpp/bin/llama-cli", home);
    let _ = fs::create_dir_all(PathBuf::from(&bin_path).parent().unwrap());

    // Create a dummy binary that will fail the --help check
    let _ = fs::write(&bin_path, b"#!/bin/bash\nexit 1");
    let _ = Command::new("chmod").args(&["+x", &bin_path]).output();

    assert!(!is_llama_cli_functional(home));
    assert!(!is_component_installed_by_id("llama-cpp", &[]));

    let _ = fs::remove_dir_all(home);
}
