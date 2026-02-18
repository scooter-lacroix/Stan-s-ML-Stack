# Multi-Distro Linux Compatibility - Implementation Plan

## Phase 1: Analysis and Discovery

**Goal:** Understand current installer architecture and identify distribution-specific code

### Tasks

- [ ] Task: Analyze existing installation scripts using LeIndex
  - [ ] Sub-task: Write test script to catalog all package manager commands
  - [ ] Sub-task: Implement comprehensive LeIndex search for apt/pacman/dnf usage
  - [ ] Sub-task: Document all hardcoded Debian/Ubuntu-specific paths and commands
  - [ ] Sub-task: Create dependency mapping table for target distributions

- [ ] Task: Identify distribution detection points
  - [ ] Sub-task: Write tests for distribution detection scenarios
  - [ ] Sub-task: Implement distribution detection utility using LeIndex analysis
  - [ ] Sub-task: Map all `/etc/os-release` variants across target distros

- [ ] Task: Catalog package name differences
  - [ ] Sub-task: Write test to compare package names across distros
  - [ ] Sub-task: Implement package name mapping database
  - [ ] Sub-task: Document all core dependencies and their distro variants

- [ ] Task: Maestro - Phase Verification and Checkpoint 'Analysis and Discovery' (Protocol in workflow.md)

## Phase 2: Package Manager Abstraction Layer

**Goal:** Create abstraction layer for package manager operations

### Tasks

- [ ] Task: Design package manager interface
  - [ ] Sub-task: Write interface contract tests
  - [ ] Sub-task: Implement PackageManager trait/abstraction
  - [ ] Sub-task: Define standard install/remove/query operations

- [ ] Task: Implement apt (Debian/Ubuntu) adapter
  - [ ] Sub-task: Write tests for apt adapter operations
  - [ ] Sub-task: Implement AptPackageManager class
  - [ ] Sub-task: Handle apt-specific quirks and options

- [ ] Task: Implement pacman (Arch Linux) adapter
  - [ ] Sub-task: Write tests for pacman adapter operations
  - [ ] Sub-task: Implement PacmanPackageManager class
  - [ ] Sub-task: Handle pacman-specific quirks and options

- [ ] Task: Implement dnf (Fedora) adapter
  - [ ] Sub-task: Write tests for dnf adapter operations
  - [ ] Sub-task: Implement DnfPackageManager class
  - [ ] Sub-task: Handle dnf-specific quirks and options

- [ ] Task: Create package manager factory
  - [ ] Sub-task: Write tests for auto-detection and instantiation
  - [ ] Sub-task: Implement PackageManagerFactory with detection logic
  - [ ] Sub-task: Add fallback and error handling

- [ ] Task: Maestro - Phase Verification and Checkpoint 'Package Manager Abstraction Layer' (Protocol in workflow.md)

## Phase 3: Distribution Detection Integration

**Goal:** Integrate distribution detection into installer scripts

### Tasks

- [ ] Task: Create distribution detection module
  - [ ] Sub-task: Write tests for /etc/os-release parsing
  - [ ] Sub-task: Implement DistributionDetector class
  - [ ] Sub-task: Add support for derivative distributions (CachyOS, Manjaro)

- [ ] Task: Update hardware detection for multi-distro
  - [ ] Sub-task: Write tests for ROCm path detection across distros
  - [ ] Sub-task: Refactor hardware detection using LeIndex analysis
  - [ ] Sub-task: Handle different ROCm installation paths

- [ ] Task: Integrate detection into Rust TUI
  - [ ] Sub-task: Write tests for TUI state with different distros
  - [ ] Sub-task: Update Rust hardware detection module
  - [ ] Sub-task: Add distribution display in TUI

- [ ] Task: Maestro - Phase Verification and Checkpoint 'Distribution Detection Integration' (Protocol in workflow.md)

## Phase 4: Installation Script Updates

**Goal:** Update all installation scripts for multi-distro compatibility

### Tasks

- [ ] Task: Update core installation scripts
  - [ ] Sub-task: Write tests for script behavior on each package manager
  - [ ] Sub-task: Refactor install_rocm.sh for multi-distro support
  - [ ] Sub-task: Refactor install_pytorch_rocm.sh for multi-distro support
  - [ ] Sub-task: Refactor install_ml_stack.sh for multi-distro support

- [ ] Task: Update extension installation scripts
  - [ ] Sub-task: Write tests for extension script compatibility
  - [ ] Sub-task: Update install_triton.sh for pacman/dnf
  - [ ] Sub-task: Update install_bitsandbytes.sh for pacman/dnf
  - [ ] Sub-task: Update install_vllm.sh for pacman/dnf

- [ ] Task: Update utility scripts
  - [ ] Sub-task: Write tests for utility script behavior
  - [ ] Sub-task: Update enhanced_setup_environment.sh
  - [ ] Sub-task: Update enhanced_verify_installation.sh
  - [ ] Sub-task: Update repair_ml_stack.sh

- [ ] Task: Maestro - Phase Verification and Checkpoint 'Installation Script Updates' (Protocol in workflow.md)

## Phase 5: Testing and Validation

**Goal:** Comprehensive testing across all supported distributions

### Tasks

- [ ] Task: Create multi-distro test suite
  - [ ] Sub-task: Write test framework for distribution-specific tests
  - [ ] Sub-task: Implement CI test matrix for Debian/Ubuntu/Arch/Fedora
  - [ ] Sub-task: Add automated dependency verification tests

- [ ] Task: Manual testing on CachyOS (Arch)
  - [ ] Sub-task: Write test checklist for Arch-based systems
  - [ ] Sub-task: Execute full installation on CachyOS
  - [ ] Sub-task: Verify all components work correctly

- [ ] Task: Manual testing on Fedora
  - [ ] Sub-task: Write test checklist for Fedora/RHEL
  - [ ] Sub-task: Execute full installation on Fedora
  - [ ] Sub-task: Verify all components work correctly

- [ ] Task: Regression testing on Debian/Ubuntu
  - [ ] Sub-task: Write regression test suite
  - [ ] Sub-task: Execute full installation on Debian/Ubuntu
  - [ ] Sub-task: Verify no regressions from changes

- [ ] Task: Maestro - Phase Verification and Checkpoint 'Testing and Validation' (Protocol in workflow.md)

## Phase 6: Documentation and Release

**Goal:** Update documentation and prepare release

### Tasks

- [ ] Task: Update installation documentation
  - [ ] Sub-task: Write Arch Linux installation guide
  - [ ] Sub-task: Write Fedora installation guide
  - [ ] Sub-task: Update README with supported distributions

- [ ] Task: Update CLAUDE.md
  - [ ] Sub-task: Write distribution-specific commands section
  - [ ] Sub-task: Document package manager abstractions
  - [ ] Sub-task: Add troubleshooting for each distribution

- [ ] Task: Create release notes
  - [ ] Sub-task: Write changelog entry for multi-distro support
  - [ ] Sub-task: Document breaking changes (if any)
  - [ ] Sub-task: Add migration guide for existing users

- [ ] Task: Maestro - Phase Verification and Checkpoint 'Documentation and Release' (Protocol in workflow.md)
