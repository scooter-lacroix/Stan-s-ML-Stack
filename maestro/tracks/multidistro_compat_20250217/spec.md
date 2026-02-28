# Multi-Distro Linux Compatibility - Specification

## Overview

Extend Rusty Stack to support Arch Linux (including CachyOS, Manjaro) and Fedora-based distributions while maintaining existing Debian/Ubuntu compatibility. All components should successfully build and install across all supported distributions with equal or better reliability than the current Debian/Ubuntu implementation.

## Background

The current Rusty Stack installer has been developed primarily for Debian/Ubuntu-based distributions. Users on Arch Linux (such as CachyOS) and Fedora-based systems cannot successfully install the ML stack due to:
- Package manager differences (pacman, dnf vs apt)
- Different package naming conventions
- Varying system paths and configurations
- Distribution-specific dependency requirements

## Requirements

### Functional Requirements

1. **Package Manager Abstraction**
   - Detect the system package manager (apt, pacman, dnf, yum)
   - Abstract package installation commands across all managers
   - Handle package name differences between distributions

2. **Distribution Detection**
   - Automatically detect the Linux distribution
   - Identify distribution version (e.g., Arch vs Ubuntu 22.04 vs Fedora 39)
   - Handle derivative distributions (CachyOS, Manjaro, etc.)

3. **Dependency Management**
   - Map dependencies to distribution-specific package names
   - Handle distribution-specific build dependencies
   - Support equivalent packages across all distributions

4. **Path and Configuration Handling**
   - Adapt to distribution-specific system paths
   - Handle different configuration file locations
   - Support distribution-specific service management

### Non-Functional Requirements

1. **Backward Compatibility**
   - Must not break existing Debian/Ubuntu functionality
   - All existing features must work identically on supported distros

2. **Code Quality**
   - Refactor existing code only where necessary for abstraction
   - Maintain code clarity and testability
   - Follow existing code style guidelines

3. **LeIndex Utilization**
   - Use LeIndex for all code analysis and exploration
   - Leverage search, analyze, and phase analysis capabilities
   - Document findings using LeIndex output

4. **Testing**
   - Test on Debian/Ubuntu (existing compatibility verification)
   - Test on Arch Linux or CachyOS
   - Test on Fedora or RHEL-based systems

## Scope

### In Scope

- Package manager abstraction layer
- Distribution detection utilities
- Package name mapping for core dependencies
- Installation script updates for all core components
- Documentation updates for new distributions
- Testing infrastructure updates

### Out of Scope

- Support for non-systemd init systems
- Support for other package formats (flatpak, snap)
- Windows platform support (separate track)
- Container-specific modifications

## Success Criteria

1. Rusty Stack TUI installer launches successfully on Arch Linux and Fedora
2. All core components install correctly on Arch Linux and Fedora
3. All core components continue to install correctly on Debian/Ubuntu
4. Installation success rate is ≥95% across all supported distributions
5. Benchmark infrastructure works on all supported distributions
6. LeIndex is used throughout development for code analysis

## Technical Considerations

### Package Manager Differences

| Distribution | Package Manager | Command Syntax |
|--------------|----------------|----------------|
| Debian/Ubuntu | apt/apt-get | `apt install <package>` |
| Arch Linux | pacman | `pacman -S <package>` |
| Fedora | dnf | `dnf install <package>` |

### Key Dependency Mappings

| Component | Debian/Ubuntu | Arch | Fedora |
|-----------|---------------|------|--------|
| Python | python3 | python | python3 |
| Pip | python3-pip | python-pip | python3-pip |
| Git | git | git | git |
| CMake | cmake | cmake | cmake |
| ROCm | rocm | rocm | rocm-opencl |

### Service Management

All target distributions use systemd, so service management commands should be consistent.

## Implementation Phases

1. **Analysis Phase**: Use LeIndex to analyze existing installation scripts
2. **Abstraction Phase**: Create package manager abstraction layer
3. **Update Phase**: Modify installation scripts for multi-distro support
4. **Testing Phase**: Test across all supported distributions
5. **Documentation Phase**: Update docs and create distribution-specific guides
