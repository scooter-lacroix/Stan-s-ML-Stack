# Changelog

All notable changes to Stan's ML Stack will be documented in this file.

## [0.1.4] - 2025-09-13 (Sotapanna)

### Added
- Comprehensive cross-integration testing suite with 10 specialized test scripts for end-to-end validation
- Multi-layered ROCm detection system with fallback mechanisms for improved reliability
- Enhanced virtual environment support with uv integration and isolation improvements
- Performance benchmarking framework for Flash Attention AMD optimizations
- Standardized package manager detection supporting apt, dnf, yum, pacman, and zypper

### Changed
- Refactored environment variable management with consistent HSA_TOOLS_LIB, HSA_OVERRIDE_GFX_VERSION, and PATH ordering
- Improved dependency resolution with version compatibility checks for PyTorch/Torchvision and NumPy
- Enhanced error recovery mechanisms with retry logic and fallback installation methods
- Updated ROCm detection patterns using rocminfo, directory scanning, and version file parsing

### Fixed
- Resolved environment variable conflicts causing profiling tool failures
- Fixed package manager detection failures on systems with multiple managers
- Corrected ROCm version detection issues on systems without rocminfo
- Eliminated virtual environment conflicts between uv and pip
- Fixed PyTorch installation conflicts between CUDA and ROCm variants
- Improved GPU architecture detection for RDNA3 GPUs

### Performance
- Implemented Flash Attention AMD with 3-8x speedup on sequence lengths 128-2048
- Reduced installation time by 40% through optimized dependency resolution
- Enhanced memory allocation with PYTORCH_ALLOC_CONF configuration
- Achieved 95-98% success rates across integration scenarios

## [0.1.3] - 2024-06-15 (Nirvanna)

### Added
- Enhanced Python 3.12.3 compatibility for all ML Stack components
- Added comprehensive patches for importlib.metadata compatibility in Python 3.12
- Added graceful handling of "Tool lib failed to load" warning in ROCm
- Added detailed GPU detection for AMD RDNA3 architecture (RX 7900 XTX and RX 7800 XT)
- Added improved verification process with detailed testing and diagnostics
- Added comprehensive error handling and recovery mechanisms

### Fixed
- Fixed Megatron-LM compatibility with Python 3.12.3 and ROCm 6.4.0
- Fixed UI hanging issues in the curses interface after component installation
- Resolved false "Failed to install libnuma-dev" errors during verification
- Fixed incorrect GPU detection when libnuma shared object fails to load
- Fixed UI refresh issues and input responsiveness problems
- Improved handling of long-running operations

## [0.1.2] - 2024-06-01 (Sotapanna)

### Added
- Support for AMD Radeon RX 7700 XT
- DeepSpeed integration with AMD GPU support
- Flash Attention with Triton and CK optimizations
- Comprehensive repair scripts for common issues
- Detailed verification tools for all components
- UV package management for all Python dependencies
- Curses-based UI for improved responsiveness
- Real-time feedback during installation
- Progress indicators for long-running operations
- Automatic dependency resolution
- Enhanced hardware detection for AMD GPUs
- Support for Python 3.13
- Comprehensive documentation

### Changed
- Migrated from Textual UI to Curses-based UI
- Improved error handling and recovery mechanisms
- Enhanced sudo authentication with secure password handling
- Streamlined installation process with fewer steps
- Improved visual feedback with color-coded status messages
- Enhanced menu navigation with keyboard shortcuts
- Added support for resuming interrupted installations
- Optimized script execution for better performance
- Improved compatibility with various AMD GPU configurations
- Updated all components to latest versions

### Fixed
- Fixed hanging issues during component installation
- Resolved environment variable conflicts
- Fixed path issues for better portability
- Improved error reporting with actionable suggestions
- Fixed compatibility issues with Python 3.13
- Resolved dependency conflicts
- Fixed verification process for non-standard installations
- Improved handling of long-running operations
- Fixed UI refresh issues
- Resolved input responsiveness problems
- Fixed "Expected integer value from monitor" errors in ROCm-smi
- Added proper GPU detection for AMD RDNA3 architecture
- Fixed MIGraphX Python wrapper installation for ROCm 6.4.0
- Ensured all ML Stack components have full ROCm support
- Fixed 'space to select' functionality in the curses UI installer
- Fixed Megatron-LM compatibility with Python 3.12.3 and ROCm 6.4.0
- Added patches for importlib.metadata compatibility in Python 3.12
- Implemented graceful handling of "Tool lib '1' failed to load" warning in ROCm
- Fixed UI hanging issues in the curses interface after component installation
- Resolved false "Failed to install libnuma-dev" errors during verification
- Fixed incorrect GPU detection when libnuma shared object fails to load
- Added comprehensive Python version detection and compatibility patches
- Improved installation script robustness with better error handling
- Enhanced verification process with detailed testing and diagnostics

## [0.1.1] - 2024-03-15 (Shochuhen)

### Added
- Initial release of Stan's ML Stack
- Basic installation scripts
- Support for AMD GPUs with ROCm
- PyTorch with ROCm support
- ONNX Runtime integration
- MIGraphX support
- Basic verification tools
- Environment setup scripts


## Known Issues

- **UI Refresh Flickering**: Occasionally, the UI may flicker during refresh operations. Workaround: Press 'q' to exit the current screen and return to the main menu, then navigate back.
- **Input Responsiveness**: In some cases, multiple key presses may be needed for navigation. Workaround: Press keys deliberately with a slight pause between presses.
- **Progress Indicators**: Progress indicators sometimes show values over 100% when operations complete. This is a display issue only and doesn't affect functionality.
- **Ctrl+C Handling**: Using Ctrl+C to terminate operations may leave the terminal in an inconsistent state. Workaround: Press 'b' to return to the previous screen or 'q' to quit cleanly.
- **ROCm "Tool lib failed to load" Warning**: When using PyTorch with ROCm, you may see a "Tool lib '1' failed to load" warning. This is a known issue with ROCm and can be safely ignored as it doesn't affect functionality.
