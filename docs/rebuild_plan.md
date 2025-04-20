# Plan to Rebuild ML Stack Components and Create Verification Script

**Objective:** Debug and fix ONNX Runtime and MPI build issues, and create a comprehensive script (`verify_and_build.sh`) to verify, build, and fix components of the ML stack defined in `Stans_MLStack/README.md`.

**Context and Problem Summary:**

*   **ONNX Runtime:** The build failed with `hipErrorNoBinaryForGpu`, indicating that the ONNX Runtime library was not compiled with support for your specific AMD GPU architectures (`gfx1100` for RX 7900 XTX and `gfx1101` for RX 7800 XT). A "surgical fix" for this is not possible; recompilation is required with the correct architecture flags.
*   **MPI:** The MPI installation/build appears to lack proper ROCm/HIP support. A previous attempt to recompile OpenMPI failed during the configuration step due to deprecated C++ bindings.
*   **Verification and Build Script:** A script `Stans_MLStack/scripts/verify_and_build.sh` was partially created in a previous attempt but encountered errors during execution (ONNX dynamic module import failure, MPI configuration failure). The goal is to refine this script to be robust, check component status, attempt fixes (including recompilation), skip already verified components, and provide rough time estimates. The script structure should be modeled after `Stans_MLStack/scripts/run_tests.sh`.
*   **Implementation Strategy:** The user has requested that the implementation of the script be done in smaller, more manageable steps to avoid errors.

**Detailed Plan:**

1.  **Initial Analysis and Script Review:**
    *   Review the existing `Stans_MLStack/scripts/build_onnxruntime.sh` script to understand the current ONNX build process and identify where to add the architecture flags.
    *   Examine the error logs from the previous script execution (`~/onnx_mpi_rebuild.log`) to get precise details on the ONNX dynamic module error and the MPI configuration failure (specifically, the exact error message related to C++ bindings).
    *   Study the structure and functions within `Stans_MLStack/scripts/run_tests.sh` to replicate its style and organization in the new script.

2.  **Refine ONNX Runtime Build Process:**
    *   Confirm the exact CMake flag needed to specify multiple ROCm architectures. Based on standard CMake practices and the previous attempt, `-Donnxruntime_ROCM_ARCH="gfx1100;gfx1101"` is the likely correct format to include support for both RX 7900 XTX and RX 7800 XT.
    *   **Action:** Integrate this flag into the ONNX Runtime build command within a dedicated function for ONNX building in the `verify_and_build.sh` script. This function will handle cloning (if necessary), cleaning previous builds (if incomplete/corrupted), configuring, building, and installing the Python wheel.

3.  **Refine MPI Build Process:**
    *   Identify the correct configure flag to address the deprecated C++ bindings error. Research suggests `--disable-mpi-cxx` is the appropriate flag for OpenMPI to skip building the C++ interface, which is often deprecated or causes conflicts.
    *   **Action:** Modify the OpenMPI configure command within a dedicated function for MPI building in the `verify_and_build.sh` script. This command will include `--disable-mpi-cxx` and ensure the `--with-rocm` or `--with-hip` flag correctly points to the ROCm installation path (`/opt/rocm`).
    *   Confirm the OpenMPI source handling: The previous script downloaded `openmpi-5.0.2`. The build function should check if this source directory already exists before attempting to download and extract.

4.  **Develop the `verify_and_build.sh` Script (`Stans_MLStack/scripts/verify_and_build.sh`):**
    *   **Structure:** Create the script with a clear header, environment variable setup, logging functions (`log_success`, `log_error`, `log_warning`), verification functions, build/fix functions, and a main execution logic block.
    *   **Environment Setup:** Define and export necessary variables like `ROCM_PATH`, `PATH`, and `LD_LIBRARY_PATH` at the script's start.
    *   **Verification Functions:** Implement functions for each key component (PyTorch, Flash Attention, MPI, ONNX Runtime) using appropriate commands to check for successful installation and ROCm/HIP capability (e.g., Python imports, `mpirun --version`, checking available ONNX Runtime providers). These functions should return a clear status (e.g., 0 for success, non-zero for failure).
    *   **Build/Fix Functions:** Implement functions (`build_mpi`, `build_onnxruntime`) containing the refined build logic from steps 2 and 3. These functions should include checks for existing source/build directories to support idempotency.
    *   **Main Logic:** Implement the sequential execution flow:
        *   Call environment setup.
        *   For each component:
            *   Call the verification function.
            *   If verification fails, call the corresponding build/fix function.
            *   After attempting a build/fix, call the verification function again to confirm success.
            *   If initial verification succeeds, skip the build/fix step.
        *   Include rough time estimates as comments before each major build step.
        *   Add a final verification step that runs checks for all components again at the end.
        *   Report the overall status.

5.  **Implementation in Manageable Chunks (Transition to Code Mode):**
    *   Once the plan is approved, the implementation of the `verify_and_build.sh` script will be done iteratively. This involves writing parts of the script (e.g., one verification function, then its corresponding build function), saving the changes, and potentially testing those parts before moving on. This aligns with the user's request for smaller, manageable steps.

**Mermaid Diagram of Script Logic:**

```mermaid
graph TD
    A[Start verify_and_build.sh] --> B(Setup Environment Variables);
    B --> C{Verify PyTorch};
    C -- Verified --> D{Verify Flash Attention};
    C -- Not Verified --> C_Build[Build/Fix PyTorch];
    C_Build --> D;

    D -- Verified --> E{Verify MPI};
    D -- Not Verified --> D_Build[Build/Fix Flash Attention];
    D_Build --> E;

    E -- Verified --> F{Verify ONNX Runtime};
    E -- Not Verified --> E_Build[Build/Fix MPI (Disable C++ Bindings, Add ROCm/HIP Support)];
    E_Build --> F;

    F -- Verified --> G[Run Final Verification Tests];
    F -- Not Verified --> F_Build[Build/Fix ONNX Runtime (Add gfx1100;gfx1101 Architectures)];
    F_Build --> G;

    G --> H[Report Final Status];
    H --> I[End Script];