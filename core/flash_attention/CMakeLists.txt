cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(flash_attention_amd LANGUAGES CXX)

# Find PyTorch
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

# Disable kineto warning
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(USE_KINETO OFF CACHE BOOL "Use kineto" FORCE)

# Find Torch package
find_package(Torch REQUIRED)

# Find ROCm
if(NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
else()
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to ROCm installation")
endif()

# Enable HIP
option(USE_HIP "Use HIP for AMD GPUs" ON)

# Add HIP compiler definitions
if(USE_HIP)
    add_definitions(-D__HIP_PLATFORM_AMD__)
    add_definitions(-DHIP_ENABLE_DEVICE_ENUMS=1)
    add_definitions(-DHIP_ENABLE_DEVICE_TYPES=1)
endif()

# Find Python
execute_process(
    COMMAND python3 -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Include directories
include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
    ${ROCM_PATH}/include
    ${ROCM_PATH}/hip/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set HIP flags if using HIP
if(USE_HIP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_AMD__ -DHIP_ENABLE_DEVICE_ENUMS=1 -DHIP_ENABLE_DEVICE_TYPES=1")

    # Remove invalid flags that cause warnings
    string(REPLACE "-Wno-duplicate-decl-specifier" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# Add source files
set(SOURCES
    flash_attention_amd.cpp
    flash_attention_amd_cuda.cpp
)

# Create the shared library
add_library(flash_attention_amd_cuda SHARED ${SOURCES})
target_link_libraries(flash_attention_amd_cuda PRIVATE ${TORCH_LIBRARIES})

# Set output directory
set_target_properties(flash_attention_amd_cuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..
    PREFIX ""
)

# Install the library
install(TARGETS flash_attention_amd_cuda DESTINATION ${CMAKE_BINARY_DIR}/..)
