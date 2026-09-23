cmake_minimum_required(VERSION 3.18...3.30)

project(
  dmt
  VERSION 0.2.0
  LANGUAGES CXX
  DESCRIPTION "Dispersion Measure Transform Library"
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler version requirements
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0)
    message(
      FATAL_ERROR
        "GCC 13.0 or higher is required for C++20 support. Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
    message(
      FATAL_ERROR
        "Clang 18.0 or higher is required for C++20 support. Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
  endif()
else()
  message(
    WARNING
      "Compiler '${CMAKE_CXX_COMPILER_ID}' is not officially tested. Proceed at your own risk."
  )
endif()

# Configuration options
option(DMT_BUILD_DOCS "Build documentation" OFF)
option(DMT_BUILD_PYTHON "Build Python bindings" ON)
option(DMT_BUILD_TESTING "Build C++ tests" ON)
option(DMT_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(DMT_ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(DMT_ENABLE_IPO "Enable interprocedural optimization (LTO) on Release builds" OFF)
option(DMT_ENABLE_NATIVE_ARCH "Use -march=native in Release builds" ON)
option(DMT_USE_SYSTEM_DEPS "Prefer compatible system packages over pinned bundled dependencies" OFF)
# BUILD_SHARED_LIBS is a standard CMake option (ON = shared dmt, OFF = static dmt).
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Python extensions are installed as a single module file; embed dmt statically
# instead of shipping a separate libdmt.so plus loader search-path tweaks.
if(DMT_BUILD_PYTHON AND BUILD_SHARED_LIBS)
  message(FATAL_ERROR "DMT_BUILD_PYTHON=ON requires BUILD_SHARED_LIBS=OFF. "
                      "Python wheels should statically link dmt into libdmt.cpython-*.so."
  )
endif()

# CUDA tristate: AUTO / ON / OFF
set(DMT_CUDA
    "AUTO"
    CACHE STRING
          "CUDA build mode: AUTO (use GPU if toolchain found), ON (require GPU), OFF (CPU only)"
)
set_property(CACHE DMT_CUDA PROPERTY STRINGS "AUTO" "ON" "OFF")
if(NOT DMT_CUDA MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "DMT_CUDA must be AUTO, ON, or OFF. Got: '${DMT_CUDA}'.")
endif()

# GPU architecture(s) for nvcc ("native" = detect from local GPU at build time).
set(DMT_CUDA_ARCHITECTURES
    "native"
    CACHE STRING "CUDA architectures passed to CMAKE_CUDA_ARCHITECTURES (e.g. native, 80, 86;90)"
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release when no build type is specified (packaging-friendly).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build: Debug, Release, MinSizeRel, RelWithDebInfo." FORCE
  )
  set_property(
    CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )
endif()

# Optional LTO for Release builds.
if(CMAKE_BUILD_TYPE MATCHES "Release" AND DMT_ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported)
  if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "IPO/LTO enabled for Release build.")
  else()
    message(FATAL_ERROR "DMT_ENABLE_IPO=ON but IPO/LTO is not supported by this toolchain.")
  endif()
else()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
endif()

# Project structure and paths
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(GNUInstallDirs)

# Compiler flags (Interface target to hold global compile options)
set(DMT_COMPILE_OPTIONS ${PROJECT_NAME}_COMPILE_OPTIONS)
add_library(${DMT_COMPILE_OPTIONS} INTERFACE)
target_compile_options(${DMT_COMPILE_OPTIONS} INTERFACE -Wall -Wextra -Wpedantic)
target_compile_options(
  ${DMT_COMPILE_OPTIONS}
  INTERFACE $<$<CONFIG:Debug>:-O0;-g> $<$<CONFIG:Release>:-O3;-DNDEBUG;-ffast-math>
            $<$<AND:$<CONFIG:Release>,$<BOOL:${DMT_ENABLE_NATIVE_ARCH}>>:-march=native>
            $<$<CONFIG:RelWithDebInfo>:-O2;-DNDEBUG;-g> $<$<CONFIG:MinSizeRel>:-Os;-DNDEBUG>
)

# CUDA toolchain detection (DMT_CUDA user option; DMT_ENABLE_CUDA is the C++ macro).
if(DMT_CUDA STREQUAL "ON" OR DMT_CUDA STREQUAL "AUTO")
  include(CheckLanguage)
  # nvcc usually lives outside the default PATH; seed CUDACXX before check_language.
  if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(
      DMT_NVCC_EXECUTABLE nvcc
      HINTS ENV
            CUDA_PATH
            ENV
            CUDA_HOME
            ENV
            CUDA_ROOT
            /usr/local/cuda
            /opt/cuda
      PATH_SUFFIXES bin
    )
    if(DMT_NVCC_EXECUTABLE)
      set(ENV{CUDACXX} "${DMT_NVCC_EXECUTABLE}")
    endif()
  endif()
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  elseif(DMT_CUDA STREQUAL "ON")
    message(FATAL_ERROR "DMT_CUDA=ON but no CUDA compiler (nvcc) was found. "
                        "Install CUDA >= 12.6 or use DMT_CUDA=AUTO/OFF."
    )
  else()
    message(STATUS "CUDA compiler not found. Building CPU-only (DMT_CUDA=AUTO).")
  endif()
elseif(DMT_CUDA STREQUAL "OFF")
  message(STATUS "CUDA disabled by user (DMT_CUDA=OFF). Building CPU-only.")
endif()

# When CUDA is explicitly required, verify a driver/GPU is present (not just nvcc).
if(CMAKE_CUDA_COMPILER AND DMT_CUDA STREQUAL "ON")
  find_program(DMT_NVIDIA_SMI nvidia-smi)
  if(NOT DMT_NVIDIA_SMI)
    message(
      FATAL_ERROR "DMT_CUDA=ON but nvidia-smi was not found. An NVIDIA driver and GPU are required."
    )
  endif()
  execute_process(
    COMMAND ${DMT_NVIDIA_SMI} --query-gpu=name --format=csv,noheader
    OUTPUT_VARIABLE _dmt_gpu_names
    RESULT_VARIABLE _dmt_gpu_status
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
  )
  if(NOT _dmt_gpu_status EQUAL 0 OR _dmt_gpu_names STREQUAL "")
    message(
      FATAL_ERROR "DMT_CUDA=ON but nvidia-smi reports no usable GPU. Check driver installation."
    )
  endif()
  message(STATUS "DMT_CUDA=ON: detected GPU(s): ${_dmt_gpu_names}")
endif()

if(CMAKE_CUDA_COMPILER)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.6.0)
    message(FATAL_ERROR "CUDA >= 12.6 is required. Found: ${CMAKE_CUDA_COMPILER_VERSION}")
  endif()
  message(STATUS "Found CUDA ${CMAKE_CUDA_COMPILER_VERSION}.")

  # Find CUDA Toolkit for proper include dirs and linking
  find_package(CUDAToolkit REQUIRED)

  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  # Define a target for CUDA compilation options
  set(DMT_CUDA_OPTIONS ${PROJECT_NAME}_cuda_options)
  add_library(${DMT_CUDA_OPTIONS} INTERFACE)
  target_compile_options(
    ${DMT_CUDA_OPTIONS}
    INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
              -Wno-pedantic
              --expt-extended-lambda
              --expt-relaxed-constexpr
              -Xcompiler=-Wall,-Wextra
              $<$<CONFIG:Debug>:-G;-g;-O0>
              $<$<CONFIG:Release>:-O3;-use_fast_math;-DNDEBUG>
              $<$<CONFIG:RelWithDebInfo>:-O2;-g;-lineinfo;-DNDEBUG>
              $<$<CONFIG:MinSizeRel>:-O2;-DNDEBUG>
              >
  )
  set(CMAKE_CUDA_ARCHITECTURES ${DMT_CUDA_ARCHITECTURES})
  message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  # Disable response files for better IDE integration
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
endif()

# CPM.cmake Setup
include(cmake/CPM.cmake)
set(CPM_USE_LOCAL_PACKAGES
    ${DMT_USE_SYSTEM_DEPS}
    CACHE BOOL "Try to find packages locally before downloading" FORCE
)
if(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_SOURCE_CACHE
      $ENV{CPM_SOURCE_CACHE}
      CACHE PATH "Directory to cache downloaded CPM sources"
  )
endif()

# Target naming: library target is `dmt`, exported alias is `dmt::dmt`.
set(DMT_LIBRARY_NAME ${PROJECT_NAME})
set(DMT_PYTHON_LIBRARY_NAME ${PROJECT_NAME}lib)
set(DMT_CMAKE_PACKAGE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# BUILD_INTERFACE keeps flags off the installed export set. PUBLIC so in-tree
# consumers (tests, pybind modules) inherit compile and -lgcov link flags.
function(dmt_enable_coverage target)
  if(NOT DMT_ENABLE_COVERAGE)
    return()
  endif()
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(
      FATAL_ERROR
        "Code coverage requested, but not supported for compiler ${CMAKE_CXX_COMPILER_ID}."
    )
  endif()
  target_compile_options(
    ${target}
    PUBLIC $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:-O0>>
           $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:-g>>
           $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:--coverage>>
  )
  target_link_options(${target} PUBLIC $<BUILD_INTERFACE:--coverage>)
endfunction()

add_subdirectory(lib)
if(DMT_BUILD_PYTHON)
  add_subdirectory(src)
endif()
if(DMT_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
if(DMT_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
if(DMT_BUILD_DOCS)
  add_subdirectory(docs)
endif()

# Install / export: find_package(dmt) + target dmt::dmt
include(CMakePackageConfigHelpers)

set(DMT_BUILT_WITH_CUDA OFF)
if(CMAKE_CUDA_COMPILER)
  set(DMT_BUILT_WITH_CUDA ON)
endif()

set(DMT_BUILT_AS_SHARED OFF)
if(BUILD_SHARED_LIBS)
  set(DMT_BUILT_AS_SHARED ON)
endif()

# Install targets
install(
  TARGETS ${DMT_LIBRARY_NAME}
  EXPORT dmtTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(CMAKE_CUDA_COMPILER)
  install(TARGETS ${DMT_CUDA_OPTIONS} EXPORT dmtTargets)
endif()

# Install headers
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Generate and install export file
install(
  EXPORT dmtTargets
  FILE dmtTargets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${DMT_CMAKE_PACKAGE_DIR}
)

configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/dmtCUDAConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/dmtCUDAConfig.cmake @ONLY
)

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/dmtConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/dmtConfig.cmake
  INSTALL_DESTINATION ${DMT_CMAKE_PACKAGE_DIR}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/dmtConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dmtConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/dmtConfigVersion.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/dmtCUDAConfig.cmake DESTINATION ${DMT_CMAKE_PACKAGE_DIR}
)

# Print feature summary
include(FeatureSummary)
feature_summary(WHAT ALL)

if(CMAKE_CUDA_COMPILER)
  message(STATUS "dmt build mode: CPU + CUDA (DMT_CUDA=${DMT_CUDA}, DMT_ENABLE_CUDA macro defined)")
else()
  message(STATUS "dmt build mode: CPU-only (DMT_CUDA=${DMT_CUDA})")
endif()
