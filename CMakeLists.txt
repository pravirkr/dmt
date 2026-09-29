cmake_minimum_required(VERSION 3.18...3.30)

project(
  dmt
  VERSION 0.5.0
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
# Python bindings and tests default to ON only when dmt is the top-level
# project, not when it is pulled in via add_subdirectory/FetchContent/CPM.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(DMT_IS_TOP_LEVEL ON)
else()
  set(DMT_IS_TOP_LEVEL OFF)
endif()
option(DMT_BUILD_PYTHON "Build Python bindings" ${DMT_IS_TOP_LEVEL})
option(DMT_BUILD_TESTING "Build C++ tests" ${DMT_IS_TOP_LEVEL})
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
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS
             "Debug"
             "Release"
             "MinSizeRel"
             "RelWithDebInfo"
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
# Host-compiler flags; HIP sources get their own (dmt_gpu_options): no
# -ffast-math or -march=native reaches device code there.
set(_dmt_not_hip "$<NOT:$<COMPILE_LANGUAGE:HIP>>")
# No -Wpedantic on CUDA: nvcc forwards it to the host compiler, which then
# flags the line directives in nvcc's own generated code.
target_compile_options(
  ${DMT_COMPILE_OPTIONS} INTERFACE $<${_dmt_not_hip}:-Wall;-Wextra>
                                   $<$<NOT:$<COMPILE_LANGUAGE:CUDA,HIP>>:-Wpedantic>
)
target_compile_options(
  ${DMT_COMPILE_OPTIONS}
  INTERFACE
    $<${_dmt_not_hip}:$<$<CONFIG:Debug>:-O0;-g>>
    $<${_dmt_not_hip}:$<$<CONFIG:Release>:-O3;-DNDEBUG;-ffast-math>>
    $<${_dmt_not_hip}:$<$<AND:$<CONFIG:Release>,$<BOOL:${DMT_ENABLE_NATIVE_ARCH}>>:-march=native>>
    $<${_dmt_not_hip}:$<$<CONFIG:RelWithDebInfo>:-O2;-DNDEBUG;-g>>
    $<${_dmt_not_hip}:$<$<CONFIG:MinSizeRel>:-Os;-DNDEBUG>>
)

# OpenMP detection
find_package(OpenMP REQUIRED COMPONENTS CXX)

# GPU backend: DMT_GPU selects the toolchain of lib/cuda/ (CUDA or HIP). The
# library's private C++ macros (DMT_ENABLE_CUDA/HIP/GPU) follow from it
# (lib/CMakeLists.txt).
include(dmtGPU)

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
  if(NOT
     CMAKE_CXX_COMPILER_ID
     MATCHES
     "GNU|Clang"
  )
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

set(DMT_BUILT_GPU "${DMT_GPU_BACKEND}")
set(DMT_BUILT_WITH_LIBHIPCXX OFF)
if(DMT_GPU_BACKEND STREQUAL "HIP" AND libhipcxx_FOUND)
  set(DMT_BUILT_WITH_LIBHIPCXX ON)
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

if(DMT_GPU_OPTIONS)
  install(TARGETS ${DMT_GPU_OPTIONS} EXPORT dmtTargets)
endif()

# Install headers
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# dmtConfig.cmake's find_dependency(FFTW) (static builds) needs the finder.
install(FILES ${PROJECT_SOURCE_DIR}/cmake/FindFFTW.cmake DESTINATION ${DMT_CMAKE_PACKAGE_DIR})

# Generate and install export file
install(
  EXPORT dmtTargets
  FILE dmtTargets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${DMT_CMAKE_PACKAGE_DIR}
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
        DESTINATION ${DMT_CMAKE_PACKAGE_DIR}
)

# Print feature summary
include(FeatureSummary)
feature_summary(WHAT ALL)

set(_dmt_modes "CPU")
if(DMT_GPU_BACKEND STREQUAL "CUDA")
  string(APPEND _dmt_modes " + CUDA (${CMAKE_CUDA_ARCHITECTURES})")
elseif(DMT_GPU_BACKEND STREQUAL "HIP")
  string(APPEND _dmt_modes " + HIP (${CMAKE_HIP_ARCHITECTURES})")
endif()
message(STATUS "dmt build mode: ${_dmt_modes} (DMT_GPU=${DMT_GPU})")
