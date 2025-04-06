cmake_minimum_required(VERSION 3.22)

project(
  dmt
  VERSION 0.2.0
  LANGUAGES CXX
  DESCRIPTION "Dispersion Measure Transform Library"
)

# Modern CMake policies
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 NEW)
endif()
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

# Configuration options
option(BUILD_DOCS "Build documentation" OFF)
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TESTING "Build tests" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(ENABLE_IPO "Enable IPO/LTO" ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set default build type
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

# Interprocedural Optimization (IPO/LTO)
if(CMAKE_BUILD_TYPE MATCHES "Release" AND ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported)
  if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "IPO/LTO enabled for Release build.")
  else()
    message(WARNING "IPO/LTO requested but not supported.")
  endif()
else()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
endif()

# Compiler flags (Interface target to hold global compile options)
set(DMT_COMPILE_OPTIONS ${PROJECT_NAME}_COMPILE_OPTIONS)
add_library(${DMT_COMPILE_OPTIONS} INTERFACE)
target_compile_options(${DMT_COMPILE_OPTIONS} INTERFACE -Wall -Wextra -Wpedantic)
target_compile_options(
  ${DMT_COMPILE_OPTIONS}
  INTERFACE $<$<CONFIG:Debug>:-g> $<$<CONFIG:Release>:-O3;-DNDEBUG>
            $<$<CONFIG:MinSizeRel>:-Os;-DNDEBUG>
            $<$<AND:$<NOT:$<CONFIG:Debug>>,$<COMPILE_LANGUAGE:CXX>>:-march=native;-ffast-math>
)

# Project structure and paths
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(GNUInstallDirs)

# Find CUDA and set up the project
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10.0.0)
    message(FATAL_ERROR "CUDA 10.0 or higher is required. Found: ${CMAKE_CUDA_COMPILER_VERSION}")
  else()
    message(STATUS "Found CUDA ${CMAKE_CUDA_COMPILER_VERSION}.")
  endif()

  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -c -Xcompiler=-Wall,-Wextra --expt-extended-lambda")
  set(CMAKE_CUDA_FLAGS_DEBUG "-g")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -use_fast_math -Xcompiler=-march=native")
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
  add_compile_definitions(DMT_ENABLE_CUDA)
else()
  message(STATUS "CUDA compiler not found. GPU code will not be built.")
  remove_definitions(DMT_ENABLE_CUDA)
endif()

# CPM.cmake Setup
include(cmake/CPM.cmake)
set(CPM_USE_LOCAL_PACKAGES
    ON
    CACHE BOOL "Try to find packages locally before downloading" FORCE
)
if(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_SOURCE_CACHE
      $ENV{CPM_SOURCE_CACHE}
      CACHE PATH "Directory to cache downloaded CPM sources"
  )
endif()

# Define library names and Add subdirectories
set(DMT_LIBRARY_NAME dmt)
set(DMT_PYTHON_LIBRARY_NAME dmtlib)
add_subdirectory(lib)
if(BUILD_PYTHON)
  add_subdirectory(src)
endif()
if(BUILD_TESTING)
  add_subdirectory(tests/cpp)
endif()
if(BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

# Print feature summary
include(FeatureSummary)
feature_summary(WHAT ALL)
