cmake_minimum_required(VERSION 3.18)

project(
  dmt
  VERSION 0.2.0
  LANGUAGES CXX
  DESCRIPTION "Dispersion Measure Transform Library"
)

# Configuration options
option(BUILD_DOCS "Build documentation" OFF)
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TESTING "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(ENABLE_CODE_COVERAGE "Enable coverage reporting" OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add custom modules path
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

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

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -c -Xcompiler=-Wall,-Wextra --expt-extended-lambda")
  set(CMAKE_CUDA_FLAGS_DEBUG "-g")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -use_fast_math -Xcompiler=-march=native")
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
else()
  message(STATUS "CUDA compiler not found. GPU code will not be built.")
endif()

# Initialize necessary CMake modules
include(GNUInstallDirs)
include(cmake/CPM.cmake)

# Define library names
set(LIBRARY_NAME dmt)
set(PYLIBRARY_NAME dmtlib)

# Add subdirectories
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
