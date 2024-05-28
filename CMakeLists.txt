cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0148 OLD)

project(
  dmt
  VERSION 0.2.0
  LANGUAGES CXX
)

# Configuration options
option(BUILD_DOCS "Build documentation" OFF)
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TESTING "Build tests" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(CODE_COVERAGE "Enable coverage reporting" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
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
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS_EQUAL 10.0.0)
    message(FATAL_ERROR "Found CUDA ${CMAKE_CUDA_COMPILER_VERSION}. Need CUDA 10+")
  else()
    message(STATUS "Found CUDA ${CMAKE_CUDA_COMPILER_VERSION}.")
  endif()

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_FLAGS "-c -Xcompiler=-Wall,-Wextra --expt-extended-lambda")
  set(CMAKE_CUDA_FLAGS_DEBUG "-g")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -use_fast_math -Xcompiler=-march=native")
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
elseif(NOT CMAKE_CUDA_COMPILER)
  message(STATUS "Not building gpu code. No CUDA compiler found...")
endif(CMAKE_CUDA_COMPILER)

# Initialize some default paths
include(GNUInstallDirs)
include(cmake/CPM.cmake)

# Code Coverage Configuration
add_library(coverage_config INTERFACE)
if(CODE_COVERAGE)
  target_compile_options(coverage_config INTERFACE -O0 -g --coverage)
  target_link_options(coverage_config INTERFACE --coverage)
endif()

# Build the C++ library
set(LIBRARY_NAME dmt)
add_subdirectory(lib)
# Build the Python bindings
if(BUILD_PYTHON)
  add_subdirectory(src)
endif()
# Build tests
if(BUILD_TESTING)
  add_subdirectory(tests/cpp)
endif()
# Build benchmarks
if(BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

# Install targets and configuration
install(
  TARGETS dmt
  EXPORT dmt_config
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  EXPORT dmt_config
  NAMESPACE dmt::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dmt
)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
