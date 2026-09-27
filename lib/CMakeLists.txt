# System dependencies
message(STATUS "Searching for required system dependencies...")
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(FFTW REQUIRED COMPONENTS FLOAT_LIB)

# fmt — keep version in sync with spdlog's expected fmt release.
CPMAddPackage(
  NAME fmt
  VERSION 12.1.0
  URL https://github.com/fmtlib/fmt/archive/refs/tags/12.1.0.tar.gz
  OPTIONS "FMT_INSTALL ON" "FMT_HEADER_ONLY ON"
  EXCLUDE_FROM_ALL YES
  SYSTEM YES
)

# spdlog — use the fmt package above (header-only external fmt).
CPMAddPackage(
  NAME spdlog
  VERSION 1.17.0
  URL https://github.com/gabime/spdlog/archive/refs/tags/v1.17.0.tar.gz
  OPTIONS "SPDLOG_INSTALL ON" "SPDLOG_FMT_EXTERNAL_HO ON" "SPDLOG_BUILD_SHARED OFF"
  EXCLUDE_FROM_ALL YES
  SYSTEM YES
)

# Collect source files: backend-neutral facades and plans in lib/, one
# directory per backend (lib/cpu always, lib/cuda when nvcc is found).
file(GLOB DMT_LIBRARY_CXX_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.cpp
)
set(DMT_LIBRARY_ALL_SOURCES ${DMT_LIBRARY_CXX_SOURCES})

# Add CUDA sources if available
if(CMAKE_CUDA_COMPILER)
  file(GLOB DMT_LIBRARY_CUDA_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu)
  list(APPEND DMT_LIBRARY_ALL_SOURCES ${DMT_LIBRARY_CUDA_SOURCES})

  # Explicitly set CUDA files to be compiled with CUDA
  set_source_files_properties(${DMT_LIBRARY_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

add_library(${DMT_LIBRARY_NAME} ${DMT_LIBRARY_ALL_SOURCES})
add_library(${PROJECT_NAME}::${DMT_LIBRARY_NAME} ALIAS ${DMT_LIBRARY_NAME})

target_compile_features(${DMT_LIBRARY_NAME} PUBLIC cxx_std_20)

# DMT_ENABLE_CUDA is private: it only selects which engines the facades can
# build. The installed headers are identical in every build; callers ask
# dmt::available_backends() at runtime.
if(CMAKE_CUDA_COMPILER)
  target_compile_definitions(${DMT_LIBRARY_NAME} PRIVATE DMT_ENABLE_CUDA)
endif()

dmt_enable_coverage(${DMT_LIBRARY_NAME})
if(DMT_ENABLE_COVERAGE)
  message(STATUS "Code coverage enabled.")
endif()

# Set include directories
target_include_directories(
  ${DMT_LIBRARY_NAME}
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# CUDA headers are needed by the library's own GPU sources only.
if(CMAKE_CUDA_COMPILER)
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    target_include_directories(${DMT_LIBRARY_NAME} PRIVATE "${CUDAToolkit_INCLUDE_DIRS}/cccl")
  endif()
  target_include_directories(${DMT_LIBRARY_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

# Header-only internal deps (not part of the installed export set, so even a
# static dmt does not ask consumers for these targets).
target_link_libraries(
  ${DMT_LIBRARY_NAME} PRIVATE $<BUILD_INTERFACE:${DMT_COMPILE_OPTIONS}>
                              $<BUILD_INTERFACE:spdlog::spdlog_header_only>
)

# FFTW: shared dmt links privately; static dmt exposes FFTW to downstream linkers.
if(BUILD_SHARED_LIBS)
  target_link_libraries(${DMT_LIBRARY_NAME} PRIVATE FFTW::Float)
else()
  target_link_libraries(${DMT_LIBRARY_NAME} PUBLIC FFTW::Float)
endif()

# OpenMP stays private: public headers do not include <omp.h> or pragmas.
# A static lib still propagates the runtime onto the final link line.
target_link_libraries(${DMT_LIBRARY_NAME} PRIVATE OpenMP::OpenMP_CXX)

# Add CUDA options and libraries when available. Private: a static dmt still
# carries them onto the consumer's link line (as $<LINK_ONLY:...>).
if(CMAKE_CUDA_COMPILER)
  target_link_libraries(
    ${DMT_LIBRARY_NAME} PRIVATE ${DMT_CUDA_OPTIONS} CUDA::cuda_driver CUDA::cudart CUDA::cufft
  )
endif()
