# Catch2: prefer system package (apt/brew) when available; pin 3.16.0 via CPM otherwise.
find_package(
  Catch2
  3
  CONFIG
  QUIET
)
if(Catch2_FOUND)
  message(STATUS "Using system Catch2 ${Catch2_VERSION}")
else()
  CPMFindPackage(
    NAME Catch2
    VERSION 3.16.0
    GITHUB_REPOSITORY catchorg/Catch2
    OPTIONS "CATCH_INSTALL_DOCS OFF" "CATCH_INSTALL_EXTRAS OFF"
  )
endif()
if(NOT TARGET Catch2::Catch2WithMain)
  message(
    FATAL_ERROR "Catch2 is required for DMT_BUILD_TESTING=ON but was not found or downloaded."
  )
endif()

if(Catch2_DIR)
  list(APPEND CMAKE_MODULE_PATH ${Catch2_DIR})
endif()
if(Catch2_SOURCE_DIR)
  list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
endif()

include(CTest)
include(Catch)

# Test executables:
# - dmt_tests: public API tests (compiles against include/ only, no lib/).
# - dmt_internal_tests: internal/white-box unit tests (keeps lib/ on include path).
# - dmt_alloc_tests: replaces global operator new (include/ only, no lib/).
set(TARGET_TESTS ${PROJECT_NAME}_tests)
set(TARGET_INTERNAL_TESTS ${PROJECT_NAME}_internal_tests)
set(TARGET_ALLOC_TESTS ${PROJECT_NAME}_alloc_tests)

set(ALLOC_TEST_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_alloc_cpu_t.cpp)

set(
  INTERNAL_TEST_SOURCES_CPU
  ${CMAKE_CURRENT_SOURCE_DIR}/boxcar_plans_trace_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ddmt_cpu_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ddmt_fft_cpu_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dm_grids_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dm_utils_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_boxcar_kadane_cpu_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_boxcar_mask_cpu_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_chunk_cpu_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_cpu_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_fft_api_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_fft_boxcar_cpu_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_fft_cpu_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_fused_tile_cpu_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_skew_cpu_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_cpu_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kadane_monoid_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kadane_plan_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kadane_simd_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/logging_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/planned_vector_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/simd_bitpack_cpu_t.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/unpacker_cpu_t.cpp
)

set(
  INTERNAL_TEST_SOURCES_GPU
  ${CMAKE_CURRENT_SOURCE_DIR}/cfdmt_cuda_t.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/ddmt_cuda_t.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/ddmt_fft_cuda_t.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/ddmt_tiled_parity_t.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_boxcar_cuda_t.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_boxcar_mask_cuda_t.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_fft_boxcar_cuda_t.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_skew_cuda_t.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_cuda_t.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_engines_cuda_t.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/planned_vector_cuda_t.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/unpacker_cuda_t.cu
)

# Any new test file added to tests/cpp/ defaults to the public-API target
# (no lib/ include path), enforcing the public header boundary by default.
file(GLOB PUBLIC_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
list(REMOVE_ITEM PUBLIC_TEST_SOURCES ${ALLOC_TEST_SOURCE} ${INTERNAL_TEST_SOURCES_CPU})

set(INTERNAL_TEST_SOURCES ${INTERNAL_TEST_SOURCES_CPU})
if(DMT_GPU_BACKEND)
  file(GLOB PUBLIC_TEST_SOURCES_GPU CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
  list(REMOVE_ITEM PUBLIC_TEST_SOURCES_GPU ${INTERNAL_TEST_SOURCES_GPU})
  set_source_files_properties(
    ${PUBLIC_TEST_SOURCES_GPU} ${INTERNAL_TEST_SOURCES_GPU} PROPERTIES LANGUAGE ${DMT_GPU_BACKEND}
  )
  list(APPEND PUBLIC_TEST_SOURCES ${PUBLIC_TEST_SOURCES_GPU})
  list(APPEND INTERNAL_TEST_SOURCES ${INTERNAL_TEST_SOURCES_GPU})
endif()

# Common settings of a test executable; include_lib adds lib/ to its include path.
function(dmt_configure_test_target target_name include_lib)
  if(include_lib)
    target_include_directories(
      ${target_name} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib
                             ${CMAKE_CURRENT_SOURCE_DIR}
    )
  else()
    target_include_directories(
      ${target_name} PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}
    )
  endif()
  target_compile_definitions(${target_name} PRIVATE ${DMT_BACKEND_DEFINITIONS})
  if(DMT_GPU_BACKEND STREQUAL "CUDA")
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
      target_include_directories(${target_name} PRIVATE "${CUDAToolkit_INCLUDE_DIRS}/cccl")
    endif()
    target_include_directories(${target_name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  endif()
  # DMT_COMPILE_OPTIONS (the CPM fmt) before Catch2, so a system fmt installed
  # next to Catch2 cannot shadow it (see bench/CMakeLists.txt).
  target_link_libraries(
    ${target_name}
    PRIVATE ${PROJECT_NAME}::${DMT_LIBRARY_NAME}
            ${DMT_COMPILE_OPTIONS}
            Catch2::Catch2WithMain
            OpenMP::OpenMP_CXX
  )
  # After DMT_COMPILE_OPTIONS, so the GPU flags come last on the command line.
  if(DMT_GPU_BACKEND)
    target_link_libraries(${target_name} PRIVATE ${DMT_GPU_OPTIONS} ${DMT_GPU_LIBRARIES})
  endif()
  dmt_enable_coverage(${target_name})
endfunction()

add_executable(${TARGET_TESTS} ${PUBLIC_TEST_SOURCES})
dmt_configure_test_target(${TARGET_TESTS} FALSE)

add_executable(${TARGET_INTERNAL_TESTS} ${INTERNAL_TEST_SOURCES})
dmt_configure_test_target(${TARGET_INTERNAL_TESTS} TRUE)

add_executable(${TARGET_ALLOC_TESTS} ${ALLOC_TEST_SOURCE})
target_include_directories(
  ${TARGET_ALLOC_TESTS} PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(${TARGET_ALLOC_TESTS} PRIVATE ${DMT_BACKEND_DEFINITIONS})
target_link_libraries(
  ${TARGET_ALLOC_TESTS}
  PRIVATE ${PROJECT_NAME}::${DMT_LIBRARY_NAME}
          ${DMT_COMPILE_OPTIONS}
          Catch2::Catch2WithMain
          OpenMP::OpenMP_CXX
)
dmt_enable_coverage(${TARGET_ALLOC_TESTS})

# Device tests are tagged [gpu] (or [parity]); they run on this build's GPU
# backend (test names prefixed with it) and are labelled `gpu`, so
# `ctest -LE gpu` runs the host tests only. Full-size versions of the
# expensive host tests are tagged [slow] and labelled `slow` instead of `cpu`
# (test_sized.hpp); the CI test presets skip them. One label each:
# catch_discover_tests does not pass a label list through. Every test runs
# on one thread (OMP_NUM_THREADS=1): ctest runs tests in parallel, one per
# core.
function(dmt_register_catch_tests target_name)
  if(DMT_GPU_BACKEND)
    string(TOLOWER ${DMT_GPU_BACKEND} _dmt_gpu_name)
    catch_discover_tests(
      ${target_name}
      TEST_SPEC
      "~[gpu]~[slow]"
      TEST_PREFIX
      "cpu:"
      PROPERTIES
      LABELS
      "cpu"
      ENVIRONMENT
      "OMP_NUM_THREADS=1"
    )
    catch_discover_tests(
      ${target_name}
      TEST_SPEC
      "[slow]~[gpu]"
      TEST_PREFIX
      "cpu:"
      PROPERTIES
      LABELS
      "slow"
      ENVIRONMENT
      "OMP_NUM_THREADS=1"
    )
    catch_discover_tests(
      ${target_name}
      TEST_SPEC
      "[gpu]"
      TEST_PREFIX
      "${_dmt_gpu_name}:"
      PROPERTIES
      LABELS
      "gpu"
      ENVIRONMENT
      "OMP_NUM_THREADS=1"
    )
  else()
    catch_discover_tests(
      ${target_name}
      TEST_SPEC
      "~[gpu]~[slow]"
      PROPERTIES
      LABELS
      "cpu"
      ENVIRONMENT
      "OMP_NUM_THREADS=1"
    )
    catch_discover_tests(
      ${target_name}
      TEST_SPEC
      "[slow]~[gpu]"
      PROPERTIES
      LABELS
      "slow"
      ENVIRONMENT
      "OMP_NUM_THREADS=1"
    )
  endif()
endfunction()

dmt_register_catch_tests(${TARGET_TESTS})
dmt_register_catch_tests(${TARGET_INTERNAL_TESTS})

if(DMT_GPU_BACKEND)
  catch_discover_tests(
    ${TARGET_ALLOC_TESTS}
    TEST_PREFIX
    "cpu:"
    PROPERTIES
    LABELS
    "cpu"
    ENVIRONMENT
    "OMP_NUM_THREADS=1"
  )
else()
  catch_discover_tests(
    ${TARGET_ALLOC_TESTS}
    PROPERTIES
    LABELS
    "cpu"
    ENVIRONMENT
    "OMP_NUM_THREADS=1"
  )
endif()
