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
# Test executable
set(TARGET_TESTS ${PROJECT_NAME}_tests)

# Collect C++ test sources: host tests (every build; the backend-generic
# ones loop over the backends available at run time) and the GPU tests,
# CUDA source compiled for this build's GPU backend (CUDA or HIP).
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
if(DMT_GPU_BACKEND)
  file(GLOB TEST_SOURCES_GPU CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
  set_source_files_properties(${TEST_SOURCES_GPU} PROPERTIES LANGUAGE ${DMT_GPU_BACKEND})
  list(APPEND TEST_SOURCES ${TEST_SOURCES_GPU})
endif()

add_executable(${TARGET_TESTS} ${TEST_SOURCES})

target_include_directories(
  ${TARGET_TESTS} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib
                          ${CMAKE_CURRENT_SOURCE_DIR}
)

# The GPU tests use the library's private helpers and backend guards.
target_compile_definitions(${TARGET_TESTS} PRIVATE ${DMT_BACKEND_DEFINITIONS})
if(DMT_GPU_BACKEND STREQUAL "CUDA")
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    target_include_directories(${TARGET_TESTS} PRIVATE "${CUDAToolkit_INCLUDE_DIRS}/cccl")
  endif()
  target_include_directories(${TARGET_TESTS} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

target_link_libraries(
  ${TARGET_TESTS}
  PRIVATE ${PROJECT_NAME}::${DMT_LIBRARY_NAME}
          Catch2::Catch2WithMain
          ${DMT_COMPILE_OPTIONS}
          OpenMP::OpenMP_CXX
)
# After DMT_COMPILE_OPTIONS, so the GPU flags come last on the command line.
if(DMT_GPU_BACKEND)
  target_link_libraries(${TARGET_TESTS} PRIVATE ${DMT_GPU_OPTIONS} ${DMT_GPU_LIBRARIES})
endif()

dmt_enable_coverage(${TARGET_TESTS})

# Device tests are tagged [gpu] (or [parity]); they run on this build's GPU
# backend and are labelled `gpu` plus the backend name, so `ctest -LE gpu`
# runs the host tests only.
include(Catch)
if(DMT_GPU_BACKEND)
  string(TOLOWER ${DMT_GPU_BACKEND} _dmt_gpu_name)
  catch_discover_tests(
    ${TARGET_TESTS}
    TEST_SPEC
    "~[gpu]"
    TEST_PREFIX
    "cpu:"
    PROPERTIES
    LABELS
    "cpu"
  )
  catch_discover_tests(
    ${TARGET_TESTS}
    TEST_SPEC
    "[gpu]"
    TEST_PREFIX
    "${_dmt_gpu_name}:"
    PROPERTIES
    LABELS
    "gpu;${_dmt_gpu_name}"
  )
else()
  catch_discover_tests(
    ${TARGET_TESTS}
    TEST_SPEC
    "~[gpu]"
    PROPERTIES
    LABELS
    "cpu"
  )
endif()
