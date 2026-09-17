# Catch2 test framework (CPM).
CPMFindPackage(
  NAME Catch2
  VERSION 3.16.0
  GITHUB_REPOSITORY catchorg/Catch2
  OPTIONS "CATCH_INSTALL_DOCS OFF" "CATCH_INSTALL_EXTRAS OFF"
)
if(NOT Catch2_FOUND AND NOT TARGET Catch2::Catch2WithMain)
  message(
    FATAL_ERROR "Catch2 is required for DMT_BUILD_TESTING=ON but was not found or downloaded."
  )
endif()

list(APPEND CMAKE_MODULE_PATH ${Catch2_DIR})
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

include(CTest)
# Test executable
set(TARGET_TESTS ${PROJECT_NAME}_tests)

# Collect C++ test sources
file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Add CUDA test sources if available
if(CMAKE_CUDA_COMPILER)
  file(GLOB TEST_SOURCES_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
  list(APPEND TEST_SOURCES ${TEST_SOURCES_CUDA})

  # Explicitly set CUDA files to be compiled with CUDA
  set_source_files_properties(${TEST_SOURCES_CUDA} PROPERTIES LANGUAGE CUDA)
endif()

add_executable(${TARGET_TESTS} ${TEST_SOURCES})

target_include_directories(
  ${TARGET_TESTS} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib
                          ${CMAKE_CURRENT_SOURCE_DIR}
)

if(CMAKE_CUDA_COMPILER)
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    target_include_directories(${TARGET_TESTS} PRIVATE "${CUDAToolkit_INCLUDE_DIRS}/cccl")
  endif()
  target_include_directories(${TARGET_TESTS} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

target_link_libraries(
  ${TARGET_TESTS} PRIVATE ${PROJECT_NAME}::${DMT_LIBRARY_NAME} Catch2::Catch2WithMain
                          ${DMT_COMPILE_OPTIONS} spdlog::spdlog_header_only
)
dmt_enable_coverage(${TARGET_TESTS})

if(CMAKE_CUDA_COMPILER)
  target_link_libraries(${TARGET_TESTS} PRIVATE ${DMT_CUDA_OPTIONS})
endif()

include(Catch)
if(CMAKE_CUDA_COMPILER)
  catch_discover_tests(
    ${TARGET_TESTS}
    TEST_SPEC
    "~[gpu]~[parity]"
    TEST_PREFIX
    "cpu:"
    PROPERTIES
    LABELS
    "cpu"
  )
  catch_discover_tests(
    ${TARGET_TESTS}
    TEST_SPEC
    "[gpu],[parity]"
    TEST_PREFIX
    "gpu:"
    PROPERTIES
    LABELS
    "gpu"
  )
else()
  catch_discover_tests(${TARGET_TESTS} PROPERTIES LABELS "cpu")
endif()
