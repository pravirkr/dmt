include(CTest)

CPMFindPackage(
  NAME Catch2
  VERSION 3.8.0
  GITHUB_REPOSITORY catchorg/Catch2
)
list(APPEND CMAKE_MODULE_PATH ${Catch2_DIR})
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

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
)

if(CMAKE_CUDA_COMPILER)
  target_include_directories(${TARGET_TESTS} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

target_link_libraries(
  ${TARGET_TESTS} PRIVATE ${PROJECT_NAME}::${DMT_LIBRARY_NAME} Catch2::Catch2WithMain
                          ${DMT_COMPILE_OPTIONS}
)

if(CMAKE_CUDA_COMPILER)
  target_link_libraries(${TARGET_TESTS} PRIVATE ${DMT_CUDA_OPTIONS})
endif()

# Discover tests
include(Catch)
catch_discover_tests(${TARGET_TESTS})
