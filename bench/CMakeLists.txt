CPMAddPackage(
  NAME benchmark
  VERSION 1.8.0
  GITHUB_REPOSITORY google/benchmark
  OPTIONS "BENCHMARK_ENABLE_TESTING Off"
)

set(TARGET_BENCHMARKS ${PROJECT_NAME}_bench)
file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${TARGET_BENCHMARKS} ${TEST_SOURCES})
target_include_directories(${TARGET_BENCHMARKS} PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(${TARGET_BENCHMARKS} PRIVATE ${LIBRARY_NAME} benchmark::benchmark)
