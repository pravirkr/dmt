CPMAddPackage(
  NAME benchmark
  VERSION 1.8.4
  GITHUB_REPOSITORY google/benchmark
  OPTIONS "BENCHMARK_ENABLE_WERROR OFF" "BENCHMARK_FORCE_WERROR OFF" "BENCHMARK_ENABLE_TESTING OFF"
)

set(TARGET_BENCHMARKS ${PROJECT_NAME}_bench)
file(GLOB BENCH_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Check if CUDA was found
if(CMAKE_CUDA_COMPILER)
  file(GLOB BENCH_SOURCES_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
  list(APPEND BENCH_SOURCES ${BENCH_SOURCES_CUDA})
endif()

add_executable(${TARGET_BENCHMARKS} ${BENCH_SOURCES})
target_include_directories(${TARGET_BENCHMARKS} PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(${TARGET_BENCHMARKS} PRIVATE ${LIBRARY_NAME} benchmark::benchmark_main)
