CPMFindPackage(
  NAME benchmark #VERSION 1.9
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.0.zip
  OPTIONS "BENCHMARK_ENABLE_WERROR OFF" "BENCHMARK_FORCE_WERROR OFF" "BENCHMARK_ENABLE_TESTING OFF"
)

set(TARGET_BENCHMARKS ${PROJECT_NAME}_bench)
set(TARGET_BENCHMARKS_MEM ${PROJECT_NAME}_bench_mem)
file(GLOB BENCH_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_cpu_b.cpp)
file(GLOB BENCH_SOURCES_MEM ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_cpu_b_mem.cpp)

# Check if CUDA was found
if(CMAKE_CUDA_COMPILER)
  file(GLOB BENCH_SOURCES_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
  list(APPEND BENCH_SOURCES ${BENCH_SOURCES_CUDA})
endif()

# Performance benchmarks
add_executable(${TARGET_BENCHMARKS} ${BENCH_SOURCES})
target_include_directories(
  ${TARGET_BENCHMARKS} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/lib
)
target_link_libraries(
  ${TARGET_BENCHMARKS} PRIVATE ${PROJECT_NAME}::${LIBRARY_NAME} benchmark::benchmark_main
)

# Memory benchmarks
add_executable(${TARGET_BENCHMARKS_MEM} ${BENCH_SOURCES_MEM})
target_include_directories(
  ${TARGET_BENCHMARKS_MEM} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/lib
)
target_link_libraries(
  ${TARGET_BENCHMARKS_MEM} PRIVATE ${PROJECT_NAME}::${LIBRARY_NAME} benchmark::benchmark_main
)
