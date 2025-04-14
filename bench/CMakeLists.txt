CPMFindPackage(
  NAME benchmark
  VERSION 1.9.2
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.2.tar.gz
  OPTIONS "BENCHMARK_ENABLE_WERROR OFF" "BENCHMARK_FORCE_WERROR OFF" "BENCHMARK_ENABLE_TESTING OFF"
)

# Define benchmark targets
set(TARGET_BENCHMARKS ${PROJECT_NAME}_bench)
set(TARGET_BENCHMARKS_MEM ${PROJECT_NAME}_bench_mem)

# Collect CPU benchmark sources
file(GLOB BENCH_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_cpu_b.cpp)
file(GLOB BENCH_SOURCES_MEM ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_cpu_b_mem.cpp)

# Add CUDA benchmark sources if available
if(CMAKE_CUDA_COMPILER)
  file(GLOB BENCH_SOURCES_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
  list(APPEND BENCH_SOURCES ${BENCH_SOURCES_CUDA})

  # Explicitly set CUDA files to be compiled with CUDA
  set_source_files_properties(${BENCH_SOURCES_CUDA} PROPERTIES LANGUAGE CUDA)
endif()

# Performance benchmarks executable
add_executable(${TARGET_BENCHMARKS} ${BENCH_SOURCES})

target_include_directories(
  ${TARGET_BENCHMARKS} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/lib
)

if(CMAKE_CUDA_COMPILER)
  target_include_directories(${TARGET_BENCHMARKS} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

target_link_libraries(
  ${TARGET_BENCHMARKS} PRIVATE ${PROJECT_NAME}::${DMT_LIBRARY_NAME} benchmark::benchmark_main
                               ${DMT_COMPILE_OPTIONS}
)

if(CMAKE_CUDA_COMPILER)
  target_link_libraries(${TARGET_BENCHMARKS} PRIVATE ${DMT_CUDA_OPTIONS})
endif()

# Memory benchmarks executable
add_executable(${TARGET_BENCHMARKS_MEM} ${BENCH_SOURCES_MEM})

target_include_directories(
  ${TARGET_BENCHMARKS_MEM} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/lib
)

if(CMAKE_CUDA_COMPILER)
  target_include_directories(${TARGET_BENCHMARKS_MEM} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

target_link_libraries(
  ${TARGET_BENCHMARKS_MEM} PRIVATE ${PROJECT_NAME}::${DMT_LIBRARY_NAME} benchmark::benchmark_main
                                   ${DMT_COMPILE_OPTIONS}
)

if(CMAKE_CUDA_COMPILER)
  target_link_libraries(${TARGET_BENCHMARKS_MEM} PRIVATE ${DMT_CUDA_OPTIONS})
endif()
