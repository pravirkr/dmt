CPMFindPackage(
  NAME benchmark
  VERSION 1.9.5
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.5.tar.gz
  OPTIONS "BENCHMARK_ENABLE_WERROR OFF" "BENCHMARK_FORCE_WERROR OFF" "BENCHMARK_ENABLE_TESTING OFF"
)

# Define benchmark targets
set(TARGET_BENCHMARKS ${PROJECT_NAME}_bench)
set(TARGET_BENCHMARKS_MEM ${PROJECT_NAME}_bench_mem)

# Collect CPU benchmark sources: FDMT, FDMT-FFT, and DDMT on equal footing.
set(BENCH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_cpu_b.cpp ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_fft_cpu_b.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ddmt_cpu_b.cpp
)

# Add CUDA benchmark sources if available: same three algorithms.
if(CMAKE_CUDA_COMPILER)
  set(BENCH_SOURCES_CUDA
      ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_cuda_b.cu ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_fft_cuda_b.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/ddmt_cuda_b.cu
  )
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

# Memory benchmarks executable: FDMT, FDMT-FFT, and DDMT (float + packed),
# split into one file per algorithm (mirroring the timing executable above)
# plus a shared bench_mem_utils.hpp and a single mem_b_main.cpp anchor TU
# that owns the global operator new/delete overrides and main() -- these
# must be defined exactly once across the binary (a replacement allocation
# function must not be declared inline per [new.delete]), so they cannot
# live in the shared header included by every algorithm file. Kept as a
# separate target/binary since it installs those overrides and a custom
# main() (incompatible with benchmark::benchmark_main, which the timing
# executable above links).
set(BENCH_SOURCES_MEM
    ${CMAKE_CURRENT_SOURCE_DIR}/mem_b_main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_mem_b.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_fft_mem_b.cpp ${CMAKE_CURRENT_SOURCE_DIR}/ddmt_mem_b.cpp
)

add_executable(${TARGET_BENCHMARKS_MEM} ${BENCH_SOURCES_MEM})

target_include_directories(
  ${TARGET_BENCHMARKS_MEM} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/lib
)

target_link_libraries(
  ${TARGET_BENCHMARKS_MEM} PRIVATE ${PROJECT_NAME}::${DMT_LIBRARY_NAME} benchmark::benchmark
                                   ${DMT_COMPILE_OPTIONS}
)
