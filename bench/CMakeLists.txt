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
    ${CMAKE_CURRENT_SOURCE_DIR}/ddmt_cpu_b.cpp ${CMAKE_CURRENT_SOURCE_DIR}/ddmt_fft_cpu_b.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cfdmt_cpu_b.cpp
)

# GPU benchmark sources (CUDA source, built for CUDA or HIP): same three
# algorithms.
if(DMT_GPU_BACKEND)
  set(BENCH_SOURCES_CUDA
      ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_cuda_b.cu ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_fft_cuda_b.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/ddmt_cuda_b.cu ${CMAKE_CURRENT_SOURCE_DIR}/ddmt_fft_cuda_b.cu
  )
  list(APPEND BENCH_SOURCES ${BENCH_SOURCES_CUDA})

  set_source_files_properties(${BENCH_SOURCES_CUDA} PROPERTIES LANGUAGE ${DMT_GPU_BACKEND})
endif()

# Performance benchmarks executable
add_executable(${TARGET_BENCHMARKS} ${BENCH_SOURCES})

target_include_directories(
  ${TARGET_BENCHMARKS} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/lib
)

if(DMT_GPU_BACKEND STREQUAL "CUDA")
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    target_include_directories(${TARGET_BENCHMARKS} PRIVATE "${CUDAToolkit_INCLUDE_DIRS}/cccl")
  endif()
  target_include_directories(${TARGET_BENCHMARKS} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

target_link_libraries(
  ${TARGET_BENCHMARKS}
  PRIVATE ${PROJECT_NAME}::${DMT_LIBRARY_NAME}
          benchmark::benchmark_main
          OpenMP::OpenMP_CXX
          ${DMT_COMPILE_OPTIONS}
)

target_compile_definitions(${TARGET_BENCHMARKS} PRIVATE ${DMT_BACKEND_DEFINITIONS})
if(DMT_GPU_BACKEND)
  target_link_libraries(${TARGET_BENCHMARKS} PRIVATE ${DMT_GPU_OPTIONS} ${DMT_GPU_LIBRARIES})
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
    ${CMAKE_CURRENT_SOURCE_DIR}/mem_b_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_mem_b.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fdmt_fft_mem_b.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ddmt_mem_b.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ddmt_fft_mem_b.cpp
)

add_executable(${TARGET_BENCHMARKS_MEM} ${BENCH_SOURCES_MEM})

target_include_directories(
  ${TARGET_BENCHMARKS_MEM} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/lib
)

target_link_libraries(
  ${TARGET_BENCHMARKS_MEM}
  PRIVATE ${PROJECT_NAME}::${DMT_LIBRARY_NAME}
          benchmark::benchmark
          OpenMP::OpenMP_CXX
          ${DMT_COMPILE_OPTIONS}
)

# Published benchmark suite (bench/suite): one fixed configuration shared by
# FDMT, FDMT-FFT and DDMT on CPU and GPU (CUDA or HIP), run by bench/scripts/run_suite.py
# and plotted by bench/scripts/plot_suite.py. See bench/README.md.
set(TARGET_BENCHMARKS_SUITE ${PROJECT_NAME}_bench_suite)
set(BENCH_SOURCES_SUITE ${CMAKE_CURRENT_SOURCE_DIR}/suite/suite_cpu.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/suite/suite_cfdmt_cpu.cpp
)
if(DMT_GPU_BACKEND)
  set(BENCH_SOURCES_SUITE_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/suite/suite_cuda.cu
                               ${CMAKE_CURRENT_SOURCE_DIR}/suite/suite_cfdmt_cuda.cu
  )
  set_source_files_properties(
    ${BENCH_SOURCES_SUITE_CUDA} PROPERTIES LANGUAGE ${DMT_GPU_BACKEND}
  )
  list(APPEND BENCH_SOURCES_SUITE ${BENCH_SOURCES_SUITE_CUDA})
endif()

add_executable(${TARGET_BENCHMARKS_SUITE} ${BENCH_SOURCES_SUITE})
target_include_directories(
  ${TARGET_BENCHMARKS_SUITE} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/lib
                                     ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(${TARGET_BENCHMARKS_SUITE} PRIVATE ${DMT_BACKEND_DEFINITIONS})
if(DMT_GPU_BACKEND STREQUAL "CUDA")
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    target_include_directories(
      ${TARGET_BENCHMARKS_SUITE} PRIVATE "${CUDAToolkit_INCLUDE_DIRS}/cccl"
    )
  endif()
  target_include_directories(${TARGET_BENCHMARKS_SUITE} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()
target_link_libraries(
  ${TARGET_BENCHMARKS_SUITE} PRIVATE ${PROJECT_NAME}::${DMT_LIBRARY_NAME} benchmark::benchmark_main
                                     ${DMT_COMPILE_OPTIONS}
)
if(DMT_GPU_BACKEND)
  target_link_libraries(${TARGET_BENCHMARKS_SUITE} PRIVATE ${DMT_GPU_OPTIONS} ${DMT_GPU_LIBRARIES})
endif()
