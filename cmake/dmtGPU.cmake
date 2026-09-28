# GPU toolchain of lib/cuda/ (one per build): CUDA (nvcc) or HIP (ROCm
# clang). Sets
#   DMT_GPU_BACKEND    "CUDA", "HIP" or "" (no GPU backend)
#   DMT_GPU_OPTIONS    interface target with the GPU compile options (if any)
#   DMT_GPU_LIBRARIES  the GPU runtime, FFT and thrust targets to link
# The same CUDA sources are compiled by either toolchain: dmt/gpu_compat.cuh
# maps the CUDA runtime, cuFFT and thrust names onto HIP, hipFFT and
# rocThrust.

set(DMT_GPU
    "AUTO"
    CACHE STRING "GPU backend: AUTO (CUDA if nvcc is found, else HIP if ROCm is found, else none), \
CUDA, HIP, or OFF"
)
set_property(
  CACHE DMT_GPU
  PROPERTY STRINGS
           "AUTO"
           "CUDA"
           "HIP"
           "OFF"
)
if(NOT
   DMT_GPU
   MATCHES
   "^(AUTO|CUDA|HIP|OFF)$"
)
  message(FATAL_ERROR "DMT_GPU must be AUTO, CUDA, HIP or OFF. Got: '${DMT_GPU}'.")
endif()
if(DEFINED DMT_CUDA)
  message(
    FATAL_ERROR "DMT_CUDA was replaced by DMT_GPU=AUTO|CUDA|HIP|OFF. Remove it from the cache "
                "(cmake -UDMT_CUDA ...) and pass DMT_GPU instead."
  )
endif()
option(DMT_GPU_CHECK_DEVICE
       "With DMT_GPU=CUDA or HIP, require a visible GPU at configure time (OFF for build-only CI)"
       ON
)

# GPU architecture(s): "native" = detect from the local GPU at build time.
set(DMT_CUDA_ARCHITECTURES
    "native"
    CACHE STRING "CUDA architectures passed to CMAKE_CUDA_ARCHITECTURES (e.g. native, 80, 86;90)"
)
# HIP cannot detect without a GPU; default to the HPC parts (MI200, MI300).
set(DMT_HIP_ARCHITECTURES
    "gfx90a;gfx942"
    CACHE STRING "HIP architectures passed to CMAKE_HIP_ARCHITECTURES (e.g. gfx90a;gfx942;gfx1100)"
)

set(DMT_GPU_BACKEND "")
include(CheckLanguage)

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
if(DMT_GPU STREQUAL "CUDA" OR DMT_GPU STREQUAL "AUTO")
  # nvcc usually lives outside the default PATH; seed CUDACXX before check_language.
  if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(
      DMT_NVCC_EXECUTABLE nvcc
      HINTS ENV
            CUDA_PATH
            ENV
            CUDA_HOME
            ENV
            CUDA_ROOT
            /usr/local/cuda
            /opt/cuda
      PATH_SUFFIXES bin
    )
    if(DMT_NVCC_EXECUTABLE)
      set(ENV{CUDACXX} "${DMT_NVCC_EXECUTABLE}")
    endif()
  endif()
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(DMT_GPU_BACKEND "CUDA")
  elseif(DMT_GPU STREQUAL "CUDA")
    message(FATAL_ERROR "DMT_GPU=CUDA but no CUDA compiler (nvcc) was found. "
                        "Install CUDA >= 12.6 or use DMT_GPU=AUTO/OFF."
    )
  endif()
endif()

if(DMT_GPU_BACKEND STREQUAL "CUDA")
  if(DMT_GPU STREQUAL "CUDA" AND DMT_GPU_CHECK_DEVICE)
    # Explicitly required: verify a driver/GPU is present (not just nvcc).
    find_program(DMT_NVIDIA_SMI nvidia-smi)
    if(NOT DMT_NVIDIA_SMI)
      message(FATAL_ERROR "DMT_GPU=CUDA but nvidia-smi was not found. An NVIDIA driver and GPU "
                          "are required (or set DMT_GPU_CHECK_DEVICE=OFF)."
      )
    endif()
    execute_process(
      COMMAND ${DMT_NVIDIA_SMI} --query-gpu=name --format=csv,noheader
      OUTPUT_VARIABLE _dmt_gpu_names
      RESULT_VARIABLE _dmt_gpu_status
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    if(NOT
       _dmt_gpu_status
       EQUAL
       0
       OR _dmt_gpu_names STREQUAL ""
    )
      message(FATAL_ERROR "DMT_GPU=CUDA but nvidia-smi reports no usable GPU. Check driver "
                          "installation (or set DMT_GPU_CHECK_DEVICE=OFF)."
      )
    endif()
    message(STATUS "DMT_GPU=CUDA: detected GPU(s): ${_dmt_gpu_names}")
  endif()

  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.6.0)
    message(FATAL_ERROR "CUDA >= 12.6 is required. Found: ${CMAKE_CUDA_COMPILER_VERSION}")
  endif()
  message(STATUS "Found CUDA ${CMAKE_CUDA_COMPILER_VERSION}.")

  # Find CUDA Toolkit for proper include dirs and linking
  find_package(CUDAToolkit REQUIRED)
  set(DMT_GPU_LIBRARIES CUDA::cuda_driver CUDA::cudart CUDA::cufft)

  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  set(DMT_GPU_OPTIONS ${PROJECT_NAME}_gpu_options)
  add_library(${DMT_GPU_OPTIONS} INTERFACE)
  target_compile_options(
    ${DMT_GPU_OPTIONS}
    INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
              -Wno-pedantic
              --expt-extended-lambda
              --expt-relaxed-constexpr
              -Xcompiler=-Wall,-Wextra
              $<$<CONFIG:Debug>:-G;-g;-O0>
              $<$<CONFIG:Release>:-O3;-use_fast_math;-DNDEBUG>
              $<$<CONFIG:RelWithDebInfo>:-O2;-g;-lineinfo;-use_fast_math;-DNDEBUG>
              $<$<CONFIG:MinSizeRel>:-O2;-use_fast_math;-DNDEBUG>
              >
  )
  set(CMAKE_CUDA_ARCHITECTURES ${DMT_CUDA_ARCHITECTURES})
  message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  # Disable response files for better IDE integration
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
endif()

# ---------------------------------------------------------------------------
# HIP
# ---------------------------------------------------------------------------
if(DMT_GPU_BACKEND STREQUAL "" AND (DMT_GPU STREQUAL "HIP" OR DMT_GPU STREQUAL "AUTO"))
  if(CMAKE_VERSION VERSION_LESS 3.21)
    if(DMT_GPU STREQUAL "HIP")
      message(FATAL_ERROR "DMT_GPU=HIP requires CMake >= 3.21 (found ${CMAKE_VERSION}).")
    endif()
  else()
    # ROCm's clang is outside the default PATH; seed HIPCXX before check_language.
    set(_dmt_rocm_roots $ENV{ROCM_PATH} $ENV{HIP_PATH} /opt/rocm)
    if(NOT DEFINED CMAKE_HIP_COMPILER AND NOT DEFINED ENV{HIPCXX})
      find_program(
        DMT_HIP_CLANG clang++
        HINTS ${_dmt_rocm_roots}
        PATH_SUFFIXES llvm/bin
        NO_DEFAULT_PATH
      )
      if(DMT_HIP_CLANG)
        set(ENV{HIPCXX} "${DMT_HIP_CLANG}")
      endif()
    endif()
    # Without a GPU, CMake cannot detect the architectures: always give them.
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
      set(CMAKE_HIP_ARCHITECTURES ${DMT_HIP_ARCHITECTURES})
    endif()
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
      enable_language(HIP)
      set(DMT_GPU_BACKEND "HIP")
    elseif(DMT_GPU STREQUAL "HIP")
      message(FATAL_ERROR "DMT_GPU=HIP but no HIP compiler (ROCm clang) was found. Install "
                          "ROCm >= 6.2 (set ROCM_PATH) or use DMT_GPU=AUTO/OFF."
      )
    endif()
  endif()
endif()

if(DMT_GPU_BACKEND STREQUAL "HIP")
  list(APPEND CMAKE_PREFIX_PATH ${_dmt_rocm_roots})
  find_package(hip CONFIG REQUIRED)
  if(hip_VERSION VERSION_LESS 6.2)
    message(FATAL_ERROR "ROCm/HIP >= 6.2 is required. Found: ${hip_VERSION}")
  endif()
  message(STATUS "Found HIP ${hip_VERSION} (${CMAKE_HIP_COMPILER}).")
  # The ROCm counterparts of cuFFT and thrust (both ship with ROCm).
  find_package(hipfft CONFIG REQUIRED)
  find_package(rocthrust CONFIG REQUIRED)
  set(DMT_GPU_LIBRARIES hip::host hip::hipfft roc::rocthrust)
  # libhipcxx (cuda::std on AMD) is optional: dmt only needs span and complex
  # from it and falls back to std::span / thrust::complex (gpu_compat.cuh).
  find_package(libhipcxx CONFIG QUIET)
  if(libhipcxx_FOUND)
    list(APPEND DMT_GPU_LIBRARIES libhipcxx::libhipcxx)
    message(STATUS "Using libhipcxx ${libhipcxx_VERSION} for cuda::std.")
  endif()

  if(DMT_GPU STREQUAL "HIP" AND DMT_GPU_CHECK_DEVICE)
    find_program(
      DMT_ROCMINFO rocminfo
      HINTS ${_dmt_rocm_roots}
      PATH_SUFFIXES bin
    )
    set(_dmt_amd_agents "")
    if(DMT_ROCMINFO)
      execute_process(
        COMMAND ${DMT_ROCMINFO}
        OUTPUT_VARIABLE _dmt_rocminfo
        RESULT_VARIABLE _dmt_rocminfo_status
        ERROR_QUIET
      )
      if(_dmt_rocminfo_status EQUAL 0)
        string(
          REGEX MATCHALL
                "gfx[0-9a-f]+"
                _dmt_amd_agents
                "${_dmt_rocminfo}"
        )
        list(REMOVE_DUPLICATES _dmt_amd_agents)
      endif()
    endif()
    if(_dmt_amd_agents STREQUAL "")
      message(FATAL_ERROR "DMT_GPU=HIP but rocminfo reports no AMD GPU. Check the ROCm "
                          "installation (or set DMT_GPU_CHECK_DEVICE=OFF)."
      )
    endif()
    message(STATUS "DMT_GPU=HIP: detected GPU(s): ${_dmt_amd_agents}")
  endif()

  set(CMAKE_HIP_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(CMAKE_HIP_EXTENSIONS OFF)

  set(DMT_GPU_OPTIONS ${PROJECT_NAME}_gpu_options)
  add_library(${DMT_GPU_OPTIONS} INTERFACE)
  target_compile_options(
    ${DMT_GPU_OPTIONS}
    INTERFACE $<$<COMPILE_LANGUAGE:HIP>:
              -Wall
              -Wextra
              $<$<CONFIG:Debug>:-O0;-g>
              $<$<CONFIG:Release>:-O3;-ffast-math;-DNDEBUG>
              $<$<CONFIG:RelWithDebInfo>:-O2;-g;-ffast-math;-DNDEBUG>
              $<$<CONFIG:MinSizeRel>:-O2;-ffast-math;-DNDEBUG>
              >
  )
  message(STATUS "HIP Architectures: ${CMAKE_HIP_ARCHITECTURES}")
endif()

if(DMT_GPU_BACKEND STREQUAL "")
  if(DMT_GPU STREQUAL "OFF")
    message(STATUS "GPU disabled by user (DMT_GPU=OFF).")
  else()
    message(STATUS "No CUDA or HIP compiler found. No GPU backend (DMT_GPU=AUTO).")
  endif()
endif()
