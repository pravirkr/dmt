# Build identity of libdmt (dmt::build_info(), dmtlib.build_info()): the
# configuration is fixed at configure time (dmt_build_config.hpp), the git
# commit and dirty flag are refreshed on every build (dmt_git_info.hpp, see
# dmtGitInfo.cmake).
# The FFTW version is the linked library's own (fftwf_version, at run time).
#   dmt_add_build_info(<library target>)
function(_dmt_join_flags out_var)
  set(tokens "")
  foreach(item IN LISTS ARGN)
    separate_arguments(parts NATIVE_COMMAND "${item}")
    list(APPEND tokens ${parts})
  endforeach()
  if(tokens)
    list(REMOVE_DUPLICATES tokens)
  endif()
  list(JOIN tokens " " joined)
  set(${out_var} "${joined}" PARENT_SCOPE)
endfunction()

# Generates the build-info headers for <target> and adds them to it.
function(dmt_add_build_info target)
  set(gen_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
  string(TOUPPER "${CMAKE_BUILD_TYPE}" build_type_uc)
  _dmt_join_flags(
    cxx_flags
    "${CMAKE_CXX_FLAGS}"
    "${CMAKE_CXX_FLAGS_${build_type_uc}}"
    ${DMT_CXX_CONFIG_FLAGS_${build_type_uc}}
  )
  set(dmt_bi_compiler "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
  set(dmt_bi_cxx_flags "${cxx_flags}")
  set(dmt_bi_build_type "${CMAKE_BUILD_TYPE}")
  if(DMT_GPU_BACKEND STREQUAL "CUDA")
    set(archs "${CMAKE_CUDA_ARCHITECTURES}")
    if(archs STREQUAL "native" AND DEFINED CMAKE_CUDA_ARCHITECTURES_NATIVE)
      set(archs "native: ${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
    endif()
    _dmt_join_flags(
      _gpu_flags
      "${CMAKE_CUDA_FLAGS}"
      "${CMAKE_CUDA_FLAGS_${build_type_uc}}"
      ${DMT_GPU_CONFIG_FLAGS_${build_type_uc}}
    )
    if(_gpu_flags)
      set(dmt_bi_gpu "CUDA ${CMAKE_CUDA_COMPILER_VERSION} (${archs}; ${_gpu_flags})")
    else()
      set(dmt_bi_gpu "CUDA ${CMAKE_CUDA_COMPILER_VERSION} (${archs})")
    endif()
  elseif(DMT_GPU_BACKEND STREQUAL "HIP")
    _dmt_join_flags(
      _gpu_flags
      "${CMAKE_HIP_FLAGS}"
      "${CMAKE_HIP_FLAGS_${build_type_uc}}"
      ${DMT_GPU_CONFIG_FLAGS_${build_type_uc}}
    )
    if(_gpu_flags)
      set(dmt_bi_gpu "HIP ${hip_VERSION} (${CMAKE_HIP_ARCHITECTURES}; ${_gpu_flags})")
    else()
      set(dmt_bi_gpu "HIP ${hip_VERSION} (${CMAKE_HIP_ARCHITECTURES})")
    endif()
  else()
    set(dmt_bi_gpu "none")
  endif()
  set(dmt_bi_openmp "${OpenMP_CXX_VERSION}")
  configure_file("${PROJECT_SOURCE_DIR}/cmake/dmt_build_config.hpp.in"
                 "${gen_dir}/dmt_build_config.hpp" @ONLY)

  set(git_header "${gen_dir}/dmt_git_info.hpp")
  add_custom_target(
    ${target}_git_info
    COMMAND "${CMAKE_COMMAND}" -DDMT_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DDMT_GIT_INFO_OUT=${git_header} -P "${PROJECT_SOURCE_DIR}/cmake/dmtGitInfo.cmake"
    BYPRODUCTS "${git_header}"
    COMMENT "Checking the git identity of the source tree"
    VERBATIM
  )
  add_dependencies(${target} ${target}_git_info)
  target_include_directories(${target} PRIVATE "${gen_dir}")
endfunction()
