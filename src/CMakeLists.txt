set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# CPU module
set(DMT_PYMODULE_NAME lib${DMT_LIBRARY_NAME})
pybind11_add_module(${DMT_PYMODULE_NAME} dmt_python.cpp)
target_compile_features(${DMT_PYMODULE_NAME} PRIVATE cxx_std_20)
target_include_directories(${DMT_PYMODULE_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(
  ${DMT_PYMODULE_NAME} PUBLIC ${PROJECT_NAME}::${DMT_LIBRARY_NAME} ${DMT_COMPILE_OPTIONS}
)
install(TARGETS ${DMT_PYMODULE_NAME} DESTINATION ${DMT_PYTHON_LIBRARY_NAME})

# GPU module (if CUDA is available)
if(CMAKE_CUDA_COMPILER)
  set(DMT_PYGPUMODULE_NAME libcuda${DMT_LIBRARY_NAME})
  pybind11_add_module(${DMT_PYGPUMODULE_NAME} dmt_python_cuda.cpp)
  target_compile_features(${DMT_PYGPUMODULE_NAME} PRIVATE cxx_std_20)
  target_include_directories(
    ${DMT_PYGPUMODULE_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_link_libraries(
    ${DMT_PYGPUMODULE_NAME}
    PUBLIC ${PROJECT_NAME}::${DMT_LIBRARY_NAME} ${DMT_COMPILE_OPTIONS}
    PRIVATE ${DMT_CUDA_OPTIONS}
  )
  install(TARGETS ${DMT_PYGPUMODULE_NAME} DESTINATION ${DMT_PYTHON_LIBRARY_NAME})
endif()
