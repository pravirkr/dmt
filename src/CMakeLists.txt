set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(PYMODULE_NAME libdmt)
pybind11_add_module(${PYMODULE_NAME} dmt_python.cpp)
target_include_directories(${PYMODULE_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(${PYMODULE_NAME} PUBLIC ${LIBRARY_NAME})
install(TARGETS ${PYMODULE_NAME} DESTINATION ${PYLIBRARY_NAME})

# Check if CUDA was found
if(CMAKE_CUDA_COMPILER)
  set(PYGPUMODULE_NAME libcudmt)
  pybind11_add_module(${PYGPUMODULE_NAME} dmt_python_gpu.cpp)
  target_include_directories(${PYGPUMODULE_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(${PYGPUMODULE_NAME} PUBLIC ${LIBRARY_NAME})
  install(TARGETS ${PYGPUMODULE_NAME} DESTINATION ${PYLIBRARY_NAME})
endif()
