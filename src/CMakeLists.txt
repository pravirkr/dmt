set(PYBIND11_FINDPYTHON ON)
find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED
)
find_package(pybind11 CONFIG REQUIRED)

# One module for every backend: the backend is chosen at runtime (Exec), so
# the bindings never include a GPU runtime header.
set(DMT_PYMODULE_NAME lib${DMT_LIBRARY_NAME})
pybind11_add_module(
  ${DMT_PYMODULE_NAME}
  dmt_python.cpp
  bindings/bind_plans.cpp
  bindings/bind_fdmt.cpp
  bindings/bind_fdmt_boxcar.cpp
  bindings/bind_cfdmt.cpp
  bindings/bind_ddmt.cpp
  bindings/bind_ddmt_fft.cpp
)
target_compile_features(${DMT_PYMODULE_NAME} PRIVATE cxx_std_20)
target_include_directories(
  ${DMT_PYMODULE_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
  ${DMT_PYMODULE_NAME} PUBLIC ${PROJECT_NAME}::${DMT_LIBRARY_NAME} ${DMT_COMPILE_OPTIONS}
)
# pybind11's include directory comes first on the module's command line (a
# system pybind11 brings /opt/homebrew/include or /usr/include), where another
# fmt may live. Put the CPM fmt ahead of it so one fmt version reaches every
# object of the module (see bench/CMakeLists.txt).
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
  set(_dmt_fmt_first SYSTEM BEFORE)
else()
  set(_dmt_fmt_first BEFORE) # -I rather than -isystem before 3.25
endif()
target_include_directories(
  ${DMT_PYMODULE_NAME} ${_dmt_fmt_first}
  PRIVATE $<TARGET_PROPERTY:fmt::fmt-header-only,INTERFACE_INCLUDE_DIRECTORIES>
)
unset(_dmt_fmt_first)
dmt_enable_coverage(${DMT_PYMODULE_NAME})
install(TARGETS ${DMT_PYMODULE_NAME} DESTINATION ${DMT_PYTHON_LIBRARY_NAME})
