# docs/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)

find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(
    dmt_doxygen
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating Doxygen XML for Breathe"
    VERBATIM
  )
else()
  add_custom_target(
    dmt_doxygen
    COMMAND ${CMAKE_COMMAND} -E echo "Doxygen not found; skipping XML generation"
    VERBATIM
  )
endif()

find_program(SPHINX_EXECUTABLE sphinx-build)
if(SPHINX_EXECUTABLE)
  set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")
  add_custom_target(
    docs
    COMMAND ${CMAKE_COMMAND} -E env
            DOXYGEN_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/doxygen"
            ${SPHINX_EXECUTABLE} -b html "${CMAKE_CURRENT_SOURCE_DIR}" "${SPHINX_HTML_DIR}"
    DEPENDS dmt_doxygen
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building Sphinx HTML documentation with Breathe and MyST-NB"
    VERBATIM
  )
  message(STATUS "Configured documentation target: 'ninja docs' or 'make docs'")
else()
  message(STATUS "sphinx-build not found; 'docs' target unavailable")
endif()
