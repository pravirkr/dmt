CPMFindPackage(
  NAME spdlog
  VERSION 1.13.0
  GITHUB_REPOSITORY gabime/spdlog
  OPTIONS "SPDLOG_INSTALL ON"
)

# Code Coverage Configuration
add_library(coverage_config INTERFACE)
if(ENABLE_CODE_COVERAGE)
  target_compile_options(coverage_config INTERFACE -O0 -g --coverage)
  target_link_options(coverage_config INTERFACE --coverage)
endif()

file(GLOB LIBRARY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Check if CUDA was found
if(CMAKE_CUDA_COMPILER)
  file(GLOB LIBRARY_SOURCES_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
  list(APPEND LIBRARY_SOURCES ${LIBRARY_SOURCES_CUDA})
endif()

add_library(${LIBRARY_NAME} ${LIBRARY_SOURCES})
add_library(${PROJECT_NAME}::${LIBRARY_NAME} ALIAS ${LIBRARY_NAME})

target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  ${LIBRARY_NAME}
  PUBLIC spdlog::spdlog_header_only
  PRIVATE coverage_config
)

# Find and link OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(${LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_OPENMP)
endif()

# Install targets
install(
  TARGETS ${LIBRARY_NAME} coverage_config
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Generate and install export file
install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${LIBRARY_NAME}
)
