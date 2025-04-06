# System dependencies
message(STATUS "Searching for required system dependencies...")
find_package(FFTW REQUIRED)
find_package(OpenMP)

# spdlog (Header-only logging library)
CPMFindPackage(
  NAME spdlog
  VERSION 1.15.2
  GITHUB_REPOSITORY gabime/spdlog
  OPTIONS "SPDLOG_INSTALL ON"
)

# Code coverage setup
add_library(coverage_config INTERFACE)
if(ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(coverage_config INTERFACE -O0 -g --coverage)
    target_link_options(coverage_config INTERFACE --coverage)
    message(STATUS "Code coverage enabled.")
  else()
    message(
      WARNING "Code coverage requested, but not supported for compiler ${CMAKE_CXX_COMPILER_ID}."
    )
    set(ENABLE_COVERAGE
        OFF
        CACHE BOOL "" FORCE
    )
  endif()
endif()


# Collect source files
file(GLOB DMT_LIBRARY_CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(DMT_LIBRARY_ALL_SOURCES ${DMT_LIBRARY_CXX_SOURCES})

# Add CUDA sources if available
if(CMAKE_CUDA_COMPILER)
  file(GLOB DMT_LIBRARY_CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
  list(APPEND DMT_LIBRARY_ALL_SOURCES ${DMT_LIBRARY_CUDA_SOURCES})
endif()

add_library(${DMT_LIBRARY_NAME} ${DMT_LIBRARY_ALL_SOURCES})
add_library(${PROJECT_NAME}::${DMT_LIBRARY_NAME} ALIAS ${DMT_LIBRARY_NAME})

# Set include directories
target_include_directories(
  ${DMT_LIBRARY_NAME}
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/> $<BUILD_INTERFACE:${FFTW_INCLUDE_DIRS}>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries
target_link_libraries(
  ${DMT_LIBRARY_NAME}
  PRIVATE ${DMT_COMPILE_OPTIONS} coverage_config
  PUBLIC spdlog::spdlog_header_only ${FFTW_FLOAT_OPENMP_LIB}
)

# Find and link OpenMP if available
if(OpenMP_CXX_FOUND)
  target_link_libraries(${DMT_LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(${DMT_LIBRARY_NAME} PRIVATE DMT_ENABLE_OPENMP)
endif()

# Install targets
install(
  TARGETS ${DMT_LIBRARY_NAME} ${DMT_COMPILE_OPTIONS} coverage_config
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
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
