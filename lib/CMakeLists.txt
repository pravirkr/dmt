# Find dependencies
CPMFindPackage(
  NAME spdlog
  VERSION 1.13.0
  GITHUB_REPOSITORY gabime/spdlog
  OPTIONS "SPDLOG_INSTALL ON"
)
find_package(OpenMP)

file(GLOB LIBRARY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Check if CUDA was found
if(CMAKE_CUDA_COMPILER)
  file(GLOB LIBRARY_SOURCES_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
  list(APPEND LIBRARY_SOURCES ${LIBRARY_SOURCES_CUDA})
endif()

add_library(${LIBRARY_NAME} ${LIBRARY_SOURCES})
target_include_directories(
  ${LIBRARY_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
                         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(${LIBRARY_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  ${LIBRARY_NAME} PUBLIC $<BUILD_INTERFACE:coverage_config>
                         $<BUILD_INTERFACE:spdlog::spdlog_header_only>
)

# Check if OpenMP was found
if(OpenMP_CXX_FOUND)
  target_link_libraries(${LIBRARY_NAME} PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_OPENMP)
endif()
