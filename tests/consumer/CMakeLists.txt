# A downstream project that uses an installed dmt through find_package(dmt),
# as in docs/getting_started/quickstart_cpp.md. CI builds it against
# `cmake --install`, which checks dmtConfig.cmake, the exported dmt::dmt target
# and (for the static library) the FFTW/OpenMP dependency chain.
#
#   cmake --install build-ci --prefix /tmp/dmt
#   cmake -S tests/consumer -B build-consumer -DCMAKE_PREFIX_PATH=/tmp/dmt
#   cmake --build build-consumer && ./build-consumer/dmt_consumer
cmake_minimum_required(VERSION 3.18)
project(dmt_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(dmt REQUIRED)

add_executable(dmt_consumer main.cpp)
target_link_libraries(dmt_consumer PRIVATE dmt::dmt)
