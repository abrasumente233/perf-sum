cmake_minimum_required(VERSION 3.10)

project(sum)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "-O3 -fno-vectorize")
set (CMAKE_C_FLAGS "-O3")
find_package(benchmark REQUIRED)

add_executable(sum sum.cpp)
target_link_libraries(sum benchmark::benchmark)
