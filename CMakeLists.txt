cmake_minimum_required(VERSION 3.10)
project(ParallelSpGEMM)

set(CMAKE_CXX_STANDARD 14)

# Verbose CMake
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Python COMPONENTS Interpreter Development)

add_subdirectory(extern/pybind11)
pybind11_add_module(add_test
        src/test.cpp)
#target_compile_definitions(add_test
#        PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})


#### SpGEMM
find_package(OpenMP REQUIRED)

pybind11_add_module(spgemm
        src/spgemm.cpp)
target_link_libraries(spgemm PRIVATE OpenMP::OpenMP_CXX)

#add_library(spgemm
#        MODULE
#        src/spgemm.cpp)
#target_lin