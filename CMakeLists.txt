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
        src/spgemm.module.cpp
        src/spgemm_parallel_1.cpp
        src/spgemm_parallel_2_hashmap.cpp
        src/spgemm_parallel_3_memset.cpp
        src/spgemm_parallel_4_raw_pointer.cpp
        src/spgemm_parallel_5_matrix.cpp
        src/spgemm_parallel_6_matrix_better_reset.cpp
        src/spgemm_parallel_7_raw_pointer_outside_forloop.cpp
        src/spgemm_parallel_8_GraphBLAS_Gustavson.cpp
        src/spgemm_serial_no_prealloc.cpp
        src/spgemm_serial_prealloc.cpp
        src/spgemm_serial_hashmap.cpp
        )
target_link_libraries(spgemm PRIVATE OpenMP::OpenMP_CXX)

message ( STATUS "CMAKE build type:          " ${CMAKE_BUILD_TYPE} )
message ( STATUS "CMAKE source directory:    " ${CMAKE_SOURCE_DIR} )
message ( STATUS "CMAKE build directory:     " ${CMAKE_BINARY_DIR} )

#add_library(spgemm
#        MODULE
#        src/spgemm.cpp)
#target_lin