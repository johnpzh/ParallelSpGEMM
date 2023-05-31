//
// Created by Peng, Zhen on 4/19/23.
//
#include <pybind11/pybind11.h>
#include "../include/spgemm_parallel_1.h"
#include "../include/spgemm_parallel_2_hashmap.h"
#include "../include/spgemm_parallel_3_memset.h"
#include "../include/spgemm_parallel_4_raw_pointer.h"
#include "../include/spgemm_parallel_5_matrix.h"
#include "../include/spgemm_parallel_6_matrix_better_reset.h"
#include "../include/spgemm_parallel_7_raw_pointer_outside_forloop.h"
#include "../include/spgemm_parallel_8_GraphBLAS_Gustavson.h"
#include "../include/spgemm_serial_no_prealloc.h"
#include "../include/spgemm_serial_prealloc.h"
#include "../include/spgemm_serial_hashmap.h"

namespace py = pybind11;



PYBIND11_MODULE(spgemm, m) {
  m.doc() = "spgemm kernel"; // optional module docstring

  m.def("spgemm_parallel_1", &spgemm_parallel_1, "parallel spgemm C = A * B, auto private workspace");
  m.def("spgemm_parallel_2_hashmap", &spgemm_parallel_2_hashmap, "parallel spgemm C = A * B, use hashmap workspace");
  m.def("spgemm_parallel_3_memset", &spgemm_parallel_3_memset, "parallel spgemm C = A * B, use memset to initialize the workspace");
  m.def("spgemm_parallel_4_raw_pointer", &spgemm_parallel_4_raw_pointer, "parallel spgemm C = A * B, use raw pointer instead of vector for workspace");
  m.def("spgemm_parallel_5_matrix", &spgemm_parallel_5_matrix, "parallel spgemm C = A * B, put the data structure outside the for-loop");
  m.def("spgemm_parallel_6_matrix_better_reset", &spgemm_parallel_6_matrix_better_reset, "parallel spgemm C = A * B, put the data structure outside the for-loop");
  m.def("spgemm_parallel_7_raw_pointer_outside_forloop", &spgemm_parallel_7_raw_pointer_outside_forloop, "parallel spgemm C = A * B, private vectors but outside for-loop");
  m.def("spgemm_parallel_8_GraphBLAS_Gustavson", &spgemm_parallel_8_GraphBLAS_Gustavson, "parallel spgemm C = A * B, GraphBLAS Gustavson\'s Algorithm");
  m.def("spgemm_serial_no_prealloc", &spgemm_serial_prealloc, "serial spgemm C = A * B, assuming C is pre-allocated");
  m.def("spgemm_serial_prealloc", &spgemm_serial_prealloc, "serial spgemm C = A * B, assuming C is pre-allocated");
  m.def("spgemm_serial_hashmap", &spgemm_serial_hashmap, "serial spgemm C = A * B, assuming C is pre-allocated");
}