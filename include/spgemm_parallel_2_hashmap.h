//
// Created by Peng, Zhen on 4/21/23.
//

#ifndef PARALLELSPGEMM_SPGEMM_PARALLEL_2_HASHMAP_H
#define PARALLELSPGEMM_SPGEMM_PARALLEL_2_HASHMAP_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

void spgemm_parallel_2_hashmap(
    int64_t NI,
    int64_t NJ,
    int64_t NK,
    const py::array_t<int64_t> &A_col_py,
    const py::array_t<int64_t> &A_rowptr_py,
    const py::array_t<double> &A_data_py,
    const py::array_t<int64_t> &B_col_py,
    const py::array_t<int64_t> &B_rowptr_py,
    const py::array_t<double> &B_data_py);

#endif //PARALLELSPGEMM_SPGEMM_PARALLEL_2_HASHMAP_H
