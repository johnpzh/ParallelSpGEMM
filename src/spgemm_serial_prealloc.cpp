//
// Created by Peng, Zhen on 4/19/23.
//
#include <vector>
#include <stdio.h>
#include <algorithm>
#include "../include/spgemm_serial_prealloc.h"

namespace py = pybind11;

/// C[i,k] = A[i,j] * B[j,k]
/// C is already allocated by preprocessing
void spgemm_serial_prealloc(
    int64_t NI,
    int64_t NJ,
    int64_t NK,
    const py::array_t<int64_t> &A_col_py,
    const py::array_t<int64_t> &A_rowptr_py,
    const py::array_t<double> &A_data_py,
    const py::array_t<int64_t> &B_col_py,
    const py::array_t<int64_t> &B_rowptr_py,
    const py::array_t<double> &B_data_py,
    py::array_t<int64_t> &C_col_py,
    const py::array_t<int64_t> &C_rowptr_py,
    py::array_t<double> &C_data_py) {

  const int64_t *A_col = A_col_py.data();
  const int64_t *A_rowptr = A_rowptr_py.data();
  const double *A_data = A_data_py.data();
  const int64_t *B_col = B_col_py.data();
  const int64_t *B_rowptr = B_rowptr_py.data();
  const double *B_data = A_data_py.data();
  int64_t *C_col = C_col_py.mutable_data();
  const int64_t *C_rowptr = C_rowptr_py.data();
  double *C_data = C_data_py.mutable_data();

  /// Workspace
  std::vector<double> ws_data(NK, 0.0);
  std::vector<int64_t> ws_col_list(NK);
  std::vector<int8_t> ws_bitmap(NK, 0);
  int64_t ws_col_list_size = 0;
//  int64_t c_index = 0;
    /// Every row of A
  for (int64_t a_i_id = 0; a_i_id < NI; ++a_i_id) {
    int64_t a_i_start = A_rowptr[a_i_id];
    int64_t a_i_bound = A_rowptr[a_i_id + 1];

    /// Linear combination
    for (int64_t a_i = a_i_start; a_i < a_i_bound; ++a_i) {
      int64_t a_j_id = A_col[a_i];
      double a_val = A_data[a_i];
      int64_t b_i_start = B_rowptr[a_j_id];
      int64_t b_i_bound = B_rowptr[a_j_id + 1];
      for (int64_t b_i = b_i_start; b_i < b_i_bound; ++b_i) {
        int64_t b_k_id = B_col[b_i];
        double b_val = B_data[b_i];
        ws_data[b_k_id] += a_val * b_val;
//          {//test
//            printf("W[%lld, %lld] %lf\n", a_i_id, b_k_id, ws_data[b_k_id]);
//          }
        if (!ws_bitmap[b_k_id]) {
          ws_bitmap[b_k_id] = 1;
          ws_col_list[ws_col_list_size++] = b_k_id;
//            {//test
//              printf("ws_col_list[%lld] %lld\n", ws_col_list_size - 1, ws_col_list[ws_col_list_size - 1]);
//            }
        }
      }
    }

    /// Sort the column IDs
    std::sort(ws_col_list.begin(), ws_col_list.begin() + ws_col_list_size);

    /// Store results from the workspace to the C's row
    int64_t c_index = C_rowptr[a_i_id];
    for (int64_t ws_i = 0; ws_i < ws_col_list_size; ++ws_i) {
      int64_t c_k_id = ws_col_list[ws_i];
//        {//test
//          printf("After_ws_col_list[%lld] %lld\n", ws_i, ws_col_list[ws_i]);
//        }
      double c_val = ws_data[c_k_id];
      C_col[c_index] = c_k_id;
      C_data[c_index] = c_val;
//        {//test
//          printf("C[%lld, %lld] %lf\n", a_i_id, c_k_id, C_data[c_index]);
//        }
      ++c_index;
      ws_data[c_k_id] = 0;
      ws_bitmap[c_k_id] = 0;
    }

    ws_col_list_size = 0;
  }
}