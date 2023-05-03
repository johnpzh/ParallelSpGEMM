//
// Created by Peng, Zhen on 4/19/23.
//
#include <vector>
#include <stdio.h>
#include <algorithm>
#include <omp.h>
#include "../include/spgemm_parallel_4_raw_pointer.h"

namespace py = pybind11;

static void symbolic_phase(
        int64_t NI,
        int64_t NJ,
        int64_t NK,
        const int64_t *A_col,
        const int64_t *A_rowptr,
        const double *A_data,
        const int64_t *B_col,
        const int64_t *B_rowptr,
        const double *B_data,
        int64_t *C_rowptr,
        int64_t &C_size) {
  /// Every row of A
#pragma omp parallel for default(shared) // NOLINT(openmp-use-default-none)
  for (int64_t a_i_id = 0; a_i_id < NI; ++a_i_id) {
    /// Set up workspace
//    std::vector<int8_t> ws_bitmap(NK, 0);
//    std::vector<int8_t> ws_bitmap(NK);
//    memset(ws_bitmap.data(), 0, sizeof(int8_t) * NK);

    int8_t *ws_bitmap = (int8_t *) malloc(NK * sizeof(int8_t));
    memset(ws_bitmap, 0, NK * sizeof(int8_t));

    int64_t ws_col_list_size = 0;

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
//        ws_data[b_k_id] += a_val * b_val;
//          {//test
//            printf("W[%lld, %lld] %lf\n", a_i_id, b_k_id, ws_data[b_k_id]);
//          }
        if (!ws_bitmap[b_k_id]) {
          ws_bitmap[b_k_id] = 1;
          ++ws_col_list_size;
//          ws_col_list[ws_col_list_size++] = b_k_id;
//            {//test
//              printf("ws_col_list[%lld] %lld\n", ws_col_list_size - 1, ws_col_list[ws_col_list_size - 1]);
//            }
        }
      }
    }

    C_rowptr[a_i_id] = ws_col_list_size;

    free(ws_bitmap);
  }

  /// Reduce C.rowptr after getting all row sizes
  int64_t sum = 0;
  int64_t bound = NI + 1;
  for (int64_t row_id = 0; row_id < bound; ++row_id) {
    int64_t current = C_rowptr[row_id];
    C_rowptr[row_id] = sum;
    sum += current;
//    {//test
//      printf("C_rowptr[%lld] %lld\n", row_id, C_rowptr[row_id]);
//    }
  }

  C_size = sum;

//  {//test
//    printf("C_size: %lld\n", C_size);
//  }

}


static void numeric_phase(
    int64_t NI,
    int64_t NJ,
    int64_t NK,
    const int64_t *A_col,
    const int64_t *A_rowptr,
    const double *A_data,
    const int64_t *B_col,
    const int64_t *B_rowptr,
    const double *B_data,
    int64_t *C_col,
    const int64_t *C_rowptr,
    double *C_data) {
  /// Every row of A
#pragma omp parallel for default(shared) // NOLINT(openmp-use-default-none)
  for (int64_t a_i_id = 0; a_i_id < NI; ++a_i_id) {
    /// Set up workspace
//    std::vector<double> ws_data(NK, 0.0);
//    std::vector<int64_t> ws_col_list(NK);
//    std::vector<int8_t> ws_bitmap(NK, 0);

//    std::vector<double> ws_data(NK);
//    std::vector<int64_t> ws_col_list(NK);
//    std::vector<int8_t> ws_bitmap(NK);
//
//    memset(ws_data.data(), 0, sizeof(double) * NK);
//    memset(ws_bitmap.data(), 0, sizeof(int8_t) * NK);

    double *ws_data = (double *) malloc(NK * sizeof(double));
    int64_t *ws_col_list = (int64_t *) malloc(NK * sizeof(int64_t));
    int8_t *ws_bitmap = (int8_t *) malloc(NK * sizeof(int8_t));

    memset(ws_data, 0, NK * sizeof(double));
//    memset(ws_col_list, 0, NK * sizeof(int64_t));
    memset(ws_bitmap, 0, NK * sizeof(int8_t));


    int64_t ws_col_list_size = 0;

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

//    C_rowptr[a_i_id] = ws_col_list_size;

    /// Sort the column IDs
//    std::sort(ws_col_list.begin(), ws_col_list.begin() + ws_col_list_size);
    std::sort(ws_col_list, ws_col_list + ws_col_list_size);

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
    }

    free(ws_data);
    free(ws_col_list);
    free(ws_bitmap);
  }
}


/// C[i,k] = A[i,j] * B[j,k]
void spgemm_parallel_4_raw_pointer(
    int64_t NI,
    int64_t NJ,
    int64_t NK,
    const py::array_t<int64_t> &A_col_py,
    const py::array_t<int64_t> &A_rowptr_py,
    const py::array_t<double> &A_data_py,
    const py::array_t<int64_t> &B_col_py,
    const py::array_t<int64_t> &B_rowptr_py,
    const py::array_t<double> &B_data_py) {
//    py::array_t<int64_t> &C_col_py,
//    const py::array_t<int64_t> &C_rowptr_py,
//    py::array_t<double> &C_data_py) {

  const int64_t *A_col = A_col_py.data();
  const int64_t *A_rowptr = A_rowptr_py.data();
  const double *A_data = A_data_py.data();
  const int64_t *B_col = B_col_py.data();
  const int64_t *B_rowptr = B_rowptr_py.data();
  const double *B_data = A_data_py.data();
//  int64_t *C_col = C_col_py.mutable_data();
//  const int64_t *C_rowptr = C_rowptr_py.data();
//  double *C_data = C_data_py.mutable_data();
  int64_t *C_col = nullptr;
  int64_t *C_rowptr = (int64_t *) malloc((NI + 1) * sizeof(int64_t));
  C_rowptr[NI] = 0;
  double *C_data = nullptr;
  int64_t C_size = 0;

//#pragma omp parallel default(none)
//  {
//    printf("num_threads: %d\n", omp_get_num_threads());
//  }

  /// Symbolic Phase
  symbolic_phase(NI, NJ, NK,
                 A_col, A_rowptr, A_data,
                 B_col, B_rowptr, B_data,
                 C_rowptr, C_size);

//  {// test
//    for (int64_t p_i = 0; p_i < NI + 1; ++p_i) {
//      printf("%lld ", C_rowptr[p_i]);
//    }
//    printf("\n");
//  }

  /// Allocate C
  C_col = (int64_t *) malloc(C_size * sizeof(int64_t));
  C_data = (double *) malloc(C_size * sizeof(double));

  /// Numeric Phase
  numeric_phase(NI, NJ, NK,
                A_col, A_rowptr, A_data,
                B_col, B_rowptr, B_data,
                C_col, C_rowptr, C_data);

//  {// test
//    double sum = 0.0;
//    for (int64_t i = 0; i < NI; ++i) {
//      int64_t j_start = C_rowptr[i];
//      int64_t j_bound = C_rowptr[i + 1];
//      for (int64_t j_i = j_start; j_i < j_bound; ++j_i) {
//        int64_t j = C_col[j_i];
//        double val = C_data[j_i];
////        printf("(%lld, %lld) %lf\n", i, j, val);
//        sum += val;
//      }
//    }
//    printf("sum: %lf\n", sum);
//  }

  /// Free up
  free(C_rowptr);
  free(C_col);
  free(C_data);
}