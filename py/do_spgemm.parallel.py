import sys
import os
import scipy as sp
# import numpy as np
import pandas as pd
from benchmark_helper import bench

sys.path.append("../cmake-build-debug")

import spgemm

def test_one_input(mtx_file, times):
    A = sp.io.mmread(mtx_file).tocsr()
    B = A
    # C = A @ B
    NI, NJ = A.shape
    NJ, NK = B.shape

    print(F"#### mtx: {os.path.basename(mtx_file)} ####")
    times.append(bench(lambda : spgemm.spgemm_parallel_1(NI, NJ, NK,
                             A.indices, A.indptr, A.data,
                             B.indices, B.indptr, B.data)))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(F"Usage: python {sys.argv[0]} <data_directory>")
        exit(-1)
    data_dir = sys.argv[1]

    # matrices = [
    #     "bcsstk17",
    #     "pdb1HYS",
    #     "rma10",
    #     "cant",
    #     "consph",
    #     "shipsec1",
    #     "cop20k_A",
    #     "scircuit",
    # ]

    # matrices = [ "bcsstk17" ]
    matrices = [ "tc.1T" ]

    # mtx_file = sys.argv[1]
    times =[]
    for mtx in matrices:
        test_one_input(F"{data_dir}/{mtx}/{mtx}.mtx", times)

    code = []

    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', None)

    columns = {
        'matrix': matrices,
        'avg_time': times,
        'command': [sys.argv[0]] * len(matrices),
    }
    print(pd.DataFrame(data=columns))

# import os
# nthreads = os.environ['OMP_NUM_THREADS']
# os.environ["MKL_NUM_THREADS"] = nthreads
# os.environ["NUMEXPR_NUM_THREADS"] = nthreads
#
# import numpy as np
# import scipy
# import scipy.io
# import pandas as pd
# from time import perf_counter
# from report import gen_fig
# import kernels.spmm_mm as kernel
# from benchmark_helper import bench
#
#
# def run_np(B, C, D):
#     #return B @ C
#     return B @ C @ D
#
#
# def test_one_input(mtx_file, NH=512, NJ=16):
#     '''
#     A = B @ C @ D
#     A[i,j] = B[i,k] * C[k,h] * D[h,j]
#     '''
#     B = scipy.io.mmread(mtx_file).tocsr()
#     NI, NK = B.shape
#
#
#     temp_tensor_mem = (NI / 1024) * (NH / 1024) * 8 / 1024
#     #print(f'memory required by the temporary matrix: {temp_tensor_mem} GiB')
#
#     C = np.random.rand(NK, NH)
#     D = np.random.rand(NH, NJ)
#
#     A_ref = run_np(B, C, D)
#     A_taco = A_ref.copy()
#     A_taco.fill(0)
#
#     A_react = A_ref.copy()
#     A_react.fill(0)
#
#     kernel.taco_sep_omp(NI, NK, NH, NJ, A_taco, B.indptr, B.indices, B.data, C, D)
#     assert(np.allclose(A_ref, A_taco))
#     kernel.react_omp(NI, NK, NH, NJ, A_react, B.indptr, B.indices, B.data, C, D)
#     assert(np.allclose(A_ref, A_react))
#
#     times = []
#     times.append(bench(lambda: kernel.react(NI, NK, NH, NJ, A_react, B.indptr, B.indices, B.data, C, D)))
#     times.append(bench(lambda: kernel.taco_omp(NI, NK, NH, NJ, A_react, B.indptr, B.indices, B.data, C, D)))
#     if temp_tensor_mem < 100:
#         #times.append(bench(lambda: kernel.taco_sep_omp(NI, NK, NH, NJ, A_taco, B.indptr, B.indices, B.data, C, D)))
#         times.append(bench(lambda: run_np(B, C, D)))
#     else:
#         # Will run out of memory and get killed by the OS
#         #times.append(np.inf)
#         times.append(np.inf)
#
#     print(f'runtimes for {m}: {times}')
#     #print(times)
#     return times
#
#
# matrices = [
#     'scircuit', 'mac_econ_fwd500', 'cop20k_A', 'pwtk', 'shipsec1',
#     'consph', 'rma10', 'cant', 'pdb1HYS', 'bcsstk17', 'bcsstk29'
# ]
#
# matrices.reverse()
#
# for NH in [128, 256, 512]:
#     input_times = []
#     for m in matrices:
#         times = test_one_input(f'matrices/{m}/{m}.mtx', NH=NH)
#         input_times.append(times)
#         #print(input_times)
#
#     times = np.array(input_times)
#     table = {
#         'Input': matrices,
#         'ReACT': times[:, 0],
#         'TACO': times[:, 1],
#         'SciPy': times[:, 2],
#         'Speedup over TACO': times[:, 1] / times[:, 0],
#         'Speedup over SciPy': times[:, 2] / times[:, 0],
#     }
#
#     #nthreads = os.environ['OMP_NUM_THREADS']
#     print(f'\nResults for SpMM + MM (NH = {NH}, nthreads={nthreads})')
#     my_df = pd.DataFrame(table)
#     print(my_df)
#     my_df.to_csv(f'results/gnn1_spmm_mm__NH{NH}__threads{nthreads}.csv')
