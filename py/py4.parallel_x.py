import sys
import os
import scipy.io as spio
# import numpy as np
import pandas as pd
from benchmark_helper import bench

sys.path.append("../cmake-build-debug")
sys.path.append("../build")

import spgemm


def test_one_input(mtx_file, times, rounds=10, num_threads=1):
    A = spio.mmread(mtx_file).tocsr()
    B = A
    # C = A @ B
    NI, NJ = A.shape
    NJ, NK = B.shape


    print(F"\n#### mtx: {os.path.basename(mtx_file)} ####")
    times.append(bench(lambda : spgemm.spgemm_parallel_7_raw_pointer_outside_forloop(NI, NJ, NK,
                             A.indices, A.indptr, A.data,
                             B.indices, B.indptr, B.data),
                       repeat=rounds))
    # times.append(bench(lambda : spgemm.spgemm_parallel_6_matrix_better_reset(NI, NJ, NK,
    #                          A.indices, A.indptr, A.data,
    #                          B.indices, B.indptr, B.data,
    #                          num_threads),
    #                    repeat=rounds))
    # times.append(bench(lambda : spgemm.spgemm_parallel_5_matrix(NI, NJ, NK,
    #                          A.indices, A.indptr, A.data,
    #                          B.indices, B.indptr, B.data,
    #                          num_threads),
    #                    repeat=rounds))
    # times.append(bench(lambda : spgemm.spgemm_parallel_4_raw_pointer(NI, NJ, NK,
    #                          A.indices, A.indptr, A.data,
    #                          B.indices, B.indptr, B.data),
    #                    repeat=rounds))
    # times.append(bench(lambda : spgemm.spgemm_parallel_2_hashmap(NI, NJ, NK,
    #                          A.indices, A.indptr, A.data,
    #                          B.indices, B.indptr, B.data),
    #                    repeat=rounds))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(F"Usage: python {sys.argv[0]} <input.mtx> <rounds>")
        exit(-1)
    # data_dir = sys.argv[1]
    mtx_file = sys.argv[1]
    rounds = int(sys.argv[2])
    num_threads = int(os.environ["OMP_NUM_THREADS"])

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

    # matrices = [ "sci" ]
    # matrices = [ "test_rank2" ]

    # mtx_file = sys.argv[1]

    times =[]
    # for mtx in matrices:
    test_one_input(mtx_file=mtx_file, times=times, rounds=rounds, num_threads=num_threads)

    code = []

    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', None)



    columns = {
        'matrix': [os.path.basename(mtx_file)],
        'avg_time': times,
        'command': [sys.argv[0]],
        'threads': [num_threads]
    }
    print(pd.DataFrame(data=columns))


