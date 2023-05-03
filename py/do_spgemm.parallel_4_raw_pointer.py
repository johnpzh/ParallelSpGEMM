import sys
import os
import scipy.io as spio
# import numpy as np
import pandas as pd
from benchmark_helper import bench

sys.path.append("../cmake-build-debug")
sys.path.append("../build")

import spgemm


def test_one_input(mtx_file, times, rounds=10):
    A = spio.mmread(mtx_file).tocsr()
    B = A
    # C = A @ B
    NI, NJ = A.shape
    NJ, NK = B.shape

    print(F"\n#### mtx: {os.path.basename(mtx_file)} ####")
    times.append(bench(lambda : spgemm.spgemm_parallel_4_raw_pointer(NI, NJ, NK,
                             A.indices, A.indptr, A.data,
                             B.indices, B.indptr, B.data),
                       repeat=rounds))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(F"Usage: python {sys.argv[0]} <data_directory> <rounds>")
        exit(-1)
    data_dir = sys.argv[1]
    rounds = int(sys.argv[2])

    matrices = [
        "bcsstk17",
        "pdb1HYS",
        "rma10",
        "cant",
        "consph",
        "shipsec1",
        "cop20k_A",
        "scircuit",
    ]

    # matrices = [ "bcsstk17" ]
    # matrices = [ "test_rank2" ]

    # mtx_file = sys.argv[1]
    times =[]
    for mtx in matrices:
        test_one_input(mtx_file=F"{data_dir}/{mtx}/{mtx}.mtx", times=times, rounds=rounds)

    code = []

    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', None)

    num_threads = os.environ["OMP_NUM_THREADS"]

    columns = {
        'matrix': matrices,
        'avg_time': times,
        'command': [sys.argv[0]] * len(matrices),
        'threads': [num_threads] * len(matrices)
    }
    print(pd.DataFrame(data=columns))

