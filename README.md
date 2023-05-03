# Parallel SpGEMM
## Intro
* Uses the workspace to do SpGEMM.
* Uses SciPy to read matrix market (mtx) files into CSR format.
* Uses PyBind11 to run C++ SpGEMM kernels in Python scripts.
* Uses OpenMP for parallel.

## Python Dependency
```shell
pip install scipy
pip install pandas
```

## Build
```shell
mkdir cmake-build-release
cd cmake-build-release
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

## Run
```shell
export OMP_NUM_THREADS=24
cd py
python py4.parallel_x.py <input.mtx> <rounds>
```


* `OMP_NUM_THREADS` is setting the number of threads used by OpenMP.
* `input.mtx` is the input matrix
* `rounds` is how many times to run the program.
* Please check `py4.parallel_x.py` to see which version of spgemm code it is using. For example,
```python
times.append(bench(lambda : spgemm.spgemm_parallel_7_raw_pointer_outside_forloop(NI, NJ, NK,
                         A.indices, A.indptr, A.data,
                         B.indices, B.indptr, B.data),
                   repeat=rounds))
```
is using `spgemm_parallel_7_raw_pointer_outside_forloop`, which is in `src/spgemm_parallel_7_raw_pointer_outside_forloop.cpp`.