# Parallel SpGEMM
## Intro
* Uses the workspace to do SpGEMM.
* Uses SciPy to read matrix market (mtx) files into CSR format.
* Uses PyBind11 to run C++ SpGEMM kernels in Python scripts.
* Uses OpenMP for parallel.

## Python Dependency
```shell
$ pip install scipy
$ pip install pandas
```

## Set up
```shell
$ git clone https://github.com/johnpzh/ParallelSpGEMM.git
$ cd ParallelSpGEMM
$ git submodule init
$ git submodule update
```

## Build
Under the project directory (`ParallelSpGEMM/`),
```shell
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ make -j4
```

## Run
Under the project directory (`ParallelSpGEMM/`),
```shell
$ export OMP_NUM_THREADS=24
$ cd py
$ python py4.parallel_x.py <input.mtx> <rounds>
```


* `OMP_NUM_THREADS` is setting the number of threads used by OpenMP.
* `input.mtx` is the input matrix
* `rounds` is how many times the program will run.

Please check `py4.parallel_x.py` to see which version of spgemm code it is using. For example,
```python
times.append(bench(lambda : spgemm.spgemm_parallel_7_raw_pointer_outside_forloop(NI, NJ, NK,
                         A.indices, A.indptr, A.data,
                         B.indices, B.indptr, B.data),
                   repeat=rounds))
```
is using `spgemm_parallel_7_raw_pointer_outside_forloop`, which is in `src/spgemm_parallel_7_raw_pointer_outside_forloop.cpp`.

## How to Add New Version of SpGEMM?

1. Add `spgemm_vxx.cpp` in `src/`.
2. Add `spgemm_vxx.h` in `include/`.
3. Add function entry `spgemm_vxx` in `src/spgemm.module.cpp`.
4. Add source file `src/spgemm_vxx.cpp` into `CMakeLists.txt` for building the pybind11 module.
5. Implement the new version of SpGEMM in the function `spgemm_vxx` in `spgemm_vxx.cpp`, and rebuild.
6. Update the function entry in `py/py4.parallel_x.py` to run the new version.
7. Run `py4.parallel_x.py` to check the correctness.
