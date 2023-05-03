# Parallel SpGEMM
## Intro
* Uses the workspace to do SpGEMM.
* Uses Scipy to read matrix market (mtx) files.
* Uses PyBind11 to run C++ SpGEMM kernels.
* Uses OpenMP for parallel.

## Run
```shell
cd py
python do_spemm.parallel.py
```