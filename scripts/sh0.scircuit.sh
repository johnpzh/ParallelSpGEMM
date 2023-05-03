data_dir="/qfs/people/peng599/pppp/clion/react-eval_mac/matrices"
threads=(1 2 4 8 16 24)
data="scircuit"
py="../py/test.parallel.py"
rounds=10

for t in "${threads[@]}"; do
  export OMP_NUM_THREADS="${t}"
  python "${py}" "${data_dir}/${data}/${data}.mtx" "${rounds}"
done