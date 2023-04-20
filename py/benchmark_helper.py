import numpy as np
import logging
from time import perf_counter

def bench_0(fn, iter_time=1, clear_cache=False):
    try:
        fn()
    except np.core._exceptions._ArrayMemoryError as e:
        logging.info("Numpy runs out of memory")
        return np.inf
    except Exception as e:
        raise e


    t0 = perf_counter()
    for _ in range(3):
        fn()
    t1 = perf_counter()
    estimate_sec = (t1 - t0) / 3

    n_repeat = max(10, int(iter_time / estimate_sec))

    if estimate_sec > 3:
        n_repeat = 3

    times = []
    for i in range(n_repeat):
        if clear_cache:
            _ = np.random.rand(1024 * 1024 * 20)
        t0 = perf_counter()
        fn()
        t1 = perf_counter()
        times.append(t1 - t0)

    times.sort()
    mean, std = np.mean(times), np.std(times)
    cv = std / mean
    #    print('cv:', cv)
    if cv > 0.20:  # 0.05 for single thread
        logging.info(f'Coefficient of variation too high ({cv:.5f}), rerunning')
        bench(fn, iter_time, clear_cache)
    return mean


def bench(fn, repeat=10, clear_cache=False):
    try:
        fn()
    except np.core._exceptions._ArrayMemoryError as e:
        logging.info("Numpy runs out of memory")
        return np.inf
    except Exception as e:
        raise e

    times = []
    for _ in range(repeat):
        if clear_cache:
            _ = np.random.rand(1024 * 1024 * 20)
        t0 = perf_counter()
        fn()
        t1 = perf_counter()
        times.append(t1 - t0)

    times.sort()
    mean, std = np.mean(times), np.std(times)
    cv = std / mean

    ## Print out
    head = F"#### {len(times)} Runtimes ####"
    # print("#" * len(head))
    print(head, flush=True)
    for t in times:
        print(F"{t:.6f}", flush=True)
    tail = F"#### mean: {mean:.6f} std: {std:.6f} cv: {cv:.2%} ####"
    print(tail)
    # print("#" * len(tail), flush=True)

    return mean

