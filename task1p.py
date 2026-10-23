import numpy as np
import time
import os
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

def worker_row_major(args):
    shm_name, shape, row_start, row_end = args
    existing_shm = SharedMemory(name=shm_name)
    mat = np.ndarray(shape, dtype=np.float64, buffer=existing_shm.buf)
    
    local_sum = 0.0
    for r in range(row_start, row_end):
        for c in range(shape[1]):
            local_sum += mat[r, c]
    existing_shm.close()
    return local_sum

def worker_col_major(args):
    shm_name, shape, col_start, col_end = args
    existing_shm = SharedMemory(name=shm_name)
    mat = np.ndarray(shape, dtype=np.float64, buffer=existing_shm.buf)
    
    local_sum = 0.0
    for c in range(col_start, col_end):
        for r in range(shape[0]):
            local_sum += mat[r, c]
    existing_shm.close()
    return local_sum

if __name__ == '__main__':
    student_id = 230103386
    N = 8000
    num_cores = os.cpu_count() or 4
    
    output_lines = []
    output_lines.append("==================================================")
    output_lines.append(" TASK 1: Row-Major vs. Column-Major Cache Profiling")
    output_lines.append("==================================================")
    output_lines.append(f"Student ID: {student_id}")
    output_lines.append(f"Logical Core Count: {num_cores}")
    output_lines.append(f"Matrix Dimension: {N} x {N} float64 ({N*N*8 / (1024**2):.2f} MB)")

    # Allocate and fill matrix in shared memory
    matrix_size = N * N * np.dtype(np.float64).itemsize
    shm = SharedMemory(create=True, size=matrix_size)
    mat = np.ndarray((N, N), dtype=np.float64, buffer=shm.buf)
    
    # Fill using Student ID seed
    np.random.seed(student_id)
    mat[:] = np.random.rand(N, N)

    # Mode A: Row-Major Slicing
    rows_per_core = N // num_cores
    row_tasks = []
    for i in range(num_cores):
        r_start = i * rows_per_core
        r_end = (i + 1) * rows_per_core if i < num_cores - 1 else N
        row_tasks.append((shm.name, (N, N), r_start, r_end))

    start_time = time.perf_counter()
    with mp.Pool(num_cores) as pool:
        results_a = pool.map(worker_row_major, row_tasks)
    total_sum_a = sum(results_a)
    time_a_ms = (time.perf_counter() - start_time) * 1000.0

    # Mode B: Column-Major Slicing
    cols_per_core = N // num_cores
    col_tasks = []
    for i in range(num_cores):
        c_start = i * cols_per_core
        c_end = (i + 1) * cols_per_core if i < num_cores - 1 else N
        col_tasks.append((shm.name, (N, N), c_start, c_end))

    start_time = time.perf_counter()
    with mp.Pool(num_cores) as pool:
        results_b = pool.map(worker_col_major, col_tasks)
    total_sum_b = sum(results_b)
    time_b_ms = (time.perf_counter() - start_time) * 1000.0

    shm.close()
    shm.unlink()

    penalty_factor = time_b_ms / time_a_ms if time_a_ms > 0 else 0.0

    output_lines.append("\n[Console Output]")
    output_lines.append(f"Student ID: {student_id} | Logical Core Count: {num_cores}")
    output_lines.append(f"Mode A Runtime (Row-Major):    {time_a_ms:.2f} ms")
    output_lines.append(f"Mode B Runtime (Column-Major): {time_b_ms:.2f} ms")
    output_lines.append(f"Cache Penalty Factor:          {penalty_factor:.2f}x")
    output_lines.append("==================================================")

    output_str = "\n".join(output_lines)
    print(output_str)

    # Автоматически сохраняем консольный вывод в task1_output.txt
    with open("task1p_output.txt", "w") as f:
        f.write(output_str + "\n")