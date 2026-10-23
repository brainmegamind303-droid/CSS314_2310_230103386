import numpy as np
import time
import os
import multiprocessing as mp
from multiprocessing import Queue, Process
from multiprocessing.shared_memory import SharedMemory

def worker_ipc_queue(input_q, output_q):
    data = input_q.get()
    data *= 2
    output_q.put(data)

def worker_shm(shm_name, shape, dtype):
    shm = SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    arr *= 2
    shm.close()

if __name__ == '__main__':
    student_id = 230103386
    N_sizes = [100_000, 1_000_000, 10_000_000]

    output_lines = []
    output_lines.append("==================================================")
    output_lines.append(" TASK 4: IPC Overhead vs. Zero-Copy Shared Memory")
    output_lines.append("==================================================")
    output_lines.append(f"Student ID: {student_id}")
    output_lines.append(f"Benchmark Array Sizes (N): {N_sizes}")

    table_header = f"\n{'N (elements)':<15} | {'Path A (Queue IPC ms)':<22} | {'Path B (Shared Mem ms)':<22} | {'Speedup (B vs A)':<15}"
    output_lines.append(table_header)
    output_lines.append("-" * 80)

    for N in N_sizes:
        # Generate random float64 array using Student ID seed
        np.random.seed(student_id)
        arr_data = np.random.rand(N).astype(np.float64)

        # ----------------------------------------------------
        # Path A: IPC Pipe/Queue Serialization (Pickle/IPC)
        # ----------------------------------------------------
        input_q = Queue()
        output_q = Queue()

        p_a = Process(target=worker_ipc_queue, args=(input_q, output_q))
        p_a.start()

        start_a = time.perf_counter()
        input_q.put(arr_data.copy())
        result_a = output_q.get()
        p_a.join()
        time_a_ms = (time.perf_counter() - start_a) * 1000.0

        # ----------------------------------------------------
        # Path B: Zero-Copy Shared Memory
        # ----------------------------------------------------
        shm = SharedMemory(create=True, size=arr_data.nbytes)
        shm_arr = np.ndarray(arr_data.shape, dtype=arr_data.dtype, buffer=shm.buf)
        shm_arr[:] = arr_data[:]

        p_b = Process(target=worker_shm, args=(shm.name, arr_data.shape, arr_data.dtype))

        start_b = time.perf_counter()
        p_b.start()
        p_b.join()
        time_b_ms = (time.perf_counter() - start_b) * 1000.0

        shm.close()
        shm.unlink()

        speedup = time_a_ms / time_b_ms if time_b_ms > 0 else 0.0

        line = f"{N:<15,} | {time_a_ms:<22.2f} | {time_b_ms:<22.2f} | {speedup:<15.2f}x"
        output_lines.append(line)

    output_lines.append("==================================================")

    output_str = "\n".join(output_lines)
    print(output_str)

    with open("task4p_output.txt", "w") as f:
        f.write(output_str + "\n")