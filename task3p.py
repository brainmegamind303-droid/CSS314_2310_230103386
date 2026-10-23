import numpy as np
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor

def scan_chunk(args):
    arr, start, end = args
    chunk_scan = np.cumsum(arr[start:end], dtype=np.int64)
    return chunk_scan, chunk_scan[-1]

def add_offset_inplace(args):
    chunk_scan, offset = args
    if offset != 0:
        chunk_scan += offset
    return chunk_scan

if __name__ == '__main__':
    student_id = 230103386
    N = 2_000_000
    num_threads = 4

    output_lines = []
    output_lines.append("==================================================")
    output_lines.append(" TASK 3: Prefix Scan (Parallel 2-Pass Chunked)")
    output_lines.append("==================================================")
    output_lines.append(f"Student ID: {student_id}")
    output_lines.append(f"Array Size: {N:,} elements")
    output_lines.append(f"Worker Threads: {num_threads}")

    # Generate array of 2,000,000 integers (1 to 10) using Student ID seed
    np.random.seed(student_id)
    input_arr = np.random.randint(1, 11, size=N, dtype=np.int64)

    # ----------------------------------------------------
    # 1. Serial Inclusive Prefix Sum
    # ----------------------------------------------------
    start_serial = time.perf_counter()
    serial_output = np.cumsum(input_arr, dtype=np.int64)
    time_serial_ms = (time.perf_counter() - start_serial) * 1000.0

    # ----------------------------------------------------
    # 2. Parallel 2-Pass Chunked Prefix Scan (ThreadPool)
    # ----------------------------------------------------
    start_parallel = time.perf_counter()
    chunk_size = N // num_threads
    bounds = [(i * chunk_size, (i + 1) * chunk_size if i < num_threads - 1 else N) for i in range(num_threads)]

    # Pass 1: Compute local chunk prefix sums
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        pass1_futures = [executor.submit(scan_chunk, (input_arr, b[0], b[1])) for b in bounds]
        pass1_results = [f.result() for f in pass1_futures]

    local_scans = [res[0] for res in pass1_results]
    chunk_sums = [res[1] for res in pass1_results]

    # Pass 2: Offset reduction (prefix sum over chunk totals)
    offsets = [0] * num_threads
    running_offset = 0
    for i in range(num_threads):
        offsets[i] = running_offset
        running_offset += chunk_sums[i]

    # Pass 3: Apply offsets in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        pass2_futures = [executor.submit(add_offset_inplace, (local_scans[i], offsets[i])) for i in range(num_threads)]
        adjusted_chunks = [f.result() for f in pass2_futures]

    parallel_output = np.concatenate(adjusted_chunks)
    time_parallel_ms = (time.perf_counter() - start_parallel) * 1000.0

    # ----------------------------------------------------
    # 3. Verification Constraint
    # ----------------------------------------------------
    if parallel_output[-1] != serial_output[-1]:
        print("VERIFICATION FAILED: parallel_output[-1] != serial_output[-1]")
        sys.exit(1)

    verification_status = "PASS"
    speedup = time_serial_ms / time_parallel_ms if time_parallel_ms > 0 else 0.0

    output_lines.append("\n--- Performance & Verification Output ---")
    output_lines.append(f"Serial Time (ms)   | {time_serial_ms:.2f}")
    output_lines.append(f"Parallel Time (ms) | {time_parallel_ms:.2f}")
    output_lines.append(f"Speedup            | {speedup:.2f}x")
    output_lines.append(f"Verification       | {verification_status}")
    output_lines.append(f"Assert Match       | parallel_output[-1] == serial_output[-1] ({parallel_output[-1]})")
    output_lines.append("==================================================")

    output_str = "\n".join(output_lines)
    print(output_str)

    with open("task3p_output.txt", "w") as f:
        f.write(output_str + "\n")