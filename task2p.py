import time
import math
import threading
import random

def worker_shared(num_samples_per_thread, shared_lock, shared_rng, results, idx):
    inside_count = 0
    for _ in range(num_samples_per_thread):
        with shared_lock:
            x = shared_rng.random()
            y = shared_rng.random()
        if x * x + y * y <= 1.0:
            inside_count += 1
    results[idx] = inside_count

def worker_independent(num_samples_per_thread, seed, results, idx):
    local_rng = random.Random(seed)
    inside_count = 0
    for _ in range(num_samples_per_thread):
        x = local_rng.random()
        y = local_rng.random()
        if x * x + y * y <= 1.0:
            inside_count += 1
    results[idx] = inside_count

if __name__ == '__main__':
    student_id = 230103386
    total_samples = 40_000_000
    num_threads = 4
    samples_per_thread = total_samples // num_threads

    output_lines = []
    output_lines.append("==================================================")
    output_lines.append(" TASK 2: Pseudo-Random Generator State Bottleneck")
    output_lines.append("==================================================")
    output_lines.append(f"Student ID: {student_id}")
    output_lines.append(f"Total Monte Carlo Samples: {total_samples:,}")
    output_lines.append(f"Worker Threads: {num_threads}")

    # Mode 1: Shared PRNG Instance guarded by Mutex / Lock
    shared_lock = threading.Lock()
    shared_rng = random.Random(student_id)
    results_shared = [0] * num_threads
    threads = []

    start_time = time.perf_counter()
    for i in range(num_threads):
        t = threading.Thread(target=worker_shared, args=(samples_per_thread, shared_lock, shared_rng, results_shared, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    total_inside_shared = sum(results_shared)
    time_shared_ms = (time.perf_counter() - start_time) * 1000.0
    pi_shared = 4.0 * total_inside_shared / total_samples
    error_shared = abs(pi_shared - math.pi)

    # Mode 2: Independent PRNG per Thread
    results_indep = [0] * num_threads
    threads = []

    start_time = time.perf_counter()
    for i in range(num_threads):
        thread_seed = (student_id * 10) + i
        t = threading.Thread(target=worker_independent, args=(samples_per_thread, thread_seed, results_indep, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    total_inside_indep = sum(results_indep)
    time_indep_ms = (time.perf_counter() - start_time) * 1000.0
    pi_indep = 4.0 * total_inside_indep / total_samples
    error_indep = abs(pi_indep - math.pi)

    speedup = time_shared_ms / time_indep_ms if time_indep_ms > 0 else 0.0

    output_lines.append("\n--- Mode 1: Shared PRNG (Lock Synchronized) ---")
    output_lines.append(f"Computed Pi:     {pi_shared:.8f}")
    output_lines.append(f"Absolute Error:  {error_shared:.8e}")
    output_lines.append(f"Wall Time:       {time_shared_ms:.2f} ms")

    output_lines.append("\n--- Mode 2: Independent PRNG (Thread-Local State) ---")
    output_lines.append(f"Computed Pi:     {pi_indep:.8f}")
    output_lines.append(f"Absolute Error:  {error_indep:.8e}")
    output_lines.append(f"Wall Time:       {time_indep_ms:.2f} ms")

    output_lines.append("\n--- Profiling Summary ---")
    output_lines.append(f"Speedup Factor (Lock Elimination): {speedup:.2f}x")
    output_lines.append("==================================================")

    output_str = "\n".join(output_lines)
    print(output_str)

    with open("task2p_output.txt", "w") as f:
        f.write(output_str + "\n")