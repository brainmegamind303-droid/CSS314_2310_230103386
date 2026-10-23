import os
import time
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor

class Tee:
    """Class to duplicate console output to both terminal and file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def saxpy_chunk(x_chunk, a, b):
    """Computes y = a * x + b for a given chunk."""
    return a * x_chunk + b

def main():
    sys.stdout = Tee("task5_output.txt")
    
    student_id = 230103386
    max_cores = os.cpu_count() or 4
    
    print("=== TASK 5: The Overhead Crossover Point ===")
    print(f"Student ID: {student_id}")
    print(f"Detected Logical Cores: {max_cores}")
    print("-" * 75)
    
    # Array sizes N to benchmark
    sizes = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    a, b = 2.5, 1.5
    
    results = []
    crossover_point = None
    
    for N in sizes:
        # Generate input vector x using Student ID seed
        np.random.seed(student_id)
        x = np.random.rand(N).astype(np.float64)
        
        # --- 1. Serial Execution ---
        start_serial = time.perf_counter()
        y_serial = a * x + b
        end_serial = time.perf_counter()
        time_serial_ms = (end_serial - start_serial) * 1000.0
        
        # --- 2. Parallel Execution (Multi-threaded Pool) ---
        chunks = np.array_split(x, max_cores)
        start_parallel = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=max_cores) as executor:
            futures = [executor.submit(saxpy_chunk, chunk, a, b) for chunk in chunks]
            y_parallel = np.concatenate([f.result() for f in futures])
            
        end_parallel = time.perf_counter()
        time_parallel_ms = (end_parallel - start_parallel) * 1000.0
        
        # Determine status
        is_parallel_faster = time_parallel_ms < time_serial_ms
        status = "FASTER" if is_parallel_faster else "SLOWER"
        
        results.append({
            'N': N,
            'serial_ms': time_serial_ms,
            'parallel_ms': time_parallel_ms,
            'status': status
        })
        
        # Check for crossover shift
        if crossover_point is None and is_parallel_faster:
            crossover_point = N

    # Print formatted comparison table
    print(f"{'N (Size)':<12} | {'Serial (ms)':<15} | {'Parallel (ms)':<15} | {'Parallel Status':<15}")
    print("-" * 75)
    for r in results:
        print(f"{r['N']:<12,} | {r['serial_ms']:<15.4f} | {r['parallel_ms']:<15.4f} | {r['status']:<15}")
    print("=" * 75)
    
    # Programmatic crossover point detection output
    print("\n[Crossover Analysis]")
    if crossover_point:
        idx = sizes.index(crossover_point)
        lower_bound = sizes[idx - 1] if idx > 0 else 0
        print(f"--> Crossover Range detected: Between N = {lower_bound:,} and N = {crossover_point:,}")
        print(f"--> Multi-threading becomes FASTER than serial execution starting at N = {crossover_point:,}")
    else:
        print("--> Parallel overhead exceeded serial execution across all tested sizes N.")

if __name__ == "__main__":
    main()