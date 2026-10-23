import os
import time
import sys
import numpy as np
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

def kernel_a_copy(chunk):
    """Kernel A (Memory-Bound): Vector copy B[i] = A[i]"""
    return np.copy(chunk)

def kernel_b_math(chunk):
    """Kernel B (Compute-Bound): Trigonometric math computation"""
    return np.sin(chunk) * np.cos(chunk) + np.tan(chunk)

def run_benchmark(kernel_fn, arr, core_counts, max_cores):
    results = []
    baseline_time = None
    
    for p in core_counts:
        chunks = np.array_split(arr, p)
        
        start_time = time.perf_counter()
        if p == 1:
            _ = kernel_fn(arr)
        else:
            with ThreadPoolExecutor(max_workers=p) as executor:
                futures = [executor.submit(kernel_fn, chunk) for chunk in chunks]
                _ = [f.result() for f in futures]
                
        end_time = time.perf_counter()
        wall_time_ms = (end_time - start_time) * 1000.0
        
        if p == 1 or baseline_time is None:
            baseline_time = wall_time_ms
            
        speedup = baseline_time / wall_time_ms
        efficiency = speedup / p
        
        results.append({
            'cores': p,
            'time_ms': wall_time_ms,
            'speedup': speedup,
            'efficiency': efficiency
        })
        
    return results

def main():
    sys.stdout = Tee("task7_output.txt")
    
    student_id = 230103386
    max_cores = os.cpu_count() or 4
    num_elements = 30_000_000
    
    print("=== TASK 7: Arithmetic Intensity vs. Memory Bottlenecks ===")
    print(f"Student ID: {student_id}")
    print(f"Detected Logical Cores: {max_cores}")
    print(f"Array Size: {num_elements:,} float64 elements")
    print("-" * 75)
    
    # Generate array with Student ID seed
    np.random.seed(student_id)
    A = np.random.rand(num_elements).astype(np.float64)
    
    core_counts = sorted(list(set([1, 2, max_cores])))
    
    # --- 1. Kernel A (Memory-Bound) ---
    print("\n[Kernel A: Memory-Bound Vector Copy (B[i] = A[i])]")
    results_a = run_benchmark(kernel_a_copy, A, core_counts, max_cores)
    print(f"{'Cores':<8} | {'Time (ms)':<15} | {'Speedup S(p)':<15} | {'Efficiency E(p)':<15}")
    print("-" * 65)
    for r in results_a:
        print(f"{r['cores']:<8} | {r['time_ms']:<15.2f} | {r['speedup']:<15.2f} | {r['efficiency']:<15.2%}")
        
    # --- 2. Kernel B (Compute-Bound) ---
    print("\n[Kernel B: Compute-Bound Math (sin(A) * cos(A) + tan(A))]")
    results_b = run_benchmark(kernel_b_math, A, core_counts, max_cores)
    print(f"{'Cores':<8} | {'Time (ms)':<15} | {'Speedup S(p)':<15} | {'Efficiency E(p)':<15}")
    print("-" * 65)
    for r in results_b:
        print(f"{r['cores']:<8} | {r['time_ms']:<15.2f} | {r['speedup']:<15.2f} | {r['efficiency']:<15.2%}")
        
    # --- 3. Saturation Analysis ---
    eff_a_max = results_a[-1]['efficiency']
    eff_b_max = results_b[-1]['efficiency']
    
    print("\n" + "=" * 75)
    print("[Bottleneck Diagnosis]")
    if eff_a_max < eff_b_max:
        print("--> Kernel A (Vector Copy) STALLED in scaling due to RAM bus memory bandwidth saturation.")
        print(f"    Memory-Bound Efficiency dropped to {eff_a_max:.2%} at {max_cores} cores vs Compute-Bound {eff_b_max:.2%}.")
    else:
        print("--> Kernel B reached CPU core compute throughput saturation limits.")
    print("=" * 75)

if __name__ == "__main__":
    main()