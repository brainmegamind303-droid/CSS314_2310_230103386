import os
import platform
import time
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor

class Tee:
    """Класс для дублирования вывода в консоль и в файл одновременно"""
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

def main():
    # Перенаправляем весь вывод в task2_output.txt и на экран
    sys.stdout = Tee("task2_output.txt")

    student_id = 230103386
    max_cores = os.cpu_count() or 4
    
    print("=== TASK 2: Multi-Core Scaling vs. Amdahl's Law ===")
    print(f"Student ID: {student_id}")
    print(f"Max Logical Cores detected: {max_cores}")
    
    np.random.seed(student_id)
    arr = np.random.rand(20_000_000).astype(np.float64)
    
    core_counts = [1, 2, 4, max_cores, max_cores * 2]
    core_counts = sorted(list(set(core_counts)))
    
    results = []
    baseline_time = None
    P = 0.95 
    
    for p in core_counts:
        chunks = np.array_split(arr, p)
        
        start_time = time.perf_counter()
        
        if p == 1:
            total_sum = np.sum(arr)
        else:
            with ThreadPoolExecutor(max_workers=p) as executor:
                partial_sums = list(executor.map(np.sum, chunks))
            total_sum = sum(partial_sums)
            
        end_time = time.perf_counter()
        wall_time_ms = (end_time - start_time) * 1000.0
        
        if p == 1 or baseline_time is None:
            baseline_time = wall_time_ms
            
        speedup = baseline_time / wall_time_ms
        amdahl_upper_bound = 1.0 / ((1.0 - P) + (P / p))
        
        results.append({
            'cores': p,
            'time_ms': wall_time_ms,
            'speedup': speedup,
            'amdahl': amdahl_upper_bound
        })
        
    print("\n" + "=" * 68)
    print(f"{'Cores':<8} | {'Wall Time (ms)':<15} | {'Speedup S(p)':<12} | {'Amdahl Bound':<15}")
    print("-" * 68)
    for r in results:
        print(f"{r['cores']:<8} | {r['time_ms']:<15.2f} | {r['speedup']:<12.2f} | {r['amdahl']:<15.2f}")
    print("=" * 68)

if __name__ == '__main__':
    main()