import os
import time
import multiprocessing as mp
import sys

class Tee:
    
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

def worker_unpadded(i, shared_arr, iterations):
  
    for _ in range(iterations):
        shared_arr[i] += 1

def worker_padded(i, shared_arr, iterations):
 
    local_val = 0
    for _ in range(iterations):
        local_val += 1
    shared_arr[i] = local_val

def main():
    sys.stdout = Tee("task3_output.txt")
    
    student_id = 230103386
    n_cores = os.cpu_count() or 4
    iterations = 50_000_000
    
    print("=== TASK 3: Cache Line Invalidation & False Sharing ===")
    print(f"Student ID: {student_id}")
    print(f"Worker Processes (N): {n_cores}")
    print(f"Iterations per worker: {iterations:,}")
    print("-" * 60)
    
    manager = mp.Manager()
    shared_unpadded = manager.list([0] * n_cores)
    
    start_time = time.perf_counter()
    processes = []
    for i in range(n_cores):
        p = mp.Process(target=worker_unpadded, args=(i, shared_unpadded, iterations))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
    end_time = time.perf_counter()
    time_unpadded_ms = (end_time - start_time) * 1000.0
    print(f"1. Unpadded Shared Time : {time_unpadded_ms:.2f} ms")
    
    # - Padded / Local ---
    shared_padded = manager.list([0] * n_cores)
    
    start_time = time.perf_counter()
    processes = []
    for i in range(n_cores):
        p = mp.Process(target=worker_padded, args=(i, shared_padded, iterations))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
    end_time = time.perf_counter()
    time_padded_ms = (end_time - start_time) * 1000.0
    print(f"2. Padded / Local Time   : {time_padded_ms:.2f} ms")
    
    # (execution ratio)
    ratio = time_unpadded_ms / time_padded_ms if time_padded_ms > 0 else 0
    print("-" * 60)
    print(f"Execution Ratio (Unpadded / Padded): {ratio:.2f}x")
    print("=" * 60)

if __name__ == '__main__':
    mp.freeze_support()
    main()