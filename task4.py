import time
import math
import multiprocessing as mp
import sys

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

def worker_static(chunk, queue_results):
    """Worker processing a static chunk of work."""
    start_time = time.perf_counter()
    for i in chunk:
        # Simulate irregular execution workload: sleep(0.0001 * (i % 20))
        duration = 0.0001 * (i % 20)
        if duration > 0:
            time.sleep(duration)
    end_time = time.perf_counter()
    queue_results.put(end_time)

def worker_dynamic(task_queue, queue_results):
    """Worker pulling tasks dynamically from a shared queue (work-stealing style)."""
    start_time = time.perf_counter()
    while True:
        try:
            # Non-blocking or short timeout pull from queue
            i = task_queue.get(timeout=0.05)
        except Exception:
            break
        
        duration = 0.0001 * (i % 20)
        if duration > 0:
            time.sleep(duration)
            
    end_time = time.perf_counter()
    queue_results.put(end_time)

def main():
    sys.stdout = Tee("task4_output.txt")
    
    student_id = 230103386
    num_workers = 4
    total_items = 1000
    
    print("=== TASK 4: Work Stealing & Scheduling Under Skew ===")
    print(f"Student ID: {student_id}")
    print(f"Workers: {num_workers} | Total Items: {total_items}")
    print("-" * 60)
    
    items = list(range(total_items))
    
    # --- STRATEGY A: Static Chunk Distribution ---
    print("Running Strategy A: Static Chunk Distribution...")
    # Split items equally among 4 workers
    chunk_size = math.ceil(total_items / num_workers)
    chunks = [items[i:i + chunk_size] for i in range(0, total_items, chunk_size)]
    
    manager = mp.Manager()
    results_a = manager.Queue()
    
    start_wall_a = time.perf_counter()
    processes_a = []
    for chunk in chunks:
        p = mp.Process(target=worker_static, args=(chunk, results_a))
        processes_a.append(p)
        p.start()
        
    for p in processes_a:
        p.join()
    end_wall_a = time.perf_counter()
    total_time_a = (end_wall_a - start_wall_a) * 1000.0
    
    completion_times_a = []
    while not results_a.empty():
        completion_times_a.append(results_a.get())
        
    delta_a = (max(completion_times_a) - min(completion_times_a)) * 1000.0 if completion_times_a else 0
    
    print(f"  -> Static Total Wall Time: {total_time_a:.2f} ms")
    print(f"  -> Completion Delta (Min-Max): {delta_a:.2f} ms")
    
    # --- STRATEGY B: Dynamic Scheduling / Work-Stealing Queue ---
    print("\nRunning Strategy B: Dynamic Scheduling (Chunk size = 1)...")
    task_queue = manager.Queue()
    for item in items:
        task_queue.put(item)
        
    results_b = manager.Queue()
    
    start_wall_b = time.perf_counter()
    processes_b = []
    for _ in range(num_workers):
        p = mp.Process(target=worker_dynamic, args=(task_queue, results_b))
        processes_b.append(p)
        p.start()
        
    for p in processes_b:
        p.join()
    end_wall_b = time.perf_counter()
    total_time_b = (end_wall_b - start_wall_b) * 1000.0
    
    completion_times_b = []
    while not results_b.empty():
        completion_times_b.append(results_b.get())
        
    delta_b = (max(completion_times_b) - min(completion_times_b)) * 1000.0 if completion_times_b else 0
    
    print(f"  -> Dynamic Total Wall Time: {total_time_b:.2f} ms")
    print(f"  -> Completion Delta (Min-Max): {delta_b:.2f} ms")
    
    print("-" * 60)
    print("Summary:")
    print(f"{'Strategy':<20} | {'Total Time (ms)':<18} | {'Completion Delta (ms)':<22}")
    print("-" * 64)
    print(f"{'Static':<20} | {total_time_a:<18.2f} | {delta_a:<22.2f}")
    print(f"{'Dynamic (Queue)':<20} | {total_time_b:<18.2f} | {delta_b:<22.2f}")
    print("=" * 64)

if __name__ == '__main__':
    mp.freeze_support()
    main()