import time
import threading
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

def unsafe_increment(counter_ref, iterations, lock_ref=None):
    """Worker function that increments a shared counter."""
    for _ in range(iterations):
        if lock_ref:
            with lock_ref:
                counter_ref[0] += 1
        else:
            # Intentional race condition without lock
            counter_ref[0] += 1

def main():
    sys.stdout = Tee("task6_output.txt")
    
    student_id = 230103386
    num_threads = 4
    increments_per_thread = 2_000_000
    expected_value = num_threads * increments_per_thread
    
    print("=== TASK 6: Synchronization Cost & Data Loss ===")
    print(f"Student ID: {student_id}")
    print(f"Threads: {num_threads} | Increments per thread: {increments_per_thread:,}")
    print(f"Expected Final Value: {expected_value:,}")
    print("-" * 60)
    
    # --- PART 1: Unsafe Increment (Race Condition / Data Loss) ---
    counter_unsafe = [0]
    
    start_time = time.perf_counter()
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=unsafe_increment, args=(counter_unsafe, increments_per_thread, None))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
    end_time = time.perf_counter()
    time_unsafe_ms = (end_time - start_time) * 1000.0
    
    actual_unsafe = counter_unsafe[0]
    lost_updates = expected_value - actual_unsafe
    loss_percentage = (lost_updates / expected_value) * 100.0 if expected_value > 0 else 0
    
    print("[1. Unsafe Execution (No Locks)]")
    print(f"  -> Actual Final Value: {actual_unsafe:,}")
    print(f"  -> Lost Updates: {lost_updates:,}")
    print(f"  -> Update Loss Percentage: {loss_percentage:.2f}%")
    print(f"  -> Execution Time: {time_unsafe_ms:.2f} ms")
    
    # --- PART 2: Synchronized Increment (Mutex Lock) ---
    counter_safe = [0]
    lock = threading.Lock()
    
    start_time = time.perf_counter()
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=unsafe_increment, args=(counter_safe, increments_per_thread, lock))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
    end_time = time.perf_counter()
    time_safe_ms = (end_time - start_time) * 1000.0
    
    actual_safe = counter_safe[0]
    time_penalty_ms = time_safe_ms - time_unsafe_ms
    penalty_ratio = time_safe_ms / time_unsafe_ms if time_unsafe_ms > 0 else 0
    
    print("\n[2. Synchronized Execution (Mutex Lock)]")
    print(f"  -> Actual Final Value: {actual_safe:,}")
    print(f"  -> Execution Time: {time_safe_ms:.2f} ms")
    print(f"  -> Synchronization Time Penalty: {time_penalty_ms:.2f} ms ({penalty_ratio:.2f}x slower)")
    print("-" * 60)
    print("=" * 60)

if __name__ == "__main__":
    main()