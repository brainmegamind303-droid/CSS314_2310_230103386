import os
import platform
import time
import numpy as np
import sys

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
    # Автоматически сохраняем вывод в task1_output.txt
    sys.stdout = Tee("task1_output.txt")
    
    student_id = 230103386
    
    print("=== TASK 1: Hardware Probe & Serial Baseline ===")
    print(f"Student ID: {student_id}")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Platform / Machine: {platform.machine()}")
    print(f"Processor: {platform.processor() or 'Not specified'}")
    
    logical_cores = os.cpu_count()
    print(f"Total Logical Cores: {logical_cores}")
    print("-" * 50)
    
    print("Generating 20,000,000 float64 elements with seed:", student_id)
    np.random.seed(student_id)
    arr = np.random.rand(20_000_000).astype(np.float64)
    
    print("Computing sum and standard deviation (serial)...")
    start_time = time.perf_counter()
    
    total_sum = np.sum(arr)
    std_dev = np.std(arr)
    
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000.0
    
    print(f"Sum: {total_sum}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Execution Time: {elapsed_ms:.2f} ms")

if __name__ == "__main__":
    main()