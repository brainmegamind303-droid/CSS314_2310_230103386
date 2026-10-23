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

def matmul_nontiled_block(args):
    """Computes a strip of rows for non-tiled matrix multiplication."""
    A_strip, B = args
    return np.dot(A_strip, B)

def matmul_tiled_block(args):
    """Computes matrix multiplication using blocked/tiled sub-matrices for cache locality."""
    A, B, start_row, end_row, block_size = args
    N = B.shape[0]
    C_strip = np.zeros((end_row - start_row, N), dtype=np.float64)
    
    for i_b in range(start_row, end_row, block_size):
        i_end = min(i_b + block_size, end_row)
        for j_b in range(0, N, block_size):
            j_end = min(j_b + block_size, N)
            for k_b in range(0, N, block_size):
                k_end = min(k_b + block_size, N)
                
                # Sub-matrix tile multiplication and accumulation
                C_strip[i_b - start_row:i_end - start_row, j_b:j_end] += np.dot(
                    A[i_b:i_end, k_b:k_end], 
                    B[k_b:k_end, j_b:j_end]
                )
    return C_strip, start_row

def main():
    sys.stdout = Tee("task8_output.txt")
    
    student_id = 230103386
    max_cores = os.cpu_count() or 4
    matrix_size = 512  # Set to 512 for rapid verification (use 1024 for full benchmark scale)
    
    print("=== TASK 8: CPU Blocked/Tiled Matrix Multiplication Geometry ===")
    print(f"Student ID: {student_id}")
    print(f"Detected Logical Cores: {max_cores}")
    print(f"Matrix Dimensions: {matrix_size} x {matrix_size}")
    print("-" * 75)
    
    # Initialize matrices with Student ID seed
    np.random.seed(student_id)
    A = np.random.rand(matrix_size, matrix_size).astype(np.float64)
    B = np.random.rand(matrix_size, matrix_size).astype(np.float64)
    
    strategies = [
        ("Non-Tiled Baseline", None),
        ("Blocked Tile (16x16)", 16),
        ("Blocked Tile (64x64)", 64)
    ]
    
    results = []
    
    for name, block_size in strategies:
        start_time = time.perf_counter()
        
        if block_size is None:
            # Non-tiled parallel row strip multiplication
            row_strips = np.array_split(A, max_cores, axis=0)
            with ThreadPoolExecutor(max_workers=max_cores) as executor:
                futures = [executor.submit(matmul_nontiled_block, (strip, B)) for strip in row_strips]
                _ = np.vstack([f.result() for f in futures])
        else:
            # Tiled parallel multiplication
            row_ranges = []
            chunk_size = matrix_size // max_cores
            for c in range(max_cores):
                start = c * chunk_size
                end = matrix_size if c == max_cores - 1 else (c + 1) * chunk_size
                row_ranges.append((start, end))
                
            with ThreadPoolExecutor(max_workers=max_cores) as executor:
                futures = [
                    executor.submit(matmul_tiled_block, (A, B, start, end, block_size)) 
                    for start, end in row_ranges
                ]
                # Reconstruct matrix from ordered blocks
                output_parts = sorted([f.result() for f in futures], key=lambda x: x[1])
                _ = np.vstack([part[0] for part in output_parts])
                
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0
        
        results.append({
            'strategy': name,
            'time_ms': elapsed_ms
        })
        print(f"-> {name:<25} | Execution Latency: {elapsed_ms:.2f} ms")
        
    print("-" * 75)
    print("Summary:")
    print(f"{'Strategy / Geometry':<25} | {'Runtime (ms)':<15} | {'Cache Behavior / Note'}")
    print("-" * 75)
    for r in results:
        note = "Standard row striping (higher cache miss)" if "Non-Tiled" in r['strategy'] else "Optimized L1/L2 cache locality"
        print(f"{r['strategy']:<25} | {r['time_ms']:<15.2f} | {note}")
    print("=" * 75)

if __name__ == "__main__":
    main()