"""Main benchmark runner and results handler."""

import time
from dataclasses import dataclass
from typing import Dict, Any, Optional
from zmq_benchmark import ZeroMQBenchmark
from mp_benchmark import MultiprocessingBenchmark
from zmq_shm_benchmark import ZeroMQSharedMemoryBenchmark
from shm_ringbuffer_benchmark import SharedMemoryRingBufferBenchmark
from simple_shm_benchmark import SimpleSharedMemoryBenchmark
from redis_benchmark import RedisBenchmark
from pyarrow_benchmark import PyArrowBenchmark


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""
    zmq_results: Dict[str, Any]
    mp_results: Dict[str, Any]
    n_arrays: int
    array_size: int
    zmq_shm_results: Optional[Dict[str, Any]] = None
    shm_ring_results: Optional[Dict[str, Any]] = None
    simple_shm_results: Optional[Dict[str, Any]] = None
    redis_results: Optional[Dict[str, Any]] = None
    pyarrow_results: Optional[Dict[str, Any]] = None

    def print_comparison(self):
        """Print a formatted comparison of the benchmark results."""
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Arrays sent: {self.n_arrays}")
        print(f"Array size: {self.array_size} elements")
        print(f"Data per array: {self.array_size * 4} bytes (float32)")
        print(f"Total data: {self.n_arrays * self.array_size * 4 / (1024*1024):.2f} MB")
        print()

        print(f"{'Method':<30} {'Total Time (s)':<15} {'Arrays/sec':<12} {'MB/s':<10}")
        print(f"{'-'*70}")

        zmq_throughput_mb = (self.n_arrays * self.array_size * 4) / (self.zmq_results['total_time'] * 1024 * 1024)
        mp_throughput_mb = (self.n_arrays * self.array_size * 4) / (self.mp_results['total_time'] * 1024 * 1024)

        print(f"{'ZeroMQ':<30} {self.zmq_results['total_time']:<15.4f} {self.zmq_results['arrays_per_second']:<12.0f} {zmq_throughput_mb:<10.2f}")

        if self.zmq_shm_results:
            zmq_shm_throughput_mb = (self.n_arrays * self.array_size * 4) / (self.zmq_shm_results['total_time'] * 1024 * 1024)
            print(f"{'ZeroMQ + Shared Memory':<30} {self.zmq_shm_results['total_time']:<15.4f} {self.zmq_shm_results['arrays_per_second']:<12.0f} {zmq_shm_throughput_mb:<10.2f}")

        if self.shm_ring_results:
            shm_ring_throughput_mb = (self.n_arrays * self.array_size * 4) / (self.shm_ring_results['total_time'] * 1024 * 1024)
            print(f"{'Shared Memory Ring Buffer':<30} {self.shm_ring_results['total_time']:<15.4f} {self.shm_ring_results['arrays_per_second']:<12.0f} {shm_ring_throughput_mb:<10.2f}")

        if self.simple_shm_results:
            simple_shm_throughput_mb = (self.n_arrays * self.array_size * 4) / (self.simple_shm_results['total_time'] * 1024 * 1024)
            print(f"{'Simple Shared Memory':<30} {self.simple_shm_results['total_time']:<15.4f} {self.simple_shm_results['arrays_per_second']:<12.0f} {simple_shm_throughput_mb:<10.2f}")

        if self.redis_results:
            if self.redis_results.get("error"):
                print(f"{'Redis Pub/Sub':<30} {'ERROR: ' + self.redis_results['error']:<50}")
            else:
                redis_throughput_mb = (self.n_arrays * self.array_size * 4) / (self.redis_results['total_time'] * 1024 * 1024)
                print(f"{'Redis Pub/Sub':<30} {self.redis_results['total_time']:<15.4f} {self.redis_results['arrays_per_second']:<12.0f} {redis_throughput_mb:<10.2f}")
        
        if self.pyarrow_results:
            pyarrow_throughput_mb = (self.n_arrays * self.array_size * 4) / (self.pyarrow_results['total_time'] * 1024 * 1024)
            print(f"{'PyArrow':<30} {self.pyarrow_results['total_time']:<15.4f} {self.pyarrow_results['arrays_per_second']:<12.0f} {pyarrow_throughput_mb:<10.2f}")

        print(f"{'Multiprocessing':<30} {self.mp_results['total_time']:<15.4f} {self.mp_results['arrays_per_second']:<12.0f} {mp_throughput_mb:<10.2f}")
        print()

        # Find fastest method
        methods = [
            ("ZeroMQ", self.zmq_results['total_time']),
            ("Multiprocessing", self.mp_results['total_time'])
        ]

        if self.zmq_shm_results:
            methods.append(("ZeroMQ + Shared Memory", self.zmq_shm_results['total_time']))

        if self.shm_ring_results:
            methods.append(("Shared Memory Ring Buffer", self.shm_ring_results['total_time']))

        if self.simple_shm_results:
            methods.append(("Simple Shared Memory", self.simple_shm_results['total_time']))
        
        if self.redis_results and not self.redis_results.get("error"):
            methods.append(("Redis Pub/Sub", self.redis_results['total_time']))
            
        if self.pyarrow_results:
            methods.append(("PyArrow", self.pyarrow_results['total_time']))

        methods.sort(key=lambda x: x[1])
        fastest = methods[0]

        for name, time_taken in methods[1:]:
            speedup = time_taken / fastest[1]
            print(f"{fastest[0]} is {speedup:.2f}x faster than {name}")

        print(f"{'='*70}")


def run_all_benchmarks(n_arrays: int = 1000, array_size: int = 1024, warmup_runs: int = 1, include_ring_buffer: bool = False, include_redis: bool = False, include_pyarrow: bool = False) -> BenchmarkResults:
    """
    Run all benchmarks comparing all available methods.

    Args:
        n_arrays: Number of arrays to send
        array_size: Size of each array (number of elements)
        warmup_runs: Number of warmup runs before actual benchmark
        include_ring_buffer: Whether to include shared memory ring buffer benchmark
        include_redis: Whether to include Redis Pub/Sub benchmark
        include_pyarrow: Whether to include PyArrow benchmark

    Returns:
        BenchmarkResults object containing timing and throughput data for all methods
    """
    print(f"Running all benchmarks: {n_arrays} arrays of size {array_size}")
    print(f"Warmup runs: {warmup_runs}")
    print()

    # Initialize benchmarks
    zmq_bench = ZeroMQBenchmark(array_size=array_size)
    zmq_shm_bench = ZeroMQSharedMemoryBenchmark(array_size=array_size)
    mp_bench = MultiprocessingBenchmark(array_size=array_size)
    simple_shm_bench = SimpleSharedMemoryBenchmark(array_size=array_size)

    shm_ring_bench = None
    if include_ring_buffer:
        # Use smaller buffer capacity for large arrays to avoid excessive memory usage
        buffer_capacity = min(1024, max(64, n_arrays // 4))
        shm_ring_bench = SharedMemoryRingBufferBenchmark(
            array_size=array_size,
            buffer_capacity=buffer_capacity
        )
    
    redis_bench = None
    if include_redis:
        redis_bench = RedisBenchmark(array_size=array_size)

    pyarrow_bench = None
    if include_pyarrow:
        pyarrow_bench = PyArrowBenchmark(array_size=array_size)

    # Warmup runs
    if warmup_runs > 0:
        print("Running warmup...")
        warmup_size = min(100, n_arrays)
        for i in range(warmup_runs):
            print(f"  Warmup {i+1}/{warmup_runs}")
            zmq_bench.run_benchmark(warmup_size)
            zmq_shm_bench.run_benchmark(warmup_size)
            simple_shm_bench.run_benchmark(warmup_size)
            mp_bench.run_benchmark(warmup_size)
            if shm_ring_bench:
                shm_ring_bench.run_benchmark(warmup_size)
            if redis_bench:
                redis_bench.run_benchmark(warmup_size)
            if pyarrow_bench:
                pyarrow_bench.run_benchmark(warmup_size)
            time.sleep(0.1)
        print("Warmup complete\n")

    # Run ZeroMQ benchmark
    print("Running ZeroMQ benchmark...")
    zmq_results = zmq_bench.run_benchmark(n_arrays)
    time.sleep(0.5)

    # Run ZeroMQ + Shared Memory benchmark
    print("Running ZeroMQ + Shared Memory benchmark...")
    zmq_shm_results = zmq_shm_bench.run_benchmark(n_arrays)
    time.sleep(0.5)

    # Run simple shared memory benchmark
    print("Running Simple Shared Memory benchmark...")
    simple_shm_results = simple_shm_bench.run_benchmark(n_arrays)
    time.sleep(0.5)

    # Run shared memory ring buffer benchmark
    shm_ring_results = None
    if shm_ring_bench:
        print("Running Shared Memory Ring Buffer benchmark...")
        shm_ring_results = shm_ring_bench.run_benchmark(n_arrays)
        time.sleep(0.5)

    # Run multiprocessing benchmark
    print("Running multiprocessing benchmark...")
    mp_results = mp_bench.run_benchmark(n_arrays)

    # Run Redis benchmark
    redis_results = None
    if redis_bench:
        print("Running Redis benchmark...")
        redis_results = redis_bench.run_benchmark(n_arrays)
        time.sleep(0.5)

    # Run PyArrow benchmark
    pyarrow_results = None
    if pyarrow_bench:
        print("Running PyArrow benchmark...")
        pyarrow_results = pyarrow_bench.run_benchmark(n_arrays)
        time.sleep(0.5)

    # Create and return results
    results = BenchmarkResults(
        zmq_results=zmq_results,
        mp_results=mp_results,
        n_arrays=n_arrays,
        array_size=array_size,
        zmq_shm_results=zmq_shm_results,
        shm_ring_results=shm_ring_results,
        simple_shm_results=simple_shm_results,
        redis_results=redis_results,
        pyarrow_results=pyarrow_results
    )

    results.print_comparison()
    return results


def run_benchmark(n_arrays: int = 1000, array_size: int = 1024, warmup_runs: int = 1) -> BenchmarkResults:
    """
    Run benchmarks comparing ZeroMQ and multiprocessing queues.

    Args:
        n_arrays: Number of arrays to send
        array_size: Size of each array (number of elements)
        warmup_runs: Number of warmup runs before actual benchmark

    Returns:
        BenchmarkResults object containing timing and throughput data
    """
    print(f"Running benchmark: {n_arrays} arrays of size {array_size}")
    print(f"Warmup runs: {warmup_runs}")
    print()

    # Initialize benchmarks
    zmq_bench = ZeroMQBenchmark(array_size=array_size)
    mp_bench = MultiprocessingBenchmark(array_size=array_size)

    # Warmup runs
    if warmup_runs > 0:
        print("Running warmup...")
        for i in range(warmup_runs):
            print(f"  Warmup {i+1}/{warmup_runs}")
            zmq_bench.run_benchmark(min(100, n_arrays))
            mp_bench.run_benchmark(min(100, n_arrays))
            time.sleep(0.1)
        print("Warmup complete\n")

    # Run ZeroMQ benchmark
    print("Running ZeroMQ benchmark...")
    zmq_results = zmq_bench.run_benchmark(n_arrays)

    # Small delay between benchmarks
    time.sleep(0.5)

    # Run multiprocessing benchmark
    print("Running multiprocessing benchmark...")
    mp_results = mp_bench.run_benchmark(n_arrays)

    # Create and return results
    results = BenchmarkResults(
        zmq_results=zmq_results,
        mp_results=mp_results,
        n_arrays=n_arrays,
        array_size=array_size
    )

    results.print_comparison()
    return results
