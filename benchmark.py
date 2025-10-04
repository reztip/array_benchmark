"Main benchmark runner and results handler."

import time
from dataclasses import dataclass
from typing import Any, Dict

# Import all benchmark classes
from mp_benchmark import MultiprocessingBenchmark
from pyarrow_benchmark import PyArrowBenchmark
from redis_benchmark import RedisBenchmark

from simple_shm_benchmark import SimpleSharedMemoryBenchmark
from zmq_benchmark import ZeroMQBenchmark
from zmq_shm_benchmark import ZeroMQSharedMemoryBenchmark


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""

    n_arrays: int
    array_size: int
    batch_size: int
    results: Dict[str, Dict[str, Any]]

    def _print_method_result(self, name: str, result: Dict[str, Any]):
        """Helper to print a single method's result line."""
        if not result:
            return
        if result.get("error"):
            print(f"{name:<30} {'ERROR: ' + result['error']:<50}")
            return

        total_time = result.get("total_time", 0)
        if total_time > 0:
            arrays_per_second = self.n_arrays / total_time
            mb_per_second = (
                self.n_arrays * self.array_size * 4
            ) / (total_time * 1024 * 1024)
        else:
            arrays_per_second = float("inf")
            mb_per_second = float("inf")

        print(
            f"{name:<30} {total_time:<15.4f} {arrays_per_second:<12.0f} {mb_per_second:<10.2f}"
        )

    def print_comparison(self):
        """Print a formatted comparison of the benchmark results."""
        print(f"\n{'=' * 60}")
        print("BENCHMARK RESULTS")
        print(f"{'=' * 60}")
        print(f"Arrays sent: {self.n_arrays}")
        print(f"Array size: {self.array_size} elements")
        print(f"Batch size: {self.batch_size}")
        print(f"Data per array: {self.array_size * 4} bytes (float32)")
        print(
            f"Total data: {self.n_arrays * self.array_size * 4 / (1024 * 1024):.2f} MB"
        )
        print()

        print(
            f"{'Method':<30} {'Total Time (s)':<15} {'Arrays/sec':<12} {'MB/s':<10}"
        )
        print(f"{'-' * 70}")

        method_order = [
            "ZeroMQ",
            "ZeroMQ + Shared Memory",
            "Simple Shared Memory",
            "PyArrow",
            "Redis Pub/Sub",
            "Multiprocessing",
        ]

        for name in method_order:
            if name in self.results:
                self._print_method_result(name, self.results[name])

        print()

        valid_methods = [
            (name, res["total_time"]) 
            for name, res in self.results.items()
            if res and not res.get("error") and res.get("total_time", 0) > 0
        ]

        if not valid_methods:
            print("No valid benchmark results to compare.")
            print(f"{'-' * 70}")
            return

        valid_methods.sort(key=lambda x: x[1])
        fastest_name, fastest_time = valid_methods[0]

        for name, time_taken in valid_methods[1:]:
            speedup = time_taken / fastest_time
            print(f"{fastest_name} is {speedup:.2f}x faster than {name}")

        print(f"{'-' * 70}")


def run_all_benchmarks(
    n_arrays: int = 1000,
    array_size: int = 1024,
    warmup_runs: int = 1,
    batch_size: int = 1,
    include_redis: bool = False,
    include_pyarrow: bool = False,
) -> BenchmarkResults:
    """
    Run all benchmarks comparing all available methods.

    Args:
        n_arrays: Number of arrays to send
        array_size: Size of each array (number of elements)
        warmup_runs: Number of warmup runs before actual benchmark
        batch_size: Maximum batch size for sending arrays
        include_ring_buffer: Whether to include shared memory ring buffer benchmark
        include_redis: Whether to include Redis Pub/Sub benchmark
        include_pyarrow: Whether to include PyArrow benchmark

    Returns:
        BenchmarkResults object containing timing and throughput data for all methods
    """
    print(f"Running all benchmarks: {n_arrays} arrays of size {array_size}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Batch size: {batch_size}")
    print()

    benchmarks_to_run = {
        "ZeroMQ": ZeroMQBenchmark(array_size=array_size, batch_size=batch_size),
        "ZeroMQ + Shared Memory": ZeroMQSharedMemoryBenchmark(
            array_size=array_size, batch_size=batch_size
        ),
        "Simple Shared Memory": SimpleSharedMemoryBenchmark(
            array_size=array_size, batch_size=batch_size
        ),
        "Multiprocessing": MultiprocessingBenchmark(
            array_size=array_size, batch_size=batch_size
        ),
    }


    if include_redis:
        benchmarks_to_run["Redis Pub/Sub"] = RedisBenchmark(
            array_size=array_size, batch_size=batch_size
        )
    if include_pyarrow:
        benchmarks_to_run["PyArrow"] = PyArrowBenchmark(
            array_size=array_size, batch_size=batch_size
        )

    if warmup_runs > 0:
        print("Running warmup...")
        warmup_size = min(100, n_arrays)
        for i in range(warmup_runs):
            print(f"  Warmup {i + 1}/{warmup_runs}")
            for name, bench in benchmarks_to_run.items():
                try:
                    bench.run_benchmark(warmup_size)
                except Exception as e:
                    print(f"    Error during warmup for {name}: {e}")
        print("Warmup complete\n")

    results = {}
    for name, bench in benchmarks_to_run.items():
        print(f"Running {name} benchmark...")
        try:
            results[name] = bench.run_benchmark(n_arrays)
        except Exception as e:
            print(f"  Error running benchmark for {name}: {e}")
            results[name] = {"error": str(e)}
        time.sleep(0.5)

    benchmark_results = BenchmarkResults(
        n_arrays=n_arrays,
        array_size=array_size,
        batch_size=batch_size,
        results=results,
    )

    benchmark_results.print_comparison()
    return benchmark_results


def run_benchmark(
    n_arrays: int = 1000,
    array_size: int = 1024,
    warmup_runs: int = 1,
    batch_size: int = 1,
) -> BenchmarkResults:
    """
    Run benchmarks comparing ZeroMQ and multiprocessing queues.

    Args:
        n_arrays: Number of arrays to send
        array_size: Size of each array (number of elements)
        warmup_runs: Number of warmup runs before actual benchmark
        batch_size: Maximum batch size for sending arrays

    Returns:
        BenchmarkResults object containing timing and throughput data
    """
    print(f"Running benchmark: {n_arrays} arrays of size {array_size}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Batch size: {batch_size}")
    print()

    benchmarks_to_run = {
        "ZeroMQ": ZeroMQBenchmark(array_size=array_size, batch_size=batch_size),
        "Multiprocessing": MultiprocessingBenchmark(
            array_size=array_size, batch_size=batch_size
        ),
    }

    if warmup_runs > 0:
        print("Running warmup...")
        warmup_size = min(100, n_arrays)
        for i in range(warmup_runs):
            print(f"  Warmup {i + 1}/{warmup_runs}")
            for bench in benchmarks_to_run.values():
                bench.run_benchmark(warmup_size)
        print("Warmup complete\n")

    results = {}
    for name, bench in benchmarks_to_run.items():
        print(f"Running {name} benchmark...")
        results[name] = bench.run_benchmark(n_arrays)
        time.sleep(0.5)

    benchmark_results = BenchmarkResults(
        n_arrays=n_arrays,
        array_size=array_size,
        batch_size=batch_size,
        results=results,
    )

    benchmark_results.print_comparison()
    return benchmark_results