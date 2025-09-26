"""
Array Queue Benchmark Package

A benchmarking package for comparing ZeroMQ and multiprocessing queue performance
when sending NumPy arrays.
"""

from .benchmark import run_benchmark, run_all_benchmarks, BenchmarkResults
from .zmq_benchmark import ZeroMQBenchmark
from .mp_benchmark import MultiprocessingBenchmark
from .zmq_shm_benchmark import ZeroMQSharedMemoryBenchmark
from .shm_ringbuffer_benchmark import SharedMemoryRingBufferBenchmark

__version__ = "0.1.0"
__all__ = ["run_benchmark", "run_all_benchmarks", "BenchmarkResults", "ZeroMQBenchmark", "MultiprocessingBenchmark", "ZeroMQSharedMemoryBenchmark", "SharedMemoryRingBufferBenchmark"]