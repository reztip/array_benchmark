"""
Array Queue Benchmark Package

A benchmarking package for comparing ZeroMQ and multiprocessing queue performance
when sending NumPy arrays.

Includes support for reading large HDF5 files from S3 with partial reads.
"""

from .benchmark import run_benchmark, run_all_benchmarks, BenchmarkResults
from .zmq_benchmark import ZeroMQBenchmark
from .mp_benchmark import MultiprocessingBenchmark
from .zmq_shm_benchmark import ZeroMQSharedMemoryBenchmark
from .s3_hdf5 import S3HDF5Reader, read_s3_hdf5_metadata, read_s3_hdf5_array

__version__ = "0.1.0"
__all__ = [
    "run_benchmark",
    "run_all_benchmarks",
    "BenchmarkResults",
    "ZeroMQBenchmark",
    "MultiprocessingBenchmark",
    "ZeroMQSharedMemoryBenchmark",
    "S3HDF5Reader",
    "read_s3_hdf5_metadata",
    "read_s3_hdf5_array",
]