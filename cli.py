"""Command-line interface for the array queue benchmark."""

import argparse
from benchmark import run_benchmark, run_all_benchmarks


def main():
    """Main CLI entry point for benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark ZeroMQ vs multiprocessing queues for NumPy array transmission"
    )

    parser.add_argument(
        "-n", "--arrays",
        type=int,
        default=1000,
        help="Number of arrays to send (default: 1000)"
    )

    parser.add_argument(
        "-s", "--size",
        type=int,
        default=1024,
        help="Size of each array in elements (default: 1024)"
    )

    parser.add_argument(
        "-w", "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs (default: 1)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks including ZeroMQ + Shared Memory"
    )

    parser.add_argument(
        "--redis",
        action="store_true",
        help="Include Redis Pub/Sub benchmark in 'all' benchmarks"
    )

    parser.add_argument(
        "--pyarrow",
        action="store_true",
        help="Include PyArrow benchmark in 'all' benchmarks"
    )

    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=1,
        help="Maximum batch size for sending arrays (default: 1)"
    )

    args = parser.parse_args()

    if args.all:
        run_all_benchmarks(
            n_arrays=args.arrays,
            array_size=args.size,
            warmup_runs=args.warmup,
            batch_size=args.batch_size,
            include_redis=args.redis,
            include_pyarrow=args.pyarrow
        )
    else:
        run_benchmark(
            n_arrays=args.arrays,
            array_size=args.size,
            warmup_runs=args.warmup,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
