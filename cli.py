"""Command-line interface for the array queue benchmark."""

import argparse
from .benchmark import run_benchmark, run_all_benchmarks


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

    args = parser.parse_args()

    if args.all:
        run_all_benchmarks(
            n_arrays=args.arrays,
            array_size=args.size,
            warmup_runs=args.warmup
        )
    else:
        run_benchmark(
            n_arrays=args.arrays,
            array_size=args.size,
            warmup_runs=args.warmup
        )


if __name__ == "__main__":
    main()
