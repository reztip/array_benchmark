# Array IPC Benchmark

A collection of benchmarks comparing different methods for inter-process communication (IPC) of NumPy arrays in Python.

## Description

This project benchmarks and compares the performance of several methods for passing NumPy arrays between processes:

*   **ZeroMQ:** Using `pyzmq` for message passing.
*   **Multiprocessing:** Using the standard `multiprocessing` module's queues.
*   **ZeroMQ + Shared Memory:** Using ZeroMQ to pass metadata for arrays stored in shared memory.
*   **Simple Shared Memory:** A basic implementation using `multiprocessing.shared_memory`.
*   **Shared Memory Ring Buffer:** A more complex implementation using a ring buffer in shared memory.

## Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/your-username/array_benchmark.git
    cd array_benchmark
    ```

2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the benchmarks using the `cli.py` script.

### Basic Benchmark

To run a basic comparison between ZeroMQ and `multiprocessing`:

```bash
python cli.py -n 1000 -s 1024
```

### Full Benchmark

To run all available benchmarks, including shared memory implementations:

```bash
python cli.py --all -n 1000 -s 1024
```

### Options

*   `-n`, `--arrays`: The number of arrays to send in the benchmark (default: 1000).
*   `-s`, `--size`: The size of each array in elements (default: 1024).
*   `-w`, `--warmup`: The number of warmup runs before the benchmark (default: 1).
*   `--all`: Run all available benchmarks.
