# Array IPC Benchmark

A collection of benchmarks comparing different methods for inter-process communication (IPC) of NumPy arrays in Python.

## Description

This project benchmarks and compares the performance of several methods for passing NumPy arrays between processes:

*   **ZeroMQ:** Using `pyzmq` for message passing over IPC sockets.
*   **Multiprocessing:** Using the standard `multiprocessing` module's queues.
*   **ZeroMQ + Shared Memory:** Using ZeroMQ to pass metadata for arrays stored in shared memory (zero-copy approach).
*   **Simple Shared Memory:** A basic implementation using `mmap` for file-backed shared memory.
*   **PyArrow:** Using Apache Arrow's IPC format with shared memory for efficient serialization.
*   **Redis Pub/Sub:** Using Redis Pub/Sub to send arrays with msgpack serialization.

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

Run the benchmarks using the `cli.py` script.

### Basic Benchmark

To run a basic comparison between ZeroMQ and multiprocessing:

```bash
python cli.py -n 1000 -s 1024
```

This will run only ZeroMQ and Multiprocessing benchmarks and display:
- Total time for each method
- Throughput in arrays/second
- Throughput in MB/second
- Performance comparison between methods

### Full Benchmark Suite

To run all available benchmarks (ZeroMQ, ZeroMQ + Shared Memory, Simple Shared Memory, and Multiprocessing):

```bash
python cli.py --all -n 1000 -s 1024
```

### PyArrow Benchmark

To include the PyArrow IPC benchmark with the full suite:

```bash
python cli.py --all --pyarrow -n 1000 -s 1024
```

### Redis Benchmark

To include the Redis Pub/Sub benchmark, you first need a running Redis server.
You can start one easily with Docker:

```bash
# Make sure to use host networking for the benchmark
docker run -d --name redis-host --network host redis:latest
```

Then, run the benchmark with the `--redis` flag:

```bash
python cli.py --all --redis -n 1000 -s 1024
```

### Batch Size Configuration

To test performance with batching (sending multiple arrays per message):

```bash
python cli.py --all -n 10000 -s 1024 -b 10
```

This sends arrays in batches of 10, which can significantly improve throughput for some methods.

### Warmup Runs

Control the number of warmup iterations to ensure stable measurements:

```bash
python cli.py --all -n 1000 -s 1024 -w 3
```

### Command-Line Options

*   `-n`, `--arrays`: Number of arrays to send (default: 1000)
*   `-s`, `--size`: Size of each array in elements (default: 1024)
*   `-b`, `--batch-size`: Maximum batch size for sending arrays (default: 1)
*   `-w`, `--warmup`: Number of warmup runs before benchmarking (default: 1)
*   `--all`: Run all available benchmarks (ZeroMQ, Shared Memory implementations, Multiprocessing)
*   `--pyarrow`: Include PyArrow benchmark (requires `--all`)
*   `--redis`: Include Redis Pub/Sub benchmark (requires `--all`)

### Example Output

```
Running all benchmarks: 10000 arrays of size 1024
Warmup runs: 1
Batch size: 1

Running warmup...
  Warmup 1/1
Warmup complete

Running ZeroMQ benchmark...
Running ZeroMQ + Shared Memory benchmark...
Running Simple Shared Memory benchmark...
Running Multiprocessing benchmark...

============================================================
BENCHMARK RESULTS
============================================================
Arrays sent: 10000
Array size: 1024 elements
Batch size: 1
Data per array: 4096 bytes (float32)
Total data: 40.00 MB

Method                         Total Time (s)  Arrays/sec   MB/s
----------------------------------------------------------------------
ZeroMQ                         0.8234          12144        47.43
ZeroMQ + Shared Memory         0.3521          28394        110.92
Simple Shared Memory           0.2156          46384        181.19
Multiprocessing                2.4531          4076         15.92

Simple Shared Memory is 1.63x faster than ZeroMQ + Shared Memory
Simple Shared Memory is 3.82x faster than ZeroMQ
Simple Shared Memory is 11.38x faster than Multiprocessing
----------------------------------------------------------------------
```

## Performance Tips

1. **Batch Size**: Increase batch size for better throughput when sending many small arrays
2. **Array Size**: Shared memory methods excel with larger arrays due to zero-copy design
3. **Warmup**: Use 2-3 warmup runs for more stable measurements
4. **System Load**: Run benchmarks on an idle system for consistent results
