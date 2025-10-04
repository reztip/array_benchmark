# Array IPC Benchmark

A comprehensive benchmarking suite comparing different inter-process communication (IPC) methods for NumPy array transmission in Python. This project helps you understand the performance trade-offs between different IPC mechanisms and choose the right approach for your use case.

## Description

High-performance data exchange between processes is critical for parallel computing, data pipelines, and distributed systems. This project provides detailed benchmarks comparing six different IPC methods, each with unique characteristics:

### Local IPC Methods (Same Machine)

*   **Multiprocessing Queues:** Python's standard `multiprocessing.Queue` - simple but involves data serialization/pickling
*   **ZeroMQ IPC:** High-performance message passing using `pyzmq` over Unix domain sockets
*   **ZeroMQ + Shared Memory:** Zero-copy approach using shared memory for data and ZeroMQ for coordination
*   **Simple Shared Memory:** File-backed shared memory using `mmap` with event synchronization
*   **PyArrow IPC:** Apache Arrow's columnar format with shared memory - good for structured data

### Network-Capable Methods (Different Machines)

*   **Redis Pub/Sub:** Message broker approach using Redis with msgpack serialization - works over network
*   **ZeroMQ TCP:** Can be configured for network communication (not benchmarked by default, but ZeroMQ supports it)

Each method is tested with:
- Configurable array sizes
- Batch processing capabilities
- Warmup runs for stable measurements
- Detailed throughput metrics (arrays/sec and MB/sec)

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

### Try It Yourself

Run the benchmarks on your system to see which method performs best for your hardware and use case:

```bash
# Quick test with 1000 arrays
python cli.py --all -n 1000 -s 1024

# Test with batching for better throughput
python cli.py --all -n 10000 -s 1024 -b 10

# Large arrays test
python cli.py --all -n 100 -s 1048576

# Include all methods (requires Redis running)
python cli.py --all --pyarrow --redis -n 5000 -s 4096 -b 5
```

Performance varies significantly based on:
- Array size (small vs large)
- Batch size configuration
- CPU architecture and cache sizes
- System memory bandwidth
- OS kernel version

## Performance Characteristics

### When to Use Each Method

**Shared Memory Methods** (ZeroMQ + SHM, Simple SHM, PyArrow)
- ✅ Best for: Large arrays, high throughput requirements
- ✅ Zero-copy transfers, minimal serialization overhead
- ❌ Limited to: Same machine only
- ❌ More complex: Resource cleanup required

**Multiprocessing Queues**
- ✅ Best for: Simple use cases, small to medium arrays
- ✅ Easy to use, built-in to Python
- ❌ Limited to: Same machine only
- ❌ Slower: Full data serialization (pickle)

**ZeroMQ IPC**
- ✅ Best for: Balanced performance and flexibility
- ✅ Good throughput, battle-tested library
- ✅ Can switch to TCP for network use
- ❌ Limited to: Unix sockets for best performance

**Redis Pub/Sub**
- ✅ Best for: Distributed systems, multiple consumers
- ✅ Works across: Network boundaries
- ✅ Pub/Sub pattern: Many-to-many communication
- ❌ Requires: External Redis server
- ❌ Slower: Network + serialization overhead

**PyArrow IPC**
- ✅ Best for: Complex data structures, columnar data
- ✅ Cross-language: Compatible with C++, Java, etc.
- ✅ Efficient: Optimized for analytical workloads
- ❌ Limited to: Same machine (shared memory mode)
- ❌ Overhead: Better with larger batches

## Cross-Machine Communication

For data transfer between different computers, these methods are viable:

### 1. **Redis Pub/Sub** (Benchmarked)
```bash
# On Machine A (Redis server)
docker run -d -p 6379:6379 redis:latest

# On Machine B (modify redis_benchmark.py)
# Change host='machine-a-ip' in RedisBenchmark.__init__
python cli.py --all --redis -n 1000 -s 1024
```

### 2. **ZeroMQ TCP** (Requires modification)
ZeroMQ can use TCP instead of IPC sockets. Modify the endpoint:
```python
# Instead of: ipc:///tmp/socket
# Use: tcp://192.168.1.100:5555
```

### 3. **Network Considerations**
- **Latency**: Network adds ~0.1-1ms per round trip (LAN) vs ~0.001ms (IPC)
- **Bandwidth**: Gigabit Ethernet ~125 MB/s vs shared memory ~10-50 GB/s
- **Serialization**: Network methods need efficient serialization (msgpack, Arrow, Protocol Buffers)

## Future Improvements

### Short-term Enhancements
- [ ] Add ZeroMQ TCP benchmark for network comparison
- [ ] Implement ring buffer with multiple producer/consumer support
- [ ] Add compression options (LZ4, Zstd) for network transfers
- [ ] Support for non-contiguous arrays and different dtypes
- [ ] Add latency measurements (p50, p95, p99)
- [ ] CSV/JSON export of results for analysis

### Advanced Features
- [ ] **RDMA Support**: Use `pyverbs` or `UCX` for InfiniBand/RoCE networks
- [ ] **GPU Direct**: Benchmark GPU-to-GPU transfers with CUDA IPC
- [ ] **Distributed Benchmarks**: Multi-node testing with MPI or Dask
- [ ] **Plasma Store**: Apache Arrow's shared memory object store
- [ ] **Shared Memory Ring Buffers**: Lock-free circular buffers for streaming
- [ ] **gRPC/Protocol Buffers**: Modern RPC framework comparison

### Serialization Alternatives
- [ ] Compare msgpack vs Protocol Buffers vs FlatBuffers vs Cap'n Proto
- [ ] Add zero-copy serialization with `pickle5` protocol
- [ ] Test NumPy native `.npy` format over network
- [ ] Evaluate Parquet for batch array transfer

### Network-Specific Optimizations
- [ ] RDMA (Remote Direct Memory Access) for ultra-low latency
- [ ] UCX (Unified Communication X) for HPC environments
- [ ] Memory-mapped files over NFS/distributed filesystems
- [ ] Custom TCP socket implementation with sendfile()

## Contributing

Contributions are welcome! Areas for improvement:
- Additional IPC methods
- Platform-specific optimizations (Windows, macOS, Linux)
- Better error handling and edge cases
- Performance analysis tools and visualizations
- Documentation and examples

## License

MIT License - See LICENSE file for details
