"""ZeroMQ with shared memory benchmark implementation for NumPy array transmission."""

import time
import multiprocessing as mp
import numpy as np
import zmq
import mmap
import os
import tempfile
from pathlib import Path


class ZeroMQSharedMemoryBenchmark:
    def __init__(self, array_size=1024, port=5556):
        self.array_size = array_size
        self.port = port
        self.element_size = 4  # float32
        self.array_bytes = array_size * self.element_size

    def create_shared_memory_file(self, n_arrays):
        """Create a memory-mapped file for shared memory."""
        temp_dir = tempfile.gettempdir()
        shm_file = Path(temp_dir) / f"benchmark_shm_{os.getpid()}_{time.time():.6f}.dat"

        total_size = n_arrays * self.array_bytes

        # Create and initialize the file
        with open(shm_file, 'wb') as f:
            f.write(b'\x00' * total_size)

        return str(shm_file)

    def producer(self, n_arrays, results_queue, shm_file_path):
        """Producer process that writes arrays to shared memory and sends notifications via ZeroMQ."""
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.bind(f"tcp://*:{self.port}")

        # Small delay to ensure consumer is ready
        time.sleep(0.1)

        # Open shared memory file
        with open(shm_file_path, 'r+b') as f:
            with mmap.mmap(f.fileno(), 0) as mm:
                start_time = time.time()

                for i in range(n_arrays):
                    # Generate array
                    array = np.random.random(self.array_size).astype(np.float32)

                    # Write to shared memory at offset
                    offset = i * self.array_bytes
                    mm[offset:offset + self.array_bytes] = array.tobytes()

                    # Send notification with array index via ZeroMQ
                    socket.send_string(str(i))

                # Send termination signal
                socket.send_string("DONE")

                end_time = time.time()

        socket.close()
        context.term()
        results_queue.put(("producer", end_time - start_time))

    def consumer(self, n_arrays, results_queue, shm_file_path):
        """Consumer process that receives notifications via ZeroMQ and reads arrays from shared memory."""
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.connect(f"tcp://localhost:{self.port}")

        # Open shared memory file
        with open(shm_file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                start_time = time.time()
                received_count = 0

                while received_count < n_arrays:
                    # Receive notification
                    message = socket.recv_string()
                    if message == "DONE":
                        break

                    # Parse array index
                    array_index = int(message)

                    # Read array from shared memory
                    offset = array_index * self.array_bytes
                    array_bytes = mm[offset:offset + self.array_bytes]
                    array = np.frombuffer(array_bytes, dtype=np.float32)

                    received_count += 1

                end_time = time.time()

        socket.close()
        context.term()
        results_queue.put(("consumer", end_time - start_time))

    def run_benchmark(self, n_arrays):
        """Run the ZeroMQ shared memory benchmark and return timing results."""
        # Create shared memory file
        shm_file_path = self.create_shared_memory_file(n_arrays)

        try:
            results_queue = mp.Queue()

            # Start consumer first
            consumer_process = mp.Process(
                target=self.consumer,
                args=(n_arrays, results_queue, shm_file_path)
            )
            consumer_process.start()

            # Start producer
            producer_process = mp.Process(
                target=self.producer,
                args=(n_arrays, results_queue, shm_file_path)
            )
            producer_process.start()

            # Wait for both processes to complete
            producer_process.join()
            consumer_process.join()

            # Collect results
            results = {}
            while not results_queue.empty():
                role, duration = results_queue.get()
                results[role] = duration

            total_time = max(results.get("producer", 0), results.get("consumer", 0))

            return {
                "total_time": total_time,
                "producer_time": results.get("producer", 0),
                "consumer_time": results.get("consumer", 0),
                "arrays_per_second": n_arrays / total_time if total_time > 0 else 0
            }

        finally:
            # Clean up shared memory file
            try:
                os.unlink(shm_file_path)
            except (OSError, FileNotFoundError):
                pass  # File might already be deleted