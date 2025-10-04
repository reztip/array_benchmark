"""Simple shared memory benchmark using mmap for maximum performance."""

import time
import multiprocessing as mp
import numpy as np
import mmap
import os
import tempfile
import struct


class SimpleSharedMemoryBenchmark:
    def __init__(self, array_size=1024, batch_size=1):
        self.array_size = array_size
        self.batch_size = batch_size
        self.element_size = 4  # float32
        self.array_bytes = array_size * self.element_size

    def producer(self, n_arrays, results_queue, shm_path, ready_event, done_event):
        """Producer process that writes arrays sequentially to shared memory."""

        # Wait for consumer to be ready
        ready_event.wait()

        with open(shm_path, 'r+b') as f:
            with mmap.mmap(f.fileno(), 0) as mm:
                start_time = time.time()

                for i in range(0, n_arrays, self.batch_size):
                    batch_size = min(self.batch_size, n_arrays - i)
                    batch = np.random.random((batch_size, self.array_size)).astype(np.float32)
                    
                    # Write batch to shared memory
                    offset = i * self.array_bytes
                    size = batch_size * self.array_bytes
                    mm[offset:offset + size] = batch.tobytes()

                # Signal completion
                done_event.set()
                end_time = time.time()

        results_queue.put(("producer", end_time - start_time))

    def consumer(self, n_arrays, results_queue, shm_path, ready_event, done_event):
        """Consumer process that reads arrays sequentially from shared memory."""

        with open(shm_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Signal ready
                ready_event.set()

                # Wait for producer to signal completion
                done_event.wait()

                start_time = time.time()
                received_count = 0

                # Read all arrays in batches
                for i in range(0, n_arrays, self.batch_size):
                    batch_size = min(self.batch_size, n_arrays - i)
                    offset = i * self.array_bytes
                    size = batch_size * self.array_bytes
                    batch_bytes = mm[offset:offset + size]
                    batch_array = np.frombuffer(batch_bytes, dtype=np.float32).reshape(batch_size, self.array_size)
                    received_count += batch_array.shape[0]

                end_time = time.time()

        results_queue.put(("consumer", end_time - start_time))

    def run_benchmark(self, n_arrays):
        """Run the simple shared memory benchmark."""
        # Create temporary file for shared memory
        temp_dir = tempfile.gettempdir()
        shm_path = os.path.join(temp_dir, f"simple_shm_{os.getpid()}_{time.time():.6f}.dat")

        # Calculate total size needed
        total_size = n_arrays * self.array_bytes

        try:
            # Create and initialize the file using seek for better performance
            with open(shm_path, 'wb') as f:
                f.seek(total_size - 1)
                f.write(b'\x00')

            results_queue = mp.Queue()
            ready_event = mp.Event()
            done_event = mp.Event()

            # Start both processes
            consumer_process = mp.Process(
                target=self.consumer,
                args=(n_arrays, results_queue, shm_path, ready_event, done_event)
            )
            producer_process = mp.Process(
                target=self.producer,
                args=(n_arrays, results_queue, shm_path, ready_event, done_event)
            )

            consumer_process.start()
            producer_process.start()

            # Wait for completion with timeout
            producer_process.join(timeout=300)
            consumer_process.join(timeout=300)

            # Check if processes are still alive
            if producer_process.is_alive():
                producer_process.terminate()
                producer_process.join()
            if consumer_process.is_alive():
                consumer_process.terminate()
                consumer_process.join()

            # Collect results
            results = {}
            while not results_queue.empty():
                role, duration = results_queue.get()
                results[role] = duration

            # Clean up queue
            results_queue.close()
            results_queue.join_thread()

            total_time = max(results.get("producer", 0), results.get("consumer", 0))

            return {
                "total_time": total_time,
                "producer_time": results.get("producer", 0),
                "consumer_time": results.get("consumer", 0),
                "arrays_per_second": n_arrays / total_time if total_time > 0 else 0
            }

        finally:
            # Clean up
            try:
                os.unlink(shm_path)
            except (OSError, FileNotFoundError):
                pass