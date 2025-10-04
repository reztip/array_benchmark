"""Shared memory ring buffer benchmark implementation for NumPy array transmission."""

import time
import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory
import struct
import threading


class SharedMemoryRingBufferBenchmark:
    def __init__(self, array_size=1024, batch_size=1, buffer_capacity=1024):
        self.array_size = array_size
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity  # Number of arrays the ring buffer can hold
        self.element_size = 4  # float32
        self.array_bytes = array_size * self.element_size

        # Ring buffer metadata size (producer_idx, consumer_idx, flags)
        self.metadata_size = 64  # Cache line aligned, extra space for safety

        # Total shared memory size: metadata + ring buffer data
        self.total_size = self.metadata_size + (buffer_capacity * self.array_bytes)

    def producer(self, n_arrays, results_queue, shm_name):
        """Producer process that writes arrays to shared memory ring buffer."""
        # Connect to existing shared memory
        shm = shared_memory.SharedMemory(name=shm_name)

        try:
            # Map metadata section
            metadata_view = shm.buf[:self.metadata_size]

            # Map ring buffer data section
            data_start = self.metadata_size
            data_size = self.buffer_capacity * self.array_bytes
            data_array = np.frombuffer(data_view, dtype=np.float32).reshape(
                self.buffer_capacity, self.array_size
            )

            start_time = time.time()

            for i in range(0, n_arrays, self.batch_size):
                batch_size = min(self.batch_size, n_arrays - i)
                batch = [np.random.random(self.array_size).astype(np.float32) for _ in range(batch_size)]

                # Wait for space in ring buffer
                while True:
                    producer_idx = struct.unpack('Q', metadata_view[:8])[0]
                    consumer_idx = struct.unpack('Q', metadata_view[8:16])[0]

                    if (producer_idx + batch_size) % self.buffer_capacity != consumer_idx:
                        break
                    time.sleep(0.0001)

                # Write batch to ring buffer
                for j in range(batch_size):
                    buffer_idx = (producer_idx + j) % self.buffer_capacity
                    data_array[buffer_idx] = batch[j]

                # Update producer index
                next_producer_idx = (producer_idx + batch_size) % self.buffer_capacity
                struct.pack_into('Q', metadata_view, 0, next_producer_idx)

            # Set completion flag
            struct.pack_into('Q', metadata_view, 16, 1)

            end_time = time.time()
            results_queue.put(("producer", end_time - start_time))

        finally:
            shm.close()

    def consumer(self, n_arrays, results_queue, shm_name):
        """Consumer process that reads arrays from shared memory ring buffer."""
        # Connect to existing shared memory
        shm = shared_memory.SharedMemory(name=shm_name)

        try:
            # Map metadata section
            metadata_view = shm.buf[:self.metadata_size]

            # Map ring buffer data section
            data_start = self.metadata_size
            data_size = self.buffer_capacity * self.array_bytes
            data_array = np.frombuffer(data_view, dtype=np.float32).reshape(
                self.buffer_capacity, self.array_size
            )

            start_time = time.time()
            received_count = 0

            while received_count < n_arrays:
                producer_idx = struct.unpack('Q', metadata_view[:8])[0]
                consumer_idx = struct.unpack('Q', metadata_view[8:16])[0]

                if consumer_idx != producer_idx:
                    num_to_read = (producer_idx - consumer_idx + self.buffer_capacity) % self.buffer_capacity
                    for i in range(num_to_read):
                        buffer_idx = (consumer_idx + i) % self.buffer_capacity
                        array = data_array[buffer_idx].copy()
                        received_count += 1

                    next_consumer_idx = (consumer_idx + num_to_read) % self.buffer_capacity
                    struct.pack_into('Q', metadata_view, 8, next_consumer_idx)
                else:
                    done_flag = struct.unpack('Q', metadata_view[16:24])[0]
                    if done_flag and consumer_idx == producer_idx:
                        break
                    time.sleep(0.0001)

            end_time = time.time()
            results_queue.put(("consumer", end_time - start_time))

        finally:
            shm.close()

    def run_benchmark(self, n_arrays):
        """Run the shared memory ring buffer benchmark and return timing results."""
        # Create shared memory with shorter name
        shm = shared_memory.SharedMemory(
            create=True,
            size=self.total_size,
            name=f"rb{mp.current_process().pid}{int(time.time() * 1000000) % 1000000}"
        )

        try:
            # Initialize metadata (producer_idx, consumer_idx, done_flag)
            metadata_view = shm.buf[:self.metadata_size]
            struct.pack_into('Q', metadata_view, 0, 0)   # producer_idx = 0
            struct.pack_into('Q', metadata_view, 8, 0)   # consumer_idx = 0
            struct.pack_into('Q', metadata_view, 16, 0)  # done_flag = 0

            results_queue = mp.Queue()

            # Start consumer first
            consumer_process = mp.Process(
                target=self.consumer,
                args=(n_arrays, results_queue, shm.name)
            )
            consumer_process.start()

            # Start producer
            producer_process = mp.Process(
                target=self.producer,
                args=(n_arrays, results_queue, shm.name)
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
            # Clean up shared memory
            shm.close()
            shm.unlink()