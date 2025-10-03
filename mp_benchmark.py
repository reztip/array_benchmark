"""Multiprocessing queue benchmark implementation for NumPy array transmission."""

import time
import multiprocessing as mp
import numpy as np


class MultiprocessingBenchmark:
    def __init__(self, array_size=1024):
        self.array_size = array_size

    def producer(self, n_arrays, queue, results_queue):
        """Producer process that sends NumPy arrays via multiprocessing queue."""
        start_time = time.time()

        for i in range(n_arrays):
            array = np.random.random(self.array_size).astype(np.float32)
            queue.put(array)

        # Send termination signal
        queue.put("DONE")

        end_time = time.time()
        results_queue.put(("producer", end_time - start_time))

    def consumer(self, n_arrays, queue, results_queue):
        """Consumer process that receives NumPy arrays via multiprocessing queue."""
        start_time = time.time()
        received_count = 0

        while received_count < n_arrays:
            data = queue.get()
            if isinstance(data, str) and data == "DONE":
                break

            # Data is already a NumPy array (multiprocessing handles serialization)
            received_count += 1

        end_time = time.time()
        results_queue.put(("consumer", end_time - start_time))

    def run_benchmark(self, n_arrays):
        """Run the multiprocessing queue benchmark and return timing results."""
        # Use a multiprocessing queue for data transfer
        data_queue = mp.Queue()
        results_queue = mp.Queue()

        # Start consumer first
        consumer_process = mp.Process(
            target=self.consumer,
            args=(n_arrays, data_queue, results_queue)
        )
        consumer_process.start()

        # Start producer
        producer_process = mp.Process(
            target=self.producer,
            args=(n_arrays, data_queue, results_queue)
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