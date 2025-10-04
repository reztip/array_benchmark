"""Multiprocessing queue benchmark implementation for NumPy array transmission."""

import time
import multiprocessing as mp
import numpy as np


class MultiprocessingBenchmark:
    def __init__(self, array_size=1024, batch_size=1):
        self.array_size = array_size
        self.batch_size = batch_size

    def producer(self, n_arrays, queue, results_queue):
        """Producer process that sends NumPy arrays via multiprocessing queue."""
        start_time = time.time()

        for i in range(0, n_arrays, self.batch_size):
            batch_size = min(self.batch_size, n_arrays - i)
            batch = np.random.random((batch_size, self.array_size)).astype(np.float32)
            queue.put(batch)

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

            # Data is a batch of NumPy arrays
            received_count += len(data)

        end_time = time.time()
        results_queue.put(("consumer", end_time - start_time))

    def run_benchmark(self, n_arrays):
        """Run the multiprocessing queue benchmark and return timing results."""
        # Use a multiprocessing queue for data transfer with maxsize to prevent memory issues
        data_queue = mp.Queue(maxsize=100)
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

        # Wait for both processes to complete with timeout
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

        # Clean up queues
        data_queue.close()
        data_queue.join_thread()
        results_queue.close()
        results_queue.join_thread()

        total_time = max(results.get("producer", 0), results.get("consumer", 0))

        return {
            "total_time": total_time,
            "producer_time": results.get("producer", 0),
            "consumer_time": results.get("consumer", 0),
            "arrays_per_second": n_arrays / total_time if total_time > 0 else 0
        }