"""ZeroMQ benchmark implementation for NumPy array transmission."""

import time
import multiprocessing as mp
import numpy as np
import zmq
import os
import tempfile


class ZeroMQBenchmark:
    def __init__(self, array_size=1024, batch_size=1, port=5555):
        self.array_size = array_size
        self.batch_size = batch_size
        self.port = port
        # Create unique IPC endpoint
        self.ipc_endpoint = f"ipc://{tempfile.gettempdir()}/zmq_benchmark_{os.getpid()}_{time.time():.6f}.ipc"

    def producer(self, n_arrays, results_queue, ipc_endpoint, ready_event):
        """Producer process that sends NumPy arrays via ZeroMQ IPC."""
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.bind(ipc_endpoint)

        # Wait for consumer to be ready
        ready_event.wait()

        start_time = time.time()

        for i in range(0, n_arrays, self.batch_size):
            batch_size = min(self.batch_size, n_arrays - i)
            numpy_batch = np.random.random((batch_size, self.array_size)).astype(np.float32)
            # Send entire batch as single message for better performance
            socket.send(numpy_batch.tobytes())

        # Send termination signal
        socket.send(b"DONE")

        end_time = time.time()
        socket.close()
        context.term()

        results_queue.put(("producer", end_time - start_time))

    def consumer(self, n_arrays, results_queue, ipc_endpoint, ready_event):
        """Consumer process that receives NumPy arrays via ZeroMQ IPC."""
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.connect(ipc_endpoint)

        # Signal that consumer is ready
        ready_event.set()

        start_time = time.time()
        received_count = 0

        while received_count < n_arrays:
            data = socket.recv()
            if data == b"DONE":
                break

            # Deserialize batch
            batch = np.frombuffer(data, dtype=np.float32).reshape(-1, self.array_size)
            received_count += len(batch)

        end_time = time.time()
        socket.close()
        context.term()

        results_queue.put(("consumer", end_time - start_time))

    def run_benchmark(self, n_arrays):
        """Run the ZeroMQ IPC benchmark and return timing results."""
        results_queue = mp.Queue()
        ready_event = mp.Event()

        # Create unique IPC endpoint for this benchmark run
        ipc_endpoint = f"ipc://{tempfile.gettempdir()}/zmq_benchmark_{os.getpid()}_{time.time():.6f}.ipc"

        try:
            # Start consumer first
            consumer_process = mp.Process(
                target=self.consumer,
                args=(n_arrays, results_queue, ipc_endpoint, ready_event)
            )
            consumer_process.start()

            # Start producer
            producer_process = mp.Process(
                target=self.producer,
                args=(n_arrays, results_queue, ipc_endpoint, ready_event)
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
            # Clean up IPC socket file if it exists
            ipc_path = ipc_endpoint.replace("ipc://", "")
            try:
                if os.path.exists(ipc_path):
                    os.unlink(ipc_path)
            except (OSError, FileNotFoundError):
                pass