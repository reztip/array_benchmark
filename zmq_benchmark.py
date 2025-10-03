"""ZeroMQ benchmark implementation for NumPy array transmission."""

import time
import multiprocessing as mp
import numpy as np
import zmq
import os
import tempfile


class ZeroMQBenchmark:
    def __init__(self, array_size=1024, port=5555):
        self.array_size = array_size
        self.port = port
        # Create unique IPC endpoint
        self.ipc_endpoint = f"ipc://{tempfile.gettempdir()}/zmq_benchmark_{os.getpid()}_{time.time():.6f}.ipc"

    def producer(self, n_arrays, results_queue, ipc_endpoint):
        """Producer process that sends NumPy arrays via ZeroMQ IPC."""
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.bind(ipc_endpoint)

        # Small delay to ensure consumer is ready
        time.sleep(0.1)

        start_time = time.time()

        for i in range(n_arrays):
            array = np.random.random(self.array_size).astype(np.float32)
            socket.send(array.tobytes())

        # Send termination signal
        socket.send(b"DONE")

        end_time = time.time()
        socket.close()
        context.term()

        results_queue.put(("producer", end_time - start_time))

    def consumer(self, n_arrays, results_queue, ipc_endpoint):
        """Consumer process that receives NumPy arrays via ZeroMQ IPC."""
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.connect(ipc_endpoint)

        start_time = time.time()
        received_count = 0

        while received_count < n_arrays:
            data = socket.recv()
            if data == b"DONE":
                break

            # Deserialize array
            array = np.frombuffer(data, dtype=np.float32)
            received_count += 1

        end_time = time.time()
        socket.close()
        context.term()

        results_queue.put(("consumer", end_time - start_time))

    def run_benchmark(self, n_arrays):
        """Run the ZeroMQ IPC benchmark and return timing results."""
        results_queue = mp.Queue()

        # Create unique IPC endpoint for this benchmark run
        ipc_endpoint = f"ipc://{tempfile.gettempdir()}/zmq_benchmark_{os.getpid()}_{time.time():.6f}.ipc"

        try:
            # Start consumer first
            consumer_process = mp.Process(
                target=self.consumer,
                args=(n_arrays, results_queue, ipc_endpoint)
            )
            consumer_process.start()

            # Start producer
            producer_process = mp.Process(
                target=self.producer,
                args=(n_arrays, results_queue, ipc_endpoint)
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
            # Clean up IPC socket file if it exists
            ipc_path = ipc_endpoint.replace("ipc://", "")
            try:
                if os.path.exists(ipc_path):
                    os.unlink(ipc_path)
            except (OSError, FileNotFoundError):
                pass