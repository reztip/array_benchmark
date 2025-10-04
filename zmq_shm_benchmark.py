"""ZeroMQ with shared memory benchmark implementation for NumPy array transmission."""

import time
import multiprocessing as mp
import numpy as np
import zmq
import os
import tempfile
from multiprocessing import shared_memory


class ZeroMQSharedMemoryBenchmark:
    def __init__(self, array_size=1024, batch_size=1, port=5556): # port is unused but kept for interface consistency
        self.array_size = array_size
        self.batch_size = batch_size
        self.element_size = 4  # float32
        self.array_bytes = array_size * self.element_size
        # IPC endpoint will be generated per run
        self.ipc_endpoint = None

    def producer(self, n_arrays, results_queue, shm_name, ipc_endpoint, ready_event):
        """Producer process that writes arrays to shared memory and sends notifications via ZeroMQ IPC."""
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.bind(ipc_endpoint)

        # Connect to shared memory
        shm = shared_memory.SharedMemory(name=shm_name)
        # Create a NumPy array that directly uses the shared memory buffer
        shm_array = np.ndarray((n_arrays, self.array_size), dtype=np.float32, buffer=shm.buf)

        # Wait for consumer to be ready
        ready_event.wait()

        start_time = time.time()

        for i in range(0, n_arrays, self.batch_size):
            batch_size = min(self.batch_size, n_arrays - i)
            
            # Generate random data for the batch
            batch_data = np.random.random((batch_size, self.array_size)).astype(np.float32)
            
            # Write batch to shared memory slice
            shm_array[i : i + batch_size] = batch_data

            # Send notification with batch info (start index, size)
            socket.send_string(f"{i},{batch_size}")

        # Send termination signal
        socket.send_string("DONE")
        end_time = time.time()

        socket.close()
        context.term()
        shm.close()
        results_queue.put(("producer", end_time - start_time))

    def consumer(self, n_arrays, results_queue, shm_name, ipc_endpoint, ready_event):
        """Consumer process that receives notifications via ZeroMQ IPC and reads arrays from shared memory."""
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.connect(ipc_endpoint)

        # Connect to shared memory
        shm = shared_memory.SharedMemory(name=shm_name)
        # Create a NumPy array that directly uses the shared memory buffer
        shm_array = np.ndarray((n_arrays, self.array_size), dtype=np.float32, buffer=shm.buf)

        # Signal that consumer is ready
        ready_event.set()

        start_time = time.time()
        received_count = 0

        while received_count < n_arrays:
            # Receive notification
            message = socket.recv_string()
            if message == "DONE":
                break

            # Parse batch info
            start_index, batch_size = map(int, message.split(','))

            # "Read" batch from shared memory.
            # To simulate a real workload, we copy the data. In a true zero-copy
            # pipeline, the consumer might work on the data in-place.
            # Batch copy is more efficient than copying individual arrays
            batch_copy = shm_array[start_index:start_index + batch_size].copy()
            # In a real application, you would process the batch here
            received_count += batch_size

        end_time = time.time()

        socket.close()
        context.term()
        shm.close()
        results_queue.put(("consumer", end_time - start_time))

    def run_benchmark(self, n_arrays):
        """Run the ZeroMQ shared memory benchmark and return timing results."""
        total_size = n_arrays * self.array_bytes

        # Create a unique name for the shared memory block
        shm_name = f"shm_{os.getpid()}_{time.time():.6f}"
        shm = shared_memory.SharedMemory(create=True, size=total_size, name=shm_name)

        # Create a unique IPC endpoint for this benchmark run
        ipc_endpoint = f"ipc://{tempfile.gettempdir()}/zmq_shm_benchmark_{os.getpid()}_{time.time():.6f}.ipc"

        try:
            results_queue = mp.Queue()
            ready_event = mp.Event()

            # Start consumer first
            consumer_process = mp.Process(
                target=self.consumer,
                args=(n_arrays, results_queue, shm.name, ipc_endpoint, ready_event)
            )
            consumer_process.start()

            # Start producer
            producer_process = mp.Process(
                target=self.producer,
                args=(n_arrays, results_queue, shm.name, ipc_endpoint, ready_event)
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
            # Clean up shared memory
            shm.close()
            shm.unlink()

            # Clean up IPC socket file if it exists
            ipc_path = ipc_endpoint.replace("ipc://", "")
            try:
                if os.path.exists(ipc_path):
                    os.unlink(ipc_path)
            except (OSError, FileNotFoundError):
                pass
