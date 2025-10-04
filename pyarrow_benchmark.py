"""PyArrow benchmark implementation for NumPy array transmission, using RecordBatch."""

import time
import multiprocessing as mp
import numpy as np
import zmq
import os
import tempfile
import pyarrow as pa
from multiprocessing import shared_memory


class PyArrowBenchmark:
    def __init__(self, array_size=1024, batch_size=1, port=5555):
        self.array_size = array_size
        self.batch_size = batch_size
        # Use simple schema with flat float32 array for better performance
        self.schema = pa.schema([
            pa.field('data', pa.float32())
        ])
        self.ipc_endpoint = None

    def _estimate_size(self, n_arrays):
        """Estimates the shared memory size needed for batched serialization."""
        # Create a sample batch to estimate overhead
        sample_batch_size = min(self.batch_size, n_arrays)
        sample_data = np.zeros(sample_batch_size * self.array_size, dtype=np.float32)

        # Create table and serialize
        table = pa.Table.from_arrays([pa.array(sample_data)], schema=self.schema)

        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        buffer = sink.getvalue()

        # Calculate per-batch overhead and total size
        bytes_per_batch = len(buffer.to_pybytes())
        num_batches = (n_arrays + self.batch_size - 1) // self.batch_size

        # 30% safety buffer for metadata overhead
        estimated_size = int(bytes_per_batch * num_batches * 1.3)
        return estimated_size

    def producer(self, n_arrays, results_queue, shm_name, shm_size, ipc_endpoint, ready_event):
        """Producer: writes batched Arrow tables to shared memory."""
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.bind(ipc_endpoint)

        shm = shared_memory.SharedMemory(name=shm_name)

        # Wait for consumer to be ready
        ready_event.wait()

        start_time = time.time()

        current_offset = 0
        out_of_memory = False

        for i in range(0, n_arrays, self.batch_size):
            actual_batch_size = min(self.batch_size, n_arrays - i)
            numpy_batch = np.random.random((actual_batch_size, self.array_size)).astype(np.float32)

            # Flatten entire batch and serialize as single Table
            # This amortizes IPC overhead across all arrays in the batch
            flat_data = numpy_batch.flatten()
            table = pa.Table.from_arrays([pa.array(flat_data)], schema=self.schema)

            # Serialize entire batch at once
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)

            buffer = sink.getvalue()
            buffer_size = buffer.size

            if current_offset + buffer_size > shm_size:
                print(f"Producer: Not enough shared memory! Offset: {current_offset}, Size: {buffer_size}, Total: {shm_size}")
                out_of_memory = True
                break

            # Single memcpy for entire batch
            shm.buf[current_offset:current_offset + buffer_size] = buffer.to_pybytes()

            # Send metadata: offset, size, number of arrays in this batch
            socket.send_string(f"{current_offset},{buffer_size},{actual_batch_size}")
            current_offset += buffer_size

            if out_of_memory:
                break

        socket.send_string("DONE")
        end_time = time.time()

        shm.close()
        socket.close()
        context.term()
        results_queue.put(("producer", end_time - start_time))

    def consumer(self, n_arrays, results_queue, shm_name, ipc_endpoint, ready_event):
        """Consumer: reads batched Arrow tables from shared memory using zero-copy."""
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.connect(ipc_endpoint)

        shm = shared_memory.SharedMemory(name=shm_name)

        # Signal that consumer is ready
        ready_event.set()

        start_time = time.time()
        received_count = 0

        while received_count < n_arrays:
            message = socket.recv_string()
            if message == "DONE":
                break

            parts = message.split(',')
            offset, size, batch_arrays = int(parts[0]), int(parts[1]), int(parts[2])

            # Copy data from shared memory to avoid buffer reference issues
            buffer_data = bytes(shm.buf[offset:offset + size])
            buffer = pa.py_buffer(buffer_data)

            # Deserialize the Arrow table
            with pa.ipc.open_stream(buffer) as reader:
                table = reader.read_all()

            # Convert to numpy efficiently and reshape back to original dimensions
            # to_numpy() is much faster than as_py()
            flat_array = table.column(0).to_numpy()
            arrays = flat_array.reshape(batch_arrays, self.array_size)

            # In a real application, you would process the arrays here
            received_count += batch_arrays

        end_time = time.time()

        shm.close()
        socket.close()
        context.term()
        results_queue.put(("consumer", end_time - start_time))

    def run_benchmark(self, n_arrays):
        """Run the PyArrow zero-copy RecordBatch benchmark."""

        shm_size = self._estimate_size(n_arrays)
        shm_name = f"pa_shm_{os.getpid()}_{time.time():.6f}"
        shm = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)

        ipc_endpoint = f"ipc://{tempfile.gettempdir()}/pyarrow_benchmark_{os.getpid()}_{time.time():.6f}.ipc"

        try:
            results_queue = mp.Queue()
            ready_event = mp.Event()

            consumer_process = mp.Process(
                target=self.consumer,
                args=(n_arrays, results_queue, shm.name, ipc_endpoint, ready_event)
            )
            consumer_process.start()

            producer_process = mp.Process(
                target=self.producer,
                args=(n_arrays, results_queue, shm.name, shm_size, ipc_endpoint, ready_event)
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
            shm.close()
            shm.unlink()
            ipc_path = ipc_endpoint.replace("ipc://", "")
            try:
                if os.path.exists(ipc_path):
                    os.unlink(ipc_path)
            except (OSError, FileNotFoundError):
                pass
