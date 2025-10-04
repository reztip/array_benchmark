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
        # Schema for a RecordBatch containing a single array (as a list of floats)
        self.schema = pa.schema([
            pa.field('array', pa.list_(pa.float32()))
        ])
        self.ipc_endpoint = None

    def _estimate_size(self, n_arrays):
        """Estimates the shared memory size needed by serializing one RecordBatch."""
        numpy_array = np.zeros(self.array_size, dtype=np.float32)
        
        # Create a batch directly from the numpy array
        record_batch = pa.RecordBatch.from_arrays([pa.array([numpy_array])], schema=self.schema)

        # Serialize to get size
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, self.schema) as writer:
            writer.write_batch(record_batch)
        buffer = sink.getvalue()
        
        # Estimate for n_arrays with a 20% safety buffer
        estimated_size = int(len(buffer.to_pybytes()) * n_arrays * 1.2)
        return estimated_size

    def producer(self, n_arrays, results_queue, shm_name, shm_size, ipc_endpoint, ready_event):
        """Producer: writes Arrow RecordBatches to shared memory and sends notifications."""
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
            batch_size = min(self.batch_size, n_arrays - i)
            numpy_batch = np.random.random((batch_size, self.array_size)).astype(np.float32)

            for j in range(batch_size):
                numpy_array = numpy_batch[j]

                # Create a RecordBatch containing that single array
                record_batch = pa.RecordBatch.from_arrays([pa.array([numpy_array])], schema=self.schema)

                # Serialize the RecordBatch
                sink = pa.BufferOutputStream()
                with pa.ipc.new_stream(sink, self.schema) as writer:
                    writer.write_batch(record_batch)
                buffer = sink.getvalue()
                buffer_size = buffer.size

                if current_offset + buffer_size > shm_size:
                    print(f"Producer: Not enough shared memory! Offset: {current_offset}, Size: {buffer_size}, Total: {shm_size}")
                    out_of_memory = True
                    break

                shm.buf[current_offset : current_offset + buffer_size] = buffer.to_pybytes()

                socket.send_string(f"{current_offset},{buffer_size}")
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
        """Consumer: receives notifications and reads Arrow RecordBatches from shared memory."""
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

            offset, size = map(int, message.split(','))
            
            shm_slice = shm.buf[offset : offset + size]

            with pa.ipc.open_stream(shm_slice) as reader:
                record_batch = reader.read_next_batch()
            
            # Extract the array and convert to NumPy
            numpy_array = record_batch.column(0)[0].as_py()
            
            received_count += 1

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
