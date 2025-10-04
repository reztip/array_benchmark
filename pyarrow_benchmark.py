"""PyArrow benchmark implementation for NumPy array transmission."""

import time
import multiprocessing as mp
import numpy as np
import zmq
import os
import tempfile
import pyarrow as pa


class PyArrowBenchmark:
    def __init__(self, array_size=1024, batch_size=1, port=5555):
        self.array_size = array_size
        self.batch_size = batch_size
        self.port = port
        self.ipc_endpoint = f"ipc://{tempfile.gettempdir()}/pyarrow_benchmark_{os.getpid()}_{time.time():.6f}.ipc"
        # Create schema once to avoid overhead in the loop
        self.schema = pa.schema([
            pa.field('array_bytes', pa.list_(pa.binary()))
        ]).with_metadata({
            'dtype': 'float32',
            'shape': ','.join(map(str, (self.array_size,)))
        })

    def producer(self, n_arrays, results_queue, ipc_endpoint):
        """Producer process that sends NumPy arrays via PyArrow over ZeroMQ IPC."""
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.bind(ipc_endpoint)

        time.sleep(0.1)

        start_time = time.time()

        for i in range(0, n_arrays, self.batch_size):
            batch_size = min(self.batch_size, n_arrays - i)
            batch_arrays = [np.random.random(self.array_size).astype(np.float32) for _ in range(batch_size)]
            
            # Create a RecordBatch
            batch = pa.RecordBatch.from_pydict(
                {'array_bytes': [[arr.tobytes() for arr in batch_arrays]]},
                schema=self.schema
            )

            # Serialize the RecordBatch to a buffer using the IPC stream format
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, self.schema) as writer:
                writer.write_batch(batch)
            buffer = sink.getvalue()

            socket.send(buffer)

        socket.send(b"DONE")

        end_time = time.time()
        socket.close()
        context.term()

        results_queue.put(("producer", end_time - start_time))

    def consumer(self, n_arrays, results_queue, ipc_endpoint):
        """Consumer process that receives NumPy arrays via PyArrow over ZeroMQ IPC."""
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.connect(ipc_endpoint)

        start_time = time.time()
        received_count = 0
        
        numpy_dtype = None
        shape = None

        while received_count < n_arrays:
            buffer = socket.recv()
            if buffer == b"DONE":
                break

            with pa.ipc.open_stream(buffer) as reader:
                batch = reader.read_next_batch()

            if received_count == 0:
                # Extract metadata from the schema on the first message
                meta = {k.decode(): v.decode() for k, v in batch.schema.metadata.items()}
                numpy_dtype = np.dtype(meta['dtype'])
                shape = tuple(map(int, meta['shape'].split(',')))

            # Extract and reconstruct the NumPy arrays
            arr_bytes_list = batch.to_pydict()['array_bytes'][0]
            for arr_bytes in arr_bytes_list:
                array = np.frombuffer(arr_bytes, dtype=numpy_dtype).reshape(shape)
                received_count += 1

        end_time = time.time()
        socket.close()
        context.term()

        results_queue.put(("consumer", end_time - start_time))

    def run_benchmark(self, n_arrays):
        """Run the PyArrow benchmark and return timing results."""
        results_queue = mp.Queue()

        ipc_endpoint = f"ipc://{tempfile.gettempdir()}/pyarrow_benchmark_{os.getpid()}_{time.time():.6f}.ipc"

        try:
            consumer_process = mp.Process(
                target=self.consumer,
                args=(n_arrays, results_queue, ipc_endpoint)
            )
            consumer_process.start()

            producer_process = mp.Process(
                target=self.producer,
                args=(n_arrays, results_queue, ipc_endpoint)
            )
            producer_process.start()

            producer_process.join()
            consumer_process.join()

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
            ipc_path = ipc_endpoint.replace("ipc://", "")
            try:
                if os.path.exists(ipc_path):
                    os.unlink(ipc_path)
            except (OSError, FileNotFoundError):
                pass