import zmq
import numpy as np
import pyarrow as pa
import time

def main():
    """
    Producer script that generates NumPy arrays, serializes them into the
    Arrow IPC stream format, and sends them over a ZeroMQ PUSH socket.
    This method is robust across many PyArrow versions.
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind("tcp://*:5555")
    print("Producer is running, pushing data on tcp://*:5555")

    count = 0
    while True:
        try:
            # 1. Generate data
            count_val = count
            ts_val = time.time()
            array_val = np.random.rand(3, 4)
            dtype_str = str(array_val.dtype)

            # 2. Define a schema with metadata
            # Metadata must be strings.
            schema = pa.schema([
                pa.field('array_bytes', pa.binary())
            ]).with_metadata({
                'count': str(count_val),
                'timestamp': str(ts_val),
                'dtype': dtype_str,
                'shape': ','.join(map(str, array_val.shape))
            })

            # 3. Create a RecordBatch
            # The NumPy array is converted to raw bytes.
            batch = pa.RecordBatch.from_pydict(
                {'array_bytes': [array_val.tobytes()]},
                schema=schema
            )

            # 4. Serialize the RecordBatch to a buffer using the IPC stream format
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, batch.schema) as writer:
                writer.write_batch(batch)
            buffer = sink.getvalue()

            # 5. Send the buffer
            socket.send(buffer)

            print(f"Sent message: {count}")
            count += 1
            time.sleep(1)

        except KeyboardInterrupt:
            print("\nShutting down producer.")
            break

    socket.close()
    context.term()

if __name__ == "__main__":
    main()