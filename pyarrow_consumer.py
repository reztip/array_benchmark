import zmq
import numpy as np
import pyarrow as pa

def main():
    """
    Consumer script that receives Arrow IPC streams from a ZeroMQ PULL socket
    and deserializes them back into NumPy arrays.
    This method is robust across many PyArrow versions.
    """
    context = zmq.Context()
    socket = context.socket(zmq.PULL)

    producer_ip = "localhost"
    socket.connect(f"tcp://{producer_ip}:5555")
    print(f"Consumer is running, pulling data from tcp://{producer_ip}:5555")

    while True:
        try:
            # 1. Receive a message
            buffer = socket.recv()

            # 2. Use an IPC stream reader to read the RecordBatch
            with pa.ipc.open_stream(buffer) as reader:
                batch = reader.read_next_batch()

            # 3. Extract metadata from the schema
            # Metadata keys are bytes, so we decode them.
            meta = {k.decode(): v.decode() for k, v in batch.schema.metadata.items()}
            count = int(meta['count'])
            timestamp = float(meta['timestamp'])
            dtype = meta['dtype']
            shape = tuple(map(int, meta['shape'].split(',')))

            # 4. Extract and reconstruct the NumPy array
            # The array was stored as raw bytes in the first column.
            arr_bytes = batch.to_pydict()['array_bytes'][0]
            array = np.frombuffer(arr_bytes, dtype=np.dtype(dtype)).reshape(shape)

            # 5. Print the results
            print("--- Received Message ---")
            print(f"Count: {count}")
            print(f"Timestamp: {timestamp}")
            print(f"Array Shape: {array.shape}")
            print("Array Data:")
            print(array)
            print("\n")

        except KeyboardInterrupt:
            print("\nShutting down consumer.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    socket.close()
    context.term()

if __name__ == "__main__":
    main()

