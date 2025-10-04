"""Redis Pub/Sub benchmark implementation for NumPy array transmission."""

import time
import multiprocessing as mp
import numpy as np
import redis
import os

import msgpack
import msgpack_numpy as m

m.patch()

class RedisBenchmark:
    def __init__(self, array_size=1024, batch_size=1, host='localhost', port=6379):
        self.array_size = array_size
        self.batch_size = batch_size
        self.redis_host = host
        self.redis_port = port
        self.channel = f"redis_benchmark_{os.getpid()}"

    def producer(self, n_arrays, results_queue):
        """Producer process that sends NumPy arrays via Redis Pub/Sub."""
        r = redis.Redis(host=self.redis_host, port=self.redis_port)
        try:
            r.ping()
        except redis.exceptions.ConnectionError as e:
            print(f"Error connecting to Redis: {e}")
            results_queue.put(("producer", -1))
            return

        start_time = time.time()

        for i in range(0, n_arrays, self.batch_size):
            batch_size = min(self.batch_size, n_arrays - i)
            batch = [np.random.random(self.array_size).astype(np.float32) for _ in range(batch_size)]
            packed_batch = msgpack.packb(batch, use_bin_type=True)
            r.publish(self.channel, packed_batch)

        # Send termination signal
        r.publish(self.channel, b"DONE")

        end_time = time.time()
        results_queue.put(("producer", end_time - start_time))

    def consumer(self, n_arrays, results_queue):
        """Consumer process that receives NumPy arrays via Redis Pub/Sub."""
        r = redis.Redis(host=self.redis_host, port=self.redis_port)
        try:
            r.ping()
        except redis.exceptions.ConnectionError as e:
            print(f"Error connecting to Redis: {e}")
            results_queue.put(("consumer", -1))
            return

        pubsub = r.pubsub()
        pubsub.subscribe(self.channel)

        start_time = time.time()
        received_count = 0

        for message in pubsub.listen():
            if message['type'] == 'message':
                if message['data'] == b"DONE":
                    break
                
                # Deserialize batch
                unpacked_batch = msgpack.unpackb(message['data'], raw=False)
                received_count += len(unpacked_batch)
                if received_count >= n_arrays:
                    break
        
        end_time = time.time()
        pubsub.unsubscribe()
        pubsub.close()
        results_queue.put(("consumer", end_time - start_time))

    def run_benchmark(self, n_arrays):
        """Run the Redis Pub/Sub benchmark and return timing results."""
        results_queue = mp.Queue()

        # Start consumer first
        consumer_process = mp.Process(
            target=self.consumer,
            args=(n_arrays, results_queue)
        )
        consumer_process.start()

        # Small delay to ensure consumer is subscribed
        time.sleep(0.1)

        # Start producer
        producer_process = mp.Process(
            target=self.producer,
            args=(n_arrays, results_queue)
        )
        producer_process.start()

        # Wait for both processes to complete
        producer_process.join()
        consumer_process.join()

        # Collect results
        results = {}
        while not results_queue.empty():
            role, duration = results_queue.get()
            if duration == -1:
                return {
                    "total_time": -1,
                    "error": "Failed to connect to Redis"
                }
            results[role] = duration

        total_time = max(results.get("producer", 0), results.get("consumer", 0))

        return {
            "total_time": total_time,
            "producer_time": results.get("producer", 0),
            "consumer_time": results.get("consumer", 0),
            "arrays_per_second": n_arrays / total_time if total_time > 0 else 0
        }
