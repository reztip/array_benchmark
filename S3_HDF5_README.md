# S3 HDF5 Reader Package

A Python package for reading HDF5 files stored on Amazon S3 with support for partial reads and efficient slicing of large multi-dimensional arrays.

## Features

- **Direct S3 Access**: Read HDF5 files directly from S3 without downloading the entire file
- **Partial Reads**: Use HTTP range requests to only download the data you need
- **Array Slicing**: Efficiently slice large 4-dimensional arrays (~2GB+)
- **Metadata Support**: Read JSON metadata stored within HDF5 files
- **Memory Efficient**: Process large datasets in chunks without loading everything into memory
- **AWS Integration**: Works with AWS credentials, anonymous access, and S3-compatible storage

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `h5py>=3.8.0` - HDF5 file operations
- `s3fs>=2023.1.0` - S3 filesystem access with range requests
- `fsspec>=2023.1.0` - Filesystem specification
- `numpy` - Array operations

## Quick Start

### Basic Usage

```python
from s3_hdf5 import S3HDF5Reader
import numpy as np

# Open HDF5 file from S3
s3_path = "s3://my-bucket/data/experiment.h5"

with S3HDF5Reader(s3_path) as reader:
    # Read metadata
    metadata = reader.read_metadata()
    print(f"Metadata: {metadata}")

    # Get array information
    info = reader.get_array_info('data')
    print(f"Array shape: {info['shape']}")
    print(f"Array size: {info['nbytes'] / 1e9:.2f} GB")

    # Read a slice (only downloads requested portion)
    data_slice = reader.read_array_slice(np.s_[0:10, :, :, 0:5])
    print(f"Slice shape: {data_slice.shape}")
```

### Convenience Functions

For quick one-off reads:

```python
from s3_hdf5 import read_s3_hdf5_metadata, read_s3_hdf5_array
import numpy as np

# Quick metadata read
metadata = read_s3_hdf5_metadata("s3://my-bucket/data/experiment.h5")

# Quick array read
data = read_s3_hdf5_array(
    "s3://my-bucket/data/experiment.h5",
    array_slice=np.s_[0:100, :, :, :]
)
```

## HDF5 File Structure

The package expects HDF5 files with the following structure:

```
/
├── metadata (dataset) - JSON string containing metadata
└── data (dataset) - 4D array of scientific data
```

### Creating Compatible HDF5 Files

```python
import h5py
import numpy as np
import json

# Create HDF5 file
with h5py.File('experiment.h5', 'w') as f:
    # Store metadata as JSON string
    metadata = {
        'experiment_id': 'exp_001',
        'timestamp': '2025-01-15T10:00:00Z',
        'parameters': {'temperature': 298.15, 'pressure': 101.325}
    }
    f.create_dataset('metadata', data=json.dumps(metadata))

    # Store 4D array (time, x, y, channels)
    # Example: 1000 timesteps, 512x512 spatial, 4 channels
    data = np.random.randn(1000, 512, 512, 4).astype(np.float32)
    f.create_dataset('data', data=data, chunks=(10, 64, 64, 1),
                     compression='gzip', compression_opts=4)
```

**Important**: Use chunking when creating large HDF5 files to enable efficient partial reads.

## Usage Examples

### Example 1: Read Specific Time Slices

```python
with S3HDF5Reader(s3_path) as reader:
    # Read first 10 timesteps, all spatial dimensions
    time_slice = reader.read_array_slice(np.s_[0:10, :, :, :])

    # Read every 10th timestep
    sparse_time = reader.read_array_slice(np.s_[::10, :, :, :])

    # Read single timestep
    single_frame = reader.read_array_slice(np.s_[0, :, :, :])
```

### Example 2: Read Spatial Subsets

```python
with S3HDF5Reader(s3_path) as reader:
    # Read 100x100 spatial region, all timesteps
    spatial_subset = reader.read_array_slice(np.s_[:, 100:200, 100:200, :])

    # Read specific channel
    channel_0 = reader.read_array_slice(np.s_[:, :, :, 0])
```

### Example 3: Memory-Efficient Chunk Processing

```python
with S3HDF5Reader(s3_path) as reader:
    info = reader.get_array_info('data')
    total_timesteps = info['shape'][0]
    chunk_size = 100

    results = []
    for start in range(0, total_timesteps, chunk_size):
        end = min(start + chunk_size, total_timesteps)

        # Process one chunk at a time
        chunk = reader.read_array_slice(np.s_[start:end, :, :, :])
        result = np.mean(chunk, axis=(1, 2, 3))  # Compute stats
        results.append(result)

    all_results = np.concatenate(results)
```

### Example 4: List Available Datasets

```python
with S3HDF5Reader(s3_path) as reader:
    datasets = reader.list_datasets()
    for name, info in datasets.items():
        print(f"{name}: {info}")
```

## AWS Credentials Configuration

### Method 1: Environment Variables (Recommended)

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

```python
# Credentials are automatically loaded
with S3HDF5Reader(s3_path) as reader:
    data = reader.read_metadata()
```

### Method 2: AWS Credentials File

Configure `~/.aws/credentials`:

```ini
[default]
aws_access_key_id = your-access-key
aws_secret_access_key = your-secret-key
region = us-east-1
```

### Method 3: Anonymous Access (Public Buckets)

```python
with S3HDF5Reader(s3_path, anon=True) as reader:
    data = reader.read_metadata()
```

### Method 4: Custom S3 Endpoint (MinIO, etc.)

```python
with S3HDF5Reader(
    s3_path,
    s3_additional_kwargs={
        'endpoint_url': 'http://localhost:9000',
        'key': 'minioadmin',
        'secret': 'minioadmin'
    }
) as reader:
    data = reader.read_metadata()
```

## API Reference

### `S3HDF5Reader`

Main class for reading HDF5 files from S3.

#### Constructor

```python
S3HDF5Reader(s3_path, anon=False, s3_additional_kwargs=None)
```

**Parameters:**
- `s3_path` (str): S3 URI path (e.g., 's3://bucket/path/file.h5')
- `anon` (bool): Use anonymous access. Default: False
- `s3_additional_kwargs` (dict): Additional arguments for S3FileSystem

#### Methods

##### `open()`
Open the HDF5 file. Returns self for context manager support.

##### `close()`
Close the file and release resources.

##### `read_metadata(metadata_key='metadata')`
Read JSON metadata from the HDF5 file.

**Returns:** dict

##### `get_array_info(array_key='data')`
Get array information without reading data.

**Returns:** dict with keys: shape, dtype, size, nbytes, chunks, compression

##### `read_array_slice(array_slice=None, array_key='data')`
Read a slice of the 4D array using partial reads.

**Parameters:**
- `array_slice`: NumPy slice object (e.g., `np.s_[0:10, :, :, 0:5]`)
- `array_key`: Dataset key in HDF5 file

**Returns:** numpy.ndarray

##### `read_full_array(array_key='data')`
Read the entire array. Warning: Downloads all data.

**Returns:** numpy.ndarray

##### `list_datasets()`
List all datasets and groups in the file.

**Returns:** dict

### Convenience Functions

#### `read_s3_hdf5_metadata(s3_path, metadata_key='metadata', anon=False)`
Quick function to read metadata.

#### `read_s3_hdf5_array(s3_path, array_slice=None, array_key='data', anon=False)`
Quick function to read array data.

## Performance Tips

1. **Use Chunking**: When creating HDF5 files, use appropriate chunk sizes (e.g., `chunks=(10, 64, 64, 1)`)

2. **Compression**: Enable compression to reduce S3 transfer costs:
   ```python
   f.create_dataset('data', data=array, compression='gzip', compression_opts=4)
   ```

3. **Read Only What You Need**: Avoid reading the full array; use slicing:
   ```python
   # Good: Only downloads needed data
   chunk = reader.read_array_slice(np.s_[0:10, :, :, :])

   # Bad: Downloads entire 2GB array
   full_array = reader.read_full_array()
   ```

4. **Process in Chunks**: For large computations, process data in manageable chunks

5. **S3 Transfer Acceleration**: For large datasets, consider enabling S3 Transfer Acceleration

6. **Caching**: For repeated access, consider caching slices locally

## Troubleshooting

### Error: "HDF5 file not found at: s3://..."

- Check that the S3 path is correct
- Verify AWS credentials have access to the bucket
- Ensure the file exists in S3

### Error: "Metadata key 'metadata' not found"

- The HDF5 file doesn't have a dataset named 'metadata'
- Use `list_datasets()` to see available datasets
- Specify the correct key: `reader.read_metadata('your_metadata_key')`

### Error: "Expected 4D array, but dataset has N dimensions"

- The dataset is not 4-dimensional
- Check array shape with `get_array_info()`
- Adjust slicing or use a different dataset

### Slow Performance

- Check if HDF5 file was created with chunking
- Verify chunk sizes align with your access patterns
- Consider using compression to reduce transfer time
- Check network latency to S3 region

## Advanced Topics

### Custom Dataset Names

```python
with S3HDF5Reader(s3_path) as reader:
    # Read metadata from different location
    metadata = reader.read_metadata('custom_metadata')

    # Read different array
    array = reader.read_array_slice(array_key='alternative_data')
```

### Working with Multiple Arrays

```python
with S3HDF5Reader(s3_path) as reader:
    # List all available datasets
    datasets = reader.list_datasets()

    # Read different arrays
    raw_data = reader.read_array_slice(array_key='raw')
    processed_data = reader.read_array_slice(array_key='processed')
```

### Integration with Data Processing Pipelines

```python
import dask.array as da

def process_s3_hdf5_with_dask(s3_path):
    """Process large HDF5 array using Dask for parallel computation."""
    with S3HDF5Reader(s3_path) as reader:
        info = reader.get_array_info()

        # Create Dask array for lazy evaluation
        def load_chunk(block_id):
            start = block_id[0] * 100
            end = start + 100
            return reader.read_array_slice(np.s_[start:end, :, :, :])

        # Process in parallel
        result = # ... dask computation
```

## License

This package is part of the array_benchmark project.

## See Also

- [HDF5 Documentation](https://www.hdfgroup.org/solutions/hdf5/)
- [s3fs Documentation](https://s3fs.readthedocs.io/)
- [h5py Documentation](https://docs.h5py.org/)
- [Example Usage Script](./s3_hdf5_example.py)
