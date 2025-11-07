#!/usr/bin/env python3
"""
Example usage of the S3HDF5Reader package.

This script demonstrates how to read HDF5 files from S3 with partial reads
and slicing for large 4-dimensional arrays.
"""

import numpy as np
from s3_hdf5 import S3HDF5Reader, read_s3_hdf5_metadata, read_s3_hdf5_array


def example_basic_usage():
    """Basic usage example with context manager."""
    print("=" * 80)
    print("Example 1: Basic Usage with Context Manager")
    print("=" * 80)

    s3_path = "s3://my-bucket/data/experiment.h5"

    # Using context manager (recommended)
    with S3HDF5Reader(s3_path) as reader:
        # Read metadata
        metadata = reader.read_metadata()
        print(f"\nMetadata: {metadata}")

        # Get array information without reading data
        array_info = reader.get_array_info('data')
        print(f"\nArray Info:")
        print(f"  Shape: {array_info['shape']}")
        print(f"  Dtype: {array_info['dtype']}")
        print(f"  Size: {array_info['nbytes'] / 1e9:.2f} GB")

        # Read a slice of the 4D array
        data_slice = reader.read_array_slice(np.s_[0:10, :, :, 0:5])
        print(f"\nSlice shape: {data_slice.shape}")
        print(f"Slice size: {data_slice.nbytes / 1e6:.2f} MB")


def example_manual_open_close():
    """Example with manual open/close."""
    print("\n" + "=" * 80)
    print("Example 2: Manual Open/Close")
    print("=" * 80)

    s3_path = "s3://my-bucket/data/experiment.h5"

    reader = S3HDF5Reader(s3_path)
    try:
        reader.open()

        # List all datasets in the file
        datasets = reader.list_datasets()
        print(f"\nDatasets in file:")
        for name, info in datasets.items():
            print(f"  {name}: {info}")

        # Read specific slices
        # Example: Read first time slice, all spatial dimensions, first channel
        time_slice = reader.read_array_slice(np.s_[0, :, :, :])
        print(f"\nSingle time slice shape: {time_slice.shape}")

    finally:
        reader.close()


def example_convenience_functions():
    """Example using convenience functions."""
    print("\n" + "=" * 80)
    print("Example 3: Convenience Functions")
    print("=" * 80)

    s3_path = "s3://my-bucket/data/experiment.h5"

    # Quick metadata read
    metadata = read_s3_hdf5_metadata(s3_path)
    print(f"\nMetadata: {metadata}")

    # Quick array slice read
    data = read_s3_hdf5_array(
        s3_path,
        array_slice=np.s_[0:100, :, :, :],
        array_key='data'
    )
    print(f"\nData shape: {data.shape}")


def example_advanced_slicing():
    """Advanced slicing examples."""
    print("\n" + "=" * 80)
    print("Example 4: Advanced Slicing Patterns")
    print("=" * 80)

    s3_path = "s3://my-bucket/data/experiment.h5"

    with S3HDF5Reader(s3_path) as reader:
        # Get array shape first
        info = reader.get_array_info('data')
        shape = info['shape']
        print(f"\nFull array shape: {shape}")

        # Different slicing patterns
        slicing_examples = [
            ("First 10 timesteps, all other dims", np.s_[0:10, :, :, :]),
            ("Every 10th timestep", np.s_[::10, :, :, :]),
            ("Spatial subset", np.s_[:, 100:200, 100:200, :]),
            ("Single timestep and channel", np.s_[0, :, :, 0]),
            ("Non-contiguous indices", np.s_[0:50:5, :, :, 0:10:2]),
        ]

        for description, slice_spec in slicing_examples:
            data = reader.read_array_slice(slice_spec)
            print(f"\n{description}:")
            print(f"  Slice: {slice_spec}")
            print(f"  Result shape: {data.shape}")
            print(f"  Size: {data.nbytes / 1e6:.2f} MB")


def example_with_aws_credentials():
    """Example with AWS credentials configuration."""
    print("\n" + "=" * 80)
    print("Example 5: AWS Credentials Configuration")
    print("=" * 80)

    s3_path = "s3://my-bucket/data/experiment.h5"

    # Option 1: Use environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    # This is handled automatically by s3fs

    # Option 2: Use anonymous access for public buckets
    with S3HDF5Reader(s3_path, anon=True) as reader:
        print("Using anonymous access for public bucket")
        metadata = reader.read_metadata()

    # Option 3: Specify custom endpoint (e.g., for MinIO or other S3-compatible storage)
    with S3HDF5Reader(
        s3_path,
        s3_additional_kwargs={
            'endpoint_url': 'http://localhost:9000',
            'key': 'minioadmin',
            'secret': 'minioadmin'
        }
    ) as reader:
        print("Using custom S3 endpoint")
        metadata = reader.read_metadata()


def example_memory_efficient_processing():
    """Example of processing large arrays in chunks."""
    print("\n" + "=" * 80)
    print("Example 6: Memory-Efficient Chunk Processing")
    print("=" * 80)

    s3_path = "s3://my-bucket/data/experiment.h5"

    with S3HDF5Reader(s3_path) as reader:
        # Get total array dimensions
        info = reader.get_array_info('data')
        total_timesteps = info['shape'][0]
        chunk_size = 100  # Process 100 timesteps at a time

        print(f"\nProcessing {total_timesteps} timesteps in chunks of {chunk_size}")

        # Process in chunks
        for start_idx in range(0, total_timesteps, chunk_size):
            end_idx = min(start_idx + chunk_size, total_timesteps)

            # Read chunk
            chunk = reader.read_array_slice(np.s_[start_idx:end_idx, :, :, :])

            # Process chunk (example: compute statistics)
            chunk_mean = np.mean(chunk)
            chunk_std = np.std(chunk)

            print(f"  Timesteps {start_idx:4d}-{end_idx:4d}: "
                  f"mean={chunk_mean:.4f}, std={chunk_std:.4f}")

            # Chunk is freed when it goes out of scope or is reassigned


def main():
    """Run all examples."""
    print("\n")
    print("#" * 80)
    print("# S3 HDF5 Reader Examples")
    print("#" * 80)
    print("\nNote: These examples use placeholder S3 paths.")
    print("Replace with your actual S3 bucket and file paths to run.")
    print("\n")

    try:
        # Uncomment the examples you want to run
        # example_basic_usage()
        # example_manual_open_close()
        # example_convenience_functions()
        # example_advanced_slicing()
        # example_with_aws_credentials()
        # example_memory_efficient_processing()

        print("\nTo run these examples:")
        print("1. Set up your AWS credentials (environment variables or ~/.aws/credentials)")
        print("2. Replace the S3 paths with your actual bucket and file")
        print("3. Ensure your HDF5 file has 'metadata' and 'data' datasets")
        print("4. Uncomment the example functions you want to run")

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nThis is expected if using placeholder S3 paths.")


if __name__ == "__main__":
    main()
