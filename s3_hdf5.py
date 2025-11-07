"""
S3 HDF5 Reader Package

This module provides functionality for reading HDF5 files stored on Amazon S3
with support for partial reads and slicing of large datasets.

Features:
- Read HDF5 files directly from S3 using HTTP range requests
- Access JSON metadata stored in HDF5 files
- Perform partial reads on large 4D arrays using slicing
- Efficient memory usage by only reading requested data portions
"""

import json
from typing import Optional, Tuple, Dict, Any, Union
import h5py
import s3fs
import numpy as np


class S3HDF5Reader:
    """
    Reader class for HDF5 files stored on Amazon S3.

    Supports reading metadata and partial slicing of large 4-dimensional arrays
    without downloading the entire file.

    Attributes:
        s3_path (str): The S3 URI of the HDF5 file (e.g., 's3://bucket/path/file.h5')
        fs (s3fs.S3FileSystem): The S3 filesystem instance
        file (h5py.File): The opened HDF5 file handle (when opened)

    Example:
        >>> reader = S3HDF5Reader('s3://my-bucket/data/experiment.h5')
        >>> with reader.open():
        ...     metadata = reader.read_metadata()
        ...     data_slice = reader.read_array_slice(np.s_[0:10, :, :, 0:5])
    """

    def __init__(self, s3_path: str, anon: bool = False,
                 s3_additional_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize the S3 HDF5 reader.

        Args:
            s3_path: S3 URI path to the HDF5 file (e.g., 's3://bucket/path/file.h5')
            anon: If True, use anonymous access (no credentials). Default is False.
            s3_additional_kwargs: Additional keyword arguments to pass to s3fs.S3FileSystem
                                (e.g., endpoint_url, region_name, etc.)

        Raises:
            ValueError: If the s3_path doesn't start with 's3://'
        """
        if not s3_path.startswith('s3://'):
            raise ValueError(f"S3 path must start with 's3://'. Got: {s3_path}")

        self.s3_path = s3_path
        self.anon = anon
        self.s3_kwargs = s3_additional_kwargs or {}

        # Initialize S3 filesystem
        self.fs = s3fs.S3FileSystem(anon=anon, **self.s3_kwargs)
        self.file: Optional[h5py.File] = None
        self._is_open = False

    def open(self):
        """
        Open the HDF5 file from S3.

        This method opens the file and prepares it for reading. It can be used
        as a context manager or called directly.

        Returns:
            self: Returns self to support context manager protocol

        Raises:
            FileNotFoundError: If the file doesn't exist on S3
            IOError: If there's an error opening the file
        """
        if self._is_open:
            return self

        try:
            # Open the S3 file with s3fs
            s3_file = self.fs.open(self.s3_path, 'rb')

            # Open as HDF5 file
            self.file = h5py.File(s3_file, 'r')
            self._is_open = True

        except FileNotFoundError:
            raise FileNotFoundError(f"HDF5 file not found at: {self.s3_path}")
        except Exception as e:
            raise IOError(f"Error opening HDF5 file from S3: {str(e)}")

        return self

    def close(self):
        """
        Close the HDF5 file and release resources.
        """
        if self._is_open and self.file is not None:
            self.file.close()
            self.file = None
            self._is_open = False

    def __enter__(self):
        """Context manager entry."""
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def _ensure_open(self):
        """Internal method to ensure the file is open before operations."""
        if not self._is_open or self.file is None:
            raise RuntimeError(
                "HDF5 file is not open. Call open() or use as context manager."
            )

    def read_metadata(self, metadata_key: str = 'metadata') -> Dict[str, Any]:
        """
        Read JSON metadata from the HDF5 file.

        Args:
            metadata_key: The key/path to the metadata dataset in the HDF5 file.
                         Default is 'metadata'.

        Returns:
            dict: The parsed JSON metadata as a Python dictionary

        Raises:
            RuntimeError: If the file is not open
            KeyError: If the metadata key doesn't exist in the file
            json.JSONDecodeError: If the metadata is not valid JSON
        """
        self._ensure_open()

        if metadata_key not in self.file:
            raise KeyError(f"Metadata key '{metadata_key}' not found in HDF5 file. "
                         f"Available keys: {list(self.file.keys())}")

        # Read the metadata dataset
        metadata_dataset = self.file[metadata_key]

        # Handle different storage formats
        if isinstance(metadata_dataset, h5py.Dataset):
            # If stored as string dataset
            if metadata_dataset.dtype.kind in ('S', 'U', 'O'):
                metadata_str = metadata_dataset[()].decode('utf-8') if isinstance(
                    metadata_dataset[()], bytes
                ) else str(metadata_dataset[()])
            else:
                raise ValueError(f"Unexpected metadata dtype: {metadata_dataset.dtype}")
        elif hasattr(metadata_dataset, 'attrs'):
            # If stored as attributes
            metadata_str = metadata_dataset.attrs.get('json', '')
        else:
            raise ValueError(f"Cannot read metadata from: {type(metadata_dataset)}")

        # Parse JSON
        try:
            return json.loads(metadata_str)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Failed to parse metadata JSON: {str(e)}",
                e.doc, e.pos
            )

    def get_array_info(self, array_key: str = 'data') -> Dict[str, Any]:
        """
        Get information about the array dataset without reading the data.

        Args:
            array_key: The key/path to the array dataset in the HDF5 file.
                      Default is 'data'.

        Returns:
            dict: Information about the array including shape, dtype, size, and chunks
        """
        self._ensure_open()

        if array_key not in self.file:
            raise KeyError(f"Array key '{array_key}' not found in HDF5 file. "
                         f"Available keys: {list(self.file.keys())}")

        dataset = self.file[array_key]

        return {
            'shape': dataset.shape,
            'dtype': str(dataset.dtype),
            'size': dataset.size,
            'nbytes': dataset.size * dataset.dtype.itemsize,
            'chunks': dataset.chunks,
            'compression': dataset.compression,
            'compression_opts': dataset.compression_opts,
        }

    def read_array_slice(
        self,
        array_slice: Union[Tuple, slice, None] = None,
        array_key: str = 'data'
    ) -> np.ndarray:
        """
        Read a slice of the 4D array from S3 using partial reads.

        This method only downloads the requested portion of the array from S3,
        making it efficient for large datasets.

        Args:
            array_slice: Numpy slice or tuple of slices for each dimension.
                        If None, reads the entire array.
                        Examples:
                        - np.s_[0:10, :, :, 0:5]  # First 10 on dim 0, last 5 on dim 3
                        - (slice(0, 10), slice(None), slice(None), slice(0, 5))
                        - np.s_[0, :, :, :]  # Single index on first dimension
            array_key: The key/path to the array dataset in the HDF5 file.
                      Default is 'data'.

        Returns:
            np.ndarray: The requested slice of the array

        Raises:
            RuntimeError: If the file is not open
            KeyError: If the array key doesn't exist
            IndexError: If the slice is invalid for the array dimensions
        """
        self._ensure_open()

        if array_key not in self.file:
            raise KeyError(f"Array key '{array_key}' not found in HDF5 file. "
                         f"Available keys: {list(self.file.keys())}")

        dataset = self.file[array_key]

        # Validate that it's a 4D array
        if len(dataset.shape) != 4:
            raise ValueError(
                f"Expected 4D array, but dataset has {len(dataset.shape)} dimensions. "
                f"Shape: {dataset.shape}"
            )

        # Read the slice (HDF5 handles partial reads automatically)
        if array_slice is None:
            # Read entire array
            return dataset[:]
        else:
            # Read specified slice
            try:
                return dataset[array_slice]
            except (IndexError, ValueError) as e:
                raise IndexError(
                    f"Invalid slice for array with shape {dataset.shape}: {str(e)}"
                )

    def read_full_array(self, array_key: str = 'data') -> np.ndarray:
        """
        Read the entire 4D array from S3.

        Warning: This will download the entire array (~2GB) into memory.
        Consider using read_array_slice() for large arrays.

        Args:
            array_key: The key/path to the array dataset in the HDF5 file.
                      Default is 'data'.

        Returns:
            np.ndarray: The complete 4D array
        """
        return self.read_array_slice(array_slice=None, array_key=array_key)

    def list_datasets(self) -> Dict[str, Any]:
        """
        List all datasets and groups in the HDF5 file with their properties.

        Returns:
            dict: Dictionary mapping dataset names to their properties
        """
        self._ensure_open()

        datasets = {}

        def visit_func(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = {
                    'type': 'dataset',
                    'shape': obj.shape,
                    'dtype': str(obj.dtype),
                    'size': obj.size,
                }
            elif isinstance(obj, h5py.Group):
                datasets[name] = {
                    'type': 'group',
                    'members': list(obj.keys()),
                }

        self.file.visititems(visit_func)

        return datasets

    def __repr__(self) -> str:
        """String representation of the reader."""
        status = "open" if self._is_open else "closed"
        return f"S3HDF5Reader('{self.s3_path}', status='{status}')"


# Convenience function for quick reads
def read_s3_hdf5_metadata(s3_path: str, metadata_key: str = 'metadata',
                          anon: bool = False) -> Dict[str, Any]:
    """
    Convenience function to quickly read metadata from an S3 HDF5 file.

    Args:
        s3_path: S3 URI path to the HDF5 file
        metadata_key: The key to the metadata dataset
        anon: Use anonymous S3 access

    Returns:
        dict: The parsed metadata
    """
    with S3HDF5Reader(s3_path, anon=anon) as reader:
        return reader.read_metadata(metadata_key)


def read_s3_hdf5_array(s3_path: str, array_slice: Union[Tuple, slice, None] = None,
                       array_key: str = 'data', anon: bool = False) -> np.ndarray:
    """
    Convenience function to quickly read array data from an S3 HDF5 file.

    Args:
        s3_path: S3 URI path to the HDF5 file
        array_slice: Slice specification for partial reads
        array_key: The key to the array dataset
        anon: Use anonymous S3 access

    Returns:
        np.ndarray: The requested array or slice
    """
    with S3HDF5Reader(s3_path, anon=anon) as reader:
        return reader.read_array_slice(array_slice, array_key)
