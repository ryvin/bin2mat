import numpy as np
import scipy
import scipy.io as sio
import os
from typing import Optional, Generator, Dict
from loguru import logger
from tqdm import tqdm
from packaging import version
import h5py

from bin2mat_utils import detect_data_type, read_binary_file, process_data, verify_conversion, validate_data

def validate_data(original_data_generator: Generator[np.ndarray, None, None], 
                  processed_data: Dict[str, np.ndarray], 
                  chunk_size: int) -> bool:
    """
    Validate the processed data against the original data in chunks.

    Args:
        original_data_generator (Generator[np.ndarray, None, None]): Generator yielding chunks of original data.
        processed_data (Dict[str, np.ndarray]): The processed data (either continuous or spikes).
        chunk_size (int): The size of each chunk to validate.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    if 'continuous_data' in processed_data:
        for i, original_chunk in enumerate(original_data_generator):
            processed_chunk = processed_data['continuous_data'][:, i*chunk_size:(i+1)*chunk_size]
            if not np.allclose(original_chunk, processed_chunk, rtol=1e-5, atol=1e-8):
                logger.error(f"Validation failed at chunk {i}")
                return False
        return True
    elif 'spikes' in processed_data:
        # For spike data, we need a different validation approach
        # This is a placeholder and should be implemented based on your specific requirements
        logger.warning("Spike data validation not implemented yet")
        return True
    else:
        raise ValueError("Processed data contains neither 'continuous_data' nor 'spikes'")

def bin_to_mat(bin_file: str, mat_file: str, num_channels: int, window_size: int = 64, 
               threshold: Optional[float] = None, dtype: Optional[str] = None, verify: bool = False):
    """
    Convert binary file to MAT file with optional spike extraction and verification.

    Args:
        bin_file (str): Path to the input binary file.
        mat_file (str): Path to the output MAT file.
        num_channels (int): Number of channels in the data.
        window_size (int, optional): Window size for spike extraction. Defaults to 64.
        threshold (float, optional): Threshold for spike extraction. If None, full data will be saved.
        dtype (str, optional): Specify the data type. If None, it will be auto-detected.
        verify (bool, optional): Whether to verify the conversion after completion. Defaults to False.

    Raises:
        FileNotFoundError: If the binary file is not found.
        ValueError: If the data type cannot be detected or if the file size is unexpected.
        IOError: If there's an error reading the binary file or saving the MAT file.
    """
    try:
        # Check if input file exists
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"Binary file not found: {bin_file}")

        # Detect or set data type
        if dtype:
            detected_dtype = np.dtype(dtype)
            total_samples = os.path.getsize(bin_file) // (num_channels * detected_dtype.itemsize)
            logger.info(f"Using user-specified data type: {detected_dtype}, Total samples: {total_samples}")
        else:
            detected_dtype, total_samples = detect_data_type(bin_file, num_channels)
        
        # Read binary file
        data = read_binary_file(bin_file, num_channels, detected_dtype, total_samples)
        logger.info(f"Data shape: {data.shape}")

        # Process data (continuous or spike extraction)
        processed_data = process_data(data, window_size, threshold)

        # Validate processed data
        if not validate_data(data, processed_data):
            raise ValueError("Data validation failed. Processed data does not match original data.")

        # Save processed data to MAT file
        sio.savemat(mat_file, processed_data)
        logger.info(f"Data saved to {mat_file}")

        # Verify conversion if requested
        if verify:
            verify_conversion(bin_file, mat_file, num_channels, detected_dtype)

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except IOError as e:
        logger.error(f"I/O error: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

def chunked_read_binary_file(file_path: str, num_channels: int, dtype: np.dtype, total_samples: int, chunk_size: int = 1000000) -> Generator[np.ndarray, None, None]:
    """
    Read the binary file in chunks to reduce memory usage.

    Args:
        file_path (str): Path to the binary file.
        num_channels (int): Number of channels in the data.
        dtype (np.dtype): Data type of the binary file.
        total_samples (int): Total number of samples to read.
        chunk_size (int, optional): Number of samples to read in each chunk. Defaults to 1000000.

    Yields:
        np.ndarray: Chunks of data read from the binary file.

    Raises:
        IOError: If there's an error reading the binary file.
    """
    try:
        with open(file_path, 'rb') as f:
            for i in range(0, total_samples, chunk_size):
                # Read a chunk of data
                chunk = np.fromfile(f, dtype=dtype, count=num_channels * min(chunk_size, total_samples - i))
                # Reshape the chunk to (channels, samples) and yield
                yield chunk.reshape((-1, num_channels)).T
    except IOError as e:
        logger.error(f"Error reading binary file: {e}")
        raise

def bin_to_mat_chunked(bin_file: str, mat_file: str, num_channels: int, window_size: int = 64, 
                       threshold: Optional[float] = None, dtype: Optional[str] = None, verify: bool = False,
                       chunk_size: int = 1000000):
    """
    Convert binary file to MAT file using chunked processing for large files.

    Args:
        bin_file (str): Path to the input binary file.
        mat_file (str): Path to the output MAT file.
        num_channels (int): Number of channels in the data.
        window_size (int, optional): Window size for spike extraction. Defaults to 64.
        threshold (float, optional): Threshold for spike extraction. If None, full data will be saved.
        dtype (str, optional): Specify the data type. If None, it will be auto-detected.
        verify (bool, optional): Whether to verify the conversion after completion. Defaults to False.
        chunk_size (int, optional): Number of samples to process in each chunk. Defaults to 1000000.

    Raises:
        FileNotFoundError: If the binary file is not found.
        ValueError: If the data type cannot be detected or if the file size is unexpected.
        IOError: If there's an error reading the binary file or saving the MAT file.
    """
    try:
        # Check if input file exists
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"Binary file not found: {bin_file}")

        # Detect or set data type
        if dtype:
            detected_dtype = np.dtype(dtype)
            total_samples = os.path.getsize(bin_file) // (num_channels * detected_dtype.itemsize)
            logger.info(f"Using user-specified data type: {detected_dtype}, Total samples: {total_samples}")
        else:
            detected_dtype, total_samples = detect_data_type(bin_file, num_channels)

        # Initialize processed data structure
        processed_data = {'continuous_data': []} if threshold is None else {'spikes': [[] for _ in range(num_channels)]}

        # Process data in chunks
        for chunk in chunked_read_binary_file(bin_file, num_channels, detected_dtype, total_samples, chunk_size):
            chunk_processed = process_data(chunk, window_size, threshold)
            
            if threshold is None:
                processed_data['continuous_data'].append(chunk_processed['continuous_data'])
            else:
                for i, channel_spikes in enumerate(chunk_processed['spikes']):
                    processed_data['spikes'][i].extend(channel_spikes)

        # Concatenate processed data
        if threshold is None:
            processed_data['continuous_data'] = np.concatenate(processed_data['continuous_data'], axis=1)
        else:
            processed_data['spikes'] = np.array([np.array(channel_spikes) for channel_spikes in processed_data['spikes']])

        # Validate processed data
        original_data = read_binary_file(bin_file, num_channels, detected_dtype, total_samples)
        if not validate_data(original_data, processed_data):
            raise ValueError("Data validation failed. Processed data does not match original data.")

        # Save processed data to MAT file
        sio.savemat(mat_file, processed_data)
        logger.info(f"Data saved to {mat_file}")

        # Verify conversion if requested
        if verify:
            verify_conversion(bin_file, mat_file, num_channels, detected_dtype)

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except IOError as e:
        logger.error(f"I/O error: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

def bin_to_mat_with_progress(bin_file: str, mat_file: str, num_channels: int, window_size: int = 64, 
                             threshold: Optional[float] = None, dtype: Optional[str] = None, verify: bool = False,
                             chunk_size: int = 1000000):
    try:
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"Binary file not found: {bin_file}")

        if dtype:
            detected_dtype = np.dtype(dtype)
            total_samples = os.path.getsize(bin_file) // (num_channels * detected_dtype.itemsize)
            logger.info(f"Using user-specified data type: {detected_dtype}, Total samples: {total_samples}")
        else:
            detected_dtype, total_samples = detect_data_type(bin_file, num_channels)

        processed_data = {'continuous_data': []} if threshold is None else {'spikes': [[] for _ in range(num_channels)]}

        total_chunks = total_samples // chunk_size + (1 if total_samples % chunk_size else 0)
        
        with tqdm(total=total_chunks, desc="Processing") as pbar:
            for chunk in chunked_read_binary_file(bin_file, num_channels, detected_dtype, total_samples, chunk_size):
                chunk_processed = process_data(chunk, window_size, threshold)
                
                if threshold is None:
                    processed_data['continuous_data'].append(chunk_processed['continuous_data'])
                else:
                    for i, channel_spikes in enumerate(chunk_processed['spikes']):
                        processed_data['spikes'][i].extend(channel_spikes)
                
                pbar.update(1)

        if threshold is None:
            processed_data['continuous_data'] = np.concatenate(processed_data['continuous_data'], axis=1)
        else:
            processed_data['spikes'] = np.array([np.array(channel_spikes) for channel_spikes in processed_data['spikes']])

        # Validate processed data
        logger.info("Validating processed data...")
        original_data_generator = chunked_read_binary_file(bin_file, num_channels, detected_dtype, total_samples, chunk_size)
        if not validate_data(original_data_generator, processed_data, chunk_size):
            raise ValueError("Data validation failed. Processed data does not match original data.")
        logger.info("Data validation successful.")

        # Save processed data to MAT file
        try:
            logger.info(f"Saving data to {mat_file}")
            if version.parse(scipy.__version__) >= version.parse('1.8.0'):
                # Use v7.3 format for newer SciPy versions
                sio.savemat(mat_file, processed_data, do_compression=True, format='7.3')
            else:
                # Use default format for older SciPy versions
                sio.savemat(mat_file, processed_data, do_compression=True)
            logger.info(f"Data saved successfully to {mat_file}")
        except (MemoryError, ValueError):
            logger.warning("Unable to save using scipy.io.savemat. Attempting to save using h5py...")
            try:
                with h5py.File(mat_file, 'w') as f:
                    if 'continuous_data' in processed_data:
                        f.create_dataset('continuous_data', data=processed_data['continuous_data'], 
                                         compression='gzip', compression_opts=9)
                    elif 'spikes' in processed_data:
                        f.create_dataset('spikes', data=processed_data['spikes'], 
                                         compression='gzip', compression_opts=9)
                logger.info(f"Data saved successfully to {mat_file} using HDF5 format")
            except Exception as e:
                logger.error(f"Failed to save data using h5py: {e}")
                logger.info("Attempting to save data in chunks...")
                
                # Save data in chunks
                if 'continuous_data' in processed_data:
                    total_samples = processed_data['continuous_data'].shape[1]
                    for i in range(0, total_samples, chunk_size):
                        chunk_data = {'chunk_' + str(i): processed_data['continuous_data'][:, i:i+chunk_size]}
                        chunk_file = f"{mat_file[:-4]}_chunk_{i}.mat"
                        sio.savemat(chunk_file, chunk_data, do_compression=True)
                    logger.info(f"Continuous data saved in chunks to {mat_file[:-4]}_chunk_*.mat files")
                elif 'spikes' in processed_data:
                    for i, channel_spikes in enumerate(processed_data['spikes']):
                        chunk_data = {'channel_' + str(i): channel_spikes}
                        chunk_file = f"{mat_file[:-4]}_channel_{i}.mat"
                        sio.savemat(chunk_file, chunk_data, do_compression=True)
                    logger.info(f"Spike data saved in separate files for each channel to {mat_file[:-4]}_channel_*.mat files")

        if verify:
            verify_conversion(bin_file, mat_file, num_channels, detected_dtype)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise