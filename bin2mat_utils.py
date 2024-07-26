import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from typing import Dict, Generator
from loguru import logger

# Configuration constants
TOLERANCE_BYTES = 1000
MAX_CHANNELS_TO_PLOT = 5
SAMPLES_TO_PLOT = 1000

def detect_data_type(file_path: str, num_channels: int) -> Tuple[np.dtype, int]:
    """
    Attempt to detect the data type and number of samples in the binary file.

    Args:
        file_path (str): Path to the binary file.
        num_channels (int): Number of channels in the data.

    Returns:
        Tuple[np.dtype, int]: Detected data type and number of samples.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If no valid data type can be detected.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Binary file not found: {file_path}")

    file_size = os.path.getsize(file_path)
    possible_dtypes = [('int16', 2), ('int32', 4), ('float32', 4), ('float64', 8)]
    
    valid_types = []
    for dtype_name, itemsize in possible_dtypes:
        total_samples = file_size // (num_channels * itemsize)
        if abs(file_size - total_samples * num_channels * itemsize) <= TOLERANCE_BYTES:
            valid_types.append((dtype_name, total_samples))
    
    if not valid_types:
        raise ValueError(f"Could not detect a valid data type for the given number of channels. File size: {file_size} bytes, Channels: {num_channels}")
    
    logger.info("Possible data types:")
    for dtype_name, total_samples in valid_types:
        logger.info(f"  {dtype_name}: {total_samples} samples")
    
    default_type = next((dt for dt, _ in valid_types if dt == 'float32'), valid_types[0][0])
    default_samples = next(ns for dt, ns in valid_types if dt == default_type)
    
    logger.info(f"Defaulting to {default_type} with {default_samples} samples")
    return np.dtype(default_type), default_samples

def read_binary_file(file_path: str, num_channels: int, dtype: np.dtype, total_samples: int) -> np.ndarray:
    """
    Read the binary file into a numpy array.

    Args:
        file_path (str): Path to the binary file.
        num_channels (int): Number of channels in the data.
        dtype (np.dtype): Data type of the binary file.
        total_samples (int): Total number of samples to read.

    Returns:
        np.ndarray: Data read from the binary file.

    Raises:
        IOError: If there's an error reading the binary file.
        ValueError: If the file size doesn't match the expected size based on parameters.
    """
    try:
        with open(file_path, 'rb') as f:
            data = np.fromfile(f, dtype=dtype, count=num_channels * total_samples)
        
        expected_size = num_channels * total_samples * dtype.itemsize
        if data.size * dtype.itemsize != expected_size:
            raise ValueError(f"File size ({data.size * dtype.itemsize} bytes) doesn't match expected size ({expected_size} bytes)")
        
        return data.reshape((total_samples, num_channels)).T
    except Exception as e:
        raise IOError(f"Error reading binary file: {e}")

def extract_spikes(channel_data: np.ndarray, window_size: int, threshold: float) -> np.ndarray:
    """
    Extract spikes from a single channel's data using a simple threshold method.

    Args:
        channel_data (np.ndarray): Input data array for a single channel.
        window_size (int): Size of the window around each spike.
        threshold (float): Threshold for spike detection.

    Returns:
        np.ndarray: Array of extracted spikes for the channel.
    """
    peaks = np.where(channel_data > threshold)[0]
    valid_peaks = peaks[(peaks > window_size // 2) & (peaks < len(channel_data) - window_size // 2)]
    return np.array([channel_data[p - window_size // 2 : p + window_size // 2] for p in valid_peaks])

def process_data(data: np.ndarray, window_size: int, threshold: Optional[float]) -> dict:
    """
    Process the data, either extracting spikes or preparing continuous data.

    Args:
        data (np.ndarray): Input data array.
        window_size (int): Size of the window around each spike.
        threshold (Optional[float]): Threshold for spike detection. If None, continuous data is prepared.

    Returns:
        dict: Processed data, either 'spikes' or 'continuous_data'.
    """
    if threshold is not None:
        spikes = [extract_spikes(channel, window_size, threshold) for channel in data]
        logger.info(f"Extracted spikes shape: {np.array(spikes).shape}")
        return {'spikes': np.array(spikes)}
    else:
        return {'continuous_data': data}

def validate_data_chunk(original_chunk: np.ndarray, processed_chunk: np.ndarray) -> bool:
    """
    Validate a chunk of data.

    Args:
        original_chunk (np.ndarray): A chunk of the original data.
        processed_chunk (np.ndarray): The corresponding chunk of processed data.

    Returns:
        bool: True if the chunks are close enough, False otherwise.
    """
    return np.allclose(original_chunk, processed_chunk, rtol=1e-5, atol=1e-8)

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
            if not validate_data_chunk(original_chunk, processed_chunk):
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

def verify_conversion(bin_file: str, mat_file: str, num_channels: int, dtype: np.dtype):
    """
    Verify the conversion by comparing the binary and MAT files.

    Args:
        bin_file (str): Path to the binary file.
        mat_file (str): Path to the MAT file.
        num_channels (int): Number of channels in the data.
        dtype (np.dtype): Data type of the binary file.
    """
    logger.info("Verifying conversion:")
    
    mat_data = sio.loadmat(mat_file)
    
    if 'continuous_data' in mat_data:
        data = mat_data['continuous_data']
        data_type = "continuous"
    elif 'spikes' in mat_data:
        data = mat_data['spikes']
        data_type = "spikes"
    else:
        logger.error("Neither 'continuous_data' nor 'spikes' found in .mat file")
        return
    
    logger.info(f"Shape of {data_type} data in .mat file: {data.shape}")
    logger.info(f"Data type in .mat file: {data.dtype}")
    
    logger.info(f"Min value: {np.min(data)}")
    logger.info(f"Max value: {np.max(data)}")
    logger.info(f"Mean value: {np.mean(data)}")
    logger.info(f"Standard deviation: {np.std(data)}")
    
    bin_size = os.path.getsize(bin_file)
    mat_size = os.path.getsize(mat_file)
    logger.info(f"Binary file size: {bin_size} bytes")
    logger.info(f"MAT file size: {mat_size} bytes")
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    
    if data_type == "continuous":
        for i in range(min(MAX_CHANNELS_TO_PLOT, num_channels)):
            plot_channel_data(axs[0], data[i], i, 'Continuous data from MAT file')
    else:  # spikes
        for i in range(min(MAX_CHANNELS_TO_PLOT, num_channels)):
            if data[i].size > 0:
                plot_channel_data(axs[0], data[i, 0], i, 'First spike from each channel in MAT file')
    
    with open(bin_file, 'rb') as f:
        raw_data = np.fromfile(f, dtype=dtype, count=num_channels*SAMPLES_TO_PLOT)
    raw_data = raw_data.reshape((SAMPLES_TO_PLOT, num_channels)).T
    
    for i in range(min(MAX_CHANNELS_TO_PLOT, num_channels)):
        plot_channel_data(axs[1], raw_data[i], i, 'Raw binary data')
    
    plt.tight_layout()
    plt.show()

def view_mat_file(mat_file: str):
    """
    Load and visualize data from a MAT file.

    Args:
        mat_file (str): Path to the MAT file.
    """
    mat_data = sio.loadmat(mat_file)
    
    if 'continuous_data' in mat_data:
        data = mat_data['continuous_data']
        data_type = "continuous"
    elif 'spikes' in mat_data:
        data = mat_data['spikes']
        data_type = "spikes"
    else:
        logger.error("Neither 'continuous_data' nor 'spikes' found in .mat file")
        return

    num_channels = data.shape[0]
    num_subplots = min(num_channels, MAX_CHANNELS_TO_PLOT)
    
    fig, axs = plt.subplots(num_subplots, 1, figsize=(12, 4 * num_subplots))
    if num_subplots == 1:
        axs = [axs]
    
    if data_type == "continuous":
        for i in range(num_subplots):
            axs[i].plot(data[i, :SAMPLES_TO_PLOT])
            axs[i].set_title(f'Channel {i+1}')
            axs[i].set_xlabel('Sample')
            axs[i].set_ylabel('Amplitude')
        plt.suptitle('Continuous Data')
    else:  # spikes
        for i in range(num_subplots):
            if data[i].size > 0:
                axs[i].plot(data[i].T)
                axs[i].set_title(f'Channel {i+1}')
                axs[i].set_xlabel('Sample')
                axs[i].set_ylabel('Amplitude')
        plt.suptitle('Spike Data')
    
    plt.tight_layout()
    plt.show()