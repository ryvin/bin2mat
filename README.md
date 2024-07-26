# BIN2MAT
![Neural BIN2MAT project logo](/brain-visualization.jpg)

Bin2mat is a Python program that converts binary neural scan RAW data to MAT files compatible with MATLAB. It provides functionality for both continuous data storage and spike extraction.

## Author

Raul Pineda

## Features

- Converts binary (.bin/.dat) neural scanning files to MATLAB (.mat) files
- Supports continuous data storage and spike extraction
- Automatic data type detection
- Optional chunked processing for large files
- Progress bar for long-running operations
- Verification of conversion

## Requirements

The following Python packages are required to run bin2mat:

- numpy
- scipy
- matplotlib
- loguru
- tqdm

You can install these packages using the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

## How It Works

The program reads binary files containing neural scanning data and converts them to MATLAB-compatible .mat files. It can handle both continuous data and extract spikes based on a threshold. The process involves:

1. Detecting the data type and number of samples in the binary file
2. Reading the binary data
3. Processing the data (either storing as continuous or extracting spikes)
4. Saving the processed data to a .mat file
5. Optional verification of the conversion

### Spike Extraction

The spike extraction feature allows you to identify and extract spike waveforms from the continuous neural data. This is useful for analyzing individual neuron firing patterns. The spike extraction process works as follows:

1. A threshold value is set to determine what constitutes a spike.
2. The program scans through the continuous data for each channel.
3. When the signal exceeds the threshold, a spike is detected.
4. A window of data around the spike (centered on the peak) is extracted.
5. These extracted spike waveforms are stored in the output .mat file.

You can control the spike extraction process with two parameters:

- `--threshold`: This sets the voltage threshold for spike detection. Signals exceeding this value are considered spikes.
- `--window_size`: This determines the number of data points to extract around each spike. The spike peak will be at the center of this window.

If no threshold is provided, the program will save the full continuous data instead of extracting spikes.

## Usage

To use bin2mat, run the following command:

```
python bin2mat_main.py <input_file> <output_file> <num_channels> [options]
```

### Arguments:

- `<input_file>`: Path to the input binary file
- `<output_file>`: Path to the output MAT file
- `<num_channels>`: Number of channels in the data

### Options:

- `--window_size`: Window size for spike extraction (default: 64)
- `--threshold`: Threshold for spike extraction (if not provided, full data will be saved)
- `--dtype`: Specify the data type (e.g., 'int16', 'float32')
- `--verify`: Verify the conversion after completion
- `--chunked`: Use chunked processing for large files
- `--chunk_size`: Chunk size for processing (default: 1,000,000 samples)
- `--progress`: Show progress bar during processing
- `--view: View the output data after conversion

### Examples:

1. Convert a binary file to MAT format with default settings (saves continuous data):
   ```
   python bin2mat_main.py input.bin output.mat 8 --dtype int16
   ```

2. Convert a binary file with spike extraction:
   ```
   python bin2mat_main.py input.bin output.mat 64 --threshold 100 --window_size 32
   ```
   This will extract spikes that exceed a threshold of 100, saving 32 data points around each spike.

3. Convert a large file using chunked processing with a progress bar:
   ```
   python bin2mat_main.py input.bin output.mat 64 --chunked --progress
   ```

4. Convert a file, specifying the data type and verifying the conversion:
   ```
   python bin2mat_main.py input.bin output.mat 64 --dtype float32 --verify
   ```
5. Convert a file and immediately view the output:
   ```
   python bin2mat_main.py input.bin output.mat 64 --view
   ```
   This will convert the file and then display a plot of the data (either continuous data or extracted spikes).

## Output

The program generates a .mat file containing either:

1. 'continuous_data': A 2D array of shape (num_channels, num_samples) if no spike extraction is performed.
2. 'spikes': A 3D array of shape (num_channels, num_spikes, window_size) if spike extraction is performed.

You can load and analyze this data in MATLAB or using scipy.io in Python. Additionally, you can use the --view option to immediately visualize the data after conversion.

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.