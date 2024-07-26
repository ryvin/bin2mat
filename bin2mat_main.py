#!/usr/bin/env python3
import argparse
from bin2mat_core import bin_to_mat, bin_to_mat_chunked, bin_to_mat_with_progress
from bin2mat_utils import view_mat_file

def main():
    parser = argparse.ArgumentParser(description="Convert binary file to MAT file with optional spike extraction and verification.")
    parser.add_argument("bin_file", type=str, help="Path to the input binary file")
    parser.add_argument("mat_file", type=str, help="Path to the output MAT file")
    parser.add_argument("num_channels", type=int, help="Number of channels in the data")
    parser.add_argument("--window_size", type=int, default=64, help="Window size for spike extraction (default: 64)")
    parser.add_argument("--threshold", type=float, help="Threshold for spike extraction (if not provided, full data will be saved)")
    parser.add_argument("--dtype", type=str, help="Specify the data type (e.g., 'int16', 'float32')")
    parser.add_argument("--verify", action="store_true", help="Verify the conversion after completion")
    parser.add_argument("--chunked", action="store_true", help="Use chunked processing for large files")
    parser.add_argument("--chunk_size", type=int, default=1000000, help="Chunk size for processing (default: 1,000,000 samples)")
    parser.add_argument("--progress", action="store_true", help="Show progress bar during processing")
    parser.add_argument("--view", action="store_true", help="View the output data after conversion")

    args = parser.parse_args()

    if args.chunked:
        if args.progress:
            bin_to_mat_with_progress(args.bin_file, args.mat_file, args.num_channels, args.window_size, 
                                     args.threshold, args.dtype, args.verify, args.chunk_size)
        else:
            bin_to_mat_chunked(args.bin_file, args.mat_file, args.num_channels, args.window_size, 
                               args.threshold, args.dtype, args.verify, args.chunk_size)
    else:
        bin_to_mat(args.bin_file, args.mat_file, args.num_channels, args.window_size, 
                   args.threshold, args.dtype, args.verify)

    if args.view:
        view_mat_file(args.mat_file)

if __name__ == "__main__":
    main()