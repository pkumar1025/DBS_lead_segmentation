#!/bin/bash

# Check if both file path and output directory are provided
#if [ $# -ne 2 ]; then
#    echo "Usage: $0 <path_to_nifti_image> <output_directory>"
#    exit 1
#fi

# Get the absolute path of the input file and output directory
input_file=$(readlink -f "$1")
output_dir=$(readlink -f "$2")

# Check if the file exists
if [ ! -f "$input_file" ]; then
    echo "The file $input_file does not exist."
    exit 1
fi

# Check if the output directory exists, create it if it doesn't
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
    if [ $? -ne 0 ]; then
        echo "Failed to create output directory $output_dir"
        exit 1
    fi
fi

# Run the CT2MNI152Affine.py script on the input file
python3 CT2MNI152Affine.py "$input_file" "$output_dir"

# Check if the script executed successfully
if [ $? -ne 0 ]; then
    echo "An error occurred while processing the image with CT2MNI152Affine.py."
    exit 1
fi

input_dir=$(dirname "$input_file")

# Find the NIfTI file ending with MNI_152 in the current directory
MNI152_skull=$(find "$input_dir" -maxdepth 1 -name "*MNI15_skull.nii.gz" -print -quit)

if [ -z "$MNI152_skull" ]; then
    echo "Error: No NIfTI file ending with MNI_15_skull found in the current directory."
    exit 1
fi

echo "Found input file: $MNI152_skull"

# Ask user for threshold value
read -p "Enter threshold value (or press enter to skip thresholding): " THRESHOLD

# If threshold is provided, run the Python script again with the threshold
if [ ! -z "$THRESHOLD" ]; then
    python threshold.py "$MNI152_skull" "$THRESHOLD"
fi

echo "Threshold process completed."

MNI152_skull_thresholded=$(find "$input_dir" -maxdepth 1 -name "*MNI15_skull_thresholded.nii.gz" -print -quit)

if [ -z "$MNI152_skull_thresholded" ]; then
    echo "Error: No NIfTI file ending with MNI_15_skull_thresholded found in the current directory."
    exit 1
fi

# Run the atlas.py script on the input file
python3 atlas.py "$MNI152_skull_thresholded" "$output_dir"
