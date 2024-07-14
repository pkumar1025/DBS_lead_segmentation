import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def display_nifti_histogram_and_threshold(input_path, threshold=None):
    # Define output path based on input path
    output_path = input_path.replace('.nii.gz', '_thresholded.nii.gz')

    # Load the NIfTI file
    img = nib.load(input_path)
    data = img.get_fdata()
    
    # Display histogram
    flat_data = data.flatten()
    flat_data = flat_data[np.isfinite(flat_data) & (flat_data != 0)]
    
    plt.figure(figsize=(10, 6))
    plt.hist(flat_data, bins=100, edgecolor='black')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Pixel Intensities for {os.path.basename(input_path)}')
    plt.show()

    # If threshold is provided, apply it and save the new image
    if threshold is not None:
        threshold = float(threshold)
        thresholded_data = np.where(data >= threshold, data, 0)
        new_img = nib.Nifti1Image(thresholded_data, img.affine, img.header)
        nib.save(new_img, output_path)
        print(f"Thresholded image saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_nifti_path> [threshold]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    threshold = sys.argv[2] if len(sys.argv) > 2 else None
    display_nifti_histogram_and_threshold(input_path, threshold)