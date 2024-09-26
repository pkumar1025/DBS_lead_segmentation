import nibabel as nib
import numpy as np
from nilearn import datasets, image
import pandas as pd

def create_mask_and_compare_atlas(input_nifti_path):
    # Load the input NIfTI file
    img = nib.load(input_nifti_path)
    data = img.get_fdata()

    # Create binary mask of non-zero intensities
    mask = (data != 0).astype(int)

    # Example - Load Harvard-Oxford atlas (You might need to modify this based on which atlas space you want to use)
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = nib.load(atlas['filename'])
    atlas_data = atlas_img.get_fdata()

    # Ensure the input image and atlas are in the same space
    if img.shape != atlas_img.shape or not np.allclose(img.affine, atlas_img.affine):
        print("Warning: Input image and atlas are not in the same space. Resampling...")
        atlas_img = image.resample_to_img(atlas_img, img, interpolation='nearest')
        atlas_data = atlas_img.get_fdata()

    # Find overlapping regions
    overlapping_regions = np.unique(atlas_data[mask == 1]).astype(int)

    # Create a dataframe of overlapping regions
    overlap_df = pd.DataFrame({
        'Region ID': overlapping_regions,
        'Region Name': [atlas['labels'][i] for i in overlapping_regions if i < len(atlas['labels'])],
        'Overlap Voxel Count': [np.sum((atlas_data == i) & (mask == 1)) for i in overlapping_regions]
    })

    # Sort by overlap voxel count
    overlap_df = overlap_df.sort_values('Overlap Voxel Count', ascending=False).reset_index(drop=True)

    return overlap_df

# Example usage
input_nifti_path = '/Users/Prane/Documents/GitHub/DBS_lead_segmentation/code/leads/postop_ct.nii'
result = create_mask_and_compare_atlas(input_nifti_path)
print(result)

# Optionally, save to CSV
result.to_csv('atlas_overlap_results.csv', index=False)