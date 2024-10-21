#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:49:08 2017

@author: pkao
"""

import SimpleITK as sitk
import numpy as np
from skimage.filters import threshold_otsu
from skimage import measure
from scipy import ndimage
from skimage import exposure
import os

def bone_extracted(ct_img_path):
    """Extract the bone of the CT scan based on the hard thresholding on pixel value"""
    
    print('The CT scan you want to implement bone extraction: ', ct_img_path)


    ct_img = sitk.ReadImage(ct_img_path)

    bone_mask_img = sitk.Image(ct_img.GetWidth(), ct_img.GetHeight(), ct_img.GetDepth(), sitk.sitkFloat32)

    output_ct_img = sitk.Image(ct_img.GetWidth(), ct_img.GetHeight(), ct_img.GetDepth(), sitk.sitkFloat32)

    print('The size of CT scan:', ct_img.GetSize())

    ct_nda = sitk.GetArrayFromImage(ct_img)

    bone_mask_nda = sitk.GetArrayFromImage(bone_mask_img)

    output_ct_nda = sitk.GetArrayFromImage(output_ct_img)

    #print 'The minimum value of CT scan: ', np.amin(ct_nda)

    #print 'The maximum value of CT scan: ', np.amax(ct_nda)

    #print 'The pixel ID type of CT scan: ', ct_img.GetPixelIDTypeAsString()

    #m = 1.0

    #b = -1024.0

    #HU stands for Hounsfield Unit
    #bone_HU = 500.0

    #bone_pixel = (bone_HU-b)/m
    
    bone_pixel = 500
    bone_mask = ct_nda >= bone_pixel

    output_ct_nda[bone_mask] = ct_nda[bone_mask]
    bone_mask_nda[bone_mask] = 1.0

    #Output ct_image
    output_ct_image = sitk.GetImageFromArray(output_ct_nda)
    output_ct_image_name = ct_img_path[:ct_img_path.find('.nii.gz')]+'_skull.nii.gz'
    output_ct_image.CopyInformation(ct_img)
    
    print('The name of the output skull image: ', output_ct_image_name)

    sitk.WriteImage(output_ct_image, output_ct_image_name)

    #Bone_mask
    bone_mask_image = sitk.GetImageFromArray(bone_mask_nda)
    bone_mask_image_name = ct_img_path[:ct_img_path.find('.nii.gz')]+'_skullMask.nii.gz'
    bone_mask_image.CopyInformation(ct_img)

    print('The name of the output skull mask image: ', bone_mask_image_name)

    sitk.WriteImage(bone_mask_image, bone_mask_image_name)
    
    return output_ct_image_name, bone_mask_image_name  



def getMaximum3DRegion(binary):
    """ Get the Maximum 3D region from 3D multiple bindary Regions"""
    
    all_labels = measure.label(binary, background = 0)
    
    props = measure.regionprops(all_labels)
    
    areas = [prop.area for prop in props]
    
    maxArea_label = 1+np.argmax(areas)
    
    max_binary = np.float32(all_labels == maxArea_label)
    
    return max_binary


   

def normalizeCTscan(ct_nda):
    """Normalize the CT scan to range 0 to 1"""
    if np.amin(ct_nda) < 0:
        ct_normalized_nda = ct_nda - np.amin(ct_nda)
        
    ct_normalized_nda = ct_normalized_nda/np.amax(ct_normalized_nda)
    
    return ct_normalized_nda


def otsuThreshoulding(ct_normalized_nda):
    """Apply Otsu thresholding on the normalized ranging from 0 to 1 scan"""
    
    thresh = threshold_otsu(ct_normalized_nda)
    
    binary = (ct_normalized_nda > thresh)*1
    
    return binary.astype(np.float32)
    
def get2Maximum2DRegions(max_binary):
    """Get two largestest 2D region from multiple 2D regions"""
    
    xy_two_largest_binary = np.zeros(max_binary.shape, dtype = np.float32 )
    
    largest_area = np.zeros(max_binary.shape[0])
    
    second_largest_area = np.zeros(max_binary.shape[0])
    
    for i in range(max_binary.shape[0]):
        xy_binary = max_binary[i, :, :]
        xy_labels = measure.label(xy_binary, background = 0)
        xy_props = measure.regionprops(xy_labels)
        xy_areas = [prop.area for prop in xy_props]
        #print xy_areas
        
        if xy_areas == []:
            continue
        
        elif len(xy_areas) == 1:
            largest_area[i] = xy_areas[0]
            second_largest_area[i] = 0.0
            largest_label = xy_areas.index(largest_area[i]) + 1 
            xy_two_largest_binary[i, :, :] = xy_labels == largest_label
            
        else:
            xy_areas_sorted = sorted(xy_areas)
            largest_area[i] = xy_areas_sorted[-1]
            second_largest_area[i] = xy_areas_sorted[-2]
            largest_label = xy_areas.index(largest_area[i]) + 1 
            second_largest_label = xy_areas.index(second_largest_area[i])+1
            xy_largest_binary = xy_labels == largest_label
            xy_second_largest_binary = xy_labels == second_largest_label
            xy_two_largest_binary[i, :, :] = np.float32(np.logical_or(xy_largest_binary, xy_second_largest_binary))
            
    return xy_two_largest_binary

def get1Maximum2DRegion(max_second_binary):
    """Get the largest 2D region from multiple 2D regions"""
    
    new_binary = np.zeros(max_second_binary.shape, dtype = np.float32)
    for i in range(max_second_binary.shape[0]):
        xy_binary = max_second_binary[i,:,:]
        xy_labels = measure.label(xy_binary)
        xy_props = measure.regionprops(xy_labels)
        xy_areas = [prop.area for prop in xy_props]
        #print i, xy_areas_1
        if xy_areas == []:
            continue
        else:
            max_area_label = 1 + np.argmax(xy_areas)
            new_binary[i,:,:] = np.float32(xy_labels == max_area_label)
            
    return new_binary


def imageOpening2D(max_second_binary, structure=np.ones((15, 15))):
    """Applying the image opening operation on the binary mask"""
    new_max_second_binary = np.zeros(max_second_binary.shape, dtype = np.float32)
    
    for i in range(max_second_binary.shape[0]):
        
        new_max_second_binary[i,:,:] = ndimage.binary_opening(max_second_binary[i,:,:].astype(int), structure=structure).astype(np.float32)
        
    return new_max_second_binary

def removeCTscandevice(ct_img_path):
    """remove the ct scan device"""

    ct_img = sitk.ReadImage(ct_img_path)
    
    ct_nda = sitk.GetArrayFromImage(ct_img)

    print('The CT scan you want to implement CT scan device removal:', ct_img_path)

    #print 'The minimum value of CT scan: ', np.amin(ct_nda)

    #print 'The maximum value of CT scan: ', np.amax(ct_nda)

    #print 'The pixel ID type of CT scan: ', ct_img.GetPixelIDTypeAsString()
     
    ct_normalized_nda = normalizeCTscan(ct_nda)
    
    binary = otsuThreshoulding(ct_normalized_nda)
    
    max_binary = getMaximum3DRegion(binary)
    
    xy_two_largest_binary = get2Maximum2DRegions(max_binary)
    
    max_second_binary = getMaximum3DRegion(xy_two_largest_binary)
    
    new_binary = get1Maximum2DRegion(max_second_binary)
    
    new_max_second_bindary = imageOpening2D(new_binary)
    
    new_max_binary = getMaximum3DRegion(new_max_second_bindary)
    
    output_ct_image = sitk.GetImageFromArray(ct_nda * new_max_binary)
    
    output_ct_image.CopyInformation(ct_img)
    
    output_ct_image_name = ct_img_path[:ct_img_path.find('.nii.gz')]+'_woCTdevice.nii.gz'
    
    sitk.WriteImage(output_ct_image, output_ct_image_name)        

    # The mask for CT device
    
    woCTdevice_mask_image = sitk.GetImageFromArray(new_max_binary)
    
    woCTdevice_mask_image.CopyInformation(ct_img)
    
    woCTdevice_mask_image_name = ct_img_path[:ct_img_path.find('.nii.gz')]+'_woCTdeviceMask.nii.gz'
     
    sitk.WriteImage(woCTdevice_mask_image, woCTdevice_mask_image_name)
       
    return output_ct_image_name, woCTdevice_mask_image_name


def contrastStretch(ct_img_path, percent = (10,90)):
    """Apply the contrast stretching on 2D or 3D image"""
    ct_img = sitk.ReadImage(ct_img_path)
    ct_nda = sitk.GetArrayFromImage(ct_img)
    p1, p2 = np.percentile(ct_nda, percent, interpolation='nearest')
    nda_rescale = exposure.rescale_intensity(ct_nda, in_range = (p1, p2))
    ct_img_cs = sitk.GetImageFromArray(nda_rescale)
    ct_img_cs.CopyInformation(ct_img)
    output_ct_name = ct_img_path[:ct_img_path.find('.nii.gz')]+'_contrastStretching.nii.gz'
    sitk.WriteImage(ct_img_cs, output_ct_name)
    return output_ct_name



import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def display_nifti_images(nifti_path):
    original_image_data = nib.load(nifti_path).get_fdata()
    bone_extracted_data = nib.load(bone_extracted(nifti_path)[0]).get_fdata()
    bone_mask_data = nib.load(bone_extracted(nifti_path)[1]).get_fdata()

    # Choose a slice to visualize (e.g., the middle slice along the z-axis)
    slice_index = original_image_data.shape[2] // 2

    # Display the original image slice
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image_data[:, :, slice_index], cmap='gray')
    plt.title('Original CT Scan')
    plt.axis('off')

    # Display the bone-extracted image slice
    plt.subplot(1, 3, 2)
    plt.imshow(bone_extracted_data[:, :, slice_index], cmap='gray')
    plt.title('Bone Extracted CT Scan')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(bone_mask_data[:, :, slice_index], cmap='gray')
    plt.title('Bone Mask')
    plt.axis('off')

    # Show both images
    plt.show()

def display_nifti_histogram(nifti_path):
    # Load the NIfTI file
    img = nib.load(nifti_path)
    
    # Get the image data as a numpy array
    data = img.get_fdata()
    
    # Flatten the 3D/4D array to 1D for histogram
    flat_data = data.flatten()
    
    # Remove zeros and infinities for better visualization
    flat_data = flat_data[np.isfinite(flat_data) & (flat_data != 0)]
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(flat_data, bins=100, edgecolor='black')
    
    # Set labels and title
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Pixel Intensities for {nifti_path.split("/")[-1]}')
    
    # Display the plot
    plt.show()

# Example usage:
    
nifti_path_sample = os.path.join(os.getcwd(), 'code', 'leads', 'postop_ct.nii')
# display_nifti_images(nifti_path_sample)

# #Display nifti histogram for original image
# display_nifti_histogram(nifti_path_sample)

# #Display nifti histogram for image with bone extracted
# display_nifti_histogram(bone_extracted(nifti_path_sample)[0])

display_nifti_images(removeCTscandevice(nifti_path_sample)[0])


