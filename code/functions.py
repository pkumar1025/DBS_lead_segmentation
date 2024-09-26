#!/usr/bin/env python3

import sys
import numpy as np
import os
import glob
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

def sobel_edge_3d(inputTensor):
    # This function computes Sobel edge maps on 3D images
    # inputTensor: input 3D images, with size of [batchsize,W,H,D,1]
    # output: output 3D edge maps, with size of [batchsize,W-2,H-2,D-2,3], each channel represents edge map in one dimension
    sobel1 = tf.constant([1,0,-1],tf.float32) # 1D edge filter
    sobel2 = tf.constant([1,2,1],tf.float32) # 1D blur weight
    # generate sobel1 and sobel2 on x- y- and z-axis, saved in sobel1xyz and sobel2xyz
    sobel1xyz = [sobel1,sobel1,sobel1]
    sobel2xyz = [sobel2,sobel2,sobel2]
    for xyz in range(3):
        newShape = [1,1,1,1,1]
        newShape[xyz] = 3
        sobel1xyz[xyz] = tf.reshape(sobel1,newShape)
        sobel2xyz[xyz] = tf.reshape(sobel2,newShape)

    # outputTensor_x will be the Sobel edge map in x-axis
    outputTensor_x = tf.nn.conv3d(inputTensor,sobel1xyz[0],strides=[1,1,1,1,1],padding='VALID') # edge filter in x-axis
    outputTensor_x = tf.nn.conv3d(outputTensor_x,sobel2xyz[1],strides=[1,1,1,1,1],padding='VALID') # blur filter in y-axis
    outputTensor_x = tf.nn.conv3d(outputTensor_x,sobel2xyz[2],strides=[1,1,1,1,1],padding='VALID') # blur filter in z-axis
    outputTensor_y = tf.nn.conv3d(inputTensor,sobel1xyz[1],strides=[1,1,1,1,1],padding='VALID')
    outputTensor_y = tf.nn.conv3d(outputTensor_y,sobel2xyz[0],strides=[1,1,1,1,1],padding='VALID')
    outputTensor_y = tf.nn.conv3d(outputTensor_y,sobel2xyz[2],strides=[1,1,1,1,1],padding='VALID')
    outputTensor_z = tf.nn.conv3d(inputTensor,sobel1xyz[2],strides=[1,1,1,1,1],padding='VALID')
    outputTensor_z = tf.nn.conv3d(outputTensor_z,sobel2xyz[0],strides=[1,1,1,1,1],padding='VALID')
    outputTensor_z = tf.nn.conv3d(outputTensor_z,sobel2xyz[1],strides=[1,1,1,1,1],padding='VALID')
    return tf.concat([outputTensor_x,outputTensor_y,outputTensor_z],4)

def process_image(img_path):
    name = Path(img_path).stem
    img = nib.load(img_path)
    data = img.get_fdata()
    normalized = (data - np.min(data))/(np.max(data) - np.min(data))
    data_reshaped = normalized.reshape([1,normalized.shape[0],normalized.shape[1],normalized.shape[2], 1])
    output_sobel = sobel_edge_3d(data_reshaped)
    output_tensor_reshaped = tf.squeeze(output_sobel, [0])
    mag = tf.math.square(output_tensor_reshaped[:,:,:,0]) + tf.math.square(output_tensor_reshaped[:,:,:,1]) + tf.math.square(output_tensor_reshaped[:,:,:,2])
    sobel_numpy = mag.numpy()
    normalized_sobel = (sobel_numpy - np.min(sobel_numpy))/(np.max(sobel_numpy) - np.min(sobel_numpy))
    nonzero_sobel = normalized_sobel[np.nonzero(normalized_sobel)]
    threshold2 = np.mean(nonzero_sobel) + 3*np.std(nonzero_sobel)
    result = (normalized_sobel > threshold2) * normalized_sobel
    new_image = nib.Nifti1Image(result, affine=np.eye(4))
    output_dir = '/data/morrison/wip/radhika/parkinsons/rough_segmentations/testrun1/'
    os.makedirs(output_dir, exist_ok=True)
    nib.save(new_image, os.path.join(output_dir, f'{name}_sobel_thresholded_nonzero.nii'))

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python script.py <path_to_nifti_image>")
    #     sys.exit(1)
    
    img_path = '/Users/Prane/Documents/GitHub/DBS_lead_segmentation/code/leads/postop_ct.nii'
    process_image(img_path)