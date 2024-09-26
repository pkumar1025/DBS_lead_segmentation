#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:02:52 2017

@author: pkao
"""

import os
import CTtools
from subprocess import call
import sys
import SimpleITK as sitk

# if len(sys.argv) < 2:
#     print("Usage: python CT2MNI152Affine.py <path_to_ct_scan>")
#     sys.exit(1)

ct_scan_path = '/Users/Prane/Documents/GitHub/DBS_lead_segmentation/code/leads/postop_ct.nii'
MNI_152_bone = os.path.join(os.getcwd(),'MNI152_T1_1mm_bone.nii.gz')
MNI_152 = os.path.join(os.getcwd(),'MNI152_T1_1mm.nii.gz')
bspline_path = os.path.join(os.getcwd(), 'Par0000bspline.txt')
nameOfAffineMatrix = ct_scan_path[:ct_scan_path.find('.nii.gz')]+'_affine.mat'

print('The location of MNI152 bone:' , MNI_152_bone)

# if you want to do ct scan removal 
#ct_scan_wodevice = CTtools.removeCTscandevice(ct_scan_path)

ct_scan_wodevice = ct_scan_path

ct_scan_wodevice_bone = CTtools.bone_extracted(ct_scan_wodevice)

call(['flirt', '-in', ct_scan_wodevice_bone, '-ref', MNI_152_bone, '-omat', nameOfAffineMatrix, '-bins', '256', '-searchrx', '-180', '180', '-searchry', '-180', '180', '-searchrz', '-180', '180', '-dof', '12', '-interp', 'trilinear'])

call(['flirt', '-in', ct_scan_wodevice, '-ref', MNI_152, '-applyxfm', '-init', nameOfAffineMatrix, '-out', ct_scan_wodevice[:ct_scan_wodevice.find('.nii.gz')]+'_MNI152.nii.gz'])

# the code below implement the deformable transformation

ct_scan_wodevice_contraststretching = CTtools.contrastStretch(ct_scan_wodevice)

call(['flirt', '-in', ct_scan_wodevice_contraststretching, '-ref', MNI_152, '-applyxfm', '-init', nameOfAffineMatrix, '-out', ct_scan_wodevice_contraststretching[:ct_scan_wodevice_contraststretching.find('.nii.gz')]+'_MNI152.nii.gz'])

#call(['elastix', '-m', ct_scan_wodevice_contraststretching[:ct_scan_wodevice_contraststretching.find('.nii.gz')]+'_MNI152.nii.gz', '-f', MNI_152, '-out', os.path.dirname(ct_scan_path), '-p', bspline_path])

fixed_image = sitk.ReadImage(MNI_152)
moving_image = sitk.ReadImage(ct_scan_wodevice_contraststretching[:ct_scan_wodevice_contraststretching.find('.nii.gz')]+'_MNI152.nii.gz')

elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(fixed_image)
elastixImageFilter.SetMovingImage(moving_image)
elastixImageFilter.SetParameterMap(sitk.ReadParameterFile(bspline_path))
elastixImageFilter.SetOutputDirectory(os.path.dirname(ct_scan_path))
elastixImageFilter.Execute()

resultImage = elastixImageFilter.GetResultImage()
sitk.WriteImage(resultImage, os.path.join(os.path.dirname(ct_scan_path), "result.nii.gz"))

name_ct_with_MNI = ct_scan_path[:ct_scan_path.find('.nii.gz')]+'_MNI152'

ct_MNI_bone = CTtools.bone_extracted(name_ct_with_MNI)

CTtools.display_nifti_histogram(ct_MNI_bone)

