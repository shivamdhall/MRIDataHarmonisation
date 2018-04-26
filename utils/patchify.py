'''
The functions specified in this file are used to extract either 
axial slices or 3d-patches from both the input and output scans
'''

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import random


def patchify(data_scans_inp, data_scans_out1, data_scans_out2, patch_size):
    # This function extracts patches from a scan and appends them to a list.
    # This is done for all volumes of a scan (for both the inputs and targets)

    input_patches_store = []
    target_patches_store1 = []
    target_patches_store2 = []
    
    for scan_no in range(0, len(data_scans_inp)):
        input_scan = data_scans_inp[scan_no]
        target_scan1 = data_scans_out1[scan_no]
        target_scan2 = data_scans_out2[scan_no]
        
        (x,y,z,v) = input_scan.shape
        if((patch_size-1) % 2 != 0):
            print "The patch size is not compatible"
            return
        padding = ((patch_size-1)/2)
        #pad the input scan
        full_padding = ((padding, padding), (padding, padding), (padding,padding), (0,0))
        padded_scan = np.pad(input_scan, full_padding, mode='constant', constant_values=0)

        #extract patches from the input scan

        #iterate through each volume to extract the patches
        for volume in range(0, v):
            for pos_x in range(0, x):
                for pos_y in range(0,y):
                    for pos_z in range(0, z):
                        input_patch = padded_scan[pos_x:pos_x+patch_size, pos_y:pos_y+patch_size, pos_z:pos_z+patch_size, volume]
                        target_voxel1 = target_scan1[pos_x,pos_y,pos_z,volume]
                        target_voxel2 = target_scan2[pos_x,pos_y,pos_z,volume]
                        
                        #store the patch and the target
                        input_patches_store.append(input_patch)
                        target_patches_store1.append(target_voxel1)
                        target_patches_store2.append(target_voxel2)
                        
    return (input_patches_store, target_patches_store1, target_patches_store2)



def patchify_brain_only(data_scans_inp, data_scans_out1, data_scans_out2, patch_size):
    #This function extracts patches from a scan and appends them to a list.
    #This is done for all volumes of a scan (for both the inputs and targets)
    #This version of the functions only considers voxels wholly contained within the brain
    
    input_patches_store = []
    target_patches_store1 = []
    target_patches_store2 = []
    
    for scan_no in range(0, len(data_scans_inp)):
        input_scan = data_scans_inp[scan_no]
        target_scan1 = data_scans_out1[scan_no]
        target_scan2 = data_scans_out2[scan_no]
        
        (x,y,z,v) = input_scan.shape
        if((patch_size-1) % 2 != 0):
            print "The patch size is not compatible"
            return
        padding = ((patch_size-1)/2)
        #pad the input scan
        full_padding = ((padding, padding), (padding, padding), (padding,padding), (0,0))
        padded_scan = np.pad(input_scan, full_padding, mode='constant', constant_values=0)

        #extract patches from the input scan

        #iterate through each volume to extract the patches
        for volume in range(0, v):
            for pos_x in range(0, x):
                for pos_y in range(0,y):
                    for pos_z in range(0, z):
                        # Exclude all background voxels
                        if(input_scan[pos_x,pos_y,pos_z,volume] == 0):
                            continue
                        input_patch = padded_scan[pos_x:pos_x+patch_size, pos_y:pos_y+patch_size, pos_z:pos_z+patch_size, volume]
                        target_voxel1 = target_scan1[pos_x,pos_y,pos_z,volume]
                        target_voxel2 = target_scan2[pos_x,pos_y,pos_z,volume]
                        # Exclude all patches that contain artefacts
                        if input_patch.min() < 0:
                            continue
                            
                        #store the patch and the target
                        input_patches_store.append(input_patch)
                        target_patches_store1.append(target_voxel1)
                        target_patches_store2.append(target_voxel2)
                        
    return (input_patches_store, target_patches_store1, target_patches_store2)



def pad_data(dataset, numb_slices):
    # This function is used for paddiny a dataset with 0 values, we only pad the x and y
    # dimension, it is not necessary to pad the z dimension as we will be extracting axial slices

    max_width = 0
    max_hight = 0
    
    #iterate through the scans and update the above stats
    for scan in dataset:
        (width, hight, depth, volume) = scan.shape
        if width > max_width:
            max_width = width
        if hight > max_hight:
            max_hight = hight
            
    #iterate throug the scans again and pad them based on the max stats
    for index, scan in enumerate(dataset):
        #get padding dimensions
        (width, hight, depth, volume) = scan.shape
        pad_width = max_width-width
        pad_hight = max_hight-hight
        # depth dimension should be compatible with numb_slices
        pad_depth = (numb_slices - (depth % numb_slices)) % numb_slices
        
        pad_w_b = pad_width/2
        pad_w_a = pad_width-pad_w_b
        
        pad_h_b = pad_hight/2
        pad_h_a = pad_hight-pad_h_b
        
        pad_d_b = pad_depth/2
        pad_d_a = pad_depth-pad_d_b
        
        
        padding = ((pad_w_b, pad_w_a), (pad_h_b, pad_h_a), (pad_d_b, pad_d_a), (0,0))
        aug_scan = np.pad(scan, padding, mode='constant', constant_values=0)
        dataset[index] = aug_scan
        
    return dataset



def sliceify(scans_inp, numb_slices, overlap=False):
    # This function extracts non-overlapping axial slices of thickness (numb_slices) from a scan and appends them to a list.
    # This is done for all volumes of a scan

    slices_store = []

    # Since each scan can have different dimensions, it is first neccessary to pad each 
    # each scan to ensure that they have the same x and y dimension 
    # It is also necessary to ensure that the z component is divisible by numb_slices
    # This helps with training and testing because we can use batches (must be same dimensions)
    padded_scans = pad_data(scans_inp, numb_slices)


    for scan in padded_scans:
        
        (x,y,z,v) = scan.shape

        for volume in range(0, v):
            if overlap == True:
                for i in range(0, z-numb_slices+1):
                    slices_store.append(scan[:,:,i:i+numb_slices,volume])
            else:
                for i in range(0, z, numb_slices):
                    slices_store.append(scan[:,:,i:i+numb_slices,volume])
        
    return (padded_scans, slices_store)



