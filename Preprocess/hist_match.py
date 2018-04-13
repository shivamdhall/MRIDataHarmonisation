'''
This file contains functions that are used to perform histogram 
matching between a reference scan and any other scan.

Please note matching only takes place for b=0 volumes

The histogram matching script generates a new scan consisting of the
same number of voluems as the original input scan. It then stores this
scan with the file name "Brain_Matched.nii.gz" in the directory where
the input scans are stored
'''

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import argparse


def threshold(full_scan, gtab, threshold):
    # We threshold using b=700 values as this is more robust in identifying artefacts
    # Use all b=700 values then take the union of them
    
    # Make the assumption that the first b=700 volume lies at postion 2
    b_700 = full_scan[:,:,:, (gtab.bvals == 700)]

    # Identify the voxels that lie above the threshold level
    bool_matrix = b_700 > threshold
    
    # Combine the identified voxels using a union opertaion in a single scan
    bool_matrix = np.logical_or.reduce(bool_matrix, axis=3)
    
    return bool_matrix



def hist_match(general_scan, reference_scan):
	# Algorithm for performing histogram matching
    # Matching is only performed on b=0 volumes
    # We also only consider non-zero voxels when performing the matching
    
    oldshape = general_scan.shape
    source = general_scan.ravel()
    reference = reference_scan.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    
    #bin_idx returns the index of the unique element in terms of the unique array
    #returns the index of a value in s_values
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    r_values, r_counts = np.unique(reference, return_counts=True)
    
    # remove the first element from s_counts, t_counts and t_values as these values correspond to 0 intensity
    # we do not want to consider background voxels

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts[1:]).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    r_quantiles = np.cumsum(r_counts[1:]).astype(np.float64)
    r_quantiles /= r_quantiles[-1]
    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image

    interp_r_values = np.interp(s_quantiles, r_quantiles, r_values[1:])
    
    # the interpoltaed values do not contain the intensity value 0
    # add this to the start of the list
    interp_r_values = np.insert(interp_r_values, 0, 0)

    return interp_r_values[bin_idx].reshape(oldshape)


def threshold_and_match(ref_scan, ref_scan_bval, ref_scan_bvec, input_scan, input_scan_bval, input_scan_bvec, threshold):

    ref_scan_dir = os.path.dirname(ref_scan)
    input_scan_dir = os.path.dirname(input_scan)
    ref_scan_name = os.path.basename(ref_scan)
    input_scan_name = os.path.basename(input_scan)

    ref_scan_image = nib.load(ref_scan)
    ref_scan_data = ref_scan_image.get_data()
    ref_bvals, ref_bvecs = read_bvals_bvecs(ref_scan_bval, ref_scan_bvec)
    ref_gtab = gradient_table(ref_bvals, ref_bvecs, b0_threshold=5)
    ref_scan_data_b0s = ref_scan_data[:,:,:,ref_gtab.b0s_mask]


    os.chdir(input_scan_dir)
    scan_image = nib.load(input_scan)
    scan_data = scan_image.get_data()
    affine_mat = scan_image.affine
    bvals_scan, bvecs_scan = read_bvals_bvecs(input_scan_bval, input_scan_bvec)
    gtab_scan = gradient_table(bvals_scan, bvecs_scan, b0_threshold=5)

    if threshold == "True":
    # identify threshold voxels - replace with 0
        threshold_matrix = threshold(scan_data, gtab_scan, 1500)
        scan_data[threshold_matrix, :] = 0

    # Perform histogram matching on the thresholded scan (use b=0 values only)
    scan_b_0s = scan_data[:,:,:,gtab_scan.b0s_mask]
    matched_scan = hist_match(scan_b_0s, ref_scan_data_b0s)

    # Replace the new matched b=0 volumes into the full thresholded scan
    scan_data[:,:,:,gtab_scan.b0s_mask] = matched_scan

    if threshold == "True":
    # Replace the threshodled voxels with a value of 0
        scan_data[threshold_matrix, :] = 0

    # Save this new scan
    new_scan_img = nib.Nifti1Image(scan_data.astype(np.float32), affine_mat)
    nib.save(new_scan_img, input_scan_dir + "/Brain_Matched.nii.gz")




parser = argparse.ArgumentParser(description='Perform histogram matching between a reference scan and any other scan.')
parser.add_argument('-threshold', required=True, type=bool, metavar=('BOOL'),\
 	help='If true then thresholding is appplied to the input scan in order to extract artifact voxels')

parser.add_argument('ref_scan', metavar=('ref_scan'),\
 	help='File path of reference scan')
parser.add_argument('ref_bval', metavar=('ref_bval'),\
 	help='Path to reference scan bval file')
parser.add_argument('ref_bvec',  metavar=('ref_bvec'),\
 	help='Path to reference scan bvec file')

parser.add_argument('scan', metavar=('scan'),\
 	help='Path to input scan')
parser.add_argument('scan_bval', metavar=('scan_bval'),\
 	help='Path to input scan bval file')
parser.add_argument('scan_bvec', metavar=('scan_bvec'),\
 	help='Path to input scan bvec file')
params = parser.parse_args()

threshold_and_match(params.ref_scan, params.ref_bval, params.ref_bvec, params.scan, params.scan_bval, params.scan_bvec, params.threshold)




