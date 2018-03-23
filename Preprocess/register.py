'''
This script takes a two sets of DW-MRI scans and performs 
non-linear Symmetric Diffeomorphic Registration between the scans
This is a vital step is a supervised learning approach is used for
creating a machine learning model

After registration has been performed, brain masks are then generated
for each scan using FSL Brain Extraction tool (BET). The intersection
of the brain masks associated with each of a subject's scans are then created, 
these unioned mask are then used to extract the brains from each pair of registered
scans.
'''

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess
import argparse

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
from dipy.segment.mask import applymask
from dipy.segment.mask import bounding_box
from dipy.segment.mask import crop
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric

from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

parser = argparse.ArgumentParser(description='Perform registration between a reference scan and any other scan.')
parser.add_argument('-reg_type', required=True, metavar=('REG'),\
 	help='Select the type of registration to perform, either non-linear:"SDR", or linear: "Affine"')
parser.add_argument('ref_scan', metavar=('ref_scan'),\
 	help='Reference scan')
parser.add_argument('scan', metavar=('scan'),\
 	help='Scan to be registered')

params = parser.parse_args()


# Get the directories where the files are stored, they will be used for storing the generated scans
ref_dir_path = os.path.dirname(params.ref_scan)
other_dir_path = os.path.dirname(params.scan)

def affine_registration(reference, reference_grid2world, scan, scan_grid2world):
    #get first b0 volumes for both scans
    reference_b0 = reference[:,:,:,0]
    scan_b0 = scan[:,:,:,0]
    
    #In this function we use multiple stages to register the 2 scans
    #providng previous results as initialisation to the next stage, 
    #the reason we do this is because registration is a non-convex 
    #problem thus it is important to initialise as close to the 
    #optiaml value as possible
    
    #Stage1: we obtain a very rough (and fast) registration by just aligning 
    #the centers of mass of the two images
    center_of_mass = transform_centers_of_mass(reference_b0, reference_grid2world, scan_b0, scan_grid2world)
    
    #create the similarity metric (Mutual Information) to be used:
    nbins = 32
    sampling_prop = None #use all voxels to perform registration
    metric = MutualInformationMetric(nbins, sampling_prop)
    
    #We use a multi-resolution stratergy to accelerate convergence and avoid
    #getting stuck at local optimas (below are the parameters)
    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0] #parameters for gaussian kernel smoothing at each resolution
    factors = [4, 2, 1] #subsampling factor
    
    #optimisation algorithm used is L-BFGS-B
    affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)
    
    #Stage2: Perform a basic translation transform
    transform = TranslationTransform3D()
    translation = affreg.optimize(reference_b0, scan_b0, transform, None, reference_grid2world, scan_grid2world, starting_affine=center_of_mass.affine)
    
    #Stage3 : optimize previous result with a rigid transform
    #(Includes translation, rotation)
    transform = RigidTransform3D()
    rigid = affreg.optimize(reference_b0, scan_b0, transform, None, reference_grid2world, scan_grid2world, starting_affine=translation.affine)
    
    #Stage4 : optimize previous result with a affine transform
    #(Includes translation, rotation, scale, shear)
    transform = AffineTransform3D()
    affine = affreg.optimize(reference_b0, scan_b0, transform, None, reference_grid2world, scan_grid2world, starting_affine=rigid.affine)
    
    if params.reg_type == "SDR":
        #Stage 5 : Symmetric Diffeomorphic Registration
        metric = CCMetric(3)
        level_iters = [400, 200, 100]
        sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
        mapping = sdr.optimize(reference_b0, scan_b0, reference_grid2world, scan_grid2world, affine.affine)
    else:
        mapping = affine
    #Once this is completed we can perform the affine transformation on each 
    #volume of scan2
   
    for volume in range(0, scan.shape[3]):
        #note affine is an AffineMap object,
        #The transform method transforms the input image from co-domain to domain space
        #By default, the transformed image is sampled at a grid defined by the shape of the domain
        #The sampling is performed using linear interpolation (refer to comp vision lab on homographies)
        scan[:,:,:,volume] = mapping.transform(scan[:,:,:,volume])
        
    return scan



def compute_masks_crop_bet(reference_scan, other_scan1, ref_scan_path, ref_dir_path, other_dir_path):
    # Use bet for generating brain masks for each of the scans

    # Get the mask of the reference scan 
    os.chdir(ref_dir_path)
    subprocess.call(["bet", os.path.basename(ref_scan_path), "Brain_temp", "-m", "-n", "-R", "-f", "0.2", "-t"])
    reference_scan_mask = nib.load("Brain_temp_mask.nii.gz")
    reference_scan_mask = reference_scan_mask.get_data()
    # Delete the created files
    os.remove('Brain_temp.nii.gz')
    os.remove('Brain_temp_mask.nii.gz')

    # Similarly get the masks of the other scans
    os.chdir(other_dir_path)
    subprocess.call(["bet", "Full_Registered_Scan.nii.gz", "Brain_temp", "-m", "-n", "-R", "-f", "0.2", "-t"])
    other_scan1_mask = nib.load("Brain_temp_mask.nii.gz")
    other_scan1_mask = other_scan1_mask.get_data()
    os.remove('Brain_temp.nii.gz')
    os.remove('Brain_temp_mask.nii.gz')
        
    #Get the intersection of the masks
    mask_union = np.logical_and(reference_scan_mask, other_scan1_mask)
    
    #Apply the combined mask to the scans
    reference_scan_brain = applymask(reference_scan, mask_union)
    other_scan1_brain = applymask(other_scan1, mask_union)
    
    #Crop the scans using the unioned mask
    (mins, maxs) = bounding_box(mask_union)
    reference_scan_brain = crop(reference_scan_brain, mins, maxs)
    
    return (reference_scan_brain, other_scan1_brain)  



# Load the reference scan and the corresponding scan
reference_scan = nib.load(params.ref_scan)
other_scan1 = nib.load(params.scan)

# Convert the images to dipy format
reference_scan_data = reference_scan.get_data()
other_scan1_data = other_scan1.get_data()

# Get the grid2world matrices for each scan
reference_scan_grid2world = reference_scan.affine
other_scan1_grid2world = other_scan1.affine

# Perform registration using the 'reference_scan' as the static/reference scan
# i.e we apply spatial transformations to all other scans to achive spatial correspondance with the reference_scan
print ("Performing Registration for: Scan 1")
other_scan1_transformed = affine_registration(reference_scan_data, reference_scan_grid2world, other_scan1_data, other_scan1_grid2world)
print ("---Registration Completed---")

# Create images from the data
other_scan1_img = nib.Nifti1Image(other_scan1_transformed.astype(np.float32), reference_scan.affine)

# Save the fully registered scan (before brain extraction)
nib.save(other_scan1_img, other_dir_path + "/Full_Registered_Scan.nii.gz") 

# Compute brain masks for each scan - Use FSL BET for this
print ("Computing brain masks")
(reference_brain, other_brain1) = \
    compute_masks_crop_bet(reference_scan_data, other_scan1_transformed, params.ref_scan, ref_dir_path, other_dir_path)

# Save the new masked/cropped scans
print ("Saving cropped masks")  
# Create images from the cropped data
reference_brain_img = nib.Nifti1Image(reference_brain.astype(np.float32), reference_scan.affine)
other_brain1_img = nib.Nifti1Image(other_brain1.astype(np.float32), reference_scan.affine)

# Store the images -- These are cropped
nib.save(reference_brain_img, ref_dir_path + "/Brain_Extracted.nii.gz")
nib.save(other_brain1_img, other_dir_path + "/Brain_Registered_Extracted.nii.gz")
