'''
This file contains function that are required for uploading data
Note, the functions defined in this file assume that the data is stored in a particular format
''' 

import nibabel as nib
import numpy as np
import os
import sys
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import random


def get_data(petmr_path, trio_path, scans_dict, input_scanner):
    # This function simply uploads the training and testing scans into lists of numpy arrays
    # the data is not yet sliced or patched at this stage

    # scans dict should be of the form {"train_val_test": train_val_test_scans, "testing": testing_scans}
    # where train_val_test_scans and testing_scans are lists of tuples, each tuple is a pair consisting of the subject-id and the scan number
    
    train_val_test_inp = []
    train_val_test_out1 = []
    train_val_test_out2 = []
    test_data_inp = []
    test_data_out1 = []
    test_data_out2 = []
    testing_affine_mat = []
    
    if input_scanner == "PETMR":
        input_path = petmr_path
        output_path = trio_path
    else:
        input_path = trio_path
        output_path = petmr_path       
    
    for key, subjs in scans_dict.iteritems():
        for subj_scan in subjs:
                
            input_scan_image = nib.load(str(input_path) + "/Subj" + str(subj_scan[0]) + "Scan" + str(subj_scan[1]) + "/Brain_Matched.nii.gz")
            input_scan_data = input_scan_image.get_data()
                
            # Important : Upload the output scan that is registered to the appropriate input scan
            output_scan_image1 = nib.load(str(output_path) + "/Subj" + str(subj_scan[0]) + "Scan1" + "/Brain_Matched_Scan" + str(subj_scan[1]) + ".nii.gz")
            output_scan_data1 = output_scan_image1.get_data()
            
            output_scan_image2 = nib.load(str(output_path) + "/Subj" + str(subj_scan[0]) + "Scan2" + "/Brain_Matched_Scan" + str(subj_scan[1]) + ".nii.gz")
            output_scan_data2 = output_scan_image2.get_data()
                
            input_bvals_scan, input_bvecs_scan = read_bvals_bvecs(str(input_path) + "/Subj" + str(subj_scan[0]) + "Scan" + str(subj_scan[1]) + "/NODDI.bval",\
                                                          str(input_path) + "/Subj" + str(subj_scan[0]) + "Scan" + str(subj_scan[1]) + "/NODDI.bvec")
                
            output_bvals_scan1, output_bvecs_scan1 = read_bvals_bvecs(str(output_path) + "/Subj" + str(subj_scan[0]) + "Scan1/NODDI.bval",\
                                                          str(output_path) + "/Subj" + str(subj_scan[0]) + "Scan1/NODDI.bvec")
            
            output_bvals_scan2, output_bvecs_scan2 = read_bvals_bvecs(str(output_path) + "/Subj" + str(subj_scan[0]) + "Scan2/NODDI.bval",\
                                                          str(output_path) + "/Subj" + str(subj_scan[0]) + "Scan2/NODDI.bvec")
                
            #set a threshold value for b=0 values (due to TRIO dataset)
            input_gtab_scan = gradient_table(input_bvals_scan, input_bvecs_scan, b0_threshold=5)
            input_s0s_scan = input_scan_data[:, :, :, input_gtab_scan.b0s_mask]
                
            output_gtab_scan1 = gradient_table(output_bvals_scan1, output_bvecs_scan1, b0_threshold=5)
            output_s0s_scan1 = output_scan_data1[:, :, :, output_gtab_scan1.b0s_mask]
            
            output_gtab_scan2 = gradient_table(output_bvals_scan2, output_bvecs_scan2, b0_threshold=5)
            output_s0s_scan2 = output_scan_data2[:, :, :, output_gtab_scan2.b0s_mask]
                
            if(key == "train_val_test"):
                print ("Uploading Subject %s Scan %s" % (str(subj_scan[0]), str(subj_scan[1])))
                # Append the data to the lists containing the training inputs and outputs
                # upload the first volume of each slice only
                train_val_test_inp.append(input_s0s_scan[:,:,:,[0]])
                train_val_test_out1.append(output_s0s_scan1[:,:,:,[0]])
                train_val_test_out2.append(output_s0s_scan2[:,:,:,[0]])
            else:
                print ("Testing: Subject %s Scan %s" % (str(subj_scan[0]), str(subj_scan[1])))
                test_data_inp.append(input_s0s_scan[:,:,:,[0]])
                test_data_out1.append(output_s0s_scan1[:,:,:,[0]])
                test_data_out2.append(output_s0s_scan2[:,:,:,[0]])
                testing_affine_mat.append(output_scan_image1.affine)

    return (train_val_test_inp, train_val_test_out1, train_val_test_out2, test_data_inp, test_data_out1, test_data_out2, testing_affine_mat)


def get_data_unregistered(petmr_path, trio_path, scans_list, input_scanner):
    # This function simply uploads the training scans that are specified in the list "scans_list"
    # The scans that are uploaded are not registered between the scanners (hence unregistered)

    train_val_test_inp = []
    train_val_test_out = []

    
    if input_scanner == "PETMR":
        input_path = petmr_path
        output_path = trio_path
    else:
        input_path = trio_path
        output_path = petmr_path       
    
    for subj_scan in scans_list:
            
        input_scan_image = nib.load(str(input_path) + "/Subj" + str(subj_scan[0]) + "Scan" + str(subj_scan[1]) + "/Brain_Matched.nii.gz")
        input_scan_data = input_scan_image.get_data()
            
        # Upload the corresponding output scan (unregistered)
        output_scan_image1 = nib.load(str(output_path) + "/Subj" + str(subj_scan[0]) + "Scan1" + "/Brain_Matched.nii.gz")
        output_scan_data1 = output_scan_image1.get_data()
            
        input_bvals_scan, input_bvecs_scan = read_bvals_bvecs(str(input_path) + "/Subj" + str(subj_scan[0]) + "Scan" + str(subj_scan[1]) + "/NODDI.bval",\
                                                      str(input_path) + "/Subj" + str(subj_scan[0]) + "Scan" + str(subj_scan[1]) + "/NODDI.bvec")
            
        output_bvals_scan1, output_bvecs_scan1 = read_bvals_bvecs(str(output_path) + "/Subj" + str(subj_scan[0]) + "Scan1/NODDI.bval",\
                                                      str(output_path) + "/Subj" + str(subj_scan[0]) + "Scan1/NODDI.bvec")
        
        #set a threshold value for b=0 values (due to TRIO dataset)
        input_gtab_scan = gradient_table(input_bvals_scan, input_bvecs_scan, b0_threshold=5)
        input_s0s_scan = input_scan_data[:, :, :, input_gtab_scan.b0s_mask]
            
        output_gtab_scan1 = gradient_table(output_bvals_scan1, output_bvecs_scan1, b0_threshold=5)
        output_s0s_scan1 = output_scan_data1[:, :, :, output_gtab_scan1.b0s_mask]
            
        print ("Uploading unregistered Subject %s Scan %s" % (str(subj_scan[0]), str(subj_scan[1])))
        # Append the data to the lists containing the training inputs and outputs
        # upload the first volume of each slice only
        train_val_test_inp.append(input_s0s_scan[:,:,:,[0]])
        train_val_test_out.append(output_s0s_scan1[:,:,:,[0]])

    return (train_val_test_inp, train_val_test_out)
