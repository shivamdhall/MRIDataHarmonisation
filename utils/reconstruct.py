'''
This file defines helper functions that are used to convert
a list of predicted patched to a full scan
'''

import numpy as np

def replace_background(prediction, input_scan):
    # This function simpy replaces all background voxels in the predicted scan with 0

    background_mask = input_scan <= 0
    prediction[background_mask] = 0
    
    return prediction


def reconstruct_brain(predictions, scans, model):
    # Scans is a list containing the input scans where each scan is stored as a 4d numpy array
    # Iterate through all the scans and reconstruct them using the list of patches

    start_voxel = 0
    reconstructed_scans = []
    for scan in scans:
        dimensions = scan.shape
        size_x = dimensions[0]
        size_y = dimensions[1]
        size_z = dimensions[2]
        size_v = dimensions[3]
        end_voxel = start_voxel + (size_x * size_y * size_z * size_v)
        
        if model == "cnn":
            reconstructed = np.reshape(predictions[:,:,:,:,start_voxel:end_voxel], [size_v, size_x, size_y, size_z], order='C')
        elif model == "rnn":
            reconstructed = np.reshape(predictions[:,start_voxel:end_voxel], [size_v, size_x, size_y, size_z], order='C')

        reconstructed = reconstructed.transpose(1,2,3,0)
        start_voxel = end_voxel
        
        # Replace the background voxels of the scan with 0
        reconstructed = replace_background(reconstructed, scan)
        reconstructed_scans.append(reconstructed)
        
    return reconstructed_scans


def reconstruct_brain_from_slices(predictions, scans, slice_size):
    # Scans is a list containing the input scans where each scan is stored as a 4d numpy array
    # Iterate through all the scans and reconstruct them
    
    i = 0
    reconstructed_scans = []
    for scan in scans:
        dimensions = scan.shape
        size_x = dimensions[0]
        size_y = dimensions[1]
        size_z = dimensions[2]
        size_v = dimensions[3]
        predicted_scan = np.zeros((size_x, size_y, size_z, size_v))
        
        for volume in range(0, size_v):
            for pos_z in range(0, size_z-slice_size, slice_size):
                predicted_scan[:, :, pos_z:pos_z+slice_size, volume] = predictions[i,:,:,:]
                i += 1

        # Replace the background voxels of the scan with 0
        reconstructed = replace_background(predicted_scan, scan)
        reconstructed_scans.append(reconstructed)
        
    return reconstructed_scans