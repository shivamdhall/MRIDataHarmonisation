'''
This file defines functions that are used to calculate the errors between the
predicted and the actual scans
'''

import numpy as np

def get_errors(predictions, output1, output2, data_type, log_file):

    # This is a generic function that is used to get the errors between the predictions
    # and targets, data_type is a keywords that is either "patches" or "scan"

    # if "scan" then the predictions and outputs are assumed to be a list containing the predicted
    # scans and actual scans respectively

    # if "patches" then the predictions are outputs are assumed to be a single numpy array corresponding to the
    # predicted and actual scan respectively.
    
    scan = 0
    mse_avg = 0
    l1_avg = 0
    r1_avg = 0
    
    while True:
        if data_type == "scan":
            scan_predictions = predictions[scan]
            scan_output1 = output1[scan]
            scan_output2 = output2[scan]
            scan += 1
            log_file.write('\n--------------Testing Scan %d---------------\n' % (scan))
        else:
            log_file.write('\n----------------------------\n')
            scan_predictions = predictions
            scan_output1 = output1
            scan_output2 = output2
            
        mse = ((scan_predictions - scan_output1) ** 2).mean()
        log_file.write('MSE on test %s: %f \n' % (data_type, mse))

        l1 = (np.absolute(scan_predictions -  scan_output1)).mean()
        log_file.write('Mean absolute loss on test %s: %f \n' % (data_type, l1))

        r1 = 1 - (np.sum((scan_output1 - scan_predictions) ** 2) / np.sum((scan_output1 - scan_output1.mean()) ** 2))
        log_file.write('R2 loss on test %s (Coefficient of determination): %f \n' % (data_type, r1))

        min_target = np.min([scan_output1, scan_output2], axis=0)
        max_target = np.max([scan_output1, scan_output2], axis=0)
        correct_predictions = np.logical_and(scan_predictions>=min_target, scan_predictions<=max_target)
        correct_predictions = np.sum(correct_predictions)
        percentage_correct = (float(correct_predictions) / float(scan_predictions.size)) * 100
        log_file.write('Percentage of predictions within target range: %f \n' % (percentage_correct))
        
        mse_avg += mse
        l1_avg += l1
        r1_avg += r1
        
        if scan > 0 and scan < len(predictions):
            # Get averages of all the data
            continue
        else:
            break
                    
    if data_type == "scan":
        log_file.write('\n------Average of errors across all scans------ \n')
        log_file.write('Avg MSE: %f \n' % (mse_avg/scan))
        log_file.write('Avg L1: %f \n' % (l1_avg/scan))
        log_file.write('Avg R2: %f \n' % (r1_avg/scan))




def get_errors_between_targets(output1, output2, data_type, log_file):

    # This function works in a similar manner to that described above, but instead
    # calculates the errors between two corresponding sets of targets

    # data_type can either be "scan" or "patches"

    # if "scan" then the outputs are assumed to be a list of numpy arrays containing actual scans 

    # if "patches" then the outputs are assumed to be a single numpy array corresponding to the
    # actual scan. 
    
    scan = 0
    mse_avg = 0
    l1_avg = 0
    r1_avg = 0
    
    while True:
        if data_type == "scan":
            scan_output1 = output1[scan]
            scan_output2 = output2[scan]
            scan += 1
            log_file.write('\n--------------Testing Scan %d--------------- \n' % (scan))
        else:
            log_file.write('\n---------------------------- \n')
            scan_output1 = output1
            scan_output2 = output2
        
        target_mse = ((scan_output1 - scan_output2) ** 2).mean()
        log_file.write('MSE between targets: %f \n' % (target_mse))

        target_l1 = (np.absolute(scan_output2 - scan_output1)).mean()
        log_file.write('Mean absolute loss between targets: %f \n' % (target_l1))

        target_r1 = 1 - (np.sum((scan_output1 - scan_output2) ** 2) / np.sum((scan_output1 - scan_output1.mean()) ** 2))
        log_file.write('R2 loss between targets (Coefficient of determination): %f \n' % (target_r1))
        
        mse_avg += target_mse
        l1_avg += target_l1
        r1_avg += target_r1
        
        if scan > 0 and scan < len(output1):
            # Get averages of all the data
            continue
        else:
            break
                    
    if data_type == "scan":
        log_file.write('\n-------Average of errors-------- \n')
        log_file.write('Avg MSE: %f \n' % (mse_avg/scan))
        log_file.write('Avg L1: %f \n' % (l1_avg/scan))
        log_file.write('Avg R2: %f \n' % (r1_avg/scan))