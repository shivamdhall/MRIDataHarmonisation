'''
Pipeline for training and evaluating the the performance of the unsupervised modles,
this includes the cycle-GAN model
'''

import numpy as np
from utils.loader import get_data, get_data_unregistered
from utils.patchify import patchify, patchify_brain_only, sliceify
from utils.split_dataset import split_data
from models.cycle_gan.cyc_gan import cyc_gan_run


# Specify location of data
petmr_data_path = '/Volumes/Seagate Backup Plus Drive/Project/Dataset/PETMR_data'
trio_data_path = '/Volumes/Seagate Backup Plus Drive/Project/Dataset/TRIO_data'

# The below proportions should sum to 1
train_proportion = 0.80
validation_proportion = 0.10
test_proportion = 0.10

# Upload scans then split into training, validation and testing
# Enter a list of tuples (subject, scan_number)
train_val_test_scans = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2),\
					    (5,1), (5,2), (6,1), (6,2), (7,1), (7,2), (8,1), (8,2)]

# Final test scans - These subject scans have never been seen by cycle-GAN
final_testing_scans = [(9,1), (9,2), (10,1), (10,2)]

data_dict = {"train_val_test": train_val_test_scans, "testing":final_testing_scans}


# Since we are using an unsupervised approach, the input scans and output scnas do not need to be registered
# Hence we upload input scans and the corresponding unregisterd output scans
(train_val_test_inp, train_val_test_out_unregistered) = \
        get_data_unregistered(petmr_data_path, trio_data_path, train_val_test_scans, input_scanner="PETMR")


# Since we have registered scans, we can also upload them and use these for evlauation purposes
(_, train_val_test_out1_registered, train_val_test_out2_registered, final_test_inp,\
final_test_out1_registered, final_test_out2_registered, affine_mat) = \
		get_data(petmr_data_path, trio_data_path, data_dict, input_scanner="PETMR")


print("Number of scans used for training, validation and testing: %d" % len(train_val_test_inp))
print ("Number of scans used for final testing: %d" % len(final_test_inp))


# Convert the training data inputs and outputs into slices of depth 3 i.e (HxWx3)
# Do the same for the registered scans that will be used for evaluation
print("sliceifying training scans")
_, train_val_test_inp_slices = sliceify(train_val_test_inp, 3, overlap=True)
_, train_val_test_out_slices = sliceify(train_val_test_out_unregistered, 3, overlap=True)
_, train_val_test_out1_slices_registerd = sliceify(train_val_test_out1_registered, 3, overlap=True)
_, train_val_test_out2_slices_registerd = sliceify(train_val_test_out2_registered, 3, overlap=True)

# Sliceify the final testing scans
print("sliceifying final testing scans")
final_scans_inp_padded, final_test_inp_slices = sliceify(final_test_inp, 3, overlap=False)
final_scans_out1_padded, final_test_out1_slices = sliceify(final_test_out1_registered, 3, overlap=False)
final_scans_out2_padded, final_test_out2_slices = sliceify(final_test_out2_registered, 3, overlap=False)


# Randomly split the train_val_test_inp dataset into a training, validation and testing subsest
train_inp_slices, train_inp_slices_reg, validation_inp_slices, validation_out_slices, \
testing_inp_slices, testing_out1_slices, testing_out2_slices = \
        split_data(train_val_test_inp_slices, train_val_test_out1_slices_registerd, train_val_test_out2_slices_registerd,\
                   train_proportion, validation_proportion, test_proportion)

# print statistics
print ("Number of training examples : %d" % len(train_inp_slices))
print ("Number of Validation examples : %d" % len(validation_inp_slices))
print ("Number of testing examples : %d" % len(testing_inp_slices))


training_data = (train_inp_slices, train_inp_slices_reg, train_val_test_out_slices) # Registered and unregistered set of inputs and outputs
validation_data = (validation_inp_slices, validation_out_slices) # Registered set of validation examples in slice form
testing_data = (testing_inp_slices, testing_out1_slices, testing_out2_slices) # Registered set of training examples in slice form
final_testing_data = (final_scans_inp_padded, final_scans_out1_padded, final_scans_out2_padded,\
                      final_test_inp_slices, final_test_out1_slices, final_test_out2_slices)


# Train or evaluate the performance of a cycle - GAN
cyc_gan_run(3, training_data, validation_data, testing_data, final_testing_data, affine_mat, epochs=25, train=True, restore_model=False)



