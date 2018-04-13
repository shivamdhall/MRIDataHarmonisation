'''
train and evaluate the performance of all supervised deep learning algorithms on DW-MRI data
'''

import numpy as np
from utils.loader import get_data
from utils.patchify import patchify, patchify_brain_only
from utils.split_dataset import split_data
from models.convolution_nn.cnn import cnn_run
from models.residual_nn.rnn import rnn_run

# Variables
petmr_data_path = '/home/ubuntu/project/Dataset/PETMR_data'
trio_data_path = '/home/ubuntu/project/Dataset/TRIO_data'

# The below proportions should sum to 1
train_proportion = 0.80
validation_proportion = 0.10
test_proportion = 0.10

# Upload scans then split into training, validation and testing subsets
# Enter a list of tuples (subject, scan_number)
train_val_test_scans = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2), (5,1), (5,2), (6,1), (6,2), (7,1), (7,2), (8,2)]

# Final test scans - These subject scans have never been seen by the model
final_testing_scans = [(8,1), (9,1), (9,2), (10,1), (10,2)]

data_dict = {"train_val_test": train_val_test_scans, "testing":final_testing_scans}

# Upload the training and testing datasets
# Note we have 2 sets of outputs for each input (only 1 is used for training, the other is used for evaluation purposes)
(train_val_test_inp, train_val_test_out1, train_val_test_out2, final_test_inp, final_test_out1, final_test_out2, affine_mat) = \
        get_data(petmr_data_path, trio_data_path, data_dict, input_scanner="PETMR")

print("Number of scans used for training, validation and testing: %d" % len(train_val_test_inp))
print ("Number of scans used for final testing: %d" % len(final_test_inp))

# Convert the data into 3d-patches of size 9x9x9
# For training purposes we patchify the brain only
print("Patchifying training, validation and testing set")
(train_val_test_input, train_val_test_target1, train_val_test_target2) = patchify_brain_only(train_val_test_inp, train_val_test_out1, train_val_test_out2, 9)

# Randomly split the train_val_test dataset into a training, validation and testing subsest
training_input, training_target, validation_input, validation_target, testing_input, testing_target1, testing_target2 = \
        split_data(train_val_test_input, train_val_test_target1, train_val_test_target2, train_proportion, validation_proportion, test_proportion)

# In order to predict a full scan, we must convert the entire scan into patches,
# this makes it possible to reconstruct the entire scan from the predicted patches
print("Patchifying final testing scans")
(final_testing_input, final_testing_target1, final_testing_target2) = patchify(final_test_inp, final_test_out1, final_test_out2, 9)


# print statistics
print ("Number of training examples : %d" % len(training_input))
print ("Number of Validation examples : %d" % len(validation_input))
print ("Number of testing examples : %d" % len(testing_input))


training_data = (training_input, training_target)
validation_data = (validation_input, validation_target)
testing_data = (testing_input, testing_target1, testing_target2)
final_testing_data = (final_test_inp, final_test_out1, final_test_out2, final_testing_input, final_testing_target1, final_testing_target2)


# Train or evaluate the performance of a convolution NN
cnn_run(training_data, validation_data, testing_data, final_testing_data, affine_mat, epochs=25, train=True, restore_model=False)

# Train or evaluate the performance of a residual NN
rnn_run(training_data, validation_data, testing_data, final_testing_data, affine_mat, epochs=25, train=True, restore_model=False)


