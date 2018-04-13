'''
Use the functions defined in this file to split a dataset set into 
a separate training, validaton and testing datasets.

A proportion can be spcified for each of the subsets
'''

import numpy as np
import random

def split_data(input_list, output_list1, output_list2, train_prop, val_prop, test_prop):

	# Randomly split a datasets into training, validation and testing subsets
	# Perform the same split for the input and output datasets
	
    length = len(input_list)
    indexes = range(length)
    # Randomly shuffle the indexes
    random.shuffle(indexes)
    
    train_len = int(round(length * train_prop))
    val_len = int(round(length * val_prop))
    test_len = int(length - train_len - val_len)

    train_indices = indexes[:train_len]
    val_indices = indexes[train_len:train_len+val_len]
    test_indices = indexes[train_len+val_len:]

    training_input = [input_list[i] for i in train_indices]
    training_output = [output_list1[i] for i in train_indices]
    
    validation_input = [input_list[i] for i in val_indices]
    validation_output = [output_list1[i] for i in val_indices]
    
    testing_input = [input_list[i] for i in test_indices]
    testing_output1 = [output_list1[i] for i in test_indices]
    testing_output2 = [output_list2[i] for i in test_indices]
    
    return(training_input, training_output, validation_input, validation_output, testing_input, testing_output1, testing_output2)