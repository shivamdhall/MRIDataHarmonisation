'''
Define classes and functions for training and evaluating a residual neural network
'''

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib

from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.layers import *
from evaluation.evaluate import *
from utils.reconstruct import *
from visualise import *

class Residual_net(torch.nn.Module):
    # This class is used to define the architecture of a Residual neural network
    def __init__(self):
        super(Residual_net, self).__init__()

        # Residual convolution layers
        self.conv1 = Conv_Block(input_size=1, output_size=50, kernel_size=3)
        self.res1 = Resnet_Block(num_filter=50)
        self.conv2 = Conv_Block(input_size=50, output_size=50, kernel_size=3)
        self.res2 = Resnet_Block(num_filter=50)
        self.conv3 = Conv_Block(input_size=50, output_size=50, kernel_size=3)

        # MLP Regression layers
        self.fc1 = Fully_Connected_Block(input_size=50*3*3*3, output_size=1500)
        self.fc2 = Fully_Connected_Block(input_size=1500, output_size=700)
        self.fc3 = Fully_Connected_Block(input_size=700, output_size=300)
        self.fc4 = Fully_Connected_Block(input_size=300, output_size=1, batch_norm=False)

        # Reusable dropout layer
        self.drop = nn.Dropout(p=0.10)

    def forward(self, x):
        # conv blocks
        enc1 = self.drop(self.conv1(x))
        enc2 = self.drop(self.res1(enc1))
        enc3 = self.drop(self.conv2(enc2))
        enc4 = self.drop(self.res2(enc3))
        enc5 = self.drop(self.res3(enc4))
        enc5 = enc5.view(-1, 50*3*3*3)
        enc6 = self.drop(self.fc1(enc5))
        enc7 = self.drop(self.fc2(enc6))
        enc8= self.drop(self.fc3(enc7))
        out = self.fc4(enc8)

        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for layer in self.children():
            if isinstance(layer, nn.Dropout):
                continue
            layer.initialise_weights()



class MRIdataset(Dataset):
    # Define a class for holding our MRI dataset consisting of 3d patches.
    # This requires both input patches and target patches

    def __init__(self, input_patches, target_patches, transform=None):

        self.input_patches = input_patches
        self.target_patches = target_patches
        self.transform = transform

    def __len__(self):
        return len(self.input_patches)

    def __getitem__(self, idx):
        input_patch = np.array(self.input_patches[idx])
        target_patch = np.array(self.target_patches[idx])
        sample = {'input': input_patch, 'target': target_patch}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class To_Tensor(object):
    # Define a class for converting numpy n-d arrays to tensors. 
    # Note: PyTorch requires a class definition

    def __call__(self, sample):
        inp, out = sample['input'], sample['target']
        
        #first expand dimension because torch expects H x W x D x C
        #currently we only have H x W x D
        aug_inp = np.expand_dims(inp, 3)
        
        #The target is a single voxel,
        #Conver it to an array
        aug_out = np.array([out])

        # swap channel axis because
        # numpy: H x W x D x C
        # torch: C x D x H x W
        aug_inp = aug_inp.transpose((3, 2, 0, 1))
        
        return {'input': torch.Tensor(aug_inp),
                'target': torch.Tensor(aug_out)}



def train_net(net, trainloader, valiloader, epochs, log_interval, gpu, log_file):
    # This function is used for training a convolution neural network

    training_losses = []
    validation_losses = []

    # Define a loss function and a n optimizer
    # We use MSE loss 
    if gpu:
        criterion = nn.MSELoss().cuda() #returns the average over a mini-batch as opposed to the sum
    else:
        criterion = nn.MSELoss()
    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    # Use a scheduler to automatically reduce the learning rate parameter by a factor of 0.1 
    # if the loss is not minimised by 0.1 after a full epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=0, verbose=True, factor =0.1, threshold=0.1)

    for epoch in range(epochs):  # loop over the dataset multiple times
        print epoch
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0): #done in batches
            # get the inputs
            inputs = data['input']
            labels = data['target']

            # wrap them in Variable
            if gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize/update weights
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0] #loss is a variable tensor of size 1, we index to get the value out
            if i % log_interval == log_interval-1:    # log after set interval
                total_loss = running_loss / (i+1)
                log_file.write('[%d, %5d] --- Losss = %.5f \n' % (epoch + 1, i + 1, total_loss))
        
        
        # After each epoch evaluate the performance of the CNN on 10% of the training set 
        net.eval()
        training_error = 0
        iteration = 0
        limit = int(len(trainloader) * 0.1) # only evaluate on 10% of the training data
        for i, data in enumerate(trainloader, 0):
            iteration = i 
            if i == limit:
                break
            training_inputs = data['input']
            training_labels = data['target']

            if gpu:
                training_inputs, training_labels = Variable(training_inputs.cuda()), Variable(training_labels.cuda())
            else:
                training_inputs, training_labels = Variable(training_inputs), Variable(training_labels)

            training_predictions = net(training_inputs)

            if gpu:
                training_error += (torch.nn.functional.mse_loss(Variable(training_predictions.data), training_labels)).cuda().data[0]
            else:
                training_error += (torch.nn.functional.mse_loss(Variable(training_predictions.data), training_labels)).data[0]
            
        training_error /= iteration+1
        training_losses.append(training_error)        
        log_file.write('\nTraining loss iteration %d = %.5f \n' % (epoch+1, training_error ))

        # Update the learning rate dynamically based on the training error
        scheduler.step(training_error)
  
        # evaluate the performance on the full validatio set
        validation_error = 0
        iteration = 0
        for i, validation_data in enumerate(valiloader, 0): #batch processing
            iteration = i
            validation_inputs = validation_data['input']
            validation_labels = validation_data['target']
            
            if gpu:
                validation_inputs, validation_labels = Variable(validation_inputs.cuda()), Variable(validation_labels.cuda())
            else:
                validation_inputs, validation_labels = Variable(validation_inputs), Variable(validation_labels)

            validation_predictions = net(validation_inputs)
            
            if gpu:
                validation_error += (torch.nn.functional.mse_loss(Variable(validation_predictions.data), validation_labels)).cuda().data[0]
            else:
                validation_error += (torch.nn.functional.mse_loss(Variable(validation_predictions.data), validation_labels)).data[0]
            
        validation_error /= iteration+1
        log_file.write('MSE on validation set: %f \n\n' % (validation_error))
        validation_losses.append(validation_error)
 
    log_file.write("Completed training \n\n")

    return (net, training_losses, validation_losses)




def get_predictions(net, testloader, gpu):
    # This function uses a trained residual NN to generate predictions for the input patches that are contained within the training set.
    # The predictions are returned in the form of a numpy array
    net.eval()
    for index, test_data in enumerate(testloader):
        test_inputs = test_data['input']
        test_labels = test_data['target']
        
        if gpu:
            test_inputs, test_labels = Variable(test_inputs.cuda()), Variable(test_labels.cuda())
        else:
            test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)

        #store the predictions in a numpy array which we can reshape later
        test_predictions = net(test_inputs)
        if(index == 0):
            predictions = test_predictions.data.cpu().numpy() 

        else:
            predictions = np.concatenate((predictions, test_predictions.data.cpu().numpy()), axis=0)
            
    #convert back to numpy dimensions of (HxWxDxCxNumbExpls)
    predictions = predictions.transpose(1,0)

    return predictions



def rnn_run(training_data, validation_data, testing_data_patches, testing_data_scans, affine_mat,\
             epochs=100, train=True, restore_model=False):

    # training_data is a tuple of (training_inputs, training_labels)
    # validation_data is a tuple of (validation_inputs, validation_labels)
    # testing_data_patches is a tuple (testing_inputs, testing_target1, testing_target2)
    # testing_data_scans is a tuple (testing_scans_inp, testing_scans_target1, testing_scnas_target2,....
    # ....testing_scans_patches_inp, testing_scans_patches_out1, testing_scans_patches_out2)

    # First check if the current machine has GPU support
    gpu = torch.cuda.is_available()

    # Create a log file, append all output to the log file
    log_file = open("models/residual_nn/logs.txt","a+")

    # Define path for storing visualisations
    viz_path =  "models/residual_nn/visualisations"

    # Define path for storing predicted scans
    save_to = "models/residual_nn/predictions"

    rnn_model = Residual_net()

    if restore_model == False:
        # Generate a new model
        rnn_model.normal_weight_init()

    else:
        # Restore the existing model
        rnn_model.load_state_dict(torch.load('models/residual_nn/trained_rnn.pth'))

    if gpu:
        rnn_model.cuda()

    if train == True:
        # Generate a MRIdataset object using the training, validatoin and testing data
        training_dataset = MRIdataset(training_data[0], training_data[1], transform=transforms.Compose([To_Tensor()]))
        validation_dataset = MRIdataset(validation_data[0], validation_data[1], transform=transforms.Compose([To_Tensor()]))
        testing_dataset = MRIdataset(testing_data_patches[0], testing_data_patches[1], transform=transforms.Compose([To_Tensor()]))

        # Create Dataloader objects from the datasets (required by pytorch)
        trainloader = DataLoader(training_dataset, batch_size=160, shuffle=True, num_workers=16)
        valiloader = DataLoader(validation_dataset, batch_size=160, shuffle=True, num_workers=16)
        testloader = DataLoader(testing_dataset, batch_size=160, shuffle=False, num_workers=16)
        # Train the network
        print ("Training Residual NN")
        _, training_losses, validation_losses = train_net(rnn_model, trainloader, valiloader, epochs, 20, gpu, log_file)
        print ("Training completed\n")

        # Generate a learning curve plot 
        plot_losses(training_losses, validation_losses, "Learning Curves", viz_path)

        # Store the parameters of the trained model
        torch.save(rnn_model.state_dict(), 'models/residual_nn/trained_rnn.pth') 
        print ("Generating predictions for test patches")

        # Get predictions of the test patches using the trained residual NN
        test_predictions = get_predictions(rnn_model, testloader, gpu)

        # Generate error plots of predictions
        bland_altman_error(test_predictions, np.asarray(testing_data_patches[1]), np.asarray(testing_data_patches[2]),\
                         "Predictoin error plot", viz_path, "RNN")

         # Generate plot of prediction vs target
        bland_altman_prediction(test_predictions, np.asarray(testing_data_patches[1]), np.asarray(testing_data_patches[2]),\
                                 "Predicted VS target", viz_path, "RNN")

        # Get performance statistics
        print ("Evaluating performance of residual NN on test patches\n")
        log_file.write("\n\n\nEvaluating performance of residual NN on test patches\n")
        get_errors(test_predictions, np.asarray(testing_data_patches[1]), np.asarray(testing_data_patches[2]), "patches", log_file)
        # Get the errors between the 2 sets of target patches
        log_file.write("\n\n\nCalculating errors between target patches\n")
        get_errors_between_targets(np.asarray(testing_data_patches[1]), np.asarray(testing_data_patches[2]), "patches", log_file)


    # Evaluate the performance of the trained residual NN on unseen scans
    testing_dataset_scans = MRIdataset(testing_data_scans[3], testing_data_scans[4], transform=transforms.Compose([To_Tensor()]))
    final_test_loader = DataLoader(testing_dataset_scans, batch_size=160, shuffle=False, num_workers=16)

    # Generate predictions for the unseen test scans using the trained residual NN
    print ("Generating predictions for unseen test scans")
    final_test_predictions = get_predictions(rnn_model, final_test_loader, gpu)
    # Reconstruct the predicted patches into full scans
    final_predicted_brains = reconstruct_brain(final_test_predictions, testing_data_scans[0], model="rnn")
    # Get performance statistics
    print ("Evaluating performance of residual NN on unseen test scans")
    log_file.write("\n\n\nEvaluating performance of residual NN on unseen test scans\n")
    get_errors(final_predicted_brains, testing_data_scans[1], testing_data_scans[2], "scan", log_file)
    # Get the errors between the 2 sets of target scans
    log_file.write("\n\n\nCalculating errors between target scans\n")
    get_errors_between_targets(testing_data_scans[1], testing_data_scans[2], "scan", log_file)

    print ("Storing the predicted scans")
    for i, (predicted_scan, scan_affine) in enumerate(zip(final_predicted_brains, affine_mat)):

        predicted_scan_image = nib.Nifti1Image(predicted_scan, scan_affine)
        nib.save(predicted_scan_image, save_to + "/predicted_scan_" + str(i) + ".nii.gz" )

    log_file.close()


