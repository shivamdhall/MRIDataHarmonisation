'''
Define classes and functions for training and evaluating a cycle general-adversarial network
'''

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import itertools
import random
import nibabel as nib

from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.layers import *
from evaluation.evaluate import *
from utils.reconstruct import *
from visualise import *


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Encoder
        self.conv1 = Conv_Block(input_size=1, output_size=100)
        # Resnet blocks
        self.resnet_blocks = []
        for i in range(3):
            self.resnet_blocks.append(Resnet_Block(num_filter=100))
        self.resnet_blocks = torch.nn.Sequential(*self.resnet_blocks)
        # Decoder
        self.deconv1 = Deconv_Block(input_size=100, output_size=1, batch_norm=False)

    def forward(self, x):
        # Encoder
        enc1 = self.conv1(x)
        # Resnet blocks
        res = self.resnet_blocks(enc1)
        # Decoder
        out = self.deconv1(res)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for layer in self.children():
            if isinstance(layer, Conv_Block):
                layer.initialise_weights()
            if isinstance(layer, Deconv_Block):
                layer.initialise_weights()
            if isinstance(layer, Resnet_Block):
                layer.initialise_weights()
                layer.initialise_weights()


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        conv1 = Conv_Block(input_size=1, output_size=100, activation='lrelu', batch_norm=False)
        conv2 = Conv_Block(input_size=100, output_size=100, kernel_size=(1,3,3), stride=(1,3,3), activation='lrelu')
        conv3 = Conv_Block(input_size=100, output_size=1, kernel_size=(1,3,3), stride=(1,3,3), activation='no_act', batch_norm=False)

        self.conv_blocks = torch.nn.Sequential(
            conv1,
            conv2,
            conv3,
        )

    def forward(self, x):
        out = self.conv_blocks(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for layer in self.children():
            if isinstance(layer, Conv_Block):
                layer.initialise_weights()


class MRIdataset(Dataset):
    # Define a class for holding our MRI dataset consisting of 3d slices.

    def __init__(self, slices, transform=None):

        self.slices = slices
        self.transform = transform

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        example = np.array(self.slices[idx])
        sample = {'slice': example}

        if self.transform:
            sample = self.transform(sample)
            
        return sample


class To_Tensor(object):
    # Define a class for converting numpy n-d arrays to tensors. 
    # Note: PyTorch requires a class definition

    def __call__(self, sample):
        slice = sample['slice']
        
        #first expand dimension because torch expects H x W x D x C
        #currently we only have H x W x D
        aug_slice = np.expand_dims(slice, 3)

        # swap channel axis because
        # numpy: H x W x D x C
        # torch: C x D x H x W
        aug_slice = aug_slice.transpose((3, 2, 0, 1))
        
        return {'slice': torch.Tensor(aug_slice)}


class ImagePool():
    # This class is used to define an image pool that 
    # consists of a set of 'fake' scans of a particular scanner
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


def train_net(G_A, G_B, D_A, D_B, trainloader_inp, trainloader_tar_unreg, trainloader_tar_reg, valiloader_inp, valiloader_tar, epochs, log_interval, gpu, log_file):

    # Define list for storing losses after every epoch
    training_losses = []
    validation_losses = []

    # define losses and optimizers
    # Loss function
    if gpu:
        MSE_loss = nn.MSELoss().cuda()
        L1_loss = nn.L1Loss().cuda()
    else:
        MSE_loss = nn.MSELoss()
        L1_loss = nn.L1Loss()

    # optimizers
    G_optimizer = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=0.01)
    D_A_optimizer = torch.optim.Adam(D_A.parameters(), lr=0.01)
    D_B_optimizer = torch.optim.Adam(D_B.parameters(), lr=0.01)

    # schedulers for dynamic learning rate
    G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, 'min', patience=1, verbose=True, threshold=0.08)
    D_A_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(D_A_optimizer, 'min', patience=1, verbose=True, threshold=0.08)
    D_B_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(D_B_optimizer, 'min', patience=1, verbose=True, threshold=0.08)
    
    # Generated image pool
    num_pool = 50
    fake_A_pool = ImagePool(num_pool)
    fake_B_pool = ImagePool(num_pool)

    G_A.train()
    G_B.train()
    D_A.train()
    D_B.train()
    
    for epoch in range(epochs):
        print epoch
        D_A_losses = []
        D_B_losses = []
        G_A_losses = []
        G_B_losses = []
        cycle_A_losses = []
        cycle_B_losses = []
        
        running_D_A_loss = 0
        running_D_B_loss = 0
        running_G_A_loss = 0
        running_G_B_loss = 0
        running_G_loss = 0
        running_cycle_A_loss = 0
        running_cycle_B_loss =0

        # training
        iteration = 0
        for i, (data_inp, data_tar) in enumerate(zip(trainloader_inp, trainloader_tar_unreg)):
            iteration = i

            # input image data
            real_A = data_inp['slice']
            real_B = data_tar['slice']

            if gpu:
                real_A = Variable(real_A.cuda())
                real_B = Variable(real_B.cuda())
            else:
                real_A = Variable(real_A)
                real_B = Variable(real_B)

            # Train generator G_A
            # A -> B
            fake_B = G_A(real_A)
            D_B_fake_decision = D_B(fake_B)
            
            if gpu:
                G_A_loss = MSE_loss(D_B_fake_decision, Variable(torch.ones(D_B_fake_decision.size()).cuda()))
            else:
                G_A_loss = MSE_loss(D_B_fake_decision, Variable(torch.ones(D_B_fake_decision.size())))
            running_G_A_loss += G_A_loss.data[0]

            # forward cycle loss
            recon_A = G_B(fake_B)
            cycle_A_loss = L1_loss(recon_A, real_A) * 5
            running_cycle_A_loss += cycle_A_loss.data[0]

            # Train generator G_B
            # B -> A
            fake_A = G_B(real_B)
            D_A_fake_decision = D_A(fake_A)
            if gpu:
                G_B_loss = MSE_loss(D_A_fake_decision, Variable(torch.ones(D_A_fake_decision.size()).cuda()))
            else:
                G_B_loss = MSE_loss(D_A_fake_decision, Variable(torch.ones(D_A_fake_decision.size())))
            running_G_B_loss += G_B_loss.data[0]

            # backward cycle loss
            recon_B = G_A(fake_A)
            cycle_B_loss = L1_loss(recon_B, real_B) * 5
            running_cycle_B_loss += cycle_B_loss.data[0]

            # Back propagation
            G_loss = G_A_loss + G_B_loss + cycle_A_loss + cycle_B_loss
            running_G_loss += G_loss.data[0]
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # Train discriminator D_A
            D_A_real_decision = D_A(real_A)
            fake_A = fake_A_pool.query(fake_A)
            D_A_fake_decision = D_A(fake_A)

            if gpu:
                D_A_real_loss = MSE_loss(D_A_real_decision, Variable(torch.ones(D_A_real_decision.size()).cuda()))
                D_A_fake_loss = MSE_loss(D_A_fake_decision, Variable(torch.zeros(D_A_fake_decision.size()).cuda()))
            else:
                D_A_real_loss = MSE_loss(D_A_real_decision, Variable(torch.ones(D_A_real_decision.size())))
                D_A_fake_loss = MSE_loss(D_A_fake_decision, Variable(torch.zeros(D_A_fake_decision.size())))

            # Back propagation
            D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
            running_D_A_loss += D_A_loss.data[0]
            D_A_optimizer.zero_grad()
            D_A_loss.backward()
            D_A_optimizer.step()

            # Train discriminator D_B
            D_B_real_decision = D_B(real_B)
            fake_B = fake_B_pool.query(fake_A)
            D_B_fake_decision = D_B(fake_B)
            if gpu:
                D_B_real_loss = MSE_loss(D_B_real_decision, Variable(torch.ones(D_B_real_decision.size()).cuda()))
                D_B_fake_loss = MSE_loss(D_B_fake_decision, Variable(torch.zeros(D_B_fake_decision.size()).cuda()))
            else:
                D_B_real_loss = MSE_loss(D_B_real_decision, Variable(torch.ones(D_B_real_decision.size())))
                D_B_fake_loss = MSE_loss(D_B_fake_decision, Variable(torch.zeros(D_B_fake_decision.size())))

            # Back propagation
            D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
            running_D_B_loss += D_B_loss.data[0]
            D_B_optimizer.zero_grad()
            D_B_loss.backward()
            D_B_optimizer.step()
            
            # loss values
            D_A_losses.append(running_D_A_loss/(i+1))
            D_B_losses.append(running_D_B_loss/(i+1))
            G_A_losses.append(running_G_A_loss/(i+1))
            G_B_losses.append(running_G_B_loss/(i+1))
            cycle_A_losses.append(running_cycle_A_loss/(i+1))
            cycle_B_losses.append(running_cycle_B_loss/(i+1))
            
            if i%log_interval == log_interval-1:
                log_file.write('Epoch [%d/%d], Step [%d/%d], D_A_loss: %.4f, D_B_loss: %.4f, G_A_loss: %.4f, G_B_loss: %.4f \n'
                  % (epoch+1, epochs, i+1, len(trainloader_inp), running_D_A_loss/(i+1), running_D_B_loss/(i+1), running_G_A_loss/(i+1), running_G_B_loss/(i+1)))
        
        # Change learning rate using the scheduler
        G_scheduler.step(running_G_loss/(iteration+1))
        D_A_scheduler.step(running_D_A_loss/(iteration+1))
        D_B_scheduler.step(running_D_B_loss/(iteration+1))


        # Evaluate performance on 10% of training set after each epoch

        G_A.eval()
        G_B.eval()
        D_A.eval()
        D_B.eval()
        training_error = 0
        iteration = 0
        limit = int(len(trainloader_inp) * 0.1) # only evaluate on 10% of the training data
        for i, (data_inp, data_tar) in enumerate(zip(trainloader_inp, trainloader_tar_reg)):
            iteration = i
            if i == limit:
                break
            training_inputs = data_inp['slice']
            training_labels = data_tar['slice']

            if gpu:
                training_inputs, training_labels = Variable(training_inputs.cuda()), Variable(training_labels.cuda())
            else:
                training_inputs, training_labels = Variable(training_inputs), Variable(training_labels)

            training_predictions = G_A(training_inputs)
            
            if gpu:
                training_error += (torch.nn.functional.mse_loss(Variable(training_predictions.data), training_labels)).cuda().data[0]
            else:
                training_error += (torch.nn.functional.mse_loss(Variable(training_predictions.data), training_labels)).data[0]
            
        training_error /= iteration + 1
        training_losses.append(training_error)        
        log_file.write('\nTraining loss iteration %d = %.5f \n' % (epoch+1, training_error ))

  
        # evaluate the performance on the full validatio set
        validation_error = 0
        iteration = 0
        for i, (vali_inp, vali_tar) in enumerate(zip(valiloader_inp, valiloader_tar)):
            iteration = i
            validation_inputs = vali_inp['slice']
            validation_labels = vali_tar['slice']
            
            if gpu:
                validation_inputs, validation_labels = Variable(validation_inputs.cuda()), Variable(validation_labels.cuda())
            else:
                validation_inputs, validation_labels = Variable(validation_inputs), Variable(validation_labels)

            validation_predictions = G_A(validation_inputs)
            
            if gpu:
                validation_error += (torch.nn.functional.mse_loss(Variable(validation_predictions.data), validation_labels)).cuda().data[0]
            else:
                validation_error += (torch.nn.functional.mse_loss(Variable(validation_predictions.data), validation_labels)).data[0]
            
        validation_error /= iteration + 1
        log_file.write('MSE on validation set: %f \n\n' % (validation_error))
        validation_losses.append(validation_error)
 
    log_file.write("Completed training \n\n")
        
    return  (training_losses, validation_losses)



def get_predictions(G_A, testloader, gpu):
    # This function uses a trained cycle-GAN to generate predictions for the input slices that are contained within the training set.
    # The predictions are returned in the form of a numpy array
    G_A.eval()
    for i, test_data in enumerate(testloader, 0):
        real_A = test_data['slice']
        if gpu:
            real_A = Variable(real_A.cuda())
        else:
            real_A = Variable(real_A)

        # A -> B
        test_predictions = G_A(real_A)

        #store the predictions in a numpy array which we can reshape later
        if(i == 0):
            predictions = test_predictions.data.cpu().numpy() 
        else:
            predictions = np.concatenate((predictions, test_predictions.data.cpu().numpy()), axis=0)

    # Transpose predictions from (NxCxDxHxW) to (NxCxHxWxD)
    predictions = np.transpose(predictions, (0,1,3,4,2))
    # Remove the channel dimension
    predictions = np.squeeze(predictions, 1)
    return predictions


def cyc_gan_run(slice_size, training_data, validation_data, testing_data_slices, testing_data_scans, affine_mat, epochs=100, train=True, restore_model=False):

    # training_data is a tuple of (training_inputs, training_outputs_reg, training_outputs_unreg)
    # validation_data is a tuple of (validation_inputs, validation_labels)
    # testing_data_slices is a tuple (testing_inputs, testing_target1, testing_target2)
    # testing_data_scans is a tuple (testing_scans_inp, testing_scans_target1, testing_scnas_target2, testing_scans_slices_inp, testing_scans_slices_out1, testing_scans_slices_out2)

    # First check if the current machine has GPU support
    gpu = torch.cuda.is_available()

    # Create a log file, append all output to the log file
    log_file = open("models/cycle_gan/logs.txt","a+")

    # Define path for storing visualisations
    viz_path =  "models/cycle_gan/visualisations"

    # Define path for storing predicted scans
    save_to = "models/cycle_gan/predictions"

    G_A = Generator()
    G_B = Generator()
    D_A = Discriminator()
    D_B = Discriminator()

    if restore_model == False:
        G_A.normal_weight_init()
        G_B.normal_weight_init()
        D_A.normal_weight_init()
        D_B.normal_weight_init()

    else:
        # Restore the existing model
        G_A.load_state_dict(torch.load('models/cycle_gan/trained_ga.pth'))
        G_B.load_state_dict(torch.load('models/cycle_gan/trained_gb.pth'))
        D_A.load_state_dict(torch.load('models/cycle_gan/trained_da.pth'))
        D_B.load_state_dict(torch.load('models/cycle_gan/trained_db.pth'))

    if gpu:
        G_A.cuda()
        G_B.cuda()
        D_A.cuda()
        D_B.cuda()  

    if train == True:
        # Generate a MRIdataset object using the training, validatoin and testing data
        training_inputs = MRIdataset(training_data[0], transform=transforms.Compose([To_Tensor()]))
        training_targets_reg = MRIdataset(training_data[1], transform=transforms.Compose([To_Tensor()]))
        training_targets_unreg = MRIdataset(training_data[2], transform=transforms.Compose([To_Tensor()]))

        validation_inputs = MRIdataset(validation_data[0], transform=transforms.Compose([To_Tensor()]))
        validation_targets = MRIdataset(validation_data[1], transform=transforms.Compose([To_Tensor()]))

        testing_inputs = MRIdataset(testing_data_slices[0], transform=transforms.Compose([To_Tensor()]))

        # Create Dataloader objects from the datasets (required by pytorch)
        trainloader_inp = DataLoader(training_inputs, batch_size=160, shuffle=False, num_workers=16)
        trainloader_tar_reg = DataLoader(training_targets_reg, batch_size=160, shuffle=False, num_workers=16)
        trainloader_tar_unreg = DataLoader(training_targets_unreg, batch_size=160, shuffle=False, num_workers=16)
        valiloader_inp = DataLoader(validation_inputs, batch_size=160, shuffle=False, num_workers=16)
        valiloader_tar = DataLoader(validation_targets, batch_size=160, shuffle=False, num_workers=16)
        testloader_inp = DataLoader(testing_inputs, batch_size=160, shuffle=False, num_workers=16)

        # Train the network
        print ("Training cycle-GAN")
        training_losses, validation_losses = train_net(G_A, G_B, D_A, D_B, trainloader_inp, trainloader_tar_unreg, trainloader_tar_reg, \
                                                        valiloader_inp, valiloader_tar, epochs, 1, gpu, log_file)
        print ("Training completed\n")

        # Generate a learning curve plot 
        plot_losses(training_losses, validation_losses, "Learning Curves", viz_path)

        # Store the parameters of the trained model
        torch.save(G_A.state_dict(), 'models/cycle_gan/trained_ga.pth')
        torch.save(G_B.state_dict(), 'models/cycle_gan/trained_gb.pth') 
        torch.save(D_A.state_dict(), 'models/cycle_gan/trained_da.pth') 
        torch.save(D_B.state_dict(), 'models/cycle_gan/trained_db.pth') 

        print ("Generating predictions for test slices")
        # Get predictions of the test slices using the trained cycle-GAN
        test_predictions = get_predictions(G_A, testloader_inp, gpu)

        # Generate error plots of predictions
        bland_altman_error(test_predictions, np.asarray(testing_data_slices[1]), np.asarray(testing_data_slices[2]), "Predictoin error plot", viz_path, "GAN")

         # Generate plot of prediction vs target
        bland_altman_prediction(test_predictions, np.asarray(testing_data_slices[1]), np.asarray(testing_data_slices[2]), "Predicted VS target", viz_path, "GAN")

        # Get performance statistics
        print ("Evaluating performance of cycle GAN on test slices\n")
        log_file.write("\n\n\nEvaluating performance of cycle GAN on test slices\n")
        get_errors(test_predictions, np.asarray(testing_data_slices[1]), np.asarray(testing_data_slices[2]), "slices", log_file)
        # Get the errors between the 2 sets of target slices
        log_file.write("\n\n\nCalculating errors between target slices\n")
        get_errors_between_targets(np.asarray(testing_data_slices[1]), np.asarray(testing_data_slices[2]), "slices", log_file)


    # Evaluate the performance of the trained cycle-GAN on unseen scans
    final_testing_inp = MRIdataset(testing_data_scans[3], transform=transforms.Compose([To_Tensor()]))
    final_testloader_inp = DataLoader(final_testing_inp, batch_size=160, shuffle=False, num_workers=16)

    # Generate predictions for the unseen test scans using the trained cycle-GAN
    print ("Generating predictions for unseen test scans")
    final_test_predictions = get_predictions(G_A, final_testloader_inp, gpu)
    # Reconstruct the predicted slices into full scans
    final_predicted_brains = reconstruct_brain_from_slices(final_test_predictions, testing_data_scans[0], slice_size)
    # Get performance statistics
    print ("Evaluating performance of cycle-GAN on unseen test scans")
    log_file.write("\n\n\nEvaluating performance of cycle-GAN on unseen test scans\n")
    get_errors(final_predicted_brains, testing_data_scans[1], testing_data_scans[2], "scan", log_file)
    # Get the errors between the 2 sets of target scans
    log_file.write("\n\n\nCalculating errors between target scans\n")
    get_errors_between_targets(testing_data_scans[1], testing_data_scans[2], "scan", log_file)
    # Store the predicted brain scans
    print ("Storing the predicted scans")
    for i, (predicted_scan, scan_affine) in enumerate(zip(final_predicted_brains, affine_mat)):

        predicted_scan_image = nib.Nifti1Image(predicted_scan, scan_affine)
        nib.save(predicted_scan_image, save_to + "/predicted_scan_" + str(i) + ".nii.gz" )

    log_file.close()


