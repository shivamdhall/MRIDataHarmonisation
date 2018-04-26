   '''
This file contains various different classes, each of which 
implement a particular type of Neural Network clock.
These blocks can be grouped together to create different
Deep Neural Network architectures 
'''

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F



class Fully_Connected_Block(torch.nn.Module):
    def __init__(self, input_size, output_size, activation='relu', batch_norm=True):
            super(Fully_Connected_Block, self).__init__()
            self.fc = torch.nn.Linear(input_size, output_size)
            self.batch_norm = batch_norm
            self.bn = torch.nn.BatchNorm1d(output_size)
            self.activation = activation
            self.relu = torch.nn.ReLU(True)
            self.lrelu = torch.nn.LeakyReLU(0.2, True)
            self.tanh = torch.nn.Tanh()

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)
        if self.activation == 'relu':
            return self.relu(out)
        elif self.activation == 'lrelu':
            return self.lrelu(out)
        elif self.activation == 'tanh':
            return self.tanh(out)
        elif self.activation == 'no_act':
            return out

    def initialise_weights(self, mean=0.0, std=0.02):
        torch.nn.init.normal(self.fc.weight, mean, std)
        torch.nn.init.constant(self.fc.bias, 0)


class Conv_Block(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=0, activation='relu', batch_norm=True):
        super(Conv_Block, self).__init__()
        self.conv = torch.nn.Conv3d(input_size, output_size, kernel_size, stride, padding)
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm3d(output_size)
        self.activation = activation
        self.relu = torch.nn.ReLU(True)
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)
        if self.activation == 'relu':
            return self.relu(out)
        elif self.activation == 'lrelu':
            return self.lrelu(out)
        elif self.activation == 'tanh':
            return self.tanh(out)
        elif self.activation == 'no_act':
            return out

    def initialise_weights(self, mean=0.0, std=0.02):
        torch.nn.init.normal(self.conv.weight, mean, std)
        torch.nn.init.constant(self.conv.bias, 0)


class Deconv_Block(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=0, output_padding=0, activation='relu', batch_norm=True):
        super(Deconv_Block, self).__init__()
        self.deconv = torch.nn.ConvTranspose3d(input_size, output_size, kernel_size, stride, padding, output_padding)
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm3d(output_size)
        self.activation = activation
        self.relu = torch.nn.ReLU(True)
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)
        if self.activation == 'relu':
            return self.relu(out)
        elif self.activation == 'lrelu':
            return self.lrelu(out)
        elif self.activation == 'tanh':
            return self.tanh(out)
        elif self.activation == 'no_act':
            return out

    def initialise_weights(self, mean=0.0, std=0.02):
        torch.nn.init.normal(self.deconv.weight, mean, std)


class Resnet_Block(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(Resnet_Block, self).__init__()
        self.conv1 = Conv_Block(num_filter, num_filter, kernel_size, stride, padding, activation='relu', batch_norm=True)
        self.conv2 = Conv_Block(num_filter, num_filter, kernel_size, stride, padding, activation='no_act', batch_norm=True)
        self.relu = torch.nn.ReLU(True)

        self.resnet_block = torch.nn.Sequential(
            self.conv1,
            self.conv2,
        )

    def forward(self, x):
        residual = x
        out = self.resnet_block(x)
        out += residual
        out = self.relu(out)
        return out

    def initialise_weights(self, mean=0.0, std=0.02):
        self.conv1.initialise_weights()
        self.conv2.initialise_weights()

