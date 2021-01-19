# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:44:48 2020

@author: Ana Mantecon
MNIST dataset challenge - pytoch network CNN definition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNnet(nn.Module):
    """ Convolutional network definition to predicts the digit value contained 
        in an image.
    """

    def __init__(self, config):
        """ Convolutional network definition. The components defined are:
            - Two convolutional layers + two max pooling layers
            - Three fully conected layers
            
        Args:
            config: CNN definition parameters defined in configuration file
            
        Note (useful definitions):
            Stride = The amount by which the filter shifts
            Padding = Amount of pixels added to an image when it is being 
                      processed by the kernel of a CNN.
        """
        super(CNNnet, self).__init__()
        
        # Configuration parameters and default values
        self.image_size = config['image_size'] if config['image_size'] else 28
        
        self.conv1out = config['conv1out'] if config['conv1out'] else 6
        self.conv1kernel = config['conv1kernel'] if config['conv1kernel'] else (3,3)
        self.conv1stride = config['conv1stride'] if config['conv1stride'] else 1
        self.conv1pad = config['conv1pad'] if config['conv1pad'] else 0
        
        self.conv2out = config['conv2out'] if config['conv2out'] else 16
        self.conv2kernel = config['conv2kernel'] if config['conv2kernel'] else (3,3)
        self.conv2stride = config['conv2stride'] if config['conv2stride'] else 1
        self.conv2pad= config['conv2pad'] if config['conv2pad'] else 0
        
        self.maxpoolkernel = config['maxpoolkernel'] if config['maxpoolkernel'] else 2
        
        self.fc1out = config['fc1out'] if config['fc1out'] else 120
        self.fc2out = config['fc2out'] if config['fc2out'] else 84               
        
        # Conolutional layers
        #From the Pytorch documentation on convolutional layers, Conv2d layers expect input with the shape
        # (n_samples, channels, height, width)
        # Each of the convolution layers below have the arguments (input_channels, output_channels, 
        # filter/kernel_size, stride, padding)
        
        self.conv1 = nn.Conv2d(1, self.conv1out, self.conv1kernel, self.conv1stride, self.conv1pad) #same as writting it as: self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(self.conv1out, self.conv2out, self.conv2kernel, self.conv2stride)
        
        # Max pooling layer
        # Each of the pooling layers below have the arguments (kernel_size)
        self.maxpool = nn.MaxPool2d(self.maxpoolkernel)
        
        #Output sizes calculation of convolutional layers
        outsize_conv1    = math.floor(self.calc_conv_out_size(self.image_size,self.conv1kernel,0,self.conv1stride))  #Default 26x26
        outsize_maxpool1 = math.floor(self.calc_conv_out_size(outsize_conv1,self.maxpoolkernel,0,self.maxpoolkernel)) #Default 13x13
        outsize_conv2    = math.floor(self.calc_conv_out_size(outsize_maxpool1,self.conv2kernel,0,self.conv2stride)) #Default 11x11
        outsize_maxpool2 = math.floor(self.calc_conv_out_size(outsize_conv2,self.maxpoolkernel,0,self.maxpoolkernel)) #Default 5x5 -if odd image removes one row for the pooling

        #Linear layers
        # Each of the linear layers below have the arguments (input_channels, output_channels)
        self.fc1 = nn.Linear(self.conv2out * outsize_maxpool2 * outsize_maxpool2, self.fc1out)  # 5*5 from image dimension after second pooling
        self.fc2 = nn.Linear(self.fc1out, self.fc2out)
        self.fc3 = nn.Linear(self.fc2out, 10)

    def forward(self, x):
        """Forward function of the network.
        
        Args:
            x: Input data with shape (n_samples, channels, height, width)
        
        Returns:
            x: Ouput of the network
        """
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        # Pytorch reshape function, by using -1 you are telling it to adjust to the number
        # of columns necessary for your specified rows
        x = x.view(-1, self.num_flat_features(x))
        
        # No dropout implemented for the moment
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        x = F.log_softmax(x, dim=1)
    
        return x

    def num_flat_features(self, x):
        """Calculates the size of the num of features that will be generated as input
        of the first linear layer con the network
        
        Args:    
            x: input image in NxN size
            
        Returns:
            num_features: vector with size [N*N, 1]
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def calc_conv_out_size(self, w, k, p, s):
        """Calculates the output size of a convolutional layer given parameters.
        
        Args:    
            w: is input image size
            k: is kernel size (if square, which usually is)
            p: is padding
            s: is stride
            
        Returns:
            size: [(w−k+2p)/s]+1
        """
        
        size = ((w-k+2*p)/s)+1
        return size
    
def loss_fn(outputs, labels):
    """Computes the cross entropy loss given outputs and labels.
        
    Args:
        outputs: Output of the model (dimension batch_size x 10 (10 classes))
        labels: Correspondent true labels (dimension batch_size, where each 
                element is a value between 0-9)
    
    Returns:
        loss: cross entropy loss for all images in the batch
    
    Note: 
        The standard loss function from pyTorch nn.CrossEntropyLoss() can also be
        used and should turn the same result. 
        This function is an example on how you to easily define a custom loss function. 
    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples