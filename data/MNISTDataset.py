# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 20:40:56 2020

@author: Ana Mantecon
MNIST Dataset data loading functions
"""

from __future__ import print_function, division
import torch
import pandas as pd
#from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import datasets
from sklearn.model_selection import train_test_split

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def read_and_format_full_dataset():
    """ Reads data in torchvision, reshapes it for the right format for a pytorch network
        and returns the data
                       
        Returns:
            train (tuple): containig two tensors one with the training data samples and 
            the other with its correspondant labels
            test (tuple): containig two tensors one with the test data samples and 
            the other with its correspondant labels
    """
    datasets.MNIST('../data',train=True, download=True)
    
    test = torch.load('../data/MNIST/processed/test.pt')
    train = torch.load('../data/MNIST/processed/training.pt')
    
    test_X  = test[0].reshape(test[0].shape[0],1,test[0].shape[1],test[0].shape[2])
    test_y  = test[1]
    test = tuple([test_X,test_y])
    train_X = train[0].reshape(train[0].shape[0],1,train[0].shape[1],train[0].shape[2])
    train_y = train[1]
    train = tuple([train_X,train_y])
    
    return train, test

def read_and_format_kaggle_dataset():
    """ Reads data from kaggle, reshapes it for the right format for a pytorch network
        and returns the data
                       
        Returns:
            train (tuple): containig two tensors one with the training data samples and 
            the other with its correspondant labels
            test (tensor): containig the test data samples
    """
    train = pd.read_csv('../data/kaggle/train/train.csv')
    test = pd.read_csv('../data/kaggle/test/test.csv')
    
    train_X = train.iloc[:,1:]
    train_X = train_X.values.reshape(train_X.shape[0],1,28,28)
    train_y = train.iloc[:,0]
    train_X = torch.tensor(train_X)
    train_y = torch.tensor(train_y.values)
    train = tuple([train_X,train_y])
    
    test = test.values.reshape(test.shape[0],1,28,28)  
    test = torch.tensor(test)   
    
    return train, test

def split_train_val_partition(data,split_train_percentage, seed):
    """ Partitions data into two partitions named tain and val
           
        Args:
            data (tuple): Containig two tensors one with the data samples and 
            the other with its correspondant labels
            
        Returns:
            train (tuple): containig two tensors one with the training data samples and 
            the other with its correspondant labels
            val (tuple): containig two tensors one with the validation data samples and 
            the other with its correspondant labels
    """
    train_size = round(len(data[0])*split_train_percentage)
    val_size = len(data[0]) - train_size
        
    train_idx, val_idx = train_test_split(list(range(len(data[0]))), test_size=val_size, random_state=seed)
    
    train_X = data[0][train_idx]
    train_y = data[1][train_idx]
    val_X = data[0][val_idx]
    val_y = data[1][val_idx]

    train = tuple([train_X,train_y])
    val = tuple([val_X,val_y])
    
    return train, val
    
class MNISTDatasetLabels(Dataset):
    """MNIST dataset labelled data"""

    def __init__(self, data, transform=None):
        """Class that handels the training data csv of MNIST dataset provided by Kaggle
        
        Args:
            data (tuple): Containig two tensors one with the data samples and 
            the other with its correspondant labels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_images = data[0]
        self.labels = data[1]
        self.transform = transform

    def __len__(self):
        """        
        Returns:
            (int): length of training database
        """
        return len(self.input_images)
    
    def __getitem__(self, idx):
        """ Gets item of the database.
            This is memory efficient because all the images are not stored in the 
            memory at once but read as required.
            
        Args:
            idx (int): Position in the database of the sample that wants to be requested
            
        Returns:
            sample (dict): Training data sample (tensor) corresponding to idx position of
                           the database
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.input_images[idx]
        label = self.labels[idx]

        sample = {'image': image, 'labels': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

class Normalization(object):
    """ Normalize data to have range [0,1]
    """
    def __call__(self, sample):
        """ 
            Args:
                sample (tensor): Training data sample
        """
        image = sample

        return image/255

class MNISTDatasetNoLabels(Dataset):
    """MNIST dataset non-labelled data"""

    def __init__(self, data, transform=None):
        """Class that handels the training data csv of MNIST dataset provided by Kaggle
        
        Args:
            data (tensor): Containig the data samples 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_images = data
        self.transform = transform

    def __len__(self):
        """        
        Returns:
            (int): length of training database
        """
        return len(self.input_images)
    
    def __getitem__(self, idx):
        """ Gets item of the database.
            This is memory efficient because all the images are not stored in the 
            memory at once but read as required.
            
        Args:
            idx (int): Position in the database of the sample that wants to be requested
            
        Returns:
            sample (dict): Training data sample (tensor) corresponding to idx position of
                           the database
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.input_images[idx]

        sample = {'image': image}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

