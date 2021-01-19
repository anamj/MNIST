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
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class MNISTDatasetTrain(Dataset):
    """MNIST dataset training data"""

    def __init__(self, csv_file, transform=None):
        """Class that handels the training data csv of MNIST dataset provided by Kaggle
        
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data = pd.read_csv(csv_file)
        self.input_images = data.iloc[:,1:]
        self.labels = data.iloc[:,0]
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
            sample (dict): Training data sample corresponding to idx position of
                           the database
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.input_images.iloc[idx,:]
        image = image.values.reshape(1,28,28)
        label = self.labels.iloc[idx]

        sample = {'image': image, 'labels': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class NormalizeAndToTensorTrain(object):
    """ Convert ndarrays in sample to Tensors and normalize data to have 
        range [0,1]. Defined for training data.
    """
    def __call__(self, sample):
        """ 
            Args:
                sample (dict): Training data sample
        """
        image, labels = sample['image'], sample['labels']

        return {'image': torch.from_numpy(image/255),
                'labels': torch.from_numpy(np.asarray(labels))}

class MNISTDatasetTest(Dataset):
    """MNIST dataset evaluation data"""

    def __init__(self, csv_file, transform=None):
        """
        Class that handels the test data csv of MNIST dataset provided by Kaggle
        
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data = pd.read_csv(csv_file)
        self.input_images = data
        self.transform = transform

    def __len__(self):
        """        
        Returns:
            (int): length of evaluation database
        """
        return len(self.input_images)
    
    def __getitem__(self, idx):
        """ Gets item of the database.
            This is memory efficient because all the images are not stored in the 
            memory at once but read as required.
            
        Args:
            idx (int): Position in the database of the sample that wants to be requested
            
        Returns:
            sample (dict): Evaluation data sample corresponding to idx position of
                           the database
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.input_images.iloc[idx,:]
        image = image.values.reshape(1,28,28)

        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

class NormalizeAndToTensorTest(object):
    """ Convert ndarrays in sample to Tensors and normalize data to have 
        range [0,1]. Defined for evaluation data.
    """

    def __call__(self, sample):
        image = sample['image']

        return {'image': torch.from_numpy(image/255)}

def split_train_val_partition(dataset, split_train_percentage):
    
    train_size = round(len(dataset)*split_train_percentage)
    val_size = len(dataset) - train_size
    train, val = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    return train, val