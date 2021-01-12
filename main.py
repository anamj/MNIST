# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:51:46 2021

@author: ana.mantecon.jene@gmail.com
MNIST dataset challenge - main function. Use this function to run your experiments.
Configure your own experiments following the default template in config\default_config.json
"""
import argparse
import logging
import os

import torch
from torchvision import transforms
import utils.utils as utils

from data_definition.MNISTDataset import MNISTDatasetTrain, NormalizeAndToTensorTrain, split_train_val_partition
from torch.utils.data import DataLoader
from training.CNN_train import train_wraper

def main(config_file):
    """Gets and prints the spreadsheet's header columns

    Args:
        config_file (str): The file location of the json configuration file
            that defines the experiment that is wants to be executed. Default 
            config is in config\default_config.json
    """
    
    # Load the configuration from json file
    assert os.path.isfile(
        config_file), "No json configuration file found at {}".format(config_file)
    config = utils.LoadConfig(config_file)

    # use GPU if available
    config.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(config.general['seed'])
    if config.cuda:
        torch.cuda.manual_seed(config.general['seed'])
    
    #Generate output path if it does not exist
    out_dir = config.general['out_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Set the logger
    utils.set_logger(os.path.join(out_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # Load data
    dataset = MNISTDatasetTrain(csv_file=config.data['train_file'],
                           transform=transforms.Compose([NormalizeAndToTensorTrain()]))
    #Train and Validation partitions
    train, val = split_train_val_partition(dataset, config.data['split_train_percentage'])
    
    train_dataloader = DataLoader(train, batch_size=config.CNN_train['batch_size'], shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val, batch_size=config.CNN_train['batch_size'], shuffle=True, num_workers=0)
    
    logging.info("- done.")
            
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(config.CNN_train['num_epochs']))
    train_wraper(train_dataloader, val_dataloader, config)
    logging.info("- done.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments of trainig model module')
    
    parser.add_argument('--config_file', dest='config_file',
                        default=r"C:\Users\Ana\Documents\Projects\MNIST\code\config\default_config.json",
                        help='Configuration file')

    args = parser.parse_args()
    main(args.config_file)