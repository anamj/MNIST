# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:51:46 2021

@author: Ana Mantecon
MNIST dataset challenge - main function. Use this function to run your experiments.
Configure your own experiments following the default template in config\default_config.json
"""
import argparse
import logging
import os

import torch
from torchvision import transforms
import utils.utils as utils

from data.MNISTDataset import MNISTDatasetLabels, MNISTDatasetNoLabels, Normalization
from data.MNISTDataset import read_and_format_full_dataset, read_and_format_kaggle_dataset, split_train_val_partition
from torch.utils.data import DataLoader
from training.CNN_train import train_wraper
from evaluation.CNN_eval import evaluate_return_labels
from visualization.metrics import accuracy, error_rate, confusion_matrix, confusion_matrix_metrics
from visualization.performance_plots import plot_confusion_matrix

def main(config_file):
    """ Main script used for executing experiments (training,evaluating and 
        comparing the models)

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
    
    #Save config file
    config.save(os.path.join(out_dir, 'experiment_config.json'))

    # Set the logger
    utils.set_logger(os.path.join(out_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # Load data
    train, test = read_and_format_full_dataset()
    train_kaggle, test_kaggle = read_and_format_kaggle_dataset()
    
    #Using kaggle's training data for training
    train, val = split_train_val_partition(train_kaggle, config.data['split_train_percentage'],config.general['seed'])
    
    #Adding data augmentation to training
    # train = MNISTDatasetLabels(train,
    #                            transform=transforms.Compose([
    #                                        Normalization(),
    #                                        transforms.RandomHorizontalFlip(0.5),
    #                                        transforms.RandomVerticalFlip(0.5),
    #                                        transforms.RandomPerspective(),
    #                                        transforms.RandomRotation(30)]))  
    
    train = MNISTDatasetLabels(train,
                               transform=transforms.Compose([
                                           Normalization(),
                                           transforms.RandomRotation(15)])) 
    
    val = MNISTDatasetLabels(val,
                           transform=transforms.Compose([Normalization()]))  
    
    test = MNISTDatasetLabels(test,
                           transform=transforms.Compose([Normalization()]))  
    
    test_kaggle = MNISTDatasetNoLabels(test_kaggle,
                           transform=transforms.Compose([Normalization()]))  
    
    train_dataloader       = DataLoader(train, batch_size=config.CNN_train['batch_size'], shuffle=True, num_workers=config.CNN_train['num_workers'])
    val_dataloader         = DataLoader(val, batch_size=config.CNN_train['batch_size'], shuffle=True, num_workers=config.CNN_train['num_workers'])
    test_dataloader        = DataLoader(test, batch_size=config.CNN_train['batch_size'], shuffle=False, num_workers=config.CNN_train['num_workers'])
    test_kaggle_dataloader = DataLoader(test_kaggle, batch_size=config.CNN_train['batch_size'], shuffle=False, num_workers=config.CNN_train['num_workers'])

    logging.info("- done.")
            
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(config.CNN_train['num_epochs']))
    train_wraper(train_dataloader, val_dataloader, config)
    logging.info("- done.")
        
    #Evaluate the model test set 
    # Using Kaggle's test set unknown labels (can have true labels or not (Kaggle's case))
    logging.info("Starting the model evaluation on Kaggle's test data")
    eval_out_kaggle = evaluate_return_labels(test_kaggle_dataloader, config)
    #Save the results
    eval_out_kaggle.to_csv(os.path.join(out_dir, 'test_result_kaggle.csv'),index=False)
    logging.info("- done.")
    
    # Using test set with known labels
    logging.info("Starting the model evaluation on test data")
    eval_out = evaluate_return_labels(test_dataloader, config)
    #Save the results
    eval_out.to_csv(os.path.join(out_dir, 'test_result.csv'),index=False)
    logging.info("- done.")
    
    # Compute metrics
    if 'TrueLabel' in eval_out:
        #Evaluate the model with test set (known labels)
        logging.info("Calculating final metrics")
        # Get unique true labels in dataset
        classes = eval_out.TrueLabel.unique()
        # Sort them
        classes.sort()
        # Calculate accuracy
        accuracy_total = accuracy(eval_out)
        # Calculate error rate
        error_rate_total = error_rate(eval_out)
        # Confussion matrix
        c_matrix       = confusion_matrix(eval_out, classes)
        plot_confusion_matrix(c_matrix, classes, 'CNN', out_dir)
        # Overall metrics
        metrics_per_class, metrics_overall = confusion_matrix_metrics(c_matrix)
        metrics_overall['accuracy_percent'] = accuracy_total
        metrics_overall['error_rate_percent'] = error_rate_total
        
        metrics_per_class.to_csv(os.path.join(out_dir, 'CNN_results_per_class.csv'))
        metrics_overall.to_csv(os.path.join(out_dir, 'CNN_results_overall.csv'))
        
        logging.info("- done.")   
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments of trainig model module')
    
    parser.add_argument('--config_file', dest='config_file',
                        default=r"C:\Users\Ana\Documents\Projects\MNIST\code\config\default_config.json",
                        help='Configuration file')

    args = parser.parse_args()
    main(args.config_file)
    

