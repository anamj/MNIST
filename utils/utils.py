# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:48:27 2021

@author: Ana Mantecon
Original file modified from https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py
"""
import json
import logging
import os
import shutil

import torch

class LoadConfig():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
        
    
def set_logger(log_file):
    """Set the logger to log info in terminal and file `log_file`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `log_file`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_file (string): Absolute path to log file where info will be saved
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Save log info in a new folder only if log_file is different from the
    # previuos log_file. Otherwise, save it in the already generated handler
    if logger.handlers:
        remove_handler = True
        for h in range(len(logger.handlers)):
            if logger.handlers[h].__class__.__name__ == 'FileHandler':
                handler_name = logger.handlers[h].baseFilename
                if handler_name == log_file:
                    remove_handler = False
                    
        if remove_handler:
            logger.handlers = []

    # Logging to a file
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
    
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    
    Args:
        d (dict) : Float-castable values (np.float, int, float, etc.)
        json_path (string):  Path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last_model'. If is_best==True, also saves
    checkpoint + 'best_model'
    
    Args:
        state(dict): Contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best(bool) : True if it is the best model seen till now
        checkpoint(string):  Folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last_model')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_model'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    
    Args:
        checkpoint(string):  Filename which needs to be loaded
        model(torch.nn.Module):  Model for which the parameters are loaded
        optimizer(torch.optim) : (Optional) Resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint