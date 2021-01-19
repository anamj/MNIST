# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:04:43 2021

@author: Ana Mantecon
Exploratory Data Analysis
"""

import argparse
import os

import utils.utils as utils
from data_definition.data_exploration import read_data, check_null, check_type, check_range, check_balance
from visualization.data_plots import plot_data

def main(config_file):
    """ Data exploration main script

    Args:
        config_file (str): The file location of the json configuration file
            that defines the data and main variables that wants to be explored. Default 
            config is in config\default_config.json
    """
    #Configuration
    # Load the configuration from json file
    assert os.path.isfile(
        config_file), "No json configuration file found at {}".format(config_file)
    config = utils.LoadConfig(config_file)
    
    # config running plots (defalut is false)
    display_plots = config.data_exploration['display_plots']
    
    # config display first images_to_plot images (default is 20)
    images_to_plot = config.data_exploration['images_to_plot']
    
    #Data to explore
    input_data = config.data_exploration['data_to_explore']
    
    #Read data
    data = read_data(input_data)
    #Checks on data
    check_null(data)
    check_type(data, 'int64')
    check_range(data,0,255)
    check_balance(data, display_plots)
    
    #Plot images of training database
    if display_plots:
        plot_data(data, images_to_plot)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments of data exploration main script')
    
    parser.add_argument('--config_file', dest='config_file',
                        default=r"C:\Users\Ana\Documents\Projects\MNIST\code\config\default_config.json",
                        help='Configuration file')

    args = parser.parse_args()
    main(args.config_file)