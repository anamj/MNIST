# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:51:35 2021

@author: Ana Mantecon
Functions for plotting the data
"""

import math
import matplotlib.pyplot as plt

def plot_balance(output_info):
    """ Plots the balance of the samples in the data with a Pie plot
           
        Args:
            output_info (DataFrame): Three columns 
                                            'label': data labels
                                            'num': count of data samples per label
                                            'percent': percentage of data samples per label
    """
    
    plt.figure()
    plt.pie(output_info.loc[:,'percent'].values, labels=output_info.loc[:,'label'].values,
        autopct='%1.1f%%', startangle=90, counterclock=False)
    plt.title('Balance of output labels in training dataset')
    plt.show()

def plot_data(data, plot_img):
    """ Plots first N samples of the dataset or given list of images
           
        Args:
            data (DataFrame): Containig the data samples and its correspondant labels
            plot_imag: This argument can be of two options
                (list) indexes of data samples that want to be plot
                (int) first N samples that want to be plot
    """
    #Print first N images or given list of images
    if type(plot_img) is list:
        N = len(plot_img)
    else:
        N = plot_img    
    x = math.floor(math.sqrt(N))
    y = math.ceil(N/x)
    
    plt.figure()
    for i in range(N):
        row = math.floor(i/y)
        col = i - y*row
    
        if type(plot_img) is list:
            plot_num = plot_img[i]
        else:
            plot_num = i   
    
        plt.subplot2grid((x,y), (row,col))
        plt.imshow(data.iloc[plot_num,1:].to_numpy().reshape((28, 28)), cmap='gray')
        plt.title("True is "+str(data.iloc[plot_num,0]))
        plt.axis('off')
    plt.suptitle("MNIST TRAINING DATA EXAMPLES")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

