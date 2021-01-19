# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:31:54 2021

@author: Ana Mantecon
Functions for exploring the data 
"""
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from visualization.data_plots import plot_balance

def read_data(input_data):
    """ Read data from a csv
           
        Args:
            input_data (str): File to data
            
        Returns:
            data (DataFrame)
    """
    
    data = pd.read_csv(input_data)
    print("Reading database...")
    print("Database has "+str(data.shape[0])+" data points (images)")
    print("Each image has size " + str(math.sqrt(data.shape[1]-1))+"x"+str(math.sqrt(data.shape[1]-1)))
    
    return data

def train_val_partition(data,test_size):
    """ Partitions data into two partitions named tain and val
           
        Args:
            data (DataFrame): Containig the data samples and its correspondant labels
            
        Returns:
            X_train (DataFrame): Data samples of train partition
            X_val (DataFrame): Data samples of val partition
            y_train (Series): Labels of train partition
            y_val (Series): Labels of val partition
    """
    X = data.iloc[:,1:]
    y = data.iloc[:,1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=1)
    
    return X_train, X_val, y_train, y_val

def check_null(data):
    """ Check if there is NaN values in the data
           
        Args:
            data (DataFrame): Containig the data samples and its correspondant labels
    """
    if data.isnull().values.any():
        print ("Data contains null values. It should be further reviewed")
    else:
        print("Data has no null values")

def check_type(data, dtype):
    """ Check if there are values that are expected
           
        Args:
            data (DataFrame): Containig the data samples and its correspondant labels
    """
    #
    if (data.dtypes != dtype).any():
        problematic_col=(data.dtypes != 'int64').index.values
        data_badType=data.loc[:,problematic_col]
        print("There are "+str(data_badType.shape[1])+" columns with the wrong data type")
    else:
        print("All data has the correct type")

def check_range(data, dmin, dmax):
    """ Check if data has the expected range
           
        Args:
            data (DataFrame): Containig the data samples and its correspondant labels
    """
    
    min_data=data.iloc[:,1:].min().min()
    max_data=data.iloc[:,1:].max().max()
    print("Range of the data is "+str(min_data)+" to "+str(max_data))

    #Check if there values out of the expected range [0-255]
    data_outRange=data[(data.iloc[:,1:]<dmin).any(1)] 
    data_outRange.append(data[(data.iloc[:,1:]>dmax).any(1)]) 
    if data_outRange.shape[0]>0:
        print("There are "+str(data_outRange.shape[0])+" rows out of range")
    else:
        print("There is no data out of range")

def check_balance(data, plot_data=False):
    """ Check the balance of the samples in the data
           
        Args:
            data (DataFrame): Containig the data samples and its correspondant labels
            plot_data (bool): Default is False. If true, plots the balance of the data
        
        Returns:
            output_info (DataFrame): Three columns 
                                            'label': data labels
                                            'num': count of data samples per label
                                            'percent': percentage of data samples per label
    """
    #Percentage of each output label, how balanced is the dataset?
    out_label=data['label'].value_counts()
    output_info=pd.DataFrame({'label': out_label.index.values.tolist(),
                          'num' : out_label})
    for i in range(len(out_label)):
        output_info.loc[i,'percent'] = (output_info['num'][i]/sum(output_info['num']))*100
        #sort index
        output_info=output_info.sort_index()
    
    if plot_data:
        plot_balance(output_info)
    
    # Missing formula to determine numerically if data is balanced or not
    #round(output_info.loc[:,'percent'].values,2)
    
    return output_info
    
    
    