# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:10:50 2020

@author: Ana ana.mantecon.jene@gmail.com
MNIST dataset challenge - reading the training data
"""

import argparse
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def read_data(input_data):
    data = pd.read_csv(input_data)
    #Exploration of the data
    print("Reading database...")
    print("Database has "+str(data.shape[0])+" data points (images)")
    print("Each image has size " + str(math.sqrt(data.shape[1]-1))+"x"+str(math.sqrt(data.shape[1]-1)))
    
    return data

def train_val_partition(data):
    X = data.iloc[:,1:]
    y = data.iloc[:,1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=1)
    
    return X_train, X_val, y_train, y_val

def check_null(data):
    data.isnull().values.any()

def check_type(data):
    #Check if there are values that are not numbers
    if (data.dtypes != 'int64').any():
        problematic_col=(data.dtypes != 'int64').index.values
        data_badType=data.loc[:,problematic_col]
        print("There are "+str(data_badType.shape[1])+" columns with the wrong data type")
    else:
        print("All data has the correct type")

def check_range(data):
    #Range of the data
    min_data=data.iloc[:,1:].min().min()
    max_data=data.iloc[:,1:].max().max()
    print("Range of the data is "+str(min_data)+" to "+str(max_data))

    #Check if there values out of the expected range [0-255]
    data_outRange=data[(data.iloc[:,1:]<0).any(1)] 
    data_outRange.append(data[(data.iloc[:,1:]>255).any(1)]) 
    if data_outRange.shape[0]>0:
        print("There are "+str(data_outRange.shape[0])+" rows out of range")
    else:
        print("There is no data out of range")

def check_balance(data,plots):
    #Percentage of each output label, how balanced is the dataset?
    out_label=data['label'].value_counts()
    output_info=pd.DataFrame({'label': out_label.index.values.tolist(),
                          'num' : out_label})
    for i in range(len(out_label)):
        output_info.loc[i,'percent'] = (output_info['num'][i]/sum(output_info['num']))*100
        #sort index
        output_info=output_info.sort_index()
    
    if plots:
        plot_balance(output_info)
    
    #return output_info
    
    # Missing formula to state if data is balanced or not
    #round(output_info.loc[:,'percent'].values,2)

def plot_balance(output_info):
    plt.figure()
    plt.pie(output_info.loc[:,'percent'].values, labels=output_info.loc[:,'label'].values,
        startangle=90, counterclock=False)
    plt.title('Percent of presence of each output label in training dataset')
    plt.show()

def plot_data(data, plot_img):
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

def main(input_data, plot_img, plots):
    #Activate running plots (defalut is false)
    plots = True
    # Display first 15 images (default is 20)
    plot_img = 15
    
    #Read data
    data = read_data(input_data)
    #Checks on data
    check_null(data)
    check_type(data)
    check_range(data)
    check_balance(data, plots)
    
    #Plot first 15 images of training database
    if plots:
        plot_data(data, plot_img)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments of trainig data module')
    
    parser.add_argument('--input_data', dest='input_data', type=str, 
                        default=r"C:\Users\Ana\Documents\Projects\data\MNIST\train\train.csv",
                        help='CSV containing MNIST training data')
    parser.add_argument('--plot_img', dest='plot_img', default=20,
                        help='Can be an int or a list')
    parser.add_argument('--plots', dest='plots', type=bool, default=False,
                        help='Display plots or not')
    args = parser.parse_args()
    main(args.input_data,args.plot_img,args.plots)