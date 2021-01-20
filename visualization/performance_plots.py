# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:55:41 2021

@author: Ana Mantecon
Performance plots
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_train_val(train_metric, val_metric, save_dir, metric_name, model_name, text_points=False):
    """ Plots training and validation performance metrics against the number of
        epochs in the same plot
           
        Args:
            train_metric (list): training metric that wants to be displayed 
            val_metric (list): validation metric that wants to be displayed
            save_dir (str): output directory where the figure will be saved
            metric_name (str): metric name
            model_name (str): model name
    """
    assert len(train_metric)==len(val_metric), 'Missmatch in train and validation metric sizes. Training performace plot can not be displayed'
    
    num_epoch = list(i for i in range(len(train_metric)))

    plt.clf()
    
    plt.figure()
    plt.plot(num_epoch,train_metric,'o-')
    plt.plot(num_epoch,val_metric,'o-')
    plt.legend(labels=('training','validation'))
    plt.xlabel('Epoch number')
    plt.xticks(np.arange(min(num_epoch), max(num_epoch)+1, 1.0))
    plt.ylabel(metric_name)
    plt.title(model_name+' training performance') 
    plt.grid()
    
    if metric_name == 'loss':
        plt.ylim([0,max(train_metric+val_metric)+0.1])
    elif metric_name == 'accuracy':
        plt.ylim([min(train_metric+val_metric)-1,100])
        
    #Adding labels to line points - train line
    if text_points:
        for x,y in zip(num_epoch,train_metric):
        
            label = "{:.2f}".format(y)
        
            plt.annotate(label, # this is the text
                         (x,y), # this is the point to label
                         ha='left',     # horizontal alignment can be left, right or center
                         va='bottom') # vertical alignment
        
        # validation line
        for x,y in zip(num_epoch,val_metric):
        
            label = "{:.2f}".format(y)
        
            plt.annotate(label, # this is the text
                         (x,y), # this is the point to label
                         ha='left',     # horizontal alignment can be left, right or center
                         va='top') # vertical alignment

    #Save figure
    filename = os.path.join(save_dir, model_name+'_train_performance_'+metric_name+'.png')
    plt.savefig(filename,dpi=300)
    
    plt.show()
    
def plot_confusion_matrix(c_matrix, classes, model_name, save_dir):
    """ Plots confusion matrix
           
        Args:
            c_matrix (array): confusion matrix Size #classes x #classes
            classes (array): output classes of the data
            model_name (str): model name
            save_dir (str): output directory where the figure will be saved
    """

    # Plot confusion matrix in a beautiful manner
    plt.figure(figsize=(10, 6))
    ax= plt.subplot()
    sns.heatmap(c_matrix, annot=True, ax = ax, fmt = 'g', cmap='YlOrBr'); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('True Labels', fontsize=13)
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=0)
    ax.xaxis.set_ticklabels(classes, fontsize = 10)
    ax.xaxis.tick_top()
    
    ax.set_ylabel('Predicted Labels', fontsize=13)
    ax.yaxis.set_ticklabels(classes, fontsize = 10)
    plt.yticks(rotation=0)
    
    plt.title(model_name+' Confusion Matrix', fontsize=18)
    
    #Save figure
    filename = os.path.join(save_dir, model_name+'_confusion_matrix.png')
    plt.savefig(filename,dpi=300)
    
    plt.show()