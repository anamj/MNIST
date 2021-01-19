# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 17:23:42 2021

@author: Ana Mantecon
MNIST dataset challenge - pytoch model evaluation.
Evaluates the model
"""
import os

import torch
import pandas as pd
from tqdm import tqdm

import utils.utils as utils
import model.CNN_def as net

def validate(model, loss_fn, dataloader, config):
    """Evaluate the model on given a database with known true labels.
    
    Args:
        model (torch.nn.Module): The neural network
        loss_fn: Function that computes the loss of the data given the true labels 
                 and the network predictions
        dataloader (torch.utils.data.DataLoader): Points at the evaluation data
        config: Experiment configuration
    
    Returns:
        metrics_summary (dict): Dictionary with accruacy and loss calculations
                                for the evaluated data
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    loss = 0.0
    total = 0.0
    correct = 0.0

    # compute metrics over the dataset
    for data_batch in dataloader:
        inputs_batch = data_batch['image'].type(torch.float32)
        labels_batch = data_batch['labels'].type(torch.long)

        # move to GPU if available
        if config.cuda:
            inputs_batch, labels_batch = inputs_batch.cuda(
                non_blocking=True), labels_batch.cuda(non_blocking=True)

        # compute model output
        output_batch = model(inputs_batch)
        loss_batch = loss_fn(output_batch, labels_batch)
        
        #Accumulate loss
        loss += loss_batch.item()
        
        #Accumulate batch size
        total += labels_batch.size(0)
        
        #Accumulate correct/incorrect outputs for accuracy
        _, predicted_batch = torch.max(output_batch.data, 1)
        correct += (predicted_batch == labels_batch).sum().item()

    # compute mean of all metrics for summary    
    loss_total = loss/total
    accuracy_total = 100 * correct / total
    
    metrics_sumary = {'accuracy':accuracy_total,'loss':loss_total}
    
    return metrics_sumary

def evaluate_return_labels(dataloader, config):
    """Evaluate the model on given a database with no true labels. Returns the
       predicted outputs
    
    Args:
        dataloader (torch.utils.data.DataLoader): Points at the evaluation data
        config: Experiment configuration
    
    Returns:
        out (DataFrame): With three columns ('ImageId', 'Label'(predicted),'TrueLabel') if
            true label is available in dataloader. Otherwise returns only two columns
            ('ImageId', 'Label'(predicted))
    """
    
    # Define the model
    model = net.CNNnet(config.CNN_def).cuda() if config.cuda else net.CNNnet(config.CNN_def)
    # Load trained model
    model_path = os.path.join(config.general['out_dir'],'best_model')
    utils.load_checkpoint(model_path, model)

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    out = pd.DataFrame([])

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        # compute metrics over the dataset
        for data_batch in dataloader:
            inputs_batch = data_batch['image'].type(torch.float32)
    
            # move to GPU if available
            if config.cuda:
                inputs_batch = inputs_batch.cuda(
                    non_blocking=True)
    
            # compute model output
            output_batch = model(inputs_batch)
            
            #Accumulate correct/incorrect outputs for accuracy
            _, predicted_batch = torch.max(output_batch.data, 1)
            
            #To numpy and to pandas
            #Saving all the data - takes up way more time
            # data             = inputs_batch.numpy().reshape(config.CNN_train['batch_size'],inputs_batch.size()[-1]*inputs_batch.size()[-1])
            # predict_label    = predicted_batch.numpy().reshape(config.CNN_train['batch_size'],1)
            # data_dF          = pd.DataFrame(data)
            # predict_label_dF = pd.DataFrame(predict_label)
            # combine_dF       = pd.concat([predict_label_dF,data_dF],axis=1)
            # out              = out.append(combine_dF,ignore_index=True)
            
            #Only saving predicted output and labels (if exist)
            if 'labels' in data_batch:
                predict_label    = predicted_batch.numpy().reshape(config.CNN_train['batch_size'],1)
                true_lable       = data_batch['labels'].numpy().reshape(config.CNN_train['batch_size'],1)
                out              = out.append(pd.concat([pd.DataFrame(predict_label),pd.DataFrame(true_lable)],axis=1),ignore_index=True)
            else:
                predict_label    = predicted_batch.numpy().reshape(config.CNN_train['batch_size'],1)
                out              = out.append(pd.DataFrame(predict_label),ignore_index=True)
            
            t.update()
        
        # Name columns and reorder for Kaggle's format
        if out.shape[1] == 2:
            out.columns = ['Label','TrueLabel']
            out['ImageId'] = out.index.values+1
            out = out[['ImageId','Label','TrueLabel']]
        else:
            out.columns = ['Label']
            out['ImageId'] = out.index.values+1
            out = out[['ImageId','Label']]
    
    return out
