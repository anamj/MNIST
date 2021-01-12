# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 17:23:42 2021

@author: ana.mantecon.jene@gmail.com
MNIST dataset challenge - pytoch model evaluation.
Evaluates the model
"""

import torch

def evaluate_with_labels(model, loss_fn, dataloader, config):
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
