"""
Created on Tue Dec 22 18:18:28 2020

@author: Ana Mantecon
MNIST dataset challenge - CNN training
"""

import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import utils.utils as utils
from model.EarlyStopping import EarlyStopping

import model.CNN_def as net
from model.CNN_def import init_weights
from evaluation.CNN_eval import validate
from visualization.performance_plots import plot_train_val



def train(model, optimizer, loss_fn, dataloader, config):
    """ Function that trains the model using all the training data
    Args:
        model (torch.nn.Module): The neural network
        optimizer (torch.optim): Optimizer for parameters of model
        loss_fn: Function that computes the loss of the data given the true labels 
                 and the network predictions
        dataloader (torch.utils.data.DataLoader): Points at the training data
        config: Experiment configuration
        
    Returns:
        Writes to log file the accuracy and loss of the model on the
        training data
    """

    # set model to training mode
    model.train()

    # running accumulate object for loss
    running_loss = 0.0
    total = 0.0
    correct = 0.0

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['image'].type(torch.float32)
            labels = data['labels'].type(torch.long)
        
            # move to GPU if available
            if config.cuda:
                inputs, labels = inputs.cuda(
                    non_blocking=True), labels.cuda(non_blocking=True)
                
            # zero the parameter gradients - clear previous gradients
            optimizer.zero_grad()
    
            # forward pass - compute model output
            outputs = model(inputs)
            # loss
            loss = loss_fn(outputs, labels)
            # backward pass - compute gradients of all variables wrt loss
            loss.backward()
            # weights update using calculated gradients
            optimizer.step()   
            
            #Accumulate loss
            running_loss += loss.item()
        
            #Accumulate batch size
            total += labels.size(0)
        
            #Accumulate correct/incorrect outputs for accuracy
            _, predicted_batch = torch.max(outputs.data, 1)
            correct += (predicted_batch == labels).sum().item()
            
            t.update()

    # compute mean of all metrics for summary    
    loss_total = running_loss/total
    accuracy_total = 100 * correct / total
    
    metrics_sumary = {'accuracy':accuracy_total,'loss':loss_total}
    
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_sumary.items())
    logging.info("- Train metrics : " + metrics_string)
    
    return metrics_sumary

def train_wraper(train_dataloader, val_dataloader, config):
    """ Defines the model and performs the training calling to the train function for the
        number of epochs defined in the configuration. Also evaluated the model ath the
        end of each epoch using the validation data
    
    Args:
        train_dataloader (torch.utils.data.DataLoader): Training data
        val_dataloader (torch.utils.data.DataLoader):  Validation data
        config:  Experiment configuration
    """    
    # Define the model and optimizer
    model = net.CNNnet(config.CNN_def).cuda() if config.cuda else net.CNNnet(config.CNN_def)
    optimizer = optim.SGD(model.parameters(), lr=config.CNN_train['learning_rate'], momentum=config.CNN_train['momentum'])
    
    #Initialize model weights
    if config.CNN_train['enable_custom_weight_init']:
        model.apply(init_weights)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss() #net.loss_fn
    
    # Output directory
    out_dir = config.general['out_dir']
    
    # reload weights from restore_file if specified
    if config.general['pre_model']:
        restore_path = os.path.join(config.general['pre_model'] + '.pth.tar')
        logging.info("Pre loading model parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=config.CNN_train['early_stop_epoch'], verbose=True)

    best_val = 0.0
    
    # Save loss and accuracy
    train_loss = list('nan' for i in range(config.CNN_train['num_epochs']))
    val_loss   = list('nan' for i in range(config.CNN_train['num_epochs']))
    train_acc  = list('nan' for i in range(config.CNN_train['num_epochs']))
    val_acc    = list('nan' for i in range(config.CNN_train['num_epochs']))

    for epoch in range(config.CNN_train['num_epochs']):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, config.CNN_train['num_epochs']))

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics=train(model, optimizer, loss_fn, train_dataloader, config)

        # Evaluate for each epoch on validation set
        val_metrics = validate(model, loss_fn, val_dataloader, config)
        
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in val_metrics.items())
        logging.info("- Validation metrics : " + metrics_string)
        
        # Update loss and accuracy lists
        train_loss[epoch] = train_metrics['loss']
        val_loss[epoch]   = val_metrics['loss']
        train_acc[epoch]  = train_metrics['accuracy']
        val_acc[epoch]    = val_metrics['accuracy']
    
        #Update is_best accordingly to new val_acc
        is_best = val_acc[epoch] >= best_val

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=out_dir)

        # If best_eval, update best_val
        if is_best:
            logging.info("- Found new best accuracy")
            best_val = val_acc[epoch]
            best_epoch = epoch
        
        # early_stopping - uses validation loss to check if it has decresed
        early_stopping(val_loss[epoch], model)
        
        if early_stopping.early_stop:
            logging.info("Early stopping")
            #Reshape acc and loss
            train_loss = list(filter(lambda a: a != 'nan', train_loss))
            val_loss   = list(filter(lambda a: a != 'nan', val_loss))
            train_acc  = list(filter(lambda a: a != 'nan', train_acc))
            val_acc    = list(filter(lambda a: a != 'nan', val_acc))
            break
    
    if config.CNN_train['num_epochs'] > 0:
        #Plot training vs validaton performance per epoch
        plot_train_val(train_loss, val_loss, out_dir, best_epoch, 'loss','CNN')
        plot_train_val(train_acc, val_acc, out_dir, best_epoch, 'accuracy','CNN')