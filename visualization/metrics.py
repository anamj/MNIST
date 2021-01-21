# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:05:01 2021

@author: Ana Mantecon
Metrics definition
"""
import logging
import pandas as pd
import numpy as np

def accuracy(eval_out):   
    """ Computes the overall accuracy of the model
           
        Args:
            eval_out (DataFrame): With columns Label (predicted) and TrueLabel
            
        Returns:
            accuracy (int): Overall accuracy of the model for the given data
    """
    
    correct = (eval_out.TrueLabel == eval_out.Label).sum()
    total   = eval_out.shape[0]
    
    accuracy = 100*correct/total
    
    logging.info('Total Accuracy is '+'{0:.2f}%'.format(accuracy))
    
    return accuracy

def error_rate(eval_out):   
    """ Computes the overall error rate of the model
           
        Args:
            eval_out (DataFrame): With columns Label (predicted) and TrueLabel
            
        Returns:
            error_rate (int): Overall accuracy of the model for the given data
    """
    
    errors = (eval_out.TrueLabel != eval_out.Label).sum()
    total   = eval_out.shape[0]
    
    error_rate = 100*errors/total
    
    logging.info('Total Error Rate is '+'{0:.2f}%'.format(error_rate))
    
    return error_rate

    
def accuracy_per_class(eval_out, classes):
    """ Computes the accuracy per class to get an overall idea of which classes
        performed better than others. This calculation only takes into account the
        data whose true label is the one of the class for the calculation (it is not
        the real accuracy per class, that one is calculated with the confusion matrix)
           
        Args:
            eval_out (DataFrame): With columns Label (predicted) and TrueLabel
            classes (array): With the classes we are trying to predict
            
        Returns:
            class_accuracy (list): 'Accuracy' per class of the model for the given data
    """
    
    #What are the classes that performed well, and the classes that did not perform well
    
    class_accuracy = list(0. for i in range(len(classes)))
       
    for i in classes:
        
        eval_class = eval_out[eval_out.TrueLabel == classes[i]]
        
        class_correct = (eval_class.TrueLabel == eval_class.Label).sum()
        class_total   = eval_class.shape[0]
        
        class_accuracy[i] = 100*class_correct/class_total
        
        print('Accuracy of class {0:2d} : {1:.2f}%'.format(classes[i],class_accuracy[i]))
    
    return class_accuracy
    
    
def confusion_matrix(eval_out, classes):
    """ Computes the confusion matrix of the model for the given data. X-axis represents
        the true labels and Y-axis represents the predicted labels.
           
        Args:
            eval_out (DataFrame): With columns Label (predicted) and TrueLabel
            classes (array): With the classes we are trying to predict
            
        Returns:
            c_matrix (array): Confusion matrix Size #classes x #classes
    """
    
    c_matrix = np.empty([10,10], dtype=float) 
    
    for i in range(len(classes)):
        eval_class = eval_out[eval_out.TrueLabel == classes[i]]
        class_correct = (eval_class.TrueLabel == eval_class.Label).sum()
        
        c_matrix[i,i] = class_correct
        
        left_class = np.delete(classes,i)
        
        # X-axis is true labels
        # Y-axis is predicted labels
        for j in left_class:
            c_matrix[j,i] = (eval_class.Label == j).sum()
    
    return c_matrix
            

def confusion_matrix_metrics(c_matrix):
    """ Computes the main metrics that represent the performance of the model for
        the given data, using the confusion matrix computed for such data.
           
        Args:
            c_matrix (array): Confusion matrix Size #classes x #classes
            
        Returns:
            metrics_per_class (DataFrame): Table with the performance results per class, it
                returns 'TP','TN','FP','FN','accuracy','miss','precision','recall',
                'TrueNegativeRate','FAR','FRR' and 'F1_score' 
            metrics_overall (DataFrame): Table with the overall precision, recall and F1 score
                results computed in three different ways 'micro', 'macro' and 'weighted'
    """
    
    dataSamples = np.empty(10, dtype=float) 
    
    TP = np.empty(10, dtype=float) 
    TN = np.empty(10, dtype=float) 
    FP = np.empty(10, dtype=float) 
    FN = np.empty(10, dtype=float) 
    
    accuracy  = np.empty(10, dtype=float) 
    miss      = np.empty(10, dtype=float) 
    precision = np.empty(10, dtype=float) 
    recall    = np.empty(10, dtype=float) 
    TNR       = np.empty(10, dtype=float) 
    FAR       = np.empty(10, dtype=float) 
    FRR       = np.empty(10, dtype=float) 
    F1_score  = np.empty(10, dtype=float) 
    
    for i in range(c_matrix.shape[0]):
        
        #Data samples per class
        dataSamples[i] = c_matrix[:,i].sum()
        
        #True positives
        TP[i] = c_matrix[i,i]
        #True negatives
        c_matrix_TN = np.delete(c_matrix,i,0)
        c_matrix_TN2 = np.delete(c_matrix_TN,i,1)
        TN[i] = c_matrix_TN2.sum()
        
        # Error 1: False Positives (False acceptance/alarm - FA)
        c_matrix_FP = c_matrix[i,:]
        c_matrix_FP2 = np.delete(c_matrix_FP,i)
        FP[i] = c_matrix_FP2.sum()
        # Error 2: False Negative (False Reject -  FR)
        c_matrix_FN = c_matrix[:,i]
        c_matrix_FN2 = np.delete(c_matrix_FN,i)
        FN[i] = c_matrix_FN2.sum()
        
        #Accuracy
        accuracy[i] = (TP[i]+TN[i])/(TP[i]+TN[i]+FP[i]+FN[i])
        
        #Missclasification rate
        miss[i] = (FN[i]+FP[i])/(TP[i]+TN[i]+FP[i]+FN[i])
        
        #Precision: fraction of predictions as a positive class were actually positive
        precision[i] = TP[i]/(TP[i]+FP[i])
        
        #Recall: fraction of all positive samples were correctly predicted as positive 
        #by the classifier. It is also known as True Positive Rate (TPR), Sensitivity, 
        #Probability of Detection
        recall[i] = TP[i]/(TP[i]+FN[i])
        
        # True Negative Rate (TNR): fraction of all negative samples are correctly 
        #predicted as negative by the classifier
        TNR[i] = TN[i]/(TN[i]+FP[i])
        
        #FAR - Error 1
        FAR[i] = FP[i]/(TN[i]+FP[i])
        
        #FRR - Error 2
        FRR[i] = FN[i]/(FN[i]+TP[i])
        
        #F1 score
        F1_score[i] = 2*((precision[i]*recall[i])/(precision[i]+recall[i])) 
    
    metrics_per_class = pd.DataFrame({'TP': TP, 
                                      'TN': TN,
                                      'FP': FP,
                                      'FN': FN,  
                                      'accuracy': accuracy,
                                      'miss': miss,
                                      'precision': precision,
                                      'recall': recall,
                                      'TrueNegativeRate': TNR,
                                      'FAR': FAR,
                                      'FRR': FRR,
                                      'F1_score': F1_score})  
    
    #Micro averaged score
    TP_total = TP.sum()
    FP_total = FP.sum()
    FN_total = FN.sum()
    
    micro_precision = TP_total/(TP_total+FP_total)
    micro_recall    = TP_total/(TP_total+FN_total)
    micro_F1_score  = 2*((micro_precision*micro_recall)/(micro_precision+micro_recall))
    
    #Macro averaged score
    macro_precision = precision.sum()/len(precision)
    macro_recall    = recall.sum()/len(recall)
    macro_F1_score  = F1_score.sum()/len(F1_score)
    
    #Weighted averaged score
    weighted_precision = np.multiply(dataSamples,precision).sum()/dataSamples.sum()
    weighted_recall    = np.multiply(dataSamples,recall).sum()/dataSamples.sum()
    weighted_F1_score  = np.multiply(dataSamples,F1_score).sum()/dataSamples.sum()
    
    metrics_overall = pd.DataFrame({'micro_precision': [micro_precision], 
                                    'micro_recall': [micro_recall],
                                    'micro_F1_score': [micro_F1_score],
                                    'macro_precision': [macro_precision],  
                                    'macro_recall': [macro_recall],
                                    'macro_F1_score': [macro_F1_score],
                                    'weighted_precision': [weighted_precision],
                                    'weighted_recall': [weighted_recall],
                                    'weighted_F1_score': [weighted_F1_score]})  
    
    return metrics_per_class, metrics_overall   