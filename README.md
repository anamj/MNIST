# MNIST challenge 
## INTRODUCTION

This code is a framework for running experiments using the MNIST digit database with
digits from [0-9]. The goal is to have a framework where different models can be 
evaluated and compared (for the moment only a CNN has been implemented).

The goal of this task is to predict the digits in each image. The complete database 
can be obtained here http://yann.lecun.com/exdb/mnist/. For our case, the database partitions
provided by Kaggle have been used for trainig and testing, so as to participate in the competition. 
The original test partition (downloaded from torchvision) has also been used for evaluating the model
and obtainig the confusion matrix and final performance metrics.  

The kaggle database consists on:
- Training data : 42000 samples of labelled digits
- Test data     : 28000 samples of unlabelled digits. 

The original database partition consists on:
- Training data : 60000 samples of labelled digits
- Test data     : 10000 samples of unlabelled digits. 

## CODE STRUCTURE

    ├── config                     # Folder with configuration files
    │   │                  
    │   ├── default_config.json    # Default experiment configuration          
    │   ├── environment.yml        # Python environment(NEXT!)
    │
    │
    ├── data                       # Folder (python package) where the data realted functions are defined
    │   │
    │   ├── MNISTDataset.py
    │   ├── data_exploration.py     
    │
    ├── evaluation                 # Folder (python package) where the evaluation functions are defined
    │   │
    │   ├── CNN_eval.py
    │
    ├── model                      # Folder (python package) with the model definitions
    │   │
    │   ├── CNN_def.py             # CNN definition
    │   ├── EarlyStopping.py       # Class to manage when to stop the training of the model
    │
    ├── training                   # Folder (python pakage) with the training functions of each model
    │   │
    │   ├── CNN_train.py           # CNN training
    │
    ├── utils                      # Folder with utils functions
    │   │
    │   ├── utils.py
    │
    ├── visualization              # Folder with general visualization functions and metrics
    │   │
    │   ├── metrics.py
    │   ├── data_plots.py
    │   ├── performance_plots.py   
    │
    ├── .gitignore                 # For ignoring pycache during commits
    ├── LICENSE                    # MIT License 
    ├── main_explore_data.py       # Main script for exploring the data
    ├── main.py                    # Main script where experiments are launched from
    └── README.md                  # This file
    
    
    
    