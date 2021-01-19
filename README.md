# MNIST challenge 
## INTRODUCTION

This code is a framework for running experiments using the MNIST digit database with
digits from [0-9]. The goal is to have a framework where different models can be quickly 
evaluated and compared (for the moment only a CNN has been implemented).

The goal of this task is to predict the digits in each image. The complete database 
can be obtained here http://yann.lecun.com/exdb/mnist/. For our case, the database
provided by Kaggle has been used, so as to participate in the competition.

The database consists on:
- Training data : 42000 samples of labelled digits
- Test data     : 28000 samples of unlabelled digits. 

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
    │
    ├── .gitignore                 # For ignoring pycache during commits
    ├── main_explore_data.py       # Main script for exploring the data
    ├── main.py                    # Main script where experiments are launched from
    └── README.md                  # This file
    
    
    
    