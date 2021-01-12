############################## MNIST challenge ###############################
##############################################################################
INTRODUCTION
##############################################################################

This code is a framework for running experiments using the MNIST digit database with
digits from [0-9]. The goal is to a framework where different models can be quickly 
evaluated and compared (for the moment only a CNN has been implemented).

The goal of this task to predict the digits in each image. This database was obtaiend though
http://yann.lecun.com/exdb/mnist/

The database consists on:
    - Training data : 60000 sample of labelled digits
    - Test data     : 10000 sample of labelled digits. 

##############################################################################
CODE STRUCTURE
##############################################################################
    |
    |-- config: Folder with configuration files
    |   |
    |   |-- default_config.json
    |   |
    |   |-- environment.yml : Python environemnt 
    |
    |
    |-- data_definition: Folder (python package) where the data loader is defined
    |   |
    |   |--MNISTDataset.py
    |
    |
    |-- evaluation: Folder (python package) where the evaluation functions are defined
    |   |
    |   |--CNN_eval.py
    |
    |
    |--model_definition: Folder (python package) with the model definitions
    |   |
    |   |--CNN_def.py: CNN definition
    |
    |
    |--training: Folder (python pakage) with the training functions of each model
    |   |
    |   |--CNN_train.py: CNN training
    |
    |
    |--utils: Folder with utils functions
    |   |
    |   |--utils.py 
    |
    |
    |-- visualization: Folder with general visualization functions and metrics (NEXT)
    |
    |
    |-- main.py: Main script where experiments are launched from
    
    