__doc__ = r""" 
    This script will generate pre-processed data for unsupervided 
    methods that is developed in this repository. 
    The result of this code would generate .pt samples
    which then can be read by the Torch Dataloaders.  
"""



# Let's import a couple of neccessary packages 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

sns.set_style("whitegrid")


import pyshm 

from pyshm.dataloader import dataInitHealthy, dataInitAnomaly
from pyshm.dataShaper import shaper, MeanCentering
from pyshm.scaling import Ztranform , Normalization
from pyshm.filters import LowPassFilter
from pyshm.augmentation import data_augmentation








def main():

    # Define the data path 
    data_path = "../../data"
    df_train = dataInitHealthy(data_path = data_path, dataset_type="train")()  
    df_validation = dataInitHealthy(data_path = data_path,dataset_type= "validation")()
    df_test = dataInitHealthy(data_path = data_path,dataset_type="test")()
    anomalies = dataInitAnomaly(data_path = data_path )()
    print(df_train)
    print(anomalies.keys())

    exit()


    # Define the sequence length 
    sequence_length = 100
    windpw_stride = 100
    data_range = (-1, 1)
    
    # Define the data shaper 
    data_shaper = shaper(sequence_len = sequence_length, stride = windpw_stride)
    MeanCenter = MeanCentering(axis = 0)
    ztransform = Ztranform()
    minmax_norm = Normalization(feature_range= data_range, clip = False)
    
    # Define the low pass filter 

if __name__ == "__main__":
    main()