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
from pyshm.dataShaper import shaper, MeanCentering
from pyshm.scaling import Ztranform , Normalization
from pyshm.filters import LowPassFilter
from pyshm.augmentation import data_augmentation



def data_init(type:str, anomaly_level:int = 1): 

    match type:
        case "train":
            return [2,3,4]
        case "test":
            return [1] 
        case "validation":
            return [5]
        case "anomaly":
            return [anomaly_level+5]
        case _:
            raise ValueError("Invalid type of data")
        


def main():

    # Define the data path 
    data_path = "../../data"
    training_experiments = data_init("train")
    tesing_experiments = data_init("test")
    validation_experiments = data_init("validation")
    
    df_train = []
    for experiment in training_experiments:
        data_path = f"{data_path}/exp_{experiment}.feather"
        df_train.append(pd.read_feather(f"{data_path}"))


    print("Loaded data")

    df_train_ = pd.concat(df_train) #.sort_values(by = "ts")
    print(df_train_.head())



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