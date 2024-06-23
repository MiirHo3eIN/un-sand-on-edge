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
from enum import Enum
sns.set_style("whitegrid")


import pyshm 

from pyshm.dataloader import dataInitHealthy, dataInitAnomaly
from pyshm.dataShaper import shaper, MeanCentering
from pyshm.scaling import Ztranform , Normalization
from pyshm.filters import LowPassFilter
from pyshm.augmentation import data_augmentation


# Define a few global variables 
class Dataset(Enum):
    train = "train"
    validation = "validation"
    test = "test"
    anomaly = "anomaly"
    parent_path = "../../data"

# Define the configuration of the Script
class config(Enum):
    sequence_length = 100
    stride = 10
    anomaly_range = list(range(1, 10))
    data_range = (-1, 1)
    lpf_cutoff = 25 
    fs = 100
    lpf_order = 5


class torch_loggers(Enum):
    pass 


def get_raw_data(): 
    
    df_train = dataInitHealthy(data_path = Dataset.parent_path, dataset_type= Dataset.train)()  
    df_validation = dataInitHealthy(data_path = Dataset.parent_path, dataset_type= Dataset.validation)()
    df_test = dataInitHealthy(data_path = Dataset.parent_path, dataset_type = Dataset.test)()
    anomalies = dataInitAnomaly(data_path = Dataset.parent_path)()


    return df_train, df_validation, df_test, anomalies


def from_raw_to_normalized(sequence_length, stride): 

    
    # Define the data shaper 
    data_shaper = shaper(sequence_len = sequence_length, stride = stride)
    # Define the Data Transformations for the data
    ztransform = Ztranform()
    minmax_norm = Normalization(feature_range= config.data_range, clip = False)
    
    # Define the low pass filter
    lpf = LowPassFilter(cutoff = config.lpf_cutoff, fs = config.fs, order = config.lpf_order)
    
    # load feather data
    df_train, df_validation, df_test, anomalies = get_raw_data()
    # Extract only a specific axis of accelerometer data to work. 
    x_train, x_validation, x_test = df_train["x"].values, df_validation["x"].values, df_test["x"].values
    x_anomaly = {key: value["x"].values for key, value in anomalies.items()}

    del df_train, df_validation, df_test, anomalies
    # apply_pre_processing
    
    x_tr_shaped = data_shaper(x_train.reshape( -1, 1))
    x_val_shaped = data_shaper(x_validation.reshape( -1, 1))
    x_test_shaped = data_shaper(x_test.reshape( -1, 1))
    x_anomaly_shaped = {key: data_shaper(value.reshape(-1, 1)) for key, value in x_anomaly.items()}

    print(f"Training samples: {x_tr_shaped.shape[0]}")
    print(f"Validation samples: {x_val_shaped.shape[0]}")
    print(f"Test samples: {x_test_shaped.shape[0]}")
    for key, value in x_anomaly_shaped.items():
        print(f"{key} samples: {value.shape[0]}")

    # Apply low pass filtering 
    x_tr = lpf(x_tr_shaped, axis= 1)
    x_val = lpf(x_val_shaped, axis= 1)
    x_test = lpf(x_test_shaped, axis= 1)
    x_anomaly = {key: lpf(value, axis = 1) for key, value in x_anomaly_shaped.items()}

    # Apply z-trandform

    x_tr = ztransform(x_tr, input_type = Dataset.train)
    x_val = ztransform(x_val, input_type = Dataset.validation)
    x_test = ztransform(x_test, input_type = Dataset.test)
    x_anomaly = {key: ztransform(value, input_type = Dataset.test) for key, value in x_anomaly.items()}

    # Apply min-max normalization
    x_tr = minmax_norm(x_tr, input_type = Dataset.train)
    x_val = minmax_norm(x_val, input_type = Dataset.validation)
    x_test = minmax_norm(x_test, input_type = Dataset.test)
    x_anomaly = {key: minmax_norm(value, input_type = Dataset.test) for key, value in x_anomaly.items()}

    return x_tr, x_val, x_test, x_anomaly
    



if __name__ == "__main__":

    x_tr, x_val, x_test, x_anomaly = from_raw_to_normalized(sequence_length= config.sequence_length, stride = config.stride)

    # Data Augmentation
    x_tr_aug = data_augmentation(x_tr)
    
    print(f"Augmented Training shape: {x_tr_aug.shape}")

    # Add Save the data
    # log_data()
