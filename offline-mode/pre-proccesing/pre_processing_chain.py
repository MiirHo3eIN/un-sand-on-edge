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
from dataclasses import dataclass
import torch
sns.set_style("whitegrid")


import pyshm 

from pyshm.dataloader import dataInitHealthy, dataInitAnomaly
from pyshm.dataShaper import shaper
from pyshm.scaling import Ztranform , Normalization
from pyshm.filters import LowPassFilter
from pyshm.augmentation import data_augmentation
from pyshm.torchLogger import TorchSampleLogger

# Define a few global variables 
@dataclass
class Dataset:
    train = "train"
    validation = "validation"
    test = "test"
    anomaly = "anomaly"
    parent_path = "../../data"
    model_type = "dl" # 'ml' or "dl" or "sysid"

# Define the configuration of the Script
@dataclass
class Conf:
    sequence_length = 100
    stride = 10
    anomaly_range = list(range(1, 10))
    data_range = (-1, 1)
    lpf_cutoff = 25 
    fs = int(100)
    lpf_order = int(5)

@dataclass
class torch_loggers:
    ml_train_path_log = "../../pre-prcossed-data/ml/"
    ml_validation_path_log = "../../pre-prcossed-data/ml/"
    ml_test_path_log = "../../pre-prcossed-data/ml/"
    ml_anomaly_path_log = "../../pre-prcossed-data/ml/"

    dl_train_path_log = "../../pre-prcossed-data/dl/train/"
    dl_validation_path_log = "../../pre-prcossed-data/dl/validation/"
    dl_test_path_log = "../../pre-prcossed-data/dl/test/"
    dl_anomaly_path_log = "../../pre-prcossed-data/dl/anomaly/"

    device = "cpu"


def numpy2torch(x:np.array, data_type: str = "float"):
    __doc__ = r"""
        Here you can also change the data type; however
        it is recommended to use float32 for the data, since we will be using float32 for the
        GAP9 implementation as the first run. 
        Further for quantization, you can use directly float 16 here as well. 
        That could be changed at the inference mode as well. 
        @TODO: @ahmad-mirsalri: Let miirho3ein know if you want to add integers here as well.
    """  
    
    match data_type:
        case "double":
            return torch.from_numpy(x).double()
        case "float":
            return torch.from_numpy(x).float()
        case "half-float":
            return torch.from_numpy(x).half()


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
    minmax_norm = Normalization(feature_range= Conf.data_range, clip = False)
    
    # Define the low pass filter
    lpf = LowPassFilter(cutoff = Conf.lpf_cutoff, fs = Conf.fs, order = Conf.lpf_order)
    
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
    

def dataset_logger(x_tr, x_val, x_test, x_anomaly) -> None:
    # Notice that the application_type is set at the beginning of the script.

    # Transfor Anomalies to tensors.
    if all([isinstance(value, np.ndarray) for value in x_anomaly.values()]):     
        x_tensor_anomalies = {key: numpy2torch(value) for key, value in x_anomaly.items()}
    x_val = numpy2torch(x_val) if isinstance(x_val, np.ndarray)  else x_val
    x_test = numpy2torch(x_test) if isinstance(x_test, np.ndarray) else x_test
    
    if not isinstance(x_val,  torch.Tensor):
        raise ValueError("Validation data should be either a tensor or numpy array.")
    if not isinstance(x_test, torch.Tensor):
        raise ValueError("Test data should be either a tensor or numpy array.")
    if not isinstance(x_tr,  torch.Tensor):
        raise ValueError("Training data should be either a tensor or numpy array.")
    if not isinstance(x_tensor_anomalies,  dict) and all([not isinstance(value, torch.Tensor) for value in x_tensor_anomalies.values()]):
        raise ValueError("Anomalies should be a dictionary of tensors or numpy array.")
    

    match Dataset.model_type:
        case "ml":
            assert torch_loggers.ml_train_path_log is not None, "Please define the path for the training data."
            assert torch_loggers.ml_validation_path_log is not None, "Please define the path for the validation data."
            assert torch_loggers.ml_test_path_log is not None, "Please define the path for the test data."
            assert torch_loggers.ml_anomaly_path_log is not None, "Please define the path for the anomaly data."
            
            
            torch.save(x_val, f"{torch_loggers.ml_validation_path_log}/x_validation.pt")
            torch.save(x_test, f"{torch_loggers.ml_test_path_log}/x_test.pt")
            torch.save(x_tr_aug, f"{torch_loggers.ml_train_path_log}/x_train.pt")
            torch.save(x_tensor_anomalies, f"{torch_loggers.ml_anomaly_path_log}/x_anomalies.pt")

            print("Pre-processed data has been saved Successfully for Machine Learning Models.")      
        case "dl":
            assert torch_loggers.dl_train_path_log is not None, "Please define the path for the training data."
            assert torch_loggers.dl_validation_path_log is not None, "Please define the path for the validation data."
            assert torch_loggers.dl_test_path_log is not None, "Please define the path for the test data."
            assert torch_loggers.dl_anomaly_path_log is not None, "Please define the path for the anomaly data."

            print("Saving samples for the DNN Models.\n ")
            print("Time to have a coffee break :)")
            print("This might take a while. \n Please be patient.")
            

            tr_logger = TorchSampleLogger(save_path = torch_loggers.dl_train_path_log, device = torch_loggers.device)
            val_logger = TorchSampleLogger(save_path = torch_loggers.dl_validation_path_log, device = torch_loggers.device)
            test_logger = TorchSampleLogger(save_path = torch_loggers.dl_test_path_log, device = torch_loggers.device)
            

            tr_logger(x_tr_aug)
            print("Pre-processed Training data has been saved Successfully for DNN Models.")
            val_logger(x_val)
            print("Pre-processed validation data has been saved Successfully for DNN Models.")
            test_logger(x_test)
            print("Pre-processed Test data has been saved Successfully for DNN Models.")

            torch.save(x_tensor_anomalies, f"{torch_loggers.dl_anomaly_path_log}/x_anomalies.pt") 
            

        case "sysid":
            pass
    

if __name__ == "__main__":

    x_tr, x_val, x_test, x_anomaly = from_raw_to_normalized(sequence_length= Conf.sequence_length, stride = Conf.stride)

    # Training Data Augmentation
    x_tr_aug = data_augmentation(x_tr)
    
    # Add Save the data
    dataset_logger(x_tr_aug, x_val, x_test, x_anomaly)


