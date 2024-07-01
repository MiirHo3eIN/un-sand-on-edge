from dataclasses import dataclass
import torch 
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary

from pyshm.dataloader import UnsandDataset
from pyshm.dnnTrain import Train
from pyshm.autoencoders import FcAutoEncoder

workers_num_ = 8
local_path_train = "../../pre-prcossed-data/dl/train/*"
local_path_valid = "../../pre-prcossed-data/dl/validation/*"


@dataclass
class Config: 
    alpha: float
    batch_size: int
    epochs: int
    seq_len: int 
    latent_dim: int


@dataclass
class save_model: 
    save_path: str = "../mid-results/models/"

    



def unsand_DataLoader(device: str, dataset: torch.utils.data.Dataset, workers_num: int, batch_size: int):
    if device == "cpu":
        return  DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory=False,num_workers = workers_num)
    return DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory=True,num_workers = workers_num)

def train_(): 

    Config.alpha = 0.7
    Config.batch_size = 16
    Config.epochs = 10
    
    Config.seq_len = 500
    Config.latent_dim = 32
    #model!
    # arch_id = "aa00"
    model = FcAutoEncoder(input_dim= Config.seq_len,
                          latent_dim=Config.latent_dim, 
                            encoder_activation= "linear",
                            decoder_activation= "linear",)
    
    summary(model, input_size = (1, Config.seq_len), verbose = 1, depth = 5)
    
    
    #train metrics
    tot_train_time = []
    tot_train_loss = []
    tot_valid_loss = []

    device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    unsand_dataset_train = UnsandDataset(local_path_train, device = 'cpu')
    unsand_dataset_valid = UnsandDataset(local_path_valid, device = 'cpu')
    
    unsand_dl_train = unsand_DataLoader(device= device_, 
                                    dataset= unsand_dataset_train, 
                                    workers_num=workers_num_, 
                                    batch_size=Config.batch_size)
    
    unsand_dl_valid = unsand_DataLoader(device= device_, 
                                    dataset= unsand_dataset_valid, 
                                    workers_num=workers_num_, 
                                    batch_size=Config.batch_size)
    
    train_time, train_loss, valid_loss = Train(model = model, epochs = Config.epochs , 
                                               alpha = Config.alpha, output_filter = False, 
                                               path = save_model.save_path, early_stop = False,
                                               pre_trained_path = "")(unsand_dl_train, unsand_dl_valid)
    

    print(f"Training time: {train_time}")
    print(f"Training loss: {train_loss[-1]}")
    print(f"Validation loss: {valid_loss[-1]}")
    
    
if __name__ == "__main__":
    train_()