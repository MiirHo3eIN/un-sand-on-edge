# This script contains the basic training backends for the data compression of AeroSense Surface pressure data.
import matplotlib.pyplot as plt
import numpy as np 
import shutup 
import seaborn as sns
import time 
import torch 
import torch.nn as nn 
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader 
from model_config import ModelConfig

from models import AutoEncoder
from criterion import ReconstructionLossMixed
import utils
from low_pass import LowPassFilterG
import os
shutup.please()


PLOT  = True
train_config = ModelConfig()

device = "cuda" if torch.cuda.is_available() else "cpu" 

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        err = abs((validation_loss - self.min_validation_loss))
        self.min_validation_loss = validation_loss

        if err >= self.min_delta:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def plot_double(ax, x_org, x_rec, sensor, label):
    
    ax.plot(x_org[0, sensor, :].cpu().detach().numpy(), label = 'original data', color = "green")
    ax.plot(x_rec[0, sensor, :].cpu().detach().numpy(), label = label, color = "black")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Cp data")
    ax.set_title(f"Sensor {sensor} | Reconstructed vs original data")

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel("Cp Error", color=color)  # 50% transparent
    ax2.plot(np.subtract(x_org[0, sensor, :].cpu().detach().numpy(),
                         x_rec[0, sensor, :].cpu().detach().numpy()), 
                         color=color, linestyle="dashed", alpha=0.4)

    ax.legend()

def plot_original_vs_reconstructed(x_org: torch.tensor, x_rec: torch.tensor, sensor: list, label:str, path:str ) -> None:
    with sns.plotting_context("poster"):
        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        plot_double(ax[0], x_org, x_rec, sensor[0], label)    
        plot_double(ax[1], x_org, x_rec, sensor[1], label)
        plt.tight_layout()

        filename = "ephoc_" + str(label) + "_original_vs_reconstructed.png"
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')

        plt.show(block=False)
        plt.pause(3)
        plt.close()


class Train(nn.Module):

    def __init__(self, model: object, epochs: int, alpha:float, 
                 output_filter, path:str, early_stop: bool = True,
                 pre_trained_path: str = "",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model.to(device)
        self.epochs = epochs
        self.alpha = alpha
        self.output_filter = output_filter
        self.path = path
        self.e_stop = early_stop

        if self.e_stop == True:
            self.early_stopper = EarlyStopper(patience=3, min_delta=5)


        print("-"*50)
        print("Setup Training...")
        lr = 0.0001
        betas = (0.9, 0.98)
        eps = 1e-8
        criterion_method = ("mse", "smooth_l1")    
        reduction_type = "sum"
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
        self.criterion = ReconstructionLossMixed(criterion_method, reduction_type, alpha = self.alpha)
        self.lp = LowPassFilterG(cutoff = 10, order = 10, linear_filtering = False)

        if pre_trained_path != "":
            #pre-trained model
            self.load_checkpoint(pre_trained_path)
            self.model.train()


        print(f"Used Device: {device}")
        print(f"Optimizer | lr: {lr} | betas: {betas} | eps: {eps}")
        print(f"Criterion alpha: {self.alpha}")
        print("\n")

    def save_checkpoint(self, epoch, loss):
    
        model_number = utils.generate_hexadecimal()
   
        #save checkpoint
        # Additional information
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                }, f"{self.path}{epoch}_checkpoint.pt")


        return model_number 
    
    def load_checkpoint(self, path: str):

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print("\n")
        print("Model loaded from pre-trained, Epoch: " + str(epoch) + " Loss: " + str(loss))
        print("\n")

        return epoch, loss
   
    def forward_propagation(self, x_batch, y_batch, idx, epoch ):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if self.model.RVQ == True:
            y_train, self.indices, self.commit_loss, self.codebooks = self.model.forward(x_batch.float())
            # print(self.codebooks.size()) [8,128,100]
            # print(type(self.codebooks)) torch.Tensor
        else:
            y_train = self.model.forward(x_batch.float())
        if self.output_filter:
            y_train  = self.lp(y_train)
        train_loss = self.criterion(y_train.float(), y_batch.float())

        if self.model.RVQ == True:
            train_loss = train_loss + (self.alpha * torch.mean(self.commit_loss))
        
        if idx == 1 and epoch%10 == 0 and PLOT == 1:       
            plot_original_vs_reconstructed(y_batch, y_train, sensor = [5, 15], label = f"Epoch {epoch} | train", path = self.path)
        
        return train_loss

    def back_propagation(self, train_loss):
    
        # Backpropagation
        self.optimizer.zero_grad()
        train_loss.backward()
        # max_norm = 0.0005
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        # # Update the parameters
        self.optimizer.step()


    def evaluate(self, x_batch, y_batch, idx, epoch):
        
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.no_grad():
            self.model.eval()

            if self.model.RVQ == True:
                y_valid, indices, commit_loss, _ = self.model.forward(x_batch.float())
            else:
                y_valid = self.model.forward(x_batch.float())

            if self.output_filter:
                y_valid = self.lp(y_valid)
            if idx == 1 and epoch%10 == 0 and PLOT == 1:
                plot_original_vs_reconstructed(y_batch, y_valid, sensor = [5, 15], label = f"Epoch {epoch} | validation", path = self.path)
            valid_loss = self.criterion(y_valid.float(), y_batch.float())

            return valid_loss


    def forward(self, train_x, valid_x):
        
        
        train_epoch_loss, valid_epoch_loss = [], []
        train_batch_loss, valid_batch_loss = [], []
        train_total_loss, valid_total_loss = [], []
        
        print("-"*50)
        print("Starting Training...")
        time_start = time.time()
        print(time_start)

        
         
        for epoch in np.arange(0, self.epochs):
            #print("Inside the for loop")
            print(f"Epoch: {epoch+1}/{self.epochs}")

            ### TRAINING PHASE ###
            idx = 0
            for x_batch, y_batch in (train_x):
                # print("x_batch shape: ", x_batch.shape)
                # print("y_batch shape: ", y_batch.float().shape)
                idx += 1
                train_loss = self.forward_propagation(x_batch, y_batch, idx = idx, epoch = epoch)

                train_batch_loss += [train_loss]
                train_epoch_loss += [train_loss.item()]
                self.back_propagation(train_loss)

            idx = 0 
            ### VALIDATION PHASE ###
            for x_batch, y_batch in (valid_x):
                    idx += 1 
                    valid_loss = self.evaluate(x_batch, y_batch, idx = idx, epoch = epoch)
                    
                    valid_batch_loss += [valid_loss]
                    valid_epoch_loss += [(valid_loss.item())]

            print(f"\t Train loss = {sum(train_epoch_loss)/len(train_epoch_loss):.05}, \
                    Validation Loss = {sum(valid_epoch_loss)/len(valid_epoch_loss):.05}")
            if self.model.RVQ == True:
                print("RVQ "
                + f"cmt loss: {torch.mean(self.commit_loss).item():.3f} | "
                + f"active %: {self.indices.unique().numel() / self.model.codebook_size * 100:.3f}"
                )

            train_total_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))
            valid_total_loss.append(sum(valid_epoch_loss)/len(valid_epoch_loss))

            #save checkpoint
            self.save_checkpoint(epoch, sum(train_epoch_loss)/len(train_epoch_loss))

            #early stop condition
            if self.e_stop == True and self.early_stopper.early_stop(sum(train_epoch_loss)/len(train_epoch_loss)):
                break

            train_epoch_loss, valid_epoch_loss = [], []
            train_batch_loss, valid_batch_loss = [], []
            idx = 0

        time_end = time.time()
        train_time = time_end - time_start
        
        return train_time, train_total_loss, valid_total_loss, self.codebooks
    



def save_model(model, path, arch_id, train_loss, valid_loss):
    
    model_number = utils.generate_hexadecimal()
   
    torch.save(model.state_dict(), f"{path}{model_number}.pt")

    np.save(f"{path}{model_number}_train_loss.npy", np.array(train_loss))
    np.save(f"{path}{model_number}_valid_loss.npy", np.array(valid_loss))
    print(f"Saved the model: {model_number} with architecture Id: {arch_id}")

    return model_number 

def save_training_results(train_config):
    
    data = train_config.__dict__
    utils.write_to_csv(data)
    print("Saved the training results to the csv file.")




def plot_loss(train_loss, valid_loss, model_id, path:str):
    
    with sns.plotting_context('poster'): 
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(train_loss, color = 'green', label = 'Train ')
        ax.plot(valid_loss, color = 'red', label = 'Valid ')

        ax.set_xlabel('epochs')
        ax.set_ylabel('Loss')

        ax.set_title(f"Loss trend for Model {model_id}")
        ax.legend()
        plt.tight_layout()

        filename = "plot_loss.png"
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')

        plt.show(block=False)
        plt.pause(3)
        plt.close()

