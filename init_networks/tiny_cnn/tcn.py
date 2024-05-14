import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnx_tool
import numpy

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # 1st Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=9, padding='same')
        self.relu1 = nn.ReLU()
        
        # 1st MaxPooling Layer
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 2nd Convolutional Layer
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, padding='same')
        self.relu2 = nn.ReLU()
        
        # 2nd MaxPooling Layer
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * 10, 1024)  # Assuming the input length is 100
        self.fc2 = nn.Linear(1024, 1)
        
        # Sigmoid Activation Function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x


model = ConvNet()
print(model)

x = torch.rand( 1,1,41)
modelpath = "tiny_tcn.onnx"#'conv_tasnet.onnx'
torch.onnx.export(model, x, modelpath, input_names=["input"], output_names=["output"])