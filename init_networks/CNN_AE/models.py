
# This scripts contains the models for the data compression of AeroSense Surface pressure data.
import torch 
from torch import nn
from torchinfo import summary
from basic_tools import (
    single_conv1d_block, 
    single_trans_conv1d_block, 
    ConvBlock, 
    TransConvBlock
)
import typing as tp
import sys
# sys.path.append("../vector-quantize-pytorch")
# sys.path.append("../vector-quantize-pytorch/vector-quantize-pytorch")
# from vector_quantize_pytorch import ResidualVQ

VECTOR_QUANTIZER = False


""" 
This is the main class of the AutoEncoder. Encoder and Decoder will be defined by the given architecture ID. 

Parameters: 

    - Architecture ID: 4 digit integer, First two for Encoder and last two for Decoder.
"""

class AutoEncoder(nn.Module): 

    def __init__(self, arch_id: str, c_in: int = 36, c_factor: int = 4,
                 RVQ: bool = False, codebook_size: int = 128): 
        super().__init__()
        self.arch_id = arch_id 
        self.c_in = c_in
        self.c_out = int(c_in / c_factor)
        self.c_factor = c_factor
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.RVQ = RVQ
        self.codebook_size = codebook_size
        """
        Returns the encoder of the AutoEncoder. 

        Parameters
        ----------
        arch_id : str
            Architecture ID of the AutoEncoder. 
        c_in: int
            Number of input channels
        c_factor: int
            Compression factor in 2D, channels and samples. Select carefully, it affects the output size
        sequential_len_enc: int
            Number of Convolutional layers, used to decrease/increase the network complexity
        """


    def forward(self, x: torch.Tensor): 
        x = self.encoder(x)

        x = self.decoder(x)

        return x
    
    def get_encoder(self): 
        """
        Returns the encoder of the AutoEncoder. 

        Parameters
        ----------
        arch_id : str
            Architecture ID of the AutoEncoder. 
        """
        
        return Encoder_aa00(c_in = self.c_in, c_factor = self.c_factor, sequetial_len = 2)


    def get_decoder(self):
        """
        Returns the decoder of the AutoEncoder. 

        Parameters
        ----------
        arch_id : str
            Architecture ID of the AutoEncoder. 
        """
        
        return Decoder_aa00(c_in = self.c_out, c_factor = self.c_factor, sequetial_len = 2)
           
class Encoder_aa00(nn.Module):
    
    def __init__(self, c_in, c_factor: int = 4, sequetial_len: int = 3): 
        super().__init__()
        self.c_in = c_in
        cout = self.c_in
        self.sequetial_len = sequetial_len

        model: tp.List[nn.Module] = []

        model += [ConvBlock(c_in = int(cout), c_out = int(cout / 2), 
                                kernel_size_residual = 3, kernel_size_down_sampling = 7, 
                                stride_in = 1, strid_down_sampling = 2, padding_mode=str('zeros'), layers = self.sequetial_len)] 
        for _ in range(2, c_factor, 2):
            cout = cout / 2
            model += [ConvBlock(c_in = int(cout), c_out = int(cout / 2), 
                                kernel_size_residual = 3, kernel_size_down_sampling = 7, 
                                stride_in = 1, strid_down_sampling = 2, padding_mode=str('zeros'), layers = self.sequetial_len)]

        model += [ConvBlock(c_in = int(cout / 2), c_out = int(cout / 2), 
                                kernel_size_residual = 3, kernel_size_down_sampling = 7, 
                                stride_in = 1, strid_down_sampling = 2, padding_mode=str('zeros'), layers = self.sequetial_len)]

        self.conv = nn.Sequential(*model)

    def forward(self, x: torch.Tensor): 
        x = self.conv(x)
        return x    
    
class Decoder_aa00(nn.Module): 

    def __init__(self, c_in, c_factor: int = 4, sequetial_len: int = 3): 
        super().__init__()
        self.c_in = c_in
        cout = self.c_in
        self.sequetial_len = sequetial_len

        model: tp.List[nn.Module] = []
        
        model += [TransConvBlock(
            c_in=int(c_in), c_out=int(c_in), 
            kernel_size=3, kernel_size_up_sampling=4, 
            stride_residual=1, stride_up_sampling=2, 
            padding=1, output_padding=0, layers = self.sequetial_len)]
        
        model += [TransConvBlock(
            c_in=int(c_in), c_out=int(cout*2), 
            kernel_size=3, kernel_size_up_sampling=4, 
            stride_residual=1, stride_up_sampling=2, 
            padding=1, output_padding=0, layers = self.sequetial_len)] 
        
        for _ in range(2, c_factor, 2):
            cout = cout * 2
            model += [TransConvBlock(
                c_in=int(cout), c_out=int(cout*2), 
                kernel_size=3, kernel_size_up_sampling=4, 
                stride_residual=1, stride_up_sampling=2, 
                padding=1, output_padding=0, layers = self.sequetial_len)] 
            
        model += [nn.Conv1d(in_channels = int(cout*2), out_channels = int(cout*2), 
                            kernel_size = 7, stride = 1, padding = 3)]

        self.deconv = nn.Sequential(*model)
        
    def forward(self, x: torch.Tensor): 
        x = self.deconv(x)
        return x

if __name__ == "__main__": 

    rand_input = torch.rand(1, 36, 800)
    model  = AutoEncoder(arch_id = "aa00", c_in = 36, RVQ=False)
    summary(model, input_size = (1, 36, 800), verbose = 1, depth = 5)

    print("End of Test!")

    modelpath = "newcnnae.onnx"#'conv_tasnet.onnx'
    torch.onnx.export(model, rand_input, modelpath, input_names=["input"], output_names=["output"])
    

