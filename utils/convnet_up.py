import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Convnet_Up(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        #dimensions
        self.kernel = [2,2,2]
        self.stride = [2,2,2]
        self.padding = [0,0,0]
        self.output_padding = [0,0,0]
        self.pool_kernel = [1,1,1]
        self.pool_stride = [1,1,1]
        
        #Architecture
        self.fc_up1 = nn.Sequential(
            nn.Linear(11*11*11, 128*11*11*11),
            nn.LeakyReLU())
        self.transpose1 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, self.kernel, self.stride, self.padding, self.output_padding),
            nn.ReLU(),
            #nn.BatchNorm3d(64),
            #nn.LeakyReLU(),
            #nn.MaxUnpool3d(self.pool_kernel, self.pool_stride),
            nn.Dropout(0.3))
        self.transpose2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, self.kernel, self.stride, self.padding),
            nn.ReLU(),
            #nn.BatchNorm3d(32),
            #nn.LeakyReLU(),
            #nn.MaxUnpool3d(self.pool_kernel, self.pool_stride),
            nn.Dropout(0.3))
        self.transpose3 = nn.Sequential(
            nn.ConvTranspose3d(32, 1, self.kernel, self.stride, self.padding),
            nn.ReLU())
            #nn.LeakyReLU(),
            #nn.MaxUnpool3d(self.pool_kernel, self.pool_stride))
        
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        _in = self.fc_up1(x)
        _in = _in.view([-1,128,11,11,11])
        step1 = self.transpose1(_in)
        step2 = self.transpose2(step1)
        out = self.transpose3(step2)

        return out