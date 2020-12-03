import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Convnet_Down(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        #dimensions
        self.kernel = [2,2,2]
        self.stride = [2,2,2]
        self.padding = [0,0,0]
        self.pool_kernel = [1,1,1]
        self.pool_stride = [1,1,1]
        
        #Architecture
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, self.kernel, self.stride, self.padding),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.MaxPool3d(self.pool_kernel, self.pool_stride))
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, self.kernel, self.stride, self.padding),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.MaxPool3d(self.pool_kernel, self.pool_stride),
            nn.Dropout(0.3))
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, self.kernel, self.stride, self.padding),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.MaxPool3d(self.pool_kernel, self.pool_stride),
            nn.Dropout(0.3))
        self.fc1 = nn.Sequential(
            nn.Linear(128*11*11*11, 11*11*11),
            nn.LeakyReLU(),
            nn.Dropout(0.4))
        
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        step1 = self.conv1(x)
        step2 = self.conv2(step1)
        step3 = self.conv3(step2)
        step3 = step3.view(-1,128*11**3)
        out = self.fc1(step3)
        return out