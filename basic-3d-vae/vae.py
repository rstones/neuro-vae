import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class VAE(pl.LightningModule):

    def __init__(self):
        super().__init__()
        
        # define the ConvNet_Down, AE and ConvNet_Up bits here

    def forward(self, x):
        # link the bits together here
        pass

    def training_step(self, batch, batch_idx):
        noisy_data = batch['data']
        clean_data = batch['seg'] # seg is batchgenerators default for labels
        pred = self(noisy_data)
        criterion = nn.BCELoss()
        loss = cirterion(pred, clean_data)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy_data = batch['data']
        clean_data = batch['seg'] # seg is batchgenerators default for labels
        pred = self(noisy_data)
        criterion = nn.BCELoss()
        loss = criterion(pred, clean_data)
        self.log('val_loss', loss)
        return loss


    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1.e-3)
        return optimizer
